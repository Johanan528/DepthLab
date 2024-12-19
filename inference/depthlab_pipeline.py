import os, sys
parent_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from typing import Any, Dict, Union
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm.auto import tqdm
from PIL import Image
import torch.nn as nn
from diffusers import (
    DiffusionPipeline,
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import CLIPImageProcessor
from transformers import CLIPVisionModelWithProjection
from src.models.mutual_self_attention import ReferenceAttentionControl
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
import cv2
from diffusers.utils import BaseOutput
from transformers import CLIPTextModel, CLIPTokenizer
from utils.image_util import resize_max_res, resize_max_res_cv2, colorize_depth_maps, chw2hwc, Disparity_Normalization_mask_scale,get_filled_depth
from scipy.interpolate import griddata
sys.path.pop(0)
class DepthPipelineOutput(BaseOutput):
    """
    Output class for Marigold monocular depth prediction pipeline.

    Args:
        depth_np (`np.ndarray`):
            Predicted depth map, with depth values in the range of [0, 1].
        depth_colored (`PIL.Image.Image`):
            Colorized depth map, with the shape of [3, H, W] and values in [0, 1].
        uncertainty (`None` or `np.ndarray`):
            Uncalibrated uncertainty(MAD, median absolute deviation) coming from ensembling.
    """
    depth_np: np.ndarray
    depth_norm: Image.Image
    depth_colored: Image.Image
    depth_pred_numpy_origin:np.ndarray

class DepthLabPipeline(DiffusionPipeline):
    # two hyper-parameters
    rgb_latent_scale_factor = 0.18215
    depth_latent_scale_factor = 0.18215
    
    def __init__(self,
        reference_unet:UNet2DConditionModel,
        denoising_unet:UNet2DConditionModel_main,
        mapping_layer:My_proj,
        vae:AutoencoderKL,
        text_encoder:CLIPTextModel,
        tokenizer:CLIPTokenizer,
        image_enc:CLIPVisionModelWithProjection,
        scheduler:DDIMScheduler,
    ):
        super().__init__()
            
        self.register_modules(
            reference_unet=reference_unet,
            denoising_unet=denoising_unet, 
            mapping_layer=mapping_layer,      
            vae=vae,
            image_enc=image_enc,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder

        )
        self.empty_text_embed = None
        self.clip_image_processor = CLIPImageProcessor()
        
    @torch.no_grad()
    def __call__(self,
        input_image: str,
        denosing_steps: int = 20,
        processing_res: int = 768,
        match_input_res: bool = True,
        batch_size: int = 0,
        color_map: str="Spectral",
        show_progress_bar: bool = True,
        ensemble_kwargs: Dict = None,
        depth_numpy_origin = None,
        mask_origin = None,
        guidance_scale = 1,
        normalize_scale = 1,
        strength=0.8,
        blend=True
        ) -> DepthPipelineOutput:
        
        # inherit from thea Diffusion Pipeline
        device = self.image_enc.device

        clip_image_unresize = input_image

        try:
            depth_origin = depth_numpy_origin
        except:
            raise NotImplementedError
        assert depth_origin.min() >= 0.
        
        mask_origin = np.array(mask_origin)
        mask_origin [mask_origin <0.5] = 0.
        mask_origin [mask_origin >0.5] = 1.
        
        clip_image = self.clip_image_processor.preprocess(
            clip_image_unresize, return_tensors="pt"
        ).pixel_values
        clip_image_embeds = self.image_enc(
            clip_image.to(device, dtype=self.image_enc.dtype)
        ).image_embeds
        encoder_hidden_states = clip_image_embeds.unsqueeze(1)
        encoder_hidden_states =self.mapping_layer(encoder_hidden_states )
        prompt = ""
        text_inputs =self.tokenizer(
            prompt,
            padding="do_not_pad",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids.to(device) 
        empty_text_embed = self.text_encoder(text_input_ids)[0].to(self.dtype)
        uncond_encoder_hidden_states = empty_text_embed.repeat((1, 1, 1))[:,0,:].unsqueeze(0)

        do_classifier_free_guidance = True

        # adjust the input resolution.
        if not match_input_res:
            assert (
                processing_res is not None                
            )," Value Error: `resize_output_back` is only valid with "
        
        assert processing_res >= 0
        assert denosing_steps >= 1
        
        # --------------- Image Processing ------------------------
        # Resize image
        original_H, original_W = depth_origin.shape
        if processing_res > 0:
            image = resize_max_res(
                input_image, max_edge_resolution=processing_res, resample=Image.BICUBIC
            )
            depth = resize_max_res_cv2(
                depth_origin, max_edge_resolution=processing_res, interpolation=cv2.INTER_LINEAR
            )
            mask = resize_max_res_cv2(
                mask_origin , max_edge_resolution=processing_res, interpolation=cv2.INTER_NEAREST
            )
        # Convert the image to RGB, to 1. reomve the alpha channel.
        image = np.array(image)
        
        # Normalize RGB Values.
        rgb = np.transpose(image,(2,0,1))
        rgb_norm = rgb / 255.0 * 2.0 - 1.0
        rgb_norm = torch.from_numpy(rgb_norm).to(self.dtype).to(device)
        assert rgb_norm.min() >= -1.0 and rgb_norm.max() <= 1.0

        # depth
        depth = torch.from_numpy(depth).to(self.dtype).to(device)[None]
        assert depth.min() >= 0.

        # mask
        mask = torch.from_numpy(mask).to(self.dtype).to(device)[None]
        assert mask.min() >= 0. and mask.max() <= 1.
        
        # ----------------- predicting depth -----------------
        single_rgb_dataset = TensorDataset(rgb_norm[None], depth[None], mask[None])
        
        # find the batch size
        if batch_size>0:
            _bs = batch_size
        else:
            _bs = 1
        
        single_rgb_loader = DataLoader(single_rgb_dataset,batch_size=_bs,shuffle=False)

        # classifier guidance
        if do_classifier_free_guidance:
            encoder_hidden_states = torch.cat(
                [uncond_encoder_hidden_states, encoder_hidden_states], dim=0
            )

        reference_control_writer = ReferenceAttentionControl(
            self.reference_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        reference_control_reader = ReferenceAttentionControl(
            self.denoising_unet,
            do_classifier_free_guidance=do_classifier_free_guidance,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )

        if show_progress_bar:
            iterable_bar = tqdm(
                single_rgb_loader, desc=" " * 2 + "Inference batches", leave=False
            )
        else:
            iterable_bar = single_rgb_loader
        for batch in iterable_bar:
            (image, depth, mask)= batch  
            depth_pred_raw, max_value, min_value = self.single_infer(
                image = image,
                depth = depth,
                mask = mask,
                num_inference_steps = denosing_steps,
                show_pbar = show_progress_bar,
                guidance_scale = guidance_scale,
                encoder_hidden_states = encoder_hidden_states,
                reference_control_writer = reference_control_writer,
                reference_control_reader = reference_control_reader,
                strength = strength,
                blend = blend,
                normalize_scale = normalize_scale,
                generator = None,
            )
        
        depth_pred = depth_pred_raw
        torch.cuda.empty_cache()     

        # ----------------- Post processing -----------------
        depth_pred = (depth_pred * (max_value - min_value) + min_value)
        depth_pred_numpy = depth_pred.detach().cpu().numpy().squeeze()
        depth_pred_numpy = cv2.resize(depth_pred_numpy.astype(float), (original_W, original_H))
        depth_pred_numpy = depth_pred_numpy.clip(min=0.)
        depth_pred_numpy_origin=depth_pred_numpy.copy()
        depth_pred_norm = (depth_pred_numpy - depth_pred_numpy.min()) / (depth_pred_numpy.max() - depth_pred_numpy.min())
        depth_pred_colored = colorize_depth_maps(
            depth_pred_norm, 0, 1, cmap="Spectral"
        ).squeeze()  # [3, H, W], value in (0, 1)
        depth_pred_colored = (depth_pred_colored * 255).astype(np.uint8)
        depth_pred_colored = Image.fromarray(chw2hwc(depth_pred_colored))

        return DepthPipelineOutput(
            depth_np = depth_pred_numpy,
            depth_norm = depth_pred_norm,
            depth_colored = depth_pred_colored,
            depth_pred_numpy_origin=depth_pred_numpy_origin
        )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)

        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]

        return timesteps, num_inference_steps - t_start
        
    @torch.no_grad()
    def single_infer(self,
        image: torch.Tensor,
        depth: torch.Tensor,
        mask: torch.Tensor,
        num_inference_steps: int,
        show_pbar: bool,
        guidance_scale: float,
        encoder_hidden_states: torch.Tensor,
        reference_control_writer: ReferenceAttentionControl,
        reference_control_reader: ReferenceAttentionControl,
        strength: float,
        blend: bool,
        normalize_scale: float,
        generator=None,
    ):
        do_classifier_free_guidance = True
        try:
            device = self.image_enc.device
        except:
            import pdb; pdb.set_trace()
        h, w = image.shape[-2:]

        # Set timesteps: inherit from the diffuison pipeline
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps, _ = self.get_timesteps(num_inference_steps, strength, device)
        
        # Encode image
        rgb_latent = self.encode_RGB(image) #


        min_value = depth[mask == 0].min()
        max_value = depth[mask == 0].max()

        depth = Disparity_Normalization_mask_scale(depth, min_value, max_value, scale=normalize_scale)
        masked_depth = depth.clone()
        masked_depth[mask == 1] = 0

        masked_depth = get_filled_depth(masked_depth.squeeze().cpu().numpy(), mask.squeeze().cpu().numpy(),method='linear')
        masked_depth = torch.from_numpy(masked_depth).unsqueeze(0).unsqueeze(0).float().to(device)
        masked_depth = masked_depth.repeat(1,3,1,1)
        masked_depth_latent = self.encode_depth(masked_depth)
        depth = depth.repeat(1,3,1,1)
        depth_latent = self.encode_depth(depth)

        # process mask
        mask_latents=self.encode_RGB(mask.repeat(1,3,1,1).to(rgb_latent.dtype) * 2 - 1)
        mask_down = torch.nn.functional.interpolate(mask, size=(h//8, w//8),mode='nearest')
        noise = torch.randn_like(depth_latent)
  
        if strength < 1.:
            noisy_depth_latent = self.scheduler.add_noise(depth_latent, noise, timesteps[:1])
        else:
            noisy_depth_latent = torch.randn(
                depth_latent.shape,
                device=device,
                dtype=depth_latent.dtype,
                generator=generator,
            )
        
        # Denoising loop
        if show_pbar:
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
        else:
            iterable = enumerate(timesteps)
        
        for i, t in iterable:
            if i == 0:
                self.reference_unet(
                    rgb_latent.repeat(
                        (2 if do_classifier_free_guidance else 1), 1, 1, 1
                    ),
                    torch.zeros_like(t),
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )
                reference_control_reader.update(reference_control_writer)
            unet_input = torch.cat([noisy_depth_latent, mask_latents, masked_depth_latent], dim=1)
            noise_pred = self.denoising_unet(
                unet_input.repeat(
                        (2 if do_classifier_free_guidance else 1), 1, 1, 1
                    ).to(dtype=self.denoising_unet.dtype), t, encoder_hidden_states=encoder_hidden_states
            ).sample  # [B, 4, h, w]
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )
            # compute the previous noisy sample x_t -> x_t-1
            noisy_depth_latent = self.scheduler.step(noise_pred, t, noisy_depth_latent).prev_sample.to(self.dtype)
            if blend:
            # Blend diffusion https://arxiv.org/abs/2111.14818
                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    depth_latent_step = self.scheduler.add_noise(
                        depth_latent, noise, torch.tensor([noise_timestep])
                    )
                    mask_blend = mask_down.repeat(1,4,1,1).float()
                    noisy_depth_latent = (1 - mask_blend) * depth_latent_step + mask_blend * noisy_depth_latent

        reference_control_reader.clear()
        reference_control_writer.clear()
        torch.cuda.empty_cache()
        depth = self.decode_depth(noisy_depth_latent)
        # depth = torch.clip(depth, -1.0, 1.0)

        # shift
        depth = (depth + normalize_scale) / (normalize_scale*2)
        return depth, max_value, min_value
        
    
    def encode_RGB(self, rgb_in: torch.Tensor) -> torch.Tensor:
        """
        Encode RGB image into latent.

        Args:
            rgb_in (`torch.Tensor`):
                Input RGB image to be encoded.

        Returns:
            `torch.Tensor`: Image latent.
        """

        
        # encode
        h = self.vae.encoder(rgb_in)

        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        rgb_latent = mean * self.rgb_latent_scale_factor
        
        return rgb_latent
    
    def decode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        depth_latent = depth_latent / self.depth_latent_scale_factor
        
        depth_latent = depth_latent
        try:
            z = self.vae.post_quant_conv(depth_latent)
            stacked = self.vae.decoder(z)
        except:
            stacked = self.vae.decode(depth_latent)
        # mean of output channels
        depth_mean = stacked.mean(dim=1, keepdim=True)
        return depth_mean


    def encode_depth(self, depth_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode depth latent into depth map.

        Args:
            depth_latent (`torch.Tensor`):
                Depth latent to be decoded.

        Returns:
            `torch.Tensor`: Decoded depth map.
        """
        # scale latent
        h_disp = self.vae.encoder(depth_latent)
        moments_disp = self.vae.quant_conv(h_disp)
        mean_disp, logvar_disp = torch.chunk(moments_disp, 2, dim=1)
        disp_latents = mean_disp *self.depth_latent_scale_factor
        return disp_latents



