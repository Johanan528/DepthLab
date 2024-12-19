import argparse
import logging
import os
import random
import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
import cv2
import torch.nn as nn
from transformers import CLIPTextModel, CLIPTokenizer
from src.models.unet_2d_condition import UNet2DConditionModel
from src.models.unet_2d_condition_main import UNet2DConditionModel_main
from src.models.projection import My_proj
from transformers import CLIPVisionModelWithProjection
from inference.depthlab_pipeline import DepthLabPipeline
from utils.seed_all import seed_all
from utils.image_util import get_filled_for_latents

def load_and_process_mask(mask_path):
    image = Image.open(mask_path).convert('L')
    mask = np.array(image)
    mask = mask / 255.0
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return mask
if "__main__" == __name__:
    logging.basicConfig(level=logging.INFO)

    # -------------------- Arguments --------------------
    parser = argparse.ArgumentParser(
        description="Run single-image depth estimation using Marigold."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )

    # inference setting
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=50,  # quantitative evaluation uses 50 steps
        help="Diffusion denoising steps, more steps results in higher accuracy but slower inference speed.",
    )

    # resolution setting
    parser.add_argument(
        "--processing_res",
        type=int,
        default=0,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--normalize_scale",
        type=float,
        default=1,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.8,
        help="Maximum resolution of processing. 0 for using input image resolution. Default: 0.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--denoising_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--mapping_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--reference_unet_path", type=str, required=True, help="Path to depth inpainting model."
    )
    parser.add_argument(
        "--input_image_paths",
        nargs='+',
        default=None,
        help="input_image_paths",
    )
    parser.add_argument(
        "--known_depth_paths",
        nargs='+',
        default=None,
        help="known_depth_paths",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--masks_paths",
        nargs='+',
        default=None,
        help="masks_paths",
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    denoise_steps = args.denoise_steps
    processing_res = args.processing_res
    seed = args.seed
    output_dir_color = os.path.join(output_dir, "depth_colored")
    output_dir_npy = os.path.join(output_dir, "depth_npy")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir_color, exist_ok=True)
    os.makedirs(output_dir_npy, exist_ok=True)
    logging.info(f"output dir = {output_dir}")
    if args.input_image_paths is not None:
        assert len(args.input_image_paths) == len(args.known_depth_paths)
        assert len(args.input_image_paths) == len(args.masks_paths)
    input_image_paths = args.input_image_paths
    known_depth_paths = args.known_depth_paths
    masks_paths = args.masks_paths
    print(f"arguments: {args}")
    if seed is None:
        import time

        seed = int(time.time())
    seed_all(seed)

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"device = {device}")

    # -------------------- Model --------------------

    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='vae')
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path,
                                                     subfolder='text_encoder')
    denoising_unet = UNet2DConditionModel_main.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=12, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    reference_unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path,subfolder="unet",
                                                    in_channels=4, sample_size=96,
                                                    low_cpu_mem_usage=False,
                                                    ignore_mismatched_sizes=True)
    image_enc = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    mapping_layer=My_proj()


    mapping_layer.load_state_dict(
        torch.load(args.mapping_path, map_location="cpu"),
        strict=False,
        )
    mapping_device = torch.device("cuda")
    mapping_layer.to(mapping_device )
    reference_unet.load_state_dict(
                torch.load(args.reference_unet_path, map_location="cpu"),
        )
    denoising_unet.load_state_dict(
        torch.load(args.denoising_unet_path, map_location="cpu"),
        strict=False,
        )
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path,subfolder='tokenizer')
    scheduler = DDIMScheduler.from_pretrained(args.pretrained_model_name_or_path,subfolder='scheduler')
    pipe = DepthLabPipeline(reference_unet=reference_unet,
                                       denoising_unet = denoising_unet,  
                                       mapping_layer=mapping_layer,
                                       vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       image_enc=image_enc,
                                       scheduler=scheduler,
                                       ).to('cuda')
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ImportError:
        logging.debug("run without xformers")

    # -------------------- Inference and saving --------------------
    with torch.no_grad():
        for i in range(len(input_image_paths)):
            input_image_path = input_image_paths[i]
            mask_path = masks_paths[i]
            known_depth_path = known_depth_paths[i]

            # save path
            rgb_name_base = os.path.splitext(os.path.basename(input_image_path))[0]
            pred_name_base = rgb_name_base + "_pred"
            npy_save_path = os.path.join(output_dir_npy, f"{pred_name_base}.npy")
            colored_save_path = os.path.join(
                output_dir_color, f"{pred_name_base}_colored.png"
            )

            input_image = Image.open(input_image_path)
            try:
                mask = np.load(mask_path)
                mask[mask>0.5]=1
                mask[mask<0.5]=0
            except:
                mask = load_and_process_mask(mask_path)
            depth_numpy=np.load(known_depth_path)


            depth_numpy=get_filled_for_latents(mask,depth_numpy)
            pipe_out = pipe(
                input_image,
                denosing_steps = denoise_steps,
                processing_res = processing_res,
                match_input_res = True,
                batch_size =1,
                color_map = "Spectral",
                show_progress_bar = True,
                depth_numpy_origin = depth_numpy,
                mask_origin = mask,
                guidance_scale = 1,
                normalize_scale = args.normalize_scale,
                strength = args.strength,
                blend = args.blend)

            depth_pred: np.ndarray = pipe_out.depth_np
            if os.path.exists(colored_save_path):
                logging.warning(f"Existing file: '{colored_save_path}' will be overwritten")

            np.save(npy_save_path,depth_pred)
            pipe_out.depth_colored.save(colored_save_path)