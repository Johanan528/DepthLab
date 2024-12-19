import matplotlib
import numpy as np
import torch
from PIL import Image
import cv2
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from scipy.interpolate import griddata
from enum import Enum
import os
from scipy.interpolate import griddata as interp_grid

class DepthFileNameMode(Enum):
    """Prediction file naming modes"""

    id = 1  # id.png
    rgb_id = 2  # rgb_id.png
    i_d_rgb = 3  # i_d_1_rgb.png
    rgb_i_d = 4

def get_filled_depth(depth, mask, method):
    x, y = np.indices(depth.shape)
    known_points = mask == 0
    points = np.array((x[known_points], y[known_points])).T
    values = depth[known_points]
    # print(values.min(), values.max())
    all_points = np.array((x.flatten(), y.flatten())).T
    filled_depth = griddata(points, values, all_points, method=method, fill_value=0)
    return filled_depth.reshape(depth.shape).astype(np.float32)
def resize_max_res(img: Image.Image, max_edge_resolution: int, resample=Image.BICUBIC) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `Image.Image`: Resized image.
    """
    
    original_width, original_height = img.size
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = img.resize((new_width, new_height), resample=resample)
    return resized_img

def resize_max_res_cv2(img: np.ndarray, max_edge_resolution: int, interpolation=cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.
    Args:
        img (`np.ndarray`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
    Returns:
        `np.ndarray`: Resized image.
    """
    
    original_height, original_width = img.shape[:2]
    
    downscale_factor = min(
        max_edge_resolution / original_width, max_edge_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = cv2.resize(img, (new_width, new_height), interpolation=interpolation)
    return resized_img

def resize_max_res_tensor(input_tensor,recom_resolution=768):
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor to be resized. Expected shape: [B, C, H, W]
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `torch.Tensor`: Resized image.
    """
    assert 4 == input_tensor.dim(), f"Invalid input shape {input_tensor.shape}"

    original_height, original_width =input_tensor.shape[-2:]
    downscale_factor = min(
        recom_resolution / original_width, recom_resolution / original_height
    )

    new_width = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    resized_img = resize(input_tensor, (new_height, new_width), InterpolationMode.BILINEAR, antialias=True)
    return resized_img

def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored

def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

def Disparity_Normalization_mask_scale(disparity, min_value, max_value, scale=0.6):
    min_value = min_value.view(-1, 1, 1, 1)
    max_value = max_value.view(-1, 1, 1, 1)
    normalized_disparity = ((disparity - min_value) / (max_value - min_value + 1e-6) - 0.5) * scale*2
    return normalized_disparity

def get_pred_name(rgb_basename, name_mode, suffix=".png"):
    if DepthFileNameMode.rgb_id == name_mode:
        pred_basename = "pred_" + rgb_basename.split("_")[1]
    elif DepthFileNameMode.i_d_rgb == name_mode:
        pred_basename = rgb_basename.replace("_rgb.", "_pred.")
    elif DepthFileNameMode.id == name_mode:
        pred_basename = "pred_" + rgb_basename
    elif DepthFileNameMode.rgb_i_d == name_mode:
        pred_basename = "pred_" + "_".join(rgb_basename.split("_")[1:])
    else:
        raise NotImplementedError
    # change suffix
    pred_basename = os.path.splitext(pred_basename)[0] + suffix

    return pred_basename

def get_filled_for_latents(mask, sparse_depth):
    H, W = mask.shape
    known_depth_y_coords, known_depth_x_coords = np.where(np.array(mask)== 0)
    known_depth_coords = np.stack([known_depth_x_coords, known_depth_y_coords], axis=-1)
    known_depth = sparse_depth[known_depth_y_coords, known_depth_x_coords]
    x, y = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    grid = np.stack((x,y), axis=-1).reshape(-1,2)

    dense_depth = interp_grid(known_depth_coords, known_depth, grid, method='nearest')
    dense_depth = dense_depth.reshape(H, W)
    dense_depth = dense_depth.astype(np.float32)
    return dense_depth

