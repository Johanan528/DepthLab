#!/usr/bin/env bash
set -e
set -x
pretrained_model_name_or_path='./checkpoints/marigold-depth-v1-0'
image_encoder_path='./checkpoints/CLIP-ViT-H-14-laion2B-s32B-b79K'
denoising_unet_path='./checkpoints/DepthLab/denoising_unet.pth'
reference_unet_path='./checkpoints/DepthLab/reference_unet.pth'
mapping_path='./checkpoints/DepthLab/mapping_layer.pth'

export CUDA_VISIBLE_DEVICES=0
cd ..
python infer.py  \
    --seed 1234 \
    --denoise_steps 50 \
    --processing_res 768 \
    --normalize_scale 1 \
    --strength 0.8 \
    --pretrained_model_name_or_path $pretrained_model_name_or_path --image_encoder_path $image_encoder_path \
    --denoising_unet_path $denoising_unet_path \
    --reference_unet_path $reference_unet_path \
    --mapping_path $mapping_path \
    --output_dir 'output/in-the-wild_example' \
    --input_image_paths test_cases/RGB.JPG \
    --known_depth_paths test_cases/know_depth.npy \
    --masks_paths test_cases/mask.npy \
    --blend