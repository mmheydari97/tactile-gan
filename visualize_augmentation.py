import os
import argparse
import numpy as np
from PIL import Image
import torch
from datasets.PairedDataset import PairedDataset

def combine_channels(axes, grid, content):
    """
    Combine channel-wise images into a single image using additive blending:
    1. Start with grid as base layer
    2. Add axes on top of grid
    3. Add content on top of axes and grid
    """
    # Create a base image with black background
    height, width = axes.shape
    base = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add each channel with proper scaling
    # Grid (green channel)
    base[..., 1] = np.clip(grid.astype(np.int32), 0, 255)
    
    # Axes (red channel)
    base[..., 0] = np.clip(base[..., 0] + axes.astype(np.int32), 0, 255)
    
    # Content (blue channel)
    base[..., 2] = np.clip(base[..., 2] + content.astype(np.int32), 0, 255)
    
    return Image.fromarray(base)

def save_tensor_as_image(tensor, path):
    """Convert tensor to PIL image and save"""
    # Convert from [-1,1] to [0,255]
    tensor = tensor.mul(0.5).add(0.5).mul(255).byte()
    # Convert to numpy array and transpose dimensions
    array = tensor.numpy().transpose(1, 2, 0)
    # For single-channel images, convert to grayscale
    if array.shape[2] == 1:
        array = array.squeeze()
    Image.fromarray(array).save(path)

def main(args):
    # Create dataset instances
    raw_dataset = PairedDataset(
        img_dir=args.data_dir,
        size=args.size,
        mode='train',
        aug=False,  # No augmentation for raw version
        target=args.target_mode
    )
    
    aug_dataset = PairedDataset(
        img_dir=args.data_dir,
        size=args.size,
        mode='train',
        aug=True,  # With augmentation
        target=args.target_mode
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process each sample
    for i in range(min(args.num_samples, len(raw_dataset))):
        # Get raw version
        raw_source, raw_target = raw_dataset[i]
        
        # Get augmented version
        aug_source, aug_target = aug_dataset[i]
        
        # Save raw source
        save_tensor_as_image(
            raw_source, 
            os.path.join(args.output_dir, f"sample_{i}_source_raw.png")
        )
        
        # Save augmented source
        save_tensor_as_image(
            aug_source, 
            os.path.join(args.output_dir, f"sample_{i}_source_aug.png")
        )
        
        # Save target images
        if args.target_mode == 'rgb':
            save_tensor_as_image(
                raw_target, 
                os.path.join(args.output_dir, f"sample_{i}_target_raw.png")
            )
            save_tensor_as_image(
                aug_target, 
                os.path.join(args.output_dir, f"sample_{i}_target_aug.png")
            )
        else:
            # Convert channel-wise tensors to numpy arrays
            raw_axes = raw_target[0].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            raw_grid = raw_target[1].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            raw_content = raw_target[2].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            
            aug_axes = aug_target[0].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            aug_grid = aug_target[1].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            aug_content = aug_target[2].mul(0.5).add(0.5).mul(255).byte().numpy().squeeze()
            
            # Combine channels into single images
            raw_combined = combine_channels(raw_axes, raw_grid, raw_content)
            aug_combined = combine_channels(aug_axes, aug_grid, aug_content)
            
            # Save combined images
            raw_combined.save(os.path.join(args.output_dir, f"sample_{i}_target_raw.png"))
            aug_combined.save(os.path.join(args.output_dir, f"sample_{i}_target_aug.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize dataset augmentation')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='./augmentation_vis',
                        help='Output directory for visualization images')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size (default: 256)')
    parser.add_argument('--augment', action='store_true',
                        help='Apply data augmentation')
    parser.add_argument('--target_mode', type=str, default='non_rgb',
                        choices=['rgb', 'non_rgb'],
                        help='Target mode: rgb or non_rgb (channel-wise)')
    
    args = parser.parse_args()
    main(args)
