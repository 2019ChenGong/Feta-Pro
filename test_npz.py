import numpy as np
import cv2
import os
import argparse
import random

def sample_and_save_npz(npz_path, output_dir, num_samples=20):
    """
    Randomly sample images from a .npz file and save them as PNG files.

    Args:
        npz_path (str): Path to the .npz file
        output_dir (str): Output directory for PNG images
        num_samples (int): Number of images to sample
    """
    # Load .npz file
    try:
        data = np.load(npz_path, mmap_mode='r')
    except Exception as e:
        print(f"Failed to load NPZ file {npz_path}: {str(e)}")
        return

    # Check if 'x' key (image data) exists
    if 'x' not in data:
        print("No 'x' key (image data) found in NPZ file")
        return

    images = data['x']  # Image array
    labels = data['y'] if 'y' in data else None  # Label array (if exists)

    # Check number of images
    if len(images) == 0:
        print("No images found in NPZ file")
        return

    # Determine number of samples
    num_samples = min(num_samples, len(images))  # Ensure not exceeding total images
    print(f"Found {len(images)} images, sampling {num_samples} for PNG output")

    # Randomly sample image indices
    sample_indices = random.sample(range(len(images)), num_samples)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Process and save sampled images
    for idx in sample_indices:
        img = images[idx]  # Get single image (C, H, W) format
        img_name = f"image_{idx}.png"

        # Validate image format
        if len(img.shape) != 3:
            print(f"Image {idx} has invalid shape: {img.shape}, expected (C, H, W), skipping")
            continue

        num_channels = img.shape[0]
        if num_channels not in [1, 3, 4]:
            print(f"Image {idx} has invalid channel count: {num_channels}, expected 1, 3, or 4, skipping")
            continue

        # Convert to HWC format
        img = np.transpose(img, (1, 2, 0))  # From (C, H, W) to (H, W, C)

        # Handle data type
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).clip(0, 255).astype(np.uint8)  # Convert from [0, 1] to [0, 255]

        # Handle different channel counts
        if num_channels == 1:
            img = img.squeeze()  # Grayscale, remove single channel dimension
        elif num_channels == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for saving
        elif num_channels == 4:
            img = img[:, :, :3]  # Ignore alpha channel
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Save image
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, img)
        print(f"Saved image {img_name} to {output_path}")

    print(f"Done! Saved {num_samples} images to {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample images from a .npz file and save as PNG')
    parser.add_argument('-i', '--input', type=str, default='exp/pe/celeba_male_32_eps10.0_vd80-2025-04-30-18-55-54/gen/gen.npz', help='Path to the .npz file')
    parser.add_argument('-o', '--output', type=str, default='exp/pe/celeba_male_32_eps10.0_vd80-2025-04-30-18-55-54/gen/', help='Output directory for PNG images')
    parser.add_argument('--num_samples', type=int, default=30, help='Number of images to sample')
    args = parser.parse_args()

    sample_and_save_npz(args.input, args.output, args.num_samples)