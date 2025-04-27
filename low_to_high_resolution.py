import torch
import torch.nn as nn
from PIL import Image
import os
import argparse
from torchvision import transforms
from basicsr.archs.rrdbnet_arch import RRDBNet
import numpy as np

# EDSR Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        return x + self.conv_block(x)

# EDSR Model
class EDSR(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3, num_features=64, num_blocks=16):
        super(EDSR, self).__init__()
        self.scale_factor = scale_factor
        self.input_conv = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
        self.mid_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        upscale = []
        for _ in range(int(torch.log2(torch.tensor(scale_factor)).item())):
            upscale += [
                nn.Conv2d(num_features, num_features * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.ReLU(inplace=True)
            ]
        self.upscale = nn.Sequential(*upscale)
        self.output_conv = nn.Conv2d(num_features, num_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.input_conv(x)
        residual = x
        x = self.res_blocks(x)
        x = self.mid_conv(x) + residual
        x = self.upscale(x)
        x = self.output_conv(x)
        return x

def upscale_images(input_npz, output_npz, model_path, scale_factor):
    """
    Upsample images from a .npz file using a pre-trained EDSR model and save results with original labels to a new .npz file.

    Args:
        input_npz (str): Path to input .npz file (contains low-resolution images 'x' and labels 'y')
        output_npz (str): Path to output .npz file (to save upsampled images and original labels)
        model_path (str): Path to pre-trained EDSR weights
        scale_factor (int): Upsampling scale factor (2 or 4)
    """
    # Initialize device and model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model weights from {model_path}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return
    
    model.eval()
    
    # Image transformation
    transform = transforms.ToTensor()
    
    # Load .npz file
    try:
        rl = np.load(input_npz)
        lr_imgs, lr_labels = rl['x'], rl['y']
        print(f"Loaded {len(lr_imgs)} images from {input_npz}")
    except Exception as e:
        print(f"Failed to load .npz file: {str(e)}")
        return
    
    # Store upsampled images
    hr_imgs = []
    
    # Process each image
    for idx, lr_img in enumerate(lr_imgs):
        try:
            # Convert numpy array to PIL image
            lr_img = lr_img.astype(np.uint8)  # Ensure uint8 data type
            lr_pil = Image.fromarray(lr_img).convert('RGB')
            lr_tensor = transform(lr_pil).unsqueeze(0).to(device)
            
            # Upsample
            with torch.no_grad():
                hr_tensor = model(lr_tensor)
            
            # Convert upsampled result to numpy array
            hr_img = hr_tensor.squeeze(0).cpu().numpy()  # Shape: (C, H, W)
            hr_img = np.transpose(hr_img, (1, 2, 0))  # Convert to (H, W, C)
            hr_img = (hr_img * 255).clip(0, 255).astype(np.uint8)  # Convert to uint8
            hr_imgs.append(hr_img)
            
            print(f"Upsampled image {idx+1}/{len(lr_imgs)}")
            
        except Exception as e:
            print(f"Error processing image {idx+1}: {str(e)}")
            continue
    
    # Save upsampled images and original labels to .npz file
    hr_imgs = np.array(hr_imgs)
    try:
        np.savez_compressed(output_npz, x=hr_imgs, y=lr_labels)
        print(f"Saved upsampled results to: {output_npz}")
    except Exception as e:
        print(f"Failed to save .npz file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Upsample images from .npz file using pre-trained EDSR model")
    parser.add_argument('--input-npz', default='/exp/dp-feta2/low_resolution_images/gen/gen.npz',
                        help='Path to input .npz file')
    parser.add_argument('--output-npz', default='/exp/dp-feta2/high_resolution_images/gen.npz',
                        help='Path to output .npz file')
    parser.add_argument('--model-path', default='models/pretrained_models/RealESRGAN_x4plus.pth',
                        help='Path to pre-trained EDSR weights')
    parser.add_argument('--scale-factor', type=int, choices=[2, 4], default=4,
                        help='Upsampling scale factor (2 for 64x64, 4 for 128x128)')
    
    args = parser.parse_args()
    
    print(f"Upsampling images from {args.input_npz} to {args.scale_factor}x resolution")
    upscale_images(args.input_npz, args.output_npz, args.model_path, args.scale_factor)
    print("Upsampling completed.")

if __name__ == "__main__":
    main()