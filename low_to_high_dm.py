import os
import numpy as np
import sys
import argparse
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import save_image
from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config

def main(config, opt):
    initialize_environment(config)

    model, config = load_model(config)

    # Load synthetic data
    syn_path = os.path.join(opt.input_path, 'gen', 'gen.npz')

    try:
        syn = np.load(syn_path)
        syn_data, syn_labels = syn["x"], syn["y"]
        print(f"Loaded syn_data shape: {syn_data.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {syn_path}")
        return
    except Exception as e:
        print(f"Error loading .npz file: {e}")
        return

    # Store original dimensions for restoration
    original_shape = syn_data.shape
    was_grayscale = len(original_shape) == 4 and original_shape[1] == 1  # Check if input was grayscale (N, 1, H, W)
    was_3d = len(original_shape) == 3  # Check if input was 3D (N, H, W)
    original_height, original_width = original_shape[-2:]  # Get H, W (works for both 3D and 4D)

    # Convert NumPy array to PyTorch tensor
    device = torch.device("cuda" if torch.cuda.is_available() and config.setup.get("use_cuda", True) else "cpu")
    samples_tensor = torch.from_numpy(syn_data).float().to(device)  # Ensure float32 and move to device

    # Interpolate to 64x64
    if samples_tensor.dim() == 3:  
        samples_tensor = samples_tensor.unsqueeze(1) 
    elif samples_tensor.dim() != 4:
        raise ValueError(f"Unexpected samples shape: {samples_tensor.shape}")

    if samples_tensor.shape[1] == 1:  
        samples_tensor = samples_tensor.repeat(1, 3, 1, 1)  
        print("Converted grayscale to RGB, new shape:", samples_tensor.shape) 

    model_image_size = model.api_params.model_image_size if 'model_image_size' in model.api_params else model.api_params.network.image_size
    samples_tensor = F.interpolate(
        samples_tensor,
        size=[model_image_size, model_image_size],
        mode='bilinear',
        align_corners=False
    ).clamp(0., 1.)  

    # Define label (ensure it's a tensor if required by model)
    label = None

    # Generate new samples
    with torch.no_grad():
        new_samples = model.api._image_variation(samples_tensor, label, variation_degree=config.model.variation_degree)

    if isinstance(new_samples, np.ndarray):
        new_samples = torch.from_numpy(new_samples).float().to(device)

    # Normalize to [0, 1] for saving
    new_samples = new_samples.clamp(0., 1.)

    # Restore new_samples to original dimensions and resolution
    # Step 1: Downsample to original height and width
    new_samples = F.interpolate(
        new_samples,
        size=[original_height, original_width],
        mode='bilinear',
        align_corners=False
    ).clamp(0., 1.)

    # Step 2: Revert RGB to grayscale if input was grayscale
    if was_grayscale:
        new_samples = new_samples[:, :1, :, :]  # Keep only the first channel
        print("Restored to grayscale, new shape:", new_samples.shape)

    # Step 3: Remove channel dimension if input was 3D
    if was_3d:
        new_samples = new_samples.squeeze(1)  # Remove channel dimension (N, 1, H, W -> N, H, W)
        print("Restored to 3D shape:", new_samples.shape)

    # Random sample 2000 images
    num_samples = 2000
    indices = torch.randperm(new_samples.shape[0], device=device)[:num_samples]
    samples_generated_image = new_samples[indices]  # Sample from tensor
    syn_labels_sampled = syn_labels[indices.cpu().numpy()]  # Sample corresponding labels

    # Convert to NumPy array on CPU
    samples_generated_image = samples_generated_image.cpu().numpy()

    # Save output
    batch_size = 100
    output_dir = "sampled_images"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(0, num_samples, batch_size):
        batch = samples_generated_image[i:i + batch_size]
        # Convert back to tensor and add channel dimension if needed for save_image
        batch_tensor = torch.from_numpy(batch).float().to(device)
        if batch_tensor.dim() == 3:  # If 3D (N, H, W), add channel dimension
            batch_tensor = batch_tensor.unsqueeze(1).repeat(1, 3, 1, 1)  # Convert to RGB for saving
        elif batch_tensor.shape[1] == 1:  # If grayscale (N, 1, H, W)
            batch_tensor = batch_tensor.repeat(1, 3, 1, 1)  # Convert to RGB for saving
        output_path = os.path.join(output_dir, f"sampled_batch_{i // batch_size}.png")
        save_image(batch_tensor, output_path, nrow=10)

    # Save to npz file (adjusted for restored shape)
    os.makedirs(os.path.join(config.setup.workdir, 'gen'), exist_ok=True)
    np.savez(os.path.join(config.setup.workdir, 'gen', "gen.npz"), x=samples_generated_image, y=syn_labels_sampled)

    print('Finished')

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="PE")
    parser.add_argument('--epsilon', '-e', default="10.0")
    parser.add_argument('--data_name', '-dn', default="celeba_male_32")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    parser.add_argument('--input_path', '-ip', default="")
    parser.add_argument('--variation_degree', '-vd', type=int, default=0)
    opt, unknown = parser.parse_known_args()
    config = parse_config(opt, unknown)
    config.model['variation_degree'] = opt.variation_degree

    nowTime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if opt.resume_exp is not None:
        config.setup.workdir = "exp/{}/{}".format(str.lower(opt.method), opt.resume_exp)
    else:
        config.setup.workdir = "exp/{}/{}_eps{}{}{}-{}".format(str.lower(opt.method), opt.data_name, opt.epsilon, opt.config_suffix, opt.exp_description+'_vd'+str(opt.variation_degree), nowTime)

    run(lambda config: main(config, opt), config)