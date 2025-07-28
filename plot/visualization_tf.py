import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import argparse
import datetime
import torch
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
import torch.distributed as dist
import torch.nn.functional as F
import matplotlib.patches as patches

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default="configs")
parser.add_argument('--method', '-m', default="DP-FETA")
parser.add_argument('--epsilon', '-e', default="10.0")
parser.add_argument('--data_name', '-dn', default="cifar10_32")
parser.add_argument('--exp_description', '-ed', default="")
parser.add_argument('--resume_exp', '-re', default=None)
parser.add_argument('--config_suffix', '-cs', default="")
opt, unknown = parser.parse_known_args()

merf_path = {
    'mnist_28': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/mnist_28_eps10.0val_time5_freq7.4_1merf-2025-07-12-19-00-19/gen_merf/gen.npz',
    'fmnist_28': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/fmnist_28_eps10.0val_time5_freq7.4_1merf-2025-07-12-18-52-33/gen_merf/gen.npz',
    'celeba_male_32': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/celeba_male_32_eps10.0val_time5_freq8.2_1merf-2025-07-15-09-05-10/gen_merf/gen.npz'
}

row_titles = ["Freq.", "Spat."]

# Create a single figure and a set of subplots with a 4:3 aspect ratio
fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.5))
fig.subplots_adjust(hspace=0.25)  # Adjusted spacing for the new aspect ratio

# Turn off axes for the main subplots
for ax in axes:
    ax.axis('off')  # Disable axes for the main subplots

main_ax_positions = [ax.get_position() for ax in axes]

dataset_titles = {
    'mnist_28': 'MNIST',
    'fmnist_28': 'F-MNIST',
    'celeba_male_32': 'CelebA'
}

for i, ax in enumerate(axes):
    dataset = ['mnist_28', 'fmnist_28', 'celeba_male_32'][i]
    opt.data_name = dataset
    syn = np.load(merf_path[dataset])
    freq_x, freq_y = torch.tensor(syn["x"]), torch.tensor(syn["y"]).long()
    config = parse_config(opt, unknown)
    config.setup.local_rank = 0
    config.setup.global_rank = 0
    config.setup.global_size = config.setup.n_nodes * config.setup.n_gpus_per_node
    config.model.local_rank = config.setup.local_rank
    config.model.global_rank = config.setup.global_rank
    config.model.global_size = config.setup.global_size
    config.pretrain.batch_size = 50
    config.public_data.central.sample_num = 25
    config.public_data.central.sigma = 1.5
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)
    for time_x, time_y in public_train_loader:
        break
    
    row1_tensor = []
    row2_tensor = []
    if 'mnist' in dataset or 'fmnist' in dataset:
        for cls in range(10):
            tensor1 = time_x[time_y==cls][0:1]
            tensor2 = freq_x[freq_y==cls][0:1]
            row1_tensor.append(tensor1)
            row2_tensor.append(tensor2)
        row1_tensor = torch.cat(row1_tensor)
        row2_tensor = torch.cat(row2_tensor)
        row1_tensor = F.interpolate(row1_tensor, size=(32, 32))
        row2_tensor = F.interpolate(row2_tensor, size=(32, 32))
    else:  # CelebA
        for cls in range(2):
            tensor1 = time_x[time_y==cls][0:5]
            tensor2 = freq_x[freq_y==cls][0:5]
            row1_tensor.append(tensor1)
            row2_tensor.append(tensor2)
        row1_tensor = torch.cat(row1_tensor)
        row2_tensor = torch.cat(row2_tensor)
        # Ensure CelebA tensors have 4 dimensions (add channel dim if needed)
        if row1_tensor.dim() == 3:
            row1_tensor = row1_tensor.unsqueeze(1)  # Add channel dim: [10, H, W] -> [10, 1, H, W]
        if row2_tensor.dim() == 3:
            row2_tensor = row2_tensor.unsqueeze(1)
        row1_tensor = F.interpolate(row1_tensor, size=(32, 32))
        row2_tensor = F.interpolate(row2_tensor, size=(32, 32))
    
    # Debug: Print tensor shapes
    print(f"Dataset: {dataset}")
    print(f"row1_tensor shape before transpose: {row1_tensor.shape}")
    print(f"row2_tensor shape before transpose: {row2_tensor.shape}")

    row1_tensor = row1_tensor.numpy().transpose(1, 2, 0, 3).reshape(row1_tensor.shape[1], row1_tensor.shape[2], row1_tensor.shape[0] * row1_tensor.shape[3])
    row2_tensor = row2_tensor.numpy().transpose(1, 2, 0, 3).reshape(row2_tensor.shape[1], row2_tensor.shape[2], row2_tensor.shape[0] * row2_tensor.shape[3])

    print(f"row1_tensor shape after transpose and reshape: {row1_tensor.shape}")
    print(f"row2_tensor shape after transpose and reshape: {row2_tensor.shape}")

    row1_tensor = (row1_tensor * 255.).astype('uint8').transpose(1, 2, 0)
    row2_tensor = (row2_tensor * 255.).astype('uint8').transpose(1, 2, 0)
    images_data = [row2_tensor, row1_tensor]
    
    # Get the position of the original subplot
    pos = main_ax_positions[i]
    
    # Add the main title for this section
    fig.text((pos.x0 + pos.x1) / 2, pos.y1 + 0.015, dataset_titles[dataset],
             ha='center', va='bottom', fontsize=14.5)

    # Add the two rows of images
    for j in range(2):
        # Calculate vertical position for the image strip within the subplot's area
        y_position = pos.y0 + (pos.height * (0.6 - j * 0.5))
        
        # Add a new axis for the image strip
        inner_ax = fig.add_axes([pos.x0, y_position, pos.width, pos.height * 0.46])
        inner_ax.imshow(images_data[j], cmap='gray', aspect='auto')
        inner_ax.axis('off')  # Already present, kept for clarity

        # Add a thin black border around the image
        border = patches.Rectangle((0, 0), images_data[j].shape[1], images_data[j].shape[0],
                                  linewidth=1.5, edgecolor='black', facecolor='none')
        inner_ax.add_patch(border)

        # Add row titles ("Freq.", "Time") to the left
        inner_ax.text(-0.02, 0.5, row_titles[j], transform=inner_ax.transAxes,
                      fontsize=14, va='center', ha='right')

plt.savefig('comparison_tf.pdf', bbox_inches='tight')
plt.savefig('comparison_tf.png', bbox_inches='tight')
plt.close()