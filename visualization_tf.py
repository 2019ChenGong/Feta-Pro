import matplotlib.pyplot as plt
import numpy as np

# 示例数据：用随机数据模拟两张拼接好的图像（28x280）
# 你可以替换为你自己的 tensor 数据


import os
import sys
import argparse
import datetime
import torch
from torch.utils.data import random_split, TensorDataset, Dataset, DataLoader, ConcatDataset
import torch.distributed as dist
import numpy as np

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('--config_dir', default="configs")
parser.add_argument('--method', '-m', default="DP-FETA2")
parser.add_argument('--epsilon', '-e', default="10.0")
parser.add_argument('--data_name', '-dn', default="cifar10_32")
parser.add_argument('--exp_description', '-ed', default="")
parser.add_argument('--resume_exp', '-re', default=None)
parser.add_argument('--config_suffix', '-cs', default="")
opt, unknown = parser.parse_known_args()

merf_path = {'mnist_28': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/mnist_28_eps10.0val_time5_freq7.4_1merf-2025-07-12-19-00-19/gen_merf/gen.npz', 
             'fmnist_28': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/fmnist_28_eps10.0val_time5_freq7.4_1merf-2025-07-12-18-52-33/gen_merf/gen.npz', 
             'celeba_male_32': '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/celeba_male_32_eps10.0val_time5_freq8.2_1merf-2025-07-15-09-05-10/gen_merf/gen.npz'}
for dataset in ['mnist_28', 'fmnist_28', 'celeba_male_32']:
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
    config.public_data.central.sigma = 5
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)
    for time_x, time_y in public_train_loader:
        break
    
    row1_tensor = []
    row2_tensor = []
    if 'mnist' in dataset:
        for cls in range(10):
            tensor1 = time_x[time_y==cls][0:1]
            tensor2 = freq_x[freq_y==cls][0:1]
            row1_tensor.append(tensor1)
            row2_tensor.append(tensor2)
        row1_tensor = torch.cat(row1_tensor)
        row2_tensor = torch.cat(row2_tensor)
        row1_tensor = F.interpolate(row1_tensor, size=(32, 32))
        row2_tensor = F.interpolate(row2_tensor, size=(32, 32))
    else:
        row1_tensor = time_x[:10]
        row2_tensor = freq_x[:10]
    
    row1_tensor = row1_tensor.numpy().transpose(1, 2, 0, 3).reshape(row1_tensor.shape[1], row1_tensor.shape[2], row1_tensor.shape[3] * 10)
    row2_tensor = row2_tensor.numpy().transpose(1, 2, 0, 3).reshape(row2_tensor.shape[1], row2_tensor.shape[2], row2_tensor.shape[3] * 10)

    row1_tensor = (row1_tensor * 255.).astype('uint8').transpose(1, 2, 0)
    row2_tensor = (row2_tensor * 255.).astype('uint8').transpose(1, 2, 0)
    
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 3))

    # 设置标题
    row_titles = ['Time', 'Freq.']

    # 显示图像
    for ax, img, title in zip(axes, [row1_tensor, row2_tensor], row_titles):
        ax.imshow(img, cmap='gray')
        ax.axis('off')  # 关闭坐标轴
        ax.text(-10, 14, title, fontsize=16, va='center', ha='right')

    plt.tight_layout()
    plt.subplots_adjust(left=0.1, hspace=-0.2)  # 避免标题被截断
    # plt.savefig('comparison_tf.png')
    plt.savefig('comparison_tf_{}.png'.format(dataset), bbox_inches='tight')
    plt.savefig('comparison_tf_{}.pdf'.format(dataset), bbox_inches='tight')


    
