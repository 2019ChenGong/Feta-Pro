import numpy as np
import argparse

from utils.utils import initialize_environment, run, parse_config
from torch.utils.data import random_split, TensorDataset

from data.dataset_loader import load_data
from data.dataset_loader import CentralDataset

import torch


def rgb_to_grayscale_luminance(rgb_tensor):
    """
    使用亮度公式将 RGB 转为灰度图: Y = 0.299*R + 0.587*G + 0.114*B
    输入: (3, H, W) 或 (N, 3, H, W)
    输出: (H, W) 或 (N, 1, H, W)
    """
    coeffs = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = torch.sum(rgb_tensor.unsqueeze(0) if rgb_tensor.dim() == 3 else rgb_tensor * coeffs, dim=1, keepdim=True)
    return gray.squeeze(0).squeeze(0)  # 返回 (H, W) 或 (N, H, W)


def compute_entropy(image_tensor, method='grayscale'):
    """
    计算图像信息熵，支持：
    - method='grayscale': 转为灰度图后计算
    - method='mean_channel': 分别计算三通道熵再平均（仅适用于 RGB）
    """
    if image_tensor.dim() == 3:
        if image_tensor.shape[0] == 3:
            # 是 RGB 图像
            if method == 'grayscale':
                image_tensor = rgb_to_grayscale_luminance(image_tensor)  # (H, W)
            else:
                image_tensor = image_tensor[0]
    elif image_tensor.dim() == 2:
        pass  # 已是灰度图
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    # 现在 image_tensor 是 (H, W)
    if image_tensor.max() <= 1.0:
        image_tensor = image_tensor * 255.0
    image_tensor = image_tensor.to(torch.uint8).float()

    # 计算直方图 (256 bins)
    hist = torch.histc(image_tensor, bins=256, min=0, max=255)
    hist = hist / (hist.sum() + 1e-8)  # 概率分布，加小数避免除零

    # 计算熵: H = -sum(p * log2(p))
    entropy = -torch.sum(hist[hist > 0] * torch.log2(hist[hist > 0] + 1e-8))
    return entropy.item()


def compute_edge_complexity(image_tensor):
    """
    计算边缘纹理复杂度（基于 Sobel 梯度幅值）
    自动处理 RGB 或灰度图
    """
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:
        # RGB -> 灰度图
        gray = rgb_to_grayscale_luminance(image_tensor)  # (H, W)
    else:
        gray = image_tensor[0]  # 已是灰度图

    # 扩展为 (1, 1, H, W)
    gray = gray.unsqueeze(0).unsqueeze(0).float()

    # Sobel 算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    grad_x = torch.functional.F.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.functional.F.conv2d(gray, sobel_y, padding=1)

    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (1, 1, H, W)
    complexity = magnitude.mean().item()
    return complexity



def evaluate_dataset_entropy_and_complexity(data_loader):
    """
    遍历数据集，计算平均信息熵和平均边缘复杂度
    """
    entropies = []
    complexities = []

    for images, labels in data_loader:
        for img in images:
            entropy = compute_entropy(img)
            complexity = compute_edge_complexity(img)
            entropies.append(entropy)
            complexities.append(complexity)
            
            if len(entropies) >= 60000:
                break
        if len(entropies) >= 60000:
                break

    avg_entropy = np.mean(entropies)
    avg_complexity = np.mean(complexities)

    return avg_entropy, avg_complexity


def main(config):
    sensitive_train_loader, _, _, _, config = load_data(config)
    time_set = CentralDataset(sensitive_train_loader.dataset, num_classes=config.sensitive_data.n_classes, **config.public_data.central)
    time_dataloader = torch.utils.data.DataLoader(dataset=time_set, shuffle=True, drop_last=True, batch_size=10, num_workers=0)
    # syn = np.load("/p/fzv6enresearch/FETA-Pro/exp/dp-feta2/fmnist_28_eps10.0val_time5_freq7.4_1merf-2025-07-12-18-52-33/gen/gen.npz")
    syn = np.load("/p/fzv6enresearch/FETA-Pro/exp/dp-feta2/celeba_male_32_eps10.0val_time5_freq8.2_1merf-2025-07-15-09-05-10/gen/gen.npz")
    syn_data, syn_labels = syn["x"], syn["y"]
    freq_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
    freq_dataloader = torch.utils.data.DataLoader(dataset=freq_set, shuffle=True, drop_last=True, batch_size=100, num_workers=0)

    avg_entropy, avg_complexity = evaluate_dataset_entropy_and_complexity(sensitive_train_loader)
    print(avg_entropy, avg_complexity)
    avg_entropy, avg_complexity = evaluate_dataset_entropy_and_complexity(time_dataloader)
    print(avg_entropy, avg_complexity)
    avg_entropy, avg_complexity = evaluate_dataset_entropy_and_complexity(freq_dataloader)
    print(avg_entropy, avg_complexity)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="DP-FETA-Pro")
    parser.add_argument('--epsilon', '-e', default="10.0")
    parser.add_argument('--data_name', '-dn', default="fmnist_28")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    opt, unknown = parser.parse_known_args()

    config = parse_config(opt, unknown)
    config.setup.n_gpus_per_node = 1
    config.setup.run_type = 'normal'

    run(main, config)