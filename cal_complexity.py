import numpy as np
import argparse

from utils.utils import initialize_environment, run, parse_config
from torch.utils.data import random_split, TensorDataset

from data.dataset_loader import load_data
from data.dataset_loader import CentralDataset

import torch


from collections import defaultdict

def compute_average_class_variance(dataloader):
    """
    计算 DataLoader 中每个类别的图像像素方差，然后返回这些方差的平均值。
    
    Args:
        dataloader: PyTorch DataLoader, 返回 (image, label) 样本，image 是张量 (C, H, W)
    
    Returns:
        float: 所有类别方差的平均值
    """
    class_pixel_vars = defaultdict(list)

    # 遍历数据集
    for images, labels in dataloader:
        for image, label in zip(images, labels):
            # 将图像展平为像素向量，计算整体方差
            pixel_var = image.flatten().var().item()  # 使用 .var() 计算方差
            class_pixel_vars[label.item()].append(pixel_var)
    
    # 对每个类，计算该类所有图像方差的平均（或你可以直接用所有图像方差平均）
    class_avg_vars = []
    for class_label, vars in class_pixel_vars.items():
        avg_var = sum(vars) / len(vars)
        class_avg_vars.append(avg_var)
        # print(f"Class {class_label}: average variance = {avg_var:.6f}")
    
    # 返回所有类别平均方差的均值
    overall_avg = sum(class_avg_vars) / len(class_avg_vars)
    return overall_avg


def rgb_to_grayscale_luminance(rgb_tensor):
    """
    使用亮度公式将 RGB 转为灰度图: Y = 0.299*R + 0.587*G + 0.114*B
    输入: (3, H, W) 或 (N, 3, H, W)
    输出: (H, W) 或 (N, 1, H, W)
    """
    coeffs = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    gray = torch.sum(rgb_tensor.unsqueeze(0) if rgb_tensor.dim() == 3 else rgb_tensor * coeffs, dim=1, keepdim=True)
    return gray.squeeze(0).squeeze(0)  # 返回 (H, W) 或 (N, H, W)

import torch.nn.functional as F

def blurriness(image: torch.Tensor) -> float:
    """
    计算图像的模糊度
    
    参数:
        image (torch.Tensor): 输入图像，形状为 (C, H, W)
    
    返回:
        float: 模糊度值
    """
    # 确保输入张量在 [0, 1] 范围内
    assert torch.all((image >= 0) & (image <= 1)), "图像值应在 [0, 1] 范围内"
    
    # 计算梯度
    gradient_x = torch.abs(image[..., :-1] - image[..., 1:])
    gradient_y = torch.abs(image[..., :, :-1] - image[..., :, 1:])
    
    # 计算模糊度
    blurriness_score = (gradient_x.mean() + gradient_y.mean()) / 2.0
    
    return blurriness_score.item()

def calculate_sharpness_tensor(image_tensor):
    """
    计算 PyTorch Tensor 图像的清晰度（基于拉普拉斯方差）
    
    参数:
        image_tensor (torch.Tensor): 形状为 [C, H, W] 或 [B, C, H, W]
                                   值范围建议在 [0, 1] 或 [0, 255]
    
    返回:
        sharpness (float or torch.Tensor): 标量（单图）或形状为 [B,] 的张量
    """
    # 确保是浮点类型
    if not image_tensor.is_floating_point():
        image_tensor = image_tensor.float()

    # 判断是否是单张图像 [C, H, W]
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # 变成 [1, C, H, W]

    # 转为灰度图：加权平均 RGB 三通道
    # 使用标准灰度转换权重: 0.299*R + 0.587*G + 0.114*B
    if image_tensor.shape[1] == 3:
        gray = 0.299 * image_tensor[:, 0, :, :] + \
               0.587 * image_tensor[:, 1, :, :] + \
               0.114 * image_tensor[:, 2, :, :]
    else:
        gray = image_tensor.squeeze(1)  # 假设单通道 [B, 1, H, W] -> [B, H, W]

    # 定义拉普拉斯卷积核
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]], dtype=torch.float32, device=gray.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

    # 对每张图像应用卷积（逐通道处理）
    padding = 1
    laplacian = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=padding)
    laplacian = laplacian.squeeze(1)  # [B, H, W]

    # 计算每张图像的方差（即清晰度得分）
    sharpness = laplacian.var(dim=[1, 2])  # 在 H, W 维度上求方差 -> [B,]

    # 如果输入是单张图像，返回标量 float
    if sharpness.numel() == 1:
        return sharpness.item()

    return sharpness  # 返回 tensor

def compute_entropy(image_tensor, method='grayscale'):
    if image_tensor.dim() == 3:
        if image_tensor.shape[0] == 3:
            if method == 'grayscale':
                image_tensor = rgb_to_grayscale_luminance(image_tensor)  # (H, W)
            else:
                image_tensor = image_tensor[0]
    elif image_tensor.dim() == 2:
        pass  
    else:
        raise ValueError(f"Unexpected tensor shape: {image_tensor.shape}")

    if image_tensor.max() <= 1.0:
        image_tensor = image_tensor * 255.0
    image_tensor = image_tensor.to(torch.uint8).float()

    hist = torch.histc(image_tensor, bins=256, min=0, max=255)
    hist = hist / (hist.sum() + 1e-8)  

    entropy = -torch.sum(hist[hist > 0] * torch.log2(hist[hist > 0] + 1e-8))
    return entropy.item()


def compute_edge_complexity(image_tensor):
    if image_tensor.dim() == 3 and image_tensor.shape[0] == 3:        gray = rgb_to_grayscale_luminance(image_tensor)  # (H, W)
    else:
        gray = image_tensor[0] 

    gray = gray.unsqueeze(0).unsqueeze(0).float()

    # Sobel operators
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    grad_x = torch.functional.F.conv2d(gray, sobel_x, padding=1)
    grad_y = torch.functional.F.conv2d(gray, sobel_y, padding=1)

    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)  # (1, 1, H, W)
    complexity = magnitude.mean().item()
    return complexity

def edge_score(image: torch.Tensor) -> float:
    assert torch.all((image >= 0) & (image <= 1)), "the image values should range into [0,1]"
    if image.dim() == 3 and image.shape[0] == 3:
        image = rgb_to_grayscale_luminance(image)  # (H, W)
    else:
        image = image[0]
    image = image.unsqueeze(0).unsqueeze(0).float()
    
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.repeat(image.size(0), 1, 1, 1).to(image.device)
    sobel_y = sobel_y.repeat(image.size(0), 1, 1, 1).to(image.device)
    
    gradient_x = F.conv2d(image, sobel_x, padding=1)
    gradient_y = F.conv2d(image, sobel_y, padding=1)
    
    gradient_magnitude = torch.sqrt(gradient_x.pow(2) + gradient_y.pow(2))
    
    threshold = 0.1  
    edge_pixels = (gradient_magnitude > threshold).float().sum()
    total_pixels = image.numel()
    
    edge_score = edge_pixels / total_pixels
    
    return edge_score.item()


def evaluate_dataset_entropy_and_complexity(data_loader):
    entropies = []
    complexities = []
    sharpnesses = []
    blurrinesses = []

    for images, labels in data_loader:
        for img in images:
            entropy = compute_entropy(img)
            # complexity = edge_score(img)
            complexity = compute_edge_complexity(img)
            # complexity = 0
            # blur = blurriness(img)
            blur = 0
            sharpness = calculate_sharpness_tensor(img)
            entropies.append(entropy)
            complexities.append(complexity)
            sharpnesses.append(sharpness)
            blurrinesses.append(blur)
            
            if len(entropies) >= 60000:
                break
        if len(entropies) >= 60000:
                break

    avg_entropy = np.mean(entropies)
    avg_complexity = np.mean(complexities)
    avg_sharpness = np.mean(sharpnesses)
    avg_blur = np.mean(blurrinesses)

    return avg_entropy, avg_complexity, avg_blur, avg_sharpness, compute_average_class_variance(data_loader)


def main(config):
    sensitive_train_loader, _, _, _, config = load_data(config)
    time_set = CentralDataset(sensitive_train_loader.dataset, num_classes=config.sensitive_data.n_classes, **config.public_data.central)
    time_dataloader = torch.utils.data.DataLoader(dataset=time_set, shuffle=True, drop_last=True, batch_size=10, num_workers=0)
    syn = np.load(config.freq_path)
    syn_data, syn_labels = syn["x"], syn["y"]
    freq_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
    freq_dataloader = torch.utils.data.DataLoader(dataset=freq_set, shuffle=True, drop_last=True, batch_size=100, num_workers=0)

    avg_entropy, avg_complexity, avg_blur, avg_sharpness, div = evaluate_dataset_entropy_and_complexity(time_dataloader)
    print(avg_entropy, avg_complexity, avg_blur, avg_sharpness, div)
    avg_entropy, avg_complexity, avg_blur, avg_sharpness, div = evaluate_dataset_entropy_and_complexity(freq_dataloader)
    print(avg_entropy, avg_complexity, avg_blur, avg_sharpness, div)
    # avg_entropy, avg_complexity, avg_blur, avg_sharpness, div = evaluate_dataset_entropy_and_complexity(sensitive_train_loader)
    # print(avg_entropy, avg_complexity, avg_blur, avg_sharpness, div)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="DP-FETA-Pro")
    parser.add_argument('--epsilon', '-e', default="10.0")
    parser.add_argument('--data_name', '-dn', default="mnist_28")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    opt, unknown = parser.parse_known_args()


    for dn in ['mnist_28', 'fmnist_28', 'cifar10_32', 'celeba_male_32', 'camelyon_32'][2:3]:
        print(dn)
        opt.data_name = dn
        config = parse_config(opt, unknown)
        config.setup.n_gpus_per_node = 1
        config.setup.run_type = 'normal'

        if opt.data_name == 'mnist_28':
            config['freq_path'] = '/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/mnist_28_eps1.0val_default-2025-08-03-11-06-21/gen_freq/gen.npz'
        elif opt.data_name == 'fmnist_28':
            config['freq_path'] = '/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/fmnist_28_eps1.0val_default-2025-08-04-02-38-27/gen_freq/gen.npz'
        elif opt.data_name == 'cifar10_32':
            config['freq_path'] = '/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/cifar10_32_eps1.0val_default-2025-08-04-05-27-32/gen_freq/gen.npz'
        elif opt.data_name == 'celeba_male_32':
            config['freq_path'] = '/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/celeba_male_32_eps1.0val_default-2025-08-04-05-25-57/gen_freq/gen.npz'
        elif opt.data_name == 'camelyon_32':
            config['freq_path'] = '/p/fzv6enresearch/FETA-Pro/exp/dp-feta-pro/camelyon_32_eps1.0val_default-2025-08-04-05-26-23/gen_freq/gen.npz'

        run(main, config)