import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as si
from matplotlib.ticker import MultipleLocator
import numpy as np

def normlize(f_list, scale=1):
    new_list = []
    max_len = 0
    for f in f_list:
        if len(f) > max_len:
            max_len = len(f)
    scale_list = []
    new_f_list = []
    max_len *= scale
    xnew = np.arange(0, max_len)/(max_len-1)
    for f in f_list:
        x = np.arange(0, len(f)) / (len(f) - 1)
        fun = si.interp1d(x, f)
        fnew = fun(xnew)
        new_f_list.append(fnew)
    return new_f_list

method_name = ['DP-GAN', 'DPDM', 'DP-FETA', 'FETA-Pro']
marker_shapes = ['X', 'o', '^', 's']  # X for DPGAN, Circle for DPDM, Triangle for DP-FETA, Square for DP-FETA++
colors = ['#B883D4', '#05B9E2', '#B1CE46','#D76364']  # Purple for DPGAN, Blue for DPDM, Greenish-Yellow for DP-FETA, Red for DP-FETA++

fontsize=17
lw = 4
ms = 4
matplotlib.rcParams.update({'font.size': fontsize, 'font.weight': 'normal'})
fig = plt.figure(figsize=(20, 4))
axes = fig.subplots(1, 5)

log_files = ['/p//DPImageBench/exp/dpgan/mnist_28_eps1.0trainval-2024-10-22-17-08-37/stdout.txt',
            '/p//DPImageBench/exp/dpdm/mnist_28_eps1.0trainval-2024-10-23-00-58-27/stdout.txt',
            '/p//DPImageBench/exp/dp-feta/mnist_28_eps1.0val_central_mean-2025-03-19-07-56-01/stdout.txt',
            '/p//PE-Refine/exp/dp-feta2/mnist_28_eps1.0val_merfeps10.0_fetasigma20_merf0.25-2025-05-25-23-57-45/stdout.txt',]
fid_list = []
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    
    fids = []
    for line in lines:
        if 'FID at iteration' in line:
            fid = float(line.strip().split(' ')[-1])
            fids.append(fid)
    fids.pop()
    fid_list.append(fids)
fid_list = normlize(fid_list)
small_x = 99999
for i, fids in enumerate(fid_list):
    x = np.arange(len(fids)) / (len(fids)-1) * 72000 / 32
    x = x[::2]  # Take every 2nd point
    fids = fids[::2]  # Take every 2nd point
    axes[0].plot(x, fids, label=method_name[i], marker=marker_shapes[i], markersize=ms, color=colors[i], zorder=100)
    # if i == 0:
    #     baseline_fid = fids[-1]
    #     # axes[0].axhline(y=fids[-1], color='black', linestyle='--', zorder=0)
    # else:
    #     for j in range(len(fids)-1):
    #         if fids[j] >= baseline_fid and fids[j+1] <= baseline_fid and x[j] < small_x:
    #             small_x = x[j]
print(x[-1], small_x, small_x/x[-1])

log_files = ['/p//DPImageBench/exp/dpgan/fmnist_28_eps1.0trainval-2024-10-22-15-52-27/stdout.txt',
            '/p//DPImageBench/exp/dpdm/fmnist_28_eps1.0trainval-2024-10-23-23-32-54/stdout.txt',
            '/p//DPImageBench/exp/dp-feta/fmnist_28_eps1.0val_central_mean-2025-03-20-23-02-01/stdout.txt',
            '/p//PE-Refine/exp/dp-feta2/fmnist_28_eps1.0val_merfeps1.0_fetasigma20_merf0.25-2025-05-26-16-18-46/stdout.txt',
            ]
fid_list = []
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    
    fids = []
    for line in lines:
        if 'FID at iteration' in line:
            fid = float(line.strip().split(' ')[-1])
            fids.append(fid)
    fids.pop()
    fid_list.append(fids)
fid_list = normlize(fid_list)
small_x = 99999
for i, fids in enumerate(fid_list):
    x = np.arange(len(fids)) / (len(fids)-1) * 72000 / 32
    # Filter out FID values > 200 for F-MNIST
    mask = fids <= 200
    x = x[mask]
    fids = fids[mask]
    axes[1].plot(x, fids, label=method_name[i], marker=marker_shapes[i], markersize=ms, color=colors[i], zorder=100)
    # if i == 0:
    #     baseline_fid = fids[-1]
    #     # axes[0, 0].axhline(y=fids[-1], color='black', linestyle='--', zorder=0)
    # else:
    #     for j in range(len(fids)-1):
    #         if fids[j] >= baseline_fid and fids[j+1] <= baseline_fid and x[j] < small_x:
    #             small_x = x[j]
print(x[-1], small_x, small_x/x[-1])

log_files = ['/p//DPImageBench/exp/dpgan/cifar10_32_eps1.0trainval-2024-10-22-21-23-15/stdout.txt',
            '/p//DPImageBench/exp/dpdm/cifar10_32_eps1.0trainval-2024-10-23-14-01-14/stdout.txt',
            '/p//DPImageBench/exp/dp-feta/cifar10_32_eps1.0val_central_mean-2025-03-23-07-48-09/stdout.txt',
            '/p//PE-Refine/exp/dp-feta2/cifar10_32_eps1.0val_merfeps1.0_fetasigma15_merf0.2-2025-05-29-13-52-20/stdout.txt',
            ]
fid_list = []
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    
    fids = []
    for line in lines:
        if 'FID at iteration' in line:
            fid = float(line.strip().split(' ')[-1])
            fids.append(fid)
    fids.pop()
    fid_list.append(fids)
fid_list = normlize(fid_list)
small_x = 99999
for i, fids in enumerate(fid_list):
    x = np.arange(len(fids)) / (len(fids)-1) * 105600 / 64
    # Downsample CIFAR-10 data to reduce points
    x = x[::2]  # Take every 2nd point
    fids = fids[::2]  # Take every 2nd point
    axes[2].plot(x, fids, label=method_name[i], marker=marker_shapes[i], markersize=ms, color=colors[i], zorder=100)
    if i == 0:
        baseline_fid = fids[-1]
        # axes[0, 0].axhline(y=fids[-1], color='black', linestyle='--', zorder=0)
    else:
        for j in range(len(fids)-1):
            if fids[j] >= baseline_fid and fids[j+1] <= baseline_fid and x[j] < small_x:
                small_x = x[j]
print(x[-1], small_x, small_x/x[-1])

log_files = ['/p//DPImageBench/exp/dpgan/celeba_male_32_eps1.0trainval-2024-10-22-23-05-09/stdout.txt',
            '/p//DPImageBench/exp/dpdm/celeba_male_32_eps1.0trainval-2024-10-24-00-50-04/stdout.txt',
            '/p//DPImageBench/exp/dp-feta/celeba_male_32_eps1.0sen_central_mean-2025-03-18-02-23-26/stdout.txt',
            '/p//PE-Refine/exp/dp-feta2/celeba_male_32_eps1.0val_merfeps10.0_fetasigma15_merf0.15-2025-05-29-09-37-02/stdout.txt',
            ]
fid_list = []
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    
    fids = []
    for line in lines:
        if 'FID at iteration' in line:
            fid = float(line.strip().split(' ')[-1])
            fids.append(fid)
    if 'feta2' in log:
        fids = fids[5:]
    fids.pop()
    fid_list.append(fids)
scale_r = 3
fid_list = normlize(fid_list, scale=scale_r)
print(len(fid_list[0]))
small_x = 99999
for i, fids in enumerate(fid_list):
    x = np.arange(len(fids)) / (len(fids)-1) * 345600 / 64
    mask = fids <= 300
    x = x[mask]
    fids = fids[mask]
    x = x[::19]  # Take every 13th point
    fids = fids[::19]  # Take every 13th point
    axes[3].plot(x, fids, label=method_name[i], marker=marker_shapes[i], markersize=ms, color=colors[i], zorder=100)
    if i == 0:
        baseline_fid = fids[-1]
    else:
        for j in range(len(fids)-1):
            if fids[j] >= baseline_fid and fids[j+1] <= baseline_fid and x[j] < small_x:
                small_x = x[j]
print(x[-1], small_x, small_x/x[-1])

log_files = ['/p//DPImageBench/exp/dpgan/camelyon_32_eps1.0trainval-2024-10-22-21-20-01/stdout.txt',
            '/p//DPImageBench/exp/dpdm/camelyon_32_eps1.0trainval-2024-10-24-08-46-55/stdout.txt',
            '/p//DPImageBench/exp/dp-feta/camelyon_32_eps1.0val_central_mean-2025-03-21-05-02-55/stdout.txt',
            '/p//PE-Refine/exp/dp-feta2/camelyon_32_eps1.0val_merfeps1.0_fetasigma10_merf0.15-2025-05-27-20-59-08/stdout.txt',
            ]
fid_list = []
for log in log_files:
    with open(log, 'r') as f:
        lines = f.readlines()
    
    fids = []
    for line in lines:
        if 'FID at iteration' in line:
            fid = float(line.strip().split(' ')[-1])
            fids.append(fid)
    if 'feta2' in log:
        fids = fids[5:]
    fid_list.append(fids)

scale_r = 3
fid_list = normlize(fid_list, scale=scale_r)
print(len(fid_list[0]))
small_x = 99999
for i, fids in enumerate(fid_list):
    x = np.arange(len(fids)) / (len(fids)-1) * 211200 / 64
    # Filter out FID values > 140 for Camelyon
    mask = fids <= 140
    x = x[mask]
    fids = fids[mask]
    # Downsample Camelyon data to reduce points
    x = x[::12]  # Take every 12th point
    fids = fids[::12]  # Take every 12th point
    axes[4].plot(x, fids, label=method_name[i], marker=marker_shapes[i], markersize=ms, color=colors[i], zorder=100)
    if i == 0:
        baseline_fid = fids[-1]
        # axes[0, 0].axhline(y=fids[-1], color='black', linestyle='--', zorder=0)
    else:
        for j in range(len(fids)-1):
            if fids[j] >= baseline_fid and fids[j+1] <= baseline_fid and x[j] < small_x:
                small_x = x[j]
print(x[-1], small_x, small_x/x[-1])

axes[0].set_title('MNIST')
axes[1].set_title('F-MNIST')
axes[2].set_title('CIFAR-10')
axes[3].set_title('CelebA')
axes[4].set_title('Camelyon')

axes[0].set_xlabel('Fine-tuning Iteration')
axes[1].set_xlabel('Fine-tuning Iteration')
axes[2].set_xlabel('Fine-tuning Iteration')
axes[3].set_xlabel('Fine-tuning Iteration')
axes[4].set_xlabel('Fine-tuning Iteration')
axes[0].set_ylabel('FID')

axes[0].legend(prop={'size': 11}, facecolor='white', framealpha=1, loc='upper right')
axes[0].get_legend().set_zorder(200)
axes[1].legend(prop={'size': 11}, facecolor='white', framealpha=1, loc='upper right')
axes[1].get_legend().set_zorder(200)
axes[2].legend(prop={'size': 11}, facecolor='white', framealpha=1, loc='upper right')
axes[2].get_legend().set_zorder(200)
axes[3].legend(prop={'size': 11}, facecolor='white', framealpha=1, loc='upper right')
axes[3].get_legend().set_zorder(200)
axes[4].legend(prop={'size': 11}, facecolor='white', framealpha=1, loc='upper right')
axes[4].get_legend().set_zorder(200)

axes[0].grid(color='lightgrey', linewidth=1, zorder=0)
axes[1].grid(color='lightgrey', linewidth=1, zorder=0)
axes[2].grid(color='lightgrey', linewidth=1, zorder=0)
axes[3].grid(color='lightgrey', linewidth=1, zorder=0)
axes[4].grid(color='lightgrey', linewidth=1, zorder=0)

axes[0].set_yticks([20, 120, 220, 300], ['20', '120', '220', '300'])
axes[1].set_yticks([30, 80, 130, 180], ['30', '80', '130', '180'])
axes[2].set_yticks([100, 200, 300, 400], ['100', '200', '300', '400'])
axes[3].set_yticks([40, 120, 200, 300], ['40', '120', '200', '300'])
axes[4].set_yticks([35, 70, 105, 140], ['35', '70', '105', '140'])

plt.tight_layout()
plt.savefig('test.png')
plt.savefig('fid_curve.pdf')

# axes[0, 0].set_xlim([0, 74000/32])
# axes[0, 1].set_xlim([0, 74000/32])
# axes[1, 0].set_xlim([0, 440000/32])
# axes[1, 1].set_xlim([0, 48000/32])
# axes[0, 0].set_ylim([10, 240])
# axes[0, 1].set_ylim([30, 280])
# axes[1, 0].set_ylim([45, 370])
# axes[1, 1].set_ylim([48, 110])
# axes[0, 0].xaxis.set_major_locator(MultipleLocator(500)) 
# axes[0, 1].xaxis.set_major_locator(MultipleLocator(500)) 
# axes[1, 0].xaxis.set_major_locator(MultipleLocator(3000)) 
# axes[1, 1].xaxis.set_major_locator(MultipleLocator(500)) 
# axes[0, 0].set_xticks([0, 20000, 40000, 60000], ['0', '20', '40', '60'])
# axes[0, 1].set_xticks([0, 20000, 40000, 60000], ['0', '20', '40', '60'])
# axes[1, 0].set_xticks([0, 100000, 200000, 300000, 400000], ['0', '100', '200', '300', '400'])
# axes[1, 1].set_xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])
# axes[0, 3].set_yticks([100, 200, 300, 400], ['100', '200', '300', '400'])
# axes[0, 1].set_yticks([30, 100, 200], ['30', '100', '200'])
# axes[0, 1].set_xticks([0, 20000, 40000, 60000], ['0', '20', '40', '60'])
# axes[1, 0].set_xticks([0, 100000, 200000, 300000, 400000], ['0', '100', '200', '300', '400'])
# axes[1, 1].set_xticks([0, 10000, 20000, 30000, 40000], ['0', '10', '20', '30', '40'])

# axes[0].grid(color='lightgrey', linewidth=1, zorder=0)
# axes[1].grid(color='lightgrey', linewidth=1, zorder=0)
# axes[2].grid(color='lightgrey', linewidth=1, zorder=0)
# axes[3].grid(color='lightgrey', linewidth=1, zorder=0)
# axes[4].grid(color='lightgrey', linewidth=1, zorder=0)

# plt.tight_layout()
# plt.savefig('test.png')
# plt.savefig('fid_curve.pdf')
