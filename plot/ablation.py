import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import mpl_toolkits.axisartist as axisartist

fontsize=19
matplotlib.rcParams.update({'font.size': fontsize, 'font.weight': 'normal'})
#matplotlib.rcParams.update({'font.size': 18, 'font.family': 'Times New Roman'})

colors = ['#FFBE7A', '#8ECFC9', '#82B0D2', '#E0CBEF', '#FA7F6F', '#F7B7D2']
def create_multi_bars(ax, labels, datas, method, tick_step=1, group_gap=0.2, bar_gap=0, metric='fid'):

    ax.grid(color='lightgrey', linewidth=1, zorder=0)
    bwith = 0.7
    #ax = plt.gca()#获取边框
    
    # ax.axis["bottom"].set_axisline_style("->", size = 1)
    # ax.axis["left"].set_axisline_style("->", size = 1)
    # ax.axis['right'].set_visible(False)
    # ax.axis['top'].set_visible(False)

    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)


    ticks = np.arange(len(labels)) * tick_step
    group_num = len(datas)
    group_width = tick_step - group_gap
    bar_span = group_width / group_num
    bar_width = bar_span - bar_gap
    baseline_x = ticks - (group_width - bar_span) / 2

    if metric == 'fid':
        best_idx = np.argmax(-datas[:, 0])
    else:
        best_idx = np.argmax(datas[:, 0])
    for index, y in enumerate(datas):
        # error_params=dict(elinewidth=1.5,ecolor=edge_colors[method[index]],capsize=5)
        # error_params=dict(elinewidth=1,ecolor='black',capsize=5)
        # ax.bar(baseline_x + index*bar_span, y, bar_width, label=method[index], color=colors[method[index]], 
        #         zorder=100, edgecolor=edge_colors[method[index]],
        #         yerr=error[index], error_kw=error_params, hatch=hatches[index])
        # ax.bar(baseline_x + index*bar_span, y, bar_width, label=method[index], color=colors[method[index]], 
        #         zorder=100, edgecolor='black', hatch=hatches[method[index]])
        ax.bar(baseline_x + index*bar_span, y, bar_width*0.7, label=method[index], color=colors[index], 
                zorder=200, edgecolor='black')
        # if index != 0:
        #     if metric == 'fid':
        #         improve = (datas[index, 0] - datas[0, 0]) / datas[0, 0] * 100
        #         improve = str(int(improve)) + '%'
        #         x_shift = 0.06
        #     else:
        #         improve = (datas[index, 0] - datas[0, 0])
        #         improve = str(round(improve, 1)) + '%'
        #         x_shift = 0.07
        #     if improve[0] != '-':
        #         improve = '+' + improve
        #     if index == best_idx:
        #         ax.text(baseline_x + index*bar_span - x_shift, y+1.0, str(improve), fontsize=13, fontweight='bold')
        #     else:
        #         ax.text(baseline_x + index*bar_span - x_shift, y+1.0, str(improve), fontsize=13)
        if metric == 'fid':
            improve = datas[index, 0]
            if improve < 10:
                x_shift = 0.04
            else:
                x_shift = 0.05
        else:
            improve = datas[index, 0]
            # improve = (datas[index, 0] - datas[0, 0])
            # improve = str(round(improve, 1)) + '%'
            x_shift = 0.05
        if metric == 'fid':
            y_shift = 1.3
            if datas[0, 0] > 100:
                y_shift = 3
        else:
            y_shift = 0.8
        # if improve[0] != '-':
        #     improve = '+' + improve
        if index == best_idx:
            ax.text(baseline_x + index*bar_span - x_shift, y+y_shift, str(improve), fontsize=13, fontweight='bold')
        else:
            ax.text(baseline_x + index*bar_span - x_shift, y+y_shift, str(improve), fontsize=13)

    ax.set_xticks([])
    ax.set_title(labels[0], fontsize=fontsize)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    for spine in ax.spines.values():
        spine.set_zorder(300)

# plt.tight_layout()
# plt.subplots_adjust(left=None, bottom=None, right=None, top=0.75, wspace=0.2, hspace=0.3)

methods = ['DPDM', 'DP-FETA', 'FETA-Pro$_f$', 'FETA-Pro$_{mix}$', 'FETA-Pro$_{ft}$', 'FETA-Pro']
colors = ['#FFBE7A', '#E0CBEF', '#82B0D2', '#8ECFC9', '#FA7F6F', '#F7B7D2']
xlabels = ['MNIST', 'F-MNIST', 'CelebA', 'Camelyon']


accs_mnist = [89.2, 96.7, 96.7, 95.6, 96.8, 97.3]
fids_mnist = [36.1, 9.8, 10.1, 13.7, 11.1, 8.0]
accs_fmnist = [76.4, 81.5, 81.8, 81.7, 82.2, 83.4]
fids_fmnist = [53.5, 31.1, 29.8, 31.4, 31.0, 27.8]
accs_celeba = [74.5, 78.0, 78.2, 82.3, 84.5, 90.0]
fids_celeba = [153.9, 58.0, 60.8, 60.2, 79.3, 48.0]
accs_camelyon = [80.6, 78.2, 79.9, 77.3, 83.3, 84.0]
fids_camelyon = [111.9, 45.2, 39.6, 52.8, 37.7, 31.0]


raw_data = '''89.2	36.1	95.6	13.7	96.7	10.1	96.7	9.8	96.8	11.1	97.3	8.0
76.4	53.5	81.7	31.4	81.8	29.8	81.5	31.1	82.2	31	83.4	27.8
74.5	153.9	82.3	60.2	78.2	60.8	78	58	84.5	79.3	90.0	48.0
80.6	111.9	77.3	52.8	77.9	39.6	79.9	45.2	83.3	37.7	84.0	31.0'''
raw_data = raw_data.split('\n')
for i, raw_data_i in enumerate(raw_data):
    raw_data_i = raw_data_i.split()
    if i == 0:
        for j, raw_data_i_j in enumerate(raw_data_i):
            idx = j // 2
            if idx == 2:
                idx = 3
            elif idx == 3:
                idx = 2
            if j % 2 == 0:
                accs_mnist[idx] = float(raw_data_i_j)
            else:
                fids_mnist[idx] = float(raw_data_i_j)
    elif i == 1:
        for j, raw_data_i_j in enumerate(raw_data_i):
            idx = j // 2
            if idx == 2:
                idx = 3
            elif idx == 3:
                idx = 2
            if j % 2 == 0:
                accs_fmnist[idx] = float(raw_data_i_j)
            else:
                fids_fmnist[idx] = float(raw_data_i_j)
    elif i == 2:
        for j, raw_data_i_j in enumerate(raw_data_i):
            idx = j // 2
            if idx == 2:
                idx = 3
            elif idx == 3:
                idx = 2
            if j % 2 == 0:
                accs_celeba[idx] = float(raw_data_i_j)
            else:
                fids_celeba[idx] = float(raw_data_i_j)
    elif i == 3:
        for j, raw_data_i_j in enumerate(raw_data_i):
            idx = j // 2
            if idx == 2:
                idx = 3
            elif idx == 3:
                idx = 2
            if j % 2 == 0:
                accs_camelyon[idx] = float(raw_data_i_j)
            else:
                fids_camelyon[idx] = float(raw_data_i_j)

    
fig = plt.figure(figsize=(19, 6), dpi=200)
axes = fig.subplots(2, 4)
data = np.array([fids_mnist, fids_fmnist, fids_celeba, fids_camelyon]).T
create_multi_bars(axes[0, 0], xlabels, data, methods)
lines, labels = axes[0, 0].get_legend_handles_labels()
axes[0, 0].cla()

data = np.array([fids_mnist, fids_fmnist, fids_celeba, fids_camelyon][:1]).T
create_multi_bars(axes[0, 0], xlabels[:1], data, methods)
data = np.array([fids_mnist, fids_fmnist, fids_celeba, fids_camelyon][1:2]).T
create_multi_bars(axes[0, 1], xlabels[1:2], data, methods)
data = np.array([fids_mnist, fids_fmnist, fids_celeba, fids_camelyon][2:3]).T
create_multi_bars(axes[0, 2], xlabels[2:3], data, methods)
data = np.array([fids_mnist, fids_fmnist, fids_celeba, fids_camelyon][3:]).T
create_multi_bars(axes[0, 3], xlabels[3:], data, methods)
data = np.array([accs_mnist, accs_fmnist, accs_celeba, accs_camelyon][:1]).T
create_multi_bars(axes[1, 0], xlabels[:1], data, methods, metric='acc')
data = np.array([accs_mnist, accs_fmnist, accs_celeba, accs_camelyon][1:2]).T
create_multi_bars(axes[1, 1], xlabels[1:2], data, methods, metric='acc')
data = np.array([accs_mnist, accs_fmnist, accs_celeba, accs_camelyon][2:3]).T
create_multi_bars(axes[1, 2], xlabels[2:3], data, methods, metric='acc')
data = np.array([accs_mnist, accs_fmnist, accs_celeba, accs_camelyon][3:]).T
create_multi_bars(axes[1, 3], xlabels[3:], data, methods, metric='acc')

axes[0, 0].set_ylabel('FID', fontsize=fontsize, weight='normal', labelpad=15)
axes[1, 0].set_ylabel('Acc (%)', fontsize=fontsize, weight='normal', labelpad=4.5)

axes[0, 0].set_ylim([0, 45])
axes[0, 1].set_ylim([20, 60])
# axes[0, 2].set_ylim([20, 80])
# axes[0, 3].set_ylim([20, 80])
axes[0, 2].set_ylim([0, 180]) 
axes[0, 3].set_ylim([0, 130])
axes[1, 0].set_ylim([85, 105])
# axes[1, 0].plot([baseline_x + index*bar_span, baseline_x + index*bar_span], [70,])
axes[1, 1].set_ylim([75, 90])
axes[1, 2].set_ylim([70, 100])
axes[1, 3].set_ylim([75, 90])

# leg = fig.legend(lines, labels, loc='upper center', ncol=8, facecolor='white', edgecolor='black', shadow=True, columnspacing=3, borderaxespad=0.1)
# for legobj in leg.legendHandles:
#     legobj.set_linewidth(3.0)
leg = fig.legend(lines, labels, loc='upper center', ncol=6, facecolor='white', edgecolor='black', shadow=True, columnspacing=1, borderaxespad=0.4, fontsize=20)
for legobj in leg.legendHandles:
    legobj.set_linewidth(1.0)

plt.tight_layout()
plt.subplots_adjust(left=None, bottom=None, right=None, top=0.8, wspace=0.2, hspace=0.3)
fig.savefig("ablation.png", bbox_inches='tight')
fig.savefig("ablation.pdf", bbox_inches='tight')
print(matplotlib.get_cachedir())