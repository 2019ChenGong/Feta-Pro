from matplotlib import pyplot as plt
from matplotlib import font_manager
import seaborn as sns
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib.cm as cm

def draw_heat_w(axes_l, fig, dataf, vmin, vmax, is_left=True):
    font = {'family': 'Times New Roman', 'weight': 'bold'}

    cmap = cm.get_cmap('PuBu').copy()
    cmap.set_under(color='white', alpha=0.0)  
    cmap.set_over(color='blue', alpha=0.2)  
    cmap.set_bad(color='red', alpha=0)      
    if is_left:
        axes_l.set_ylabel(r'$\sigma_t$', fontsize=28)
        datalist_shortname = ["2", "5", "10", "20", "30"]
        axes_l.set_yticks(range(len(datalist_shortname)))
        axes_l.set_yticklabels(datalist_shortname, fontsize=24, fontdict=font)
    else:
        axes_l.set_ylabel('')
        axes_l.set_yticks([])

    axes_l.set_xlabel(r'$\sigma_f$', fontsize=28)
    axes_l.set_xticks([0, 1, 2, 3, 4])
    axes_l.set_xticklabels([115, 61, 42, 26, 20], fontsize=24, fontdict=font)

    for edge, spine in axes_l.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)  
    mid_point = (vmin+vmax)/2
    im = axes_l.imshow(dataf, cmap=cmap, norm=norm, aspect=1.0)

    for m in range(dataf.shape[0]):
        for n in range(dataf.shape[1]):
            value = dataf[m, n]
            text_color = "black" if value < mid_point else "white"
            text_value = f'{value:.2f}' if not np.isnan(value) else '-'
            axes_l.text(n, m, text_value, ha='center', va='center', color=text_color, fontsize=24, fontdict=font)

    cax = fig.add_axes([axes_l.get_position().x1+0.02,   
                         axes_l.get_position().y0,        
                         0.025,                             
                         axes_l.get_position().height])    
    clb = fig.colorbar(im, cax=cax)
    for l in clb.ax.yaxis.get_ticklabels():
        l.set_family('Times New Roman')
        l.set_fontweight('bold')
        l.set_fontsize(23)
    clb.update_ticks()

raw_acc_mnist = '''92.7	90.7	93.8	92.7	93.5
94.6	95.4	96.7	96.3	95.5
94.8	95.3	96.4	96.5	95.4
95	95	96.3	97.3	95.7
95.7	94.8	96.4	96.4	96.1'''
raw_acc_fmnist = '''76.9	73.7	75.8	76.8	81.1
78.9	81.7	81.5	82.4	82.3
79.9	80.5	80.6	82.5	80.9
79.3	80.2	80.2	83.4	80.7
79.3	80.5	81.3	83	82.1'''
raw_fid_mnist = '''27.7	31.1	21.8	19.7	14.9
16.8	15.9	10.6	9.5	11.3
24.5	14.3	10.7	9.3	10.5
19.9	17.8	10.3	8.0	9.8
16.3	18.2	11.1	9.1	9.5'''
raw_fid_fmnist = '''48.1	61.4	57.8	44.2	30
42.7	34.5	33.9	28.8	29.6
40.1	39.3	34.5	27.9	27.3
41.7	40.1	35.4	27.8	28.5
40.9	40.2	37.5	28.8	28.4'''

acc_mnist = np.array([raw_acc_mnist_i.split('\t') for raw_acc_mnist_i in raw_acc_mnist.split('\n')], dtype='float')
acc_fmnist = np.array([raw_acc_mnist_i.split('\t') for raw_acc_mnist_i in raw_acc_fmnist.split('\n')], dtype='float')
fid_mnist = np.array([raw_acc_mnist_i.split('\t') for raw_acc_mnist_i in raw_fid_mnist.split('\n')], dtype='float')
fid_fmnist = np.array([raw_acc_mnist_i.split('\t') for raw_acc_mnist_i in raw_fid_fmnist.split('\n')], dtype='float')

fig = plt.figure(figsize=(14, 6))
axes = fig.subplots(1, 2)
print(np.min(acc_mnist), np.max(acc_mnist))
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.subplots_adjust(wspace=0.35)
draw_heat_w(axes[0], fig, acc_mnist, 92, 98, is_left=True)
draw_heat_w(axes[1], fig, acc_fmnist, 73, 84, is_left=False)
fig.savefig("hm_acc.png", bbox_inches='tight')
fig.savefig("hm_acc.pdf", bbox_inches='tight')

fig = plt.figure(figsize=(14, 6))
axes = fig.subplots(1, 2)
plt.tight_layout(rect=[0, 0, 0.9, 1])
plt.subplots_adjust(wspace=0.35)
print(np.min(fid_mnist), np.max(fid_mnist))
draw_heat_w(axes[0], fig, fid_mnist, 8, 32, is_left=True)
draw_heat_w(axes[1], fig, fid_fmnist, 27, 62, is_left=False)
fig.savefig("hm_fid.png", bbox_inches='tight')
fig.savefig("hm_fid.pdf", bbox_inches='tight')