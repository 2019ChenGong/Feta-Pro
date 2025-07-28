import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

# Set up figure and subplots
fig = plt.figure(figsize=(8.5, 3.5))
axs = fig.subplots(1, 2)

# Define methods, privacy budgets, colors, and markers
methods = ["Acc", "FID"]
eps = ['0.2', '1.0', '5.0', '10', '15', '20']
colors = ['#D76364', '#05B9E2']
markers = ['v', 'D']

# MNIST data
mnists = '''86.2 97.1 98.1 98.6 98.5 98.5
76.2 8.8 3.9 2.5 2.6 2.3'''

# F-MNIST data
fmnists = '''70.2 82.3 86.0 88.3 88.1 88.7
99.7 27.9 16.1 11.9 11.2 10.5'''

# Process data
mnists = [[float(acc_i) for acc_i in acc.split()] for acc in mnists.split('\n')]
fmnists = [[float(acc_i) for acc_i in acc.split()] for acc in fmnists.split('\n')]

# Set line width and font size
lw = 1.3
fontsize = 14

# Plot MNIST subplot
method_idx = [0, 1]
for idx in method_idx:
    method = methods[idx]
    mnist = mnists[idx]
    axs[0].plot(mnist, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
axs[0].set_xticks([i for i in range(6)], eps)
axs[0].set_ylabel("Acc (%)", fontsize=fontsize, labelpad=2)
axs[0].set_xlabel("Privacy Budget $\epsilon$ (MNIST)", fontsize=fontsize)
axs[0].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
axs[0].set_ylim([-5, 105])  # Add padding to y-axis range
axs[0].tick_params(axis='both', which='major', labelsize=14.5)
axs[0].legend(fontsize=12.5)

# Plot F-MNIST subplot
for idx in method_idx:
    method = methods[idx]
    fmnist = fmnists[idx]
    axs[1].plot(fmnist, label=method, lw=lw, markersize=6.5, color=colors[idx], marker=markers[idx])
axs[1].set_xticks([i for i in range(6)], eps)
axs[1].set_xlabel("Privacy Budget $\epsilon$ (F-MNIST)", fontsize=fontsize, ha='left', position=(0, 0))
axs[1].set_ylabel("FID", fontsize=fontsize, labelpad=2)
axs[1].set_yticks([0, 20, 40, 60, 80, 100])  # Set y-axis ticks
axs[1].set_ylim([-5, 105])  # Add padding to y-axis range
axs[1].yaxis.set_label_position("right")  # FID label on right
axs[1].yaxis.tick_right()  # Move y-axis ticks to right
axs[1].tick_params(axis='both', which='major', labelsize=14.5)
axs[1].legend(fontsize=12.5)

# Add grids
axs[0].grid(color='lightgrey', linewidth=1.0, zorder=0)
axs[1].grid(color='lightgrey', linewidth=1.0, zorder=0)

# Adjust subplot spacing
fig.subplots_adjust(wspace=0.1, hspace=0.27)

# Save figures
fig.savefig("eps_change.png", bbox_inches='tight')
fig.savefig("eps_change_0117.pdf", bbox_inches='tight')