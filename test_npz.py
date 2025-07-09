import numpy as np
import cv2
import os
import argparse
import random

log_path = '/p/fzv6enresearch/PE-Refine/exp/dp-feta2/mnist_28_eps1.0val_fetasigma20_merf0.25_nogan2_checkfid-2025-06-21-05-38-23/stdout.txt'

with open(log_path, 'r') as f:
    data = f.readlines()
fid = []
flag = False
for line in data:
    if 'Number of total epochs: 150' in line:
        flag = True
    if 'FID at iteration' in line and flag:
        fid.append(str(round(float(line.strip().split(' ')[-1]), 2)))
print('\t'.join(fid[:11]))