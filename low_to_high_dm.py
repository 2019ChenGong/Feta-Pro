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

from utils.utils import parse_config

    
def main(config):
    initialize_environment(config)

    model, config = load_model(config)

    label = [0]
    syn_data = torch.randn((1, 3, 32, 32)) # your image belongs to (-1, 1)
    syn_data = F.interpolate(syn_data, size=[64, 64])

    with torch.no_grad():
        new_samples = model.api._image_variation(syn_data, label, variation_degree=50) # variation_degree belongs to (0, 100)
    new_samples = new_samples / 2 + 0.5
    new_samples = torch.Tensor(new_samples)
    save_image(new_samples.clamp(0., 1.), 'temp.png')
    print('finished')

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default="configs")
    parser.add_argument('--method', '-m', default="DP-LDM")
    parser.add_argument('--epsilon', '-e', default="10.0")
    parser.add_argument('--data_name', '-dn', default="cifar10_32")
    parser.add_argument('--exp_description', '-ed', default="")
    parser.add_argument('--resume_exp', '-re', default=None)
    parser.add_argument('--config_suffix', '-cs', default="")
    opt, unknown = parser.parse_known_args()
    config = parse_config(opt, unknown)

    run(main, config)

# python low_to_high_dm.py -m PE -dn cifar10_32