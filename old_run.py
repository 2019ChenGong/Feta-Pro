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
from evaluation.evaluator import Evaluator
os.environ['MKL_NUM_THREADS'] = "1"
from models.dp_merf import DP_MERF

def main(config):

    initialize_environment(config)
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)
    if config.pretrain.mode != 'feta_merf_nogan' and config.pretrain.mode != 'feta_merf_nogan2':
        if config.setup.global_rank == 0:
            merf_model = DP_MERF(config.model.merf, config.setup.local_rank)
            merf_model.train(sensitive_train_loader, config.train.merf)
            syn_data, syn_labels = merf_model.generate(config.gen.merf)
            torch.save(torch.tensor([merf_model.noise_factor]), os.path.join(config.gen.merf.log_dir, 'pc.pth'))
        dist.barrier()
        syn = np.load(os.path.join(config.gen.merf.log_dir, 'gen.npz'))
        syn_data, syn_labels = syn["x"], syn["y"]
        if torch.load(os.path.join(config.gen.merf.log_dir, 'pc.pth')).item() != 0:
            config.train.dp['privacy_history'].append([torch.load(os.path.join(config.gen.merf.log_dir, 'pc.pth')).item(), 1, 1])
    if config.pretrain.mode != 'feta_merf_nogan' and config.pretrain.mode != 'feta_merf_nogan2':
        merf_train_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
        merf_train_loader = torch.utils.data.DataLoader(dataset=merf_train_set, shuffle=True, drop_last=True, batch_size=config.pretrain.batch_size, num_workers=16)

    model, config = load_model(config)

    # model.curiosity_pretrain(public_train_loader, config.pretrain)
    if config.pretrain.mode == 'merf_feta':
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2
        model.pretrain(merf_train_loader, config.pretrain)
        config.pretrain.log_dir = config.pretrain.log_dir + '_feta'
        config.pretrain.n_epochs = config.pretrain.n_epochs1
        config.pretrain.batch_size = config.pretrain.batch_size1
        model.pretrain(public_train_loader, config.pretrain)
    elif config.pretrain.mode == 'feta_merf_nogan':
        config.pretrain.n_epochs = config.pretrain.n_epochs1
        config.pretrain.batch_size = config.pretrain.batch_size1
        model.pretrain(public_train_loader, config.pretrain)
        config.pretrain.log_dir = config.pretrain.log_dir + '_merf'
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2
        merf_model = DP_MERF(config.model.merf, config.setup.local_rank)
        model.pretrain_merf(sensitive_train_loader, merf_model, config.pretrain, config.train.merf)
        config.train.dp['privacy_history'].append([model.noise_factor, 1, 1])
    elif config.pretrain.mode == 'feta_merf_nogan2':
        config.pretrain.n_epochs = config.pretrain.n_epochs1
        config.pretrain.batch_size = config.pretrain.batch_size1
        model.pretrain(public_train_loader, config.pretrain)
        config.pretrain.log_dir = config.pretrain.log_dir + '_merf'
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2

        merf_model = DP_MERF(config.model.merf, config.setup.local_rank)
        model_sur, config_sur = load_model(config)
        config_sur.gen.log_dir = config.pretrain.log_dir + "/gen"
        model_sur.pretrain_merf(sensitive_train_loader, merf_model, config.pretrain, config.train.merf)
        syn_data, syn_labels = model_sur.generate(config_sur.gen, config.model.sampler)
        del model_sur
        dist.barrier()
        syn = np.load(os.path.join(config_sur.gen.log_dir, 'gen.npz'))
        syn_data, syn_labels = syn["x"], syn["y"]
        merf_train_set = TensorDataset(torch.from_numpy(syn_data).float(), torch.from_numpy(syn_labels).long())
        merf_train_loader = torch.utils.data.DataLoader(dataset=merf_train_set, shuffle=True, drop_last=True, batch_size=config.pretrain.batch_size, num_workers=16)
        config.pretrain.log_dir = config.pretrain.log_dir + '_2'
        model.pretrain(merf_train_loader, config.pretrain)
    elif config.pretrain.mode == 'feta_merf':
        config.pretrain.n_epochs = config.pretrain.n_epochs1
        config.pretrain.batch_size = config.pretrain.batch_size1
        model.pretrain(public_train_loader, config.pretrain)
        config.pretrain.log_dir = config.pretrain.log_dir + '_merf'
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2
        model.pretrain(merf_train_loader, config.pretrain)
    elif config.pretrain.mode == 'feta_merf_mix':
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2
        pretrain_set = ConcatDataset([merf_train_set, public_train_loader.dataset])
        model.pretrain(torch.utils.data.DataLoader(dataset=pretrain_set, shuffle=True, drop_last=True, batch_size=config.pretrain.batch_size, num_workers=16), config.pretrain)
    elif config.pretrain.mode == 'merf':
        config.train.dp['privacy_history'] = config.train.dp['privacy_history'][-1:]
        config.pretrain.n_epochs = config.pretrain.n_epochs2
        config.pretrain.batch_size = config.pretrain.batch_size2
        model.pretrain(merf_train_loader, config.pretrain)
    else:
        raise NotImplementedError

    if 'syn4train' in config.train:
        model.curiosity_train(sensitive_train_loader, config.train)
    elif 'cut_noise' in config.train:
        model.cut_train(sensitive_train_loader, config.train)
    else:
        model.train(sensitive_train_loader, config.train)

    syn_data, syn_labels = model.generate(config.gen)

    evaluator = Evaluator(config)
    evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader)
    


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

# python run.py setup.n_gpus_per_node=4 setup.master_port=6667 eval.mode=val pretrain.n_epochs1=100 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=1.0 public_data.central.sigma=50 train.n_splits=256 pretrain.mode=feta_merf gen.merf.cache=/p/fzv6enresearch/PE-Refine/exp/dp-feta2/celeba_male_64_eps10.0val_merfeps1.0_fetasigma50-2025-05-24-12-40-10/gen_merf -m DP-FETA2 -dn celeba_male_64 -e 10.0 -ed val_merfeps1.0_fetasigma50

# python run.py setup.n_gpus_per_node=4 setup.master_port=6667 eval.mode=val pretrain.n_epochs1=1000 pretrain.n_epochs2=10 pretrain.batch_size1=50 pretrain.batch_size2=256 train.merf.dp.epsilon=1.0 public_data.central.sigma=50 train.n_splits=256 pretrain.mode=feta_merf -m DP-FETA2 -dn celeba_male_128 -e 10.0 -ed val_merfeps1.0_fetasigma50