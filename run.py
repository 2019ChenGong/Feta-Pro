import os
import sys
import argparse
import torch
import random
import numpy as np

from models.model_loader import load_model
from data.dataset_loader import load_data
from utils.utils import initialize_environment, run, parse_config
from evaluation.evaluator import Evaluator
os.environ['MKL_NUM_THREADS'] = "1"
def set_deterministic_seeds(seed=0):
    """
    设置所有随机种子以确保可复现性
    """
    # 固定 Python 内置 random
    random.seed(seed)

    # 固定 NumPy 随机种子
    np.random.seed(seed)

    # 固定 PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # 固定 PyTorch GPU 随机种子（所有可用 GPU）
    torch.cuda.manual_seed_all(seed)

    # 强制使用确定性算法（重要！）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # ⚠️ 必须关闭，否则会引入随机性

    # 设置全局计算环境
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(config):
    # set_deterministic_seeds(0)

    initialize_environment(config)
    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)

    model, config = load_model(config)

    model.pretrain(public_train_loader, config.pretrain)

    model.train(sensitive_train_loader, config.train)

    # syn_data, syn_labels = model.generate(config.gen)

    # evaluator = Evaluator(config)
    # evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader)
    


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