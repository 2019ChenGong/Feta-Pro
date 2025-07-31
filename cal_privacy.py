from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import root_scalar
import argparse

from opacus.accountants.accountant import IAccountant
from opacus.accountants.analysis import rdp as privacy_analysis
from opacus.accountants.utils import get_noise_multiplier
from typing import Optional
from utils.utils import initialize_environment, run, parse_config

from opacus.accountants import create_accountant
from data.dataset_loader import load_data


MAX_SIGMA = 1e6


class RDPAccountant(IAccountant):
    DEFAULT_ALPHAS = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))

    def __init__(self):
        super().__init__()

    def step(self, *, noise_multiplier: float, sample_rate: float):
        if len(self.history) >= 1:
            last_noise_multiplier, last_sample_rate, num_steps = self.history.pop()
            if (
                last_noise_multiplier == noise_multiplier
                and last_sample_rate == sample_rate
            ):
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps + 1)
                )
            else:
                self.history.append(
                    (last_noise_multiplier, last_sample_rate, num_steps)
                )
                self.history.append((noise_multiplier, sample_rate, 1))

        else:
            self.history.append((noise_multiplier, sample_rate, 1))

    def get_privacy_spent(
        self, *, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ) -> Tuple[float, float]:
        if not self.history:
            return 0, 0
        flag = False
        if alphas is None:
            alphas = self.DEFAULT_ALPHAS
            flag = True
        rdp = [
                privacy_analysis.compute_rdp(
                    q=sample_rate,
                    noise_multiplier=noise_multiplier,
                    steps=num_steps,
                    orders=alphas,
                )
                for (noise_multiplier, sample_rate, num_steps) in self.history
            ]
        if not flag:
            # print(rdp)
            ratio = list(np.array(rdp) / sum(rdp) * 100)
            ratio = [str(round(float(ratio_i), 2))+'%' for ratio_i in ratio]
            print('RDP cost ratio of time, frequency, and dpsgd: ' + ' / '.join(ratio))
        rdp = sum(rdp)
        eps, best_alpha = privacy_analysis.get_privacy_spent(
            orders=alphas, rdp=rdp, delta=delta
        )
        return float(eps), float(best_alpha)

    def get_epsilon(
        self, delta: float, alphas: Optional[List[Union[float, int]]] = None
    ):
        """
        Return privacy budget (epsilon) expended so far.

        Args:
            delta: target delta
            alphas: List of RDP orders (alphas) used to search for the optimal conversion
                between RDP and (epd, delta)-DP
        """
        eps, _ = self.get_privacy_spent(delta=delta, alphas=alphas)
        return eps

    def __len__(self):
        return len(self.history)

    @classmethod
    def mechanism(cls) -> str:
        return "rdp"

def main(config):
    sensitive_train_loader, _, _, _, config = load_data(config)
    accountant = RDPAccountant()

    sigma_f = config.train.sigma_freq
    sigma_t = config.train.sigma_time
    accountant.history = [(sigma_t, config.public_data.central.batch_size/len(sensitive_train_loader.dataset), config.public_data.central.sample_num), (sigma_f, 1., 1)]
    sample_rate = 1 / len(sensitive_train_loader)
    sigma_sgd = get_noise_multiplier(
                target_epsilon=config.train.dp.epsilon,
                target_delta=config.train.dp.delta,
                sample_rate=sample_rate,
                epochs=config.train.n_epochs,
                accountant=accountant.mechanism(),
                account_history=accountant.history,
            )
    accountant.history.append((sigma_sgd, sample_rate, int(config.train.n_epochs / sample_rate)))

    eps, alpha = accountant.get_privacy_spent(delta=config.train.dp.delta)
    eps, alpha = accountant.get_privacy_spent(delta=config.train.dp.delta, alphas=alpha)

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

    config = parse_config(opt, unknown)
    config.setup.n_gpus_per_node = 1
    config.setup.run_type = 'normal'

    run(main, config)