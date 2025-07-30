from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import root_scalar
import scipy

from opacus.accountants.accountant import IAccountant
from opacus.accountants.analysis import rdp as privacy_analysis
from opacus.accountants.utils import get_noise_multiplier
from typing import Optional

from opacus.accountants import create_accountant


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
            ratio = [str(round(float(ratio_i), 2)) for ratio_i in ratio]
            print('RDP cost ratio: ' + ' / '.join(ratio))
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


accountant = RDPAccountant()
data_num = 55000
delta = 1 / (data_num * np.log(data_num))


sigma_f = 20
sigma_t = 2
sigma_sgd = 152
accountant.history = [(sigma_t, 6000/data_num, 5), (sigma_f, 1., 1)]
accountant.history.append((sigma_sgd, 4096/data_num, data_num//4096*150))

eps, alpha = accountant.get_privacy_spent(delta=delta)
eps, alpha = accountant.get_privacy_spent(delta=delta, alphas=alpha)