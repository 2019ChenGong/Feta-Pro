# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import root_scalar
import scipy

from opacus.accountants.accountant import IAccountant
from opacus.accountants.analysis import rdp as privacy_analysis
# from opacus.accountants.utils import get_noise_multiplier
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
            print(rdp)
            ratio = list(np.array(rdp) / sum(rdp) * 100)
            ratio = [str(round(float(ratio_i), 2)) for ratio_i in ratio]
            print(' / '.join(ratio))
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
def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    account_history: tuple = None,
    alpha_history: list = None,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    if alpha_history is not None:
        accountant.DEFAULT_ALPHAS = accountant.DEFAULT_ALPHAS + alpha_history

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        if account_history is None:
            accountant.history = [(sigma_high, sample_rate, steps)]
        else:
            accountant.history = account_history + [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        if account_history is None:
            accountant.history = [(sigma, sample_rate, steps)]
        else:
            accountant.history = account_history + [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    print("final sigma: ", sigma_high, "signal eps: ", eps_high)
    return sigma_high

def get_noise_multiplier2(epsilon, num_steps, delta, min_noise_multiplier=1e-1, max_noise_multiplier=500, max_epsilon=1e7):

    def delta_Gaussian(eps, mu):
        # Compute delta of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        if mu == 0:
            return 0
        if np.isinf(np.exp(eps)):
            return 0
        return scipy.stats.norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * scipy.stats.norm.cdf(-eps / mu - mu / 2)

    def eps_Gaussian(delta, mu):
        # Compute eps of Gaussian mechanism with shift mu or equivalently noise scale 1/mu
        def f(x):
            return delta_Gaussian(x, mu) - delta
        return root_scalar(f, bracket=[0, max_epsilon], method='brentq').root

    def compute_epsilon(noise_multiplier, num_steps, delta):
        return eps_Gaussian(delta, np.sqrt(num_steps) / noise_multiplier)

    def objective(x):
        return compute_epsilon(noise_multiplier=x, num_steps=num_steps, delta=delta) - epsilon

    output = root_scalar(objective, bracket=[min_noise_multiplier, max_noise_multiplier], method='brentq')

    if not output.converged:
        raise ValueError("Failed to converge")

    return output.root

accountant = RDPAccountant()
# sigma_t_list = [2, 5, 10, 20, 30]
# sigma_f_list = [11, 15, 25, 34, 67]
delta = 1 / (55000 * np.log(55000))
# for sigma_t in sigma_t_list:
#     sigma_sgd = []
#     for sigma_f in sigma_f_list:
#         sigma = get_noise_multiplier(target_epsilon=1, target_delta=delta, sample_rate=4096/55000, epochs=150, account_history=[(sigma_t, 6000/55000, 5), (sigma_f, 1.0, 1)], epsilon_tolerance=0.0000001)
#         sigma_sgd.append(str(round(sigma, 2)))
#     print('&'.join(sigma_sgd))


sigma_f = [10, 13, 21, 30, 57][0]
sigma_t = [2, 5, 10, 20, 30][0]
sigma_sgd = 152
accountant.history = [(sigma_t, 6000/55000, 5), (sigma_f, 1., 1)]
accountant.history.append((sigma_sgd, 4096/55000, 55000//4096*150))
# for _ in range(55000//4096*150):
#     accountant.history.append((sigma_sgd, 4096/55000, 1))
# delta = 1e-5

eps, alpha = accountant.get_privacy_spent(delta=delta)
print(eps, alpha)
eps, alpha = accountant.get_privacy_spent(delta=delta, alphas=alpha)
print(eps, alpha)

# print(get_noise_multiplier2(0.05, 1, delta))
# print(get_noise_multiplier(target_epsilon=0.25, target_delta=delta, sample_rate=1.0, epochs=1, epsilon_tolerance=0.0001))


# from opacus.accountants.rdp import RDPAccountant
# accountant = RDPAccountant()
# accountant.history = [(sigma, batch_size / number_of_sensitive_images, number_of_central_images_per_class)]
# eps = accountant.get_epsilon(delta=delta)
# print(eps)