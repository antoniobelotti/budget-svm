import itertools as it
import textwrap
from collections.abc import Iterator

import numpy as np

from experiments.BaseExperiment import BaseExperiment
from experiments.datasets.Base import Dataset
from experiments.datasets.PacmanDatasetBuilder import PacmanDatasetBuilder
from experiments.datasets.SinusoidDatasetBuilder import SinusoidDatasetBuilder
from experiments.storage.GDriveStorage import GDriveStorage
from budgetsvm.kernel import PrecomputedKernel, GaussianKernel, PolynomialKernel, LinearKernel
from budgetsvm.svm import SVC


class OurMethodOnSyntheticDatasets(BaseExperiment):
    USE_PRECOMPUTED_KERNEL = True
    BUDGET_PERCENTAGES = [0.01, 0.025, 0.05, 0.075, 0.1]
    CROSS_VALIDATION = 3
    EXPERIMENT_FILE_SUFFIX = "_sinusoid_batch1_3CV_no_repeat_sampling"

    def generate_datasets(self) -> Iterator[Dataset]:
        dscfg = {
            "sinusoid": [
                {"n": 1000, "beta": 0, "rho": 0.01, "theta": 20},
                {"n": 1000, "beta": 0, "rho": 0.025, "theta": 20},
                {"n": 1000, "beta": 0, "rho": 0.05, "theta": 20},
                {"n": 1000, "beta": 0, "rho": 0.075, "theta": 20},
                {"n": 1000, "beta": 0, "rho": 0.1, "theta": 20},
            ],
            "r_values": [0],
            "p_values": [1],
            "n_repeat_sampling": 1,
        }
        self.logger.info(f"generating datasets with this configuration:\n {textwrap.dedent(str(dscfg))}")

        n_try_different_seed = dscfg.get("n_repeat_sampling", 1)
        r_values = dscfg.get("r_values", [0])
        p_values = dscfg.get("p_values", [1])

        rng = np.random.default_rng(0)
        seeds = rng.integers(0, high=2**32 - 1, size=n_try_different_seed)

        if "sinusoid" in dscfg:
            builder = SinusoidDatasetBuilder(self.storage)
            for base_params, seed, r, p in it.product(dscfg["sinusoid"], seeds, r_values, p_values):
                if "gamma" not in base_params:
                    base_params["gamma"] = 10
                yield builder.build(r=r, p=p, seed=seed, **base_params)

        if "pacman" in dscfg:
            builder = PacmanDatasetBuilder(self.storage)
            for base_params, seed, r, p in it.product(dscfg["pacman"], seeds, r_values, p_values):
                yield builder.build(r=r, p=p, seed=seed, **base_params)

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.01, 0.1, 1, 10], "kernels": [PrecomputedKernel(LinearKernel())]}
        for sigma in [0.0001, 0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [2, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()


if __name__ == "__main__":
    exp = OurMethodOnSyntheticDatasets(GDriveStorage())
    exp.run()
