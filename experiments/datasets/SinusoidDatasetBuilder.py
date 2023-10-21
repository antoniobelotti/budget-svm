from typing import Callable

import numpy as np
import numpy.typing as npt

from experiments.storage.Storage import Storage
from experiments.datasets.BaseDatasetBuilder import BaseDatasetBuilder


class SinusoidDatasetBuilder(BaseDatasetBuilder):
    def get_population(self, population_size: int, **kwargs) -> npt.NDArray[float]:
        return self.rng.uniform(size=(population_size, 2), low=0, high=1)

    def get_labeling_fn(self, **kwargs) -> Callable[[npt.NDArray[float]], npt.NDArray[float]]:
        if any(req_arg not in kwargs for req_arg in ["beta", "rho", "theta"]):
            raise ValueError("specify all 'beta', 'rho', 'theta' values for sinusoid dataset generation")

        beta = kwargs.get("beta")
        rho = kwargs.get("rho")
        theta = kwargs.get("theta")

        def df(x):
            return 1 / (1 + np.exp(-beta * (x - 0.5))) + rho * np.sin(2 * np.pi * theta * x)

        def lf(x0, x1):
            return np.sign(df(x0) - x1)

        return lf

    def __init__(self, storage: Storage):
        super().__init__(storage)
