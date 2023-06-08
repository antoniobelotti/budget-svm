from typing import Tuple, Callable

import numpy as np
import numpy.typing as npt

from experiments.storage.Storage import Storage
from experiments.synthetic_datasets.BaseDatasetBuilder import BaseDatasetBuilder


class PacmanDatasetBuilder(BaseDatasetBuilder):
    def get_population(self, population_size: int, **kwargs) -> npt.ArrayLike:
        if "gamma" not in kwargs:
            raise ValueError("specify 'gamma' value for pacman dataset generation")

        gamma = kwargs.get("gamma")

        # create a population 10 times bigger than n
        mean = np.zeros(self.dim)
        cov = np.eye(self.dim) * gamma
        pop = self.rng.multivariate_normal(
            mean, cov, population_size, check_valid="raise"
        )

        # scale values
        pop = (2 * (pop - np.min(pop)) / (np.max(pop) - np.min(pop))) - 1

        return pop

    def get_labeling_fn(self, **kwargs) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
        if "a" not in kwargs:
            raise ValueError("specify 'a' value for pacman dataset generation")

        alpha = kwargs.get("a")

        def labeling_fn(*coord):
            y = coord[-1] - sum((alpha * x) ** 2 for x in coord[:-1])
            y[y > 0] = 1
            y[y <= 0] = -1
            return y

        return labeling_fn

    def __init__(self, storage: Storage):
        super().__init__(storage)
