import functools
from typing import Callable

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler

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

        if kwargs.get("scale_min_max", False):
            # scale between 0 and 1
            pop = MinMaxScaler().fit_transform(pop)

        return pop

    def get_labeling_fn(self, **kwargs) -> Callable[[npt.ArrayLike], npt.ArrayLike]:
        if "a" not in kwargs:
            raise ValueError("specify 'a' value for pacman dataset generation")

        alpha = kwargs.get("a")

        def labeling_fn(x_shift, y_shift, *coord):
            # coord = (
            #   [x1,x1,x1,x1],
            #   [x2,x2,x2,x2],
            #   [x3,x3,x3,x3],
            # )
            #   x_d = a(x1-0.5)**2 + a(x2-0.5)**2 + ... + a(x_{d-1} -0.5)**2  + 0.5
            #              ^                                                    ^
            #        shift -> on x axis                                   shift up on y axis
            #
            #   ==> vertex on (0.5,0.5)

            y = coord[-1] - np.sum(alpha * ((np.array(coord[:-1]) - x_shift) ** 2), axis=0) - y_shift
            y[y > 0] = 1
            y[y <= 0] = -1
            return y

        x_shift, y_shift = 0, 0
        if kwargs.get("scale_min_max", False):
            x_shift, y_shift = 0.5, 0.5
            
        return functools.partial(labeling_fn, x_shift, y_shift)

    def __init__(self, storage: Storage):
        super().__init__(storage)
