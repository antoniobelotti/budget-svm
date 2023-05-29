import numpy as np
from sklearn.model_selection import train_test_split

from experiments.synthetic_datasets.common import (
    flip_class,
    get_indices_with_class_balance, cache_dataset,
)


def __get_pacman_labeling_function(alpha):
    def labeling_fn(*coord):
        y = coord[-1] - sum((alpha * x) ** 2 for x in coord[:-1])
        y[y > 0] = 1
        y[y <= 0] = -1
        return y

    return labeling_fn

@cache_dataset
def get_pacman_dataset(n=300, a=1, r=0, p=1, dim=2, gamma=10, test_size=0.3, seed=42):
    rng = np.random.default_rng(seed)

    # create a population 10 times bigger than n
    mean = np.zeros(dim)
    cov = np.eye(dim) * gamma
    population_X = rng.multivariate_normal(mean, cov, n * 10, check_valid="raise")

    population_X = (2*(population_X - np.min(population_X)) / (np.max(population_X) - np.min(population_X)))-1

    # label each point in the population
    labeling_function = __get_pacman_labeling_function(a)
    population_y = labeling_function(*population_X.T)

    # sample with desired class balance
    mask = get_indices_with_class_balance(population_y, n, p, rng)
    X = population_X[mask]
    y = population_y[mask]

    # introduce desired amount of noise
    flip_class(y, r, rng)

    return train_test_split(X, y, test_size=test_size, random_state=seed)
