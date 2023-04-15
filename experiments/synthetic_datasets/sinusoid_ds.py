import numpy as np
from sklearn.model_selection import train_test_split

from experiments.synthetic_datasets.common import cache_dataset, get_indices_with_class_balance, flip_class


def __get_labeling_function(beta, rho, theta):
    def df(x):
        return 1 / (1 + np.exp(-beta * (x - 0.5))) + rho * np.sin(2 * np.pi * theta * x)

    def lf(x0, x1):
        return np.sign(df(x0) - x1)

    return lf


@cache_dataset
def get_sinusoid_dataset(
        beta=1,
        rho=0.2,
        theta=10,
        n=300,
        r=0,
        p=1,
        test_size=0.3,
        seed=42
):
    rng = np.random.default_rng(seed)
    population_X = rng.uniform(size=(n*10, 2), low=0, high=1)

    # label each point in the population
    labeling_function = __get_labeling_function(beta, rho, theta)
    population_y = labeling_function(*population_X.T)

    # sample with desired class balance
    mask = get_indices_with_class_balance(population_y, n, p, rng)
    X = population_X[mask]
    y = population_y[mask]

    # introduce desired amount of noise
    flip_class(y, r, rng)

    return train_test_split(X, y, test_size=test_size, random_state=seed)
