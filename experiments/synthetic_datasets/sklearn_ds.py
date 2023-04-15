from sklearn import preprocessing
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split

from experiments.synthetic_datasets.common import cache_dataset


@cache_dataset
def dataset_make_moons(n=300, noise=0.1, test_size=0.3, random_state=42):
    X, y = make_moons(n_samples=n, noise=noise, random_state=random_state)
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


@cache_dataset
def dataset_make_blobs(
    n=300, cluster_std=0.2, centers=2, test_size=0.3, random_state=42
):
    X, y = make_blobs(
        n_samples=n, cluster_std=cluster_std, random_state=random_state, centers=centers
    )
    X = preprocessing.MinMaxScaler().fit_transform(X)
    y = [int(x) if x == 1 else -1 for x in y]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
