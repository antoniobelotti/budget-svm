import bz2
from collections.abc import Iterator

import numpy as np
import requests
from sklearn.datasets import load_svmlight_file

from budgetsvm.kernel import PrecomputedKernel, GaussianKernel, PolynomialKernel
from budgetsvm.svm import SVC
from experiments.BaseExperiment import BaseExperiment, BaseExperimentOldStrategy
from experiments.datasets.Base import Dataset
from experiments.storage.GDriveStorage import GDriveStorage
from experiments.storage.LocalStorage import LocalStorage


class a1aExperiment(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    EXPERIMENT_FILE_SUFFIX = "_a1a_our_model_new_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        name, n_features = "a1a", 123
        cached_ds = self.storage.get_dataset_if_exists(name)
        if cached_ds:
            yield cached_ds
            return

        train_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}").text
        test_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}.t").text
        with open(f"/tmp/{name}", "w") as f:
            f.write(train_content)
        with open(f"/tmp/{name}.t", "w") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file(f"/tmp/{name}", zero_based=False, n_features=n_features)
        X_test, y_test = load_svmlight_file(f"/tmp/{name}.t", zero_based=False, n_features=n_features)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id=name,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


class GisetteExperiment(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    EXPERIMENT_FILE_SUFFIX = "_gisette_our_model_new_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        cached_ds = self.storage.get_dataset_if_exists("gisette")
        if cached_ds:
            yield cached_ds
            return
        train_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2").content
        test_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2").content
        train_content = bz2.decompress(train_content)
        test_content = bz2.decompress(test_content)

        with open("/tmp/gisette", "wb") as f:
            f.write(train_content)
        with open("/tmp/gisette.t", "wb") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file("/tmp/gisette", zero_based=False, n_features=5000)
        X_test, y_test = load_svmlight_file("/tmp/gisette.t", zero_based=False, n_features=5000)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id="gisette",
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


class SvmGuide1xperiment(BaseExperiment):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2]
    EXPERIMENT_FILE_SUFFIX = "_svmguide1_our_model_new_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        name, n_features = "svmguide1", 123
        cached_ds = self.storage.get_dataset_if_exists(name)
        if cached_ds:
            yield cached_ds
            return

        train_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}").text
        test_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}.t").text
        with open(f"/tmp/{name}", "w") as f:
            f.write(train_content)
        with open(f"/tmp/{name}.t", "w") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file(f"/tmp/{name}", zero_based=False, n_features=n_features)
        X_test, y_test = load_svmlight_file(f"/tmp/{name}.t", zero_based=False, n_features=n_features)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id=name,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


class a1aExperimentOldStrategy(BaseExperimentOldStrategy):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [.3,.4,.5,.6,.7,.8,.9]
    EXPERIMENT_FILE_SUFFIX = "_a1a_our_model_old_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        #for sigma in [0.001, 0.01, 0.1, 1, 10]:
        #    grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        name, n_features = "a1a", 123

        cached_ds = self.storage.get_dataset_if_exists(name)
        if cached_ds:
            yield cached_ds
            return

        train_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}").text
        test_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}.t").text
        with open(f"/tmp/{name}", "w") as f:
            f.write(train_content)
        with open(f"/tmp/{name}.t", "w") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file(f"/tmp/{name}", zero_based=False, n_features=n_features)
        X_test, y_test = load_svmlight_file(f"/tmp/{name}.t", zero_based=False, n_features=n_features)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id=name,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


class GisetteExperimentOldStrategy(BaseExperimentOldStrategy):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [.3,.4,.5,.6,.7,.8,.9]
    EXPERIMENT_FILE_SUFFIX = "_gisette_our_model_old_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        cached_ds = self.storage.get_dataset_if_exists("gisette")
        if cached_ds:
            yield cached_ds
            return

        train_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.bz2").content
        test_content = requests.get("https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/gisette_scale.t.bz2").content
        train_content = bz2.decompress(train_content)
        test_content = bz2.decompress(test_content)

        with open("/tmp/gisette", "wb") as f:
            f.write(train_content)
        with open("/tmp/gisette.t", "wb") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file("/tmp/gisette", zero_based=False, n_features=5000)
        X_test, y_test = load_svmlight_file("/tmp/gisette.t", zero_based=False, n_features=5000)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id="gisette",
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


class SvmGuide1xperimentOldStrategy(BaseExperimentOldStrategy):
    CROSS_VALIDATION = 5
    BUDGET_PERCENTAGES = [.3,.4,.5,.6,.7,.8,.9]
    EXPERIMENT_FILE_SUFFIX = "_svmguide1_our_model_old_strategy"

    def get_cvgrid_parameter_dict(self) -> dict:
        grid = {"C": [0.001, 0.01, 0.1, 1, 10], "kernels": []}
        for sigma in [0.001, 0.01, 0.1, 1, 10]:
            grid["kernels"].append(PrecomputedKernel(GaussianKernel(sigma=sigma)))
        for degree in [3, 5, 10]:
            grid["kernels"].append(PrecomputedKernel(PolynomialKernel(degree=degree)))
        return grid

    def get_empty_model(self):
        return SVC()

    def generate_datasets(self) -> Iterator[Dataset]:
        name, n_features = "svmguide1", 123
        cached_ds = self.storage.get_dataset_if_exists(name)
        if cached_ds:
            yield cached_ds
            return

        train_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}").text
        test_content = requests.get(f"https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/{name}.t").text
        with open(f"/tmp/{name}", "w") as f:
            f.write(train_content)
        with open(f"/tmp/{name}.t", "w") as f:
            f.write(test_content)

        X_train, y_train = load_svmlight_file(f"/tmp/{name}", zero_based=False, n_features=n_features)
        X_test, y_test = load_svmlight_file(f"/tmp/{name}.t", zero_based=False, n_features=n_features)
        X_train = X_train.toarray()
        X_test = X_test.toarray()

        X = np.concatenate([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        ds = Dataset(
            id=name,
            X=X,
            y=y,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            params=None,
            complexity_report=None,
        )
        self.storage.save_dataset(ds)
        yield ds


if __name__ == "__main__":
    #exp = a1aExperimentOldStrategy(GDriveStorage())
    storage = GDriveStorage()

    #SvmGuide1xperiment(storage).run()
    SvmGuide1xperimentOldStrategy(storage).run()

