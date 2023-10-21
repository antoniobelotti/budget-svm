import os
import pathlib
import re
import subprocess
import uuid

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import dump_svmlight_file
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from experiments.storage.DummyStorage import DummyStorage
from experiments.datasets.PacmanDatasetBuilder import PacmanDatasetBuilder
from experiments.utils import Timer


class BudgetedSVMToolbox(ClassifierMixin, BaseEstimator):
    """
    Wrapper class for the BudgetedSVM toolbox https://github.com/djurikom/BudgetedSVM.
    The original repository is added as a submodule.

    If no executable is found, __init__ tries to compile the project. It requires Make, gcc and some luck.
    Refer to ./budgetedsvmtoolbox/README.txt for details on usage.
    """

    __BASE_PATH = (
        pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / "budgetedsvmtoolbox"
    )
    __TRAIN_EXE = __BASE_PATH / "bin/budgetedsvm-train"
    __PREDICT_EXE = __BASE_PATH / "bin/budgetedsvm-predict"

    def __init__(self, gamma: float = 1, budget: int = 9999):
        """
        :param gamma: gamma parameter for the Gaussian kernel
        :type gamma: float, default=1
        :param budget: maximum number of support vectors
        :type budget: int, default=9999
        """
        tmp_dir = pathlib.Path("/tmp")
        self.all_models_path = tmp_dir
        self.all_datasets_path = tmp_dir
        self.trained = False
        if not (
            self.__TRAIN_EXE.exists()
            and self.__TRAIN_EXE.is_file()
            and self.__PREDICT_EXE.exists()
            and self.__PREDICT_EXE.is_file()
        ):
            self.__try_compiling()

        self.gamma = gamma
        self.budget = budget

        # TODO: consider not hard coding
        self.epochs = 10
        self.randomize = False

    def __try_compiling(self):
        print("BudgetedSVM Toolbox executable not found. Trying to Make...")
        comp_proc = subprocess.run(
            ["make", "--quiet", "-C", self.__BASE_PATH], capture_output=True
        )
        if comp_proc.returncode != 0:
            raise RuntimeError(
                f"Unable to compile BudgetedSVM Toolbox.\n{comp_proc.stderr}"
            )
        print("Compiled!")

    def fit(self, X: npt.NDArray[float], y: npt.NDArray[float]) -> None:
        # 1) dump dataset in libsvm format to filesystem
        # 2) call executable to train on such dataset (-> produces a model file in libsvm format)
        # 3) if successful, remember model file path

        model_name = str(uuid.uuid4()) + ".model"
        model_path = self.all_models_path / model_name

        dataset_path = self.all_datasets_path / str(uuid.uuid4())
        dump_svmlight_file(X, y, dataset_path, zero_based=False)

        params = [
            self.__TRAIN_EXE,
            "-A",
            "4",
            "-B",
            str(self.budget),
            "-v",
            "1",
            "-g",
            str(self.gamma),
            "-m",  # maintenance strategy: [0 removal, 1 merging]
            "1",
            "-e",
            str(self.epochs),
            "-r",
            "1" if self.randomize else "0",
            dataset_path,
            model_path,
        ]

        with Timer() as t:
            rc = subprocess.run(params, stdout=subprocess.PIPE, text=True).returncode

        dataset_path.unlink()

        if rc != 0:
            raise FitFailedWarning(f"solver run failed with exit code {rc}")

        self.model_name_ = model_name
        self.model_path_ = model_path
        self.time_ = t.time

    def predict(self, X: npt.NDArray[float]) -> npt.NDArray[float]:
        check_is_fitted(self)

        # 1) dump data in libsvm format to filesystem
        # 2) call executable to predict on such dataset using the model file (-> produces a predictions file)
        # 3) if successful, open predictions file and return as np array

        dataset_path = self.all_datasets_path / str(uuid.uuid4())
        dump_svmlight_file(X, np.zeros(X.shape[0]), dataset_path, zero_based=False)

        prediction_file_path = (
            self.all_datasets_path / f"{self.model_name_}_predictions"
        )

        rc = subprocess.run(
            [self.__PREDICT_EXE, dataset_path, self.model_path_, prediction_file_path],
            capture_output=True,
        ).returncode
        assert rc == 0

        with open(prediction_file_path, "r") as f:
            y_hat_test = [int(x.strip()) for x in f.readlines()]

        dataset_path.unlink()
        prediction_file_path.unlink()

        return y_hat_test

    def score(self, X: npt.NDArray[float], y: npt.NDArray[float], **kwargs):
        """Return the accuracy on given test dataset."""
        y_hat = self.predict(X)
        correct = sum(1 for act, pred in zip(y, y_hat) if act == pred)
        return correct / len(y)

    @property
    def num_sv_(self):
        check_is_fitted(self)

        with open(self.model_path_, "r") as f:
            return len(f.readlines())
