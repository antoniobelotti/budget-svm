import os
import pathlib
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
from experiments.synthetic_datasets.PacmanDatasetBuilder import PacmanDatasetBuilder
from experiments.utils import Timer


class IRWLS(ClassifierMixin, BaseEstimator):
    """
    Wrapper class for the IRWLS solver https://github.com/RobeDM/LIBIRWLS
    The original repository is added as a submodule.

    If no executable is found, __init__ tries to compile the project. It requires Make, gcc and some luck.
    Refer to ./LIBIRWLS/README.md for details on usage.
    """

    __BASE_PATH = pathlib.Path(os.path.dirname(os.path.realpath(__file__))) / "LIBIRWLS"
    __TRAIN_EXE = __BASE_PATH / "bin/budgeted-train"
    __PREDICT_EXE = __BASE_PATH / "bin/LIBIRWLS-predict"

    def __init__(self, C: float = 1, gamma: float = 1, budget: int = 50):
        """
        :param C: soft margin cost parameter
        :type C: float, default=1
        :param gamma: gamma parameter for the RBF kernel
        :type gamma: float, default=1
        :param budget: EXACT number of support vectors. Too high of a value makes training stuck.
        :type budget: int, default 50
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

        self.C = C
        self.gamma = gamma
        self.budget = budget

    def __try_compiling(self):
        print("LIBIRWLS executable not found. Trying to Make...")
        comp_proc = subprocess.run(
            [
                "make",
                "LIBS=-lm -llapack -lf77blas -lcblas -latlas -fopenmp",  # removed the option -lgfortran
                "--quiet",
                "-C",
                self.__BASE_PATH,
            ],
            capture_output=True,
        )
        if comp_proc.returncode != 0:
            raise RuntimeError(
                f"Unable to compile BudgetedSVM Toolbox.\n{comp_proc.stderr}"
            )
        print("Compiled!")

    def fit(self, X: npt.NDArray[float], y: npt.NDArray[float]) -> None:
        # 1) dump dataset in libsvm format to filesystem
        # 2) call executable to train on such dataset (-> produces a BINARY model file)
        # 3) if successful, remember model file path

        model_name = str(uuid.uuid4()) + ".model"
        model_path = self.all_models_path / model_name

        dataset_path = self.all_datasets_path / str(uuid.uuid4())
        dump_svmlight_file(X, y, dataset_path, zero_based=False)

        """
        Usage: budgeted-train [options] training_set_file model_file

        Options:
          -k kernel type: (default 1)
               0 -- Linear kernel u'*v
               1 -- radial basis function: exp(-gamma*|u-v|^2)
          -g gamma: set gamma in radial basis kernel function (default 1)
               radial basis K(u,v)= exp(-gamma*|u-v|^2)
          -c Cost: set SVM Cost (default 1)
          -t Threads: Number of threads (default 1)
          -s Classifier size: Size of the classifier (default 1)
          -a Algorithm: Algorithm for centroids selection (default 1)
               0 -- Random Selection
               1 -- SGMA (Sparse Greedy Matrix Approximation)
          -f file format: (default 1)
               0 -- CSV format (comma separator)
               1 -- libsvm format
          -p separator: csv separator character (default "," if csv format is selected)
          -v verbose: (default 1)
               0 -- No screen messages
               1 -- Screen messages
       """

        params = [
            self.__TRAIN_EXE,
            "-s",
            str(self.budget),
            "-c",
            str(self.C),
            "-g",
            str(self.gamma),
            "-v",
            "1",
            dataset_path,
            model_path,
        ]

        with Timer() as t:
            rc = subprocess.run(params, capture_output=True).returncode

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

        """
        Usage: LIBIRWLS-predict [options] data_set_file model_file output_file

        Options:
          -t Number of Threads: (default 1)
          -l type of data set: (default 0)
               0 -- Data set with no target as first dimension.
               1 -- Data set with label as first dimension (obtains accuracy too)
          -s Soft output: (default 0)
               0 -- Obtains the class of every data (It takes values of +1 or -1).
               1 -- The output before the class decision (Useful to combine in ensembles with other algorithms).
          -f file format: (default 1)
               0 -- CSV format (comma separator)
               1 -- libsvm format
          -p separator: csv separator character (default "," if csv format is selected)
          -v verbose: (default 1)
               0 -- No screen messages
               1 -- Screen messages
       """

        rc = subprocess.run(
            [
                self.__PREDICT_EXE,
                "-l",
                "1",
                "-v",
                "0",
                dataset_path,
                self.model_path_,
                prediction_file_path,
            ],
            capture_output=True,
        ).returncode
        assert rc == 0

        # for some reason predictions are float, like 1.00000000 or -1.0000000
        with open(prediction_file_path, "r") as f:
            y_hat_test = [int(x.strip().split(".")[0]) for x in f.readlines()]

        dataset_path.unlink()
        prediction_file_path.unlink()

        return y_hat_test

    def score(self, X: npt.NDArray[float], y: npt.NDArray[float], **kwargs) -> float:
        """Return the accuracy on given test dataset."""
        y_hat = self.predict(X)
        correct = sum(1 for act, pred in zip(y, y_hat) if act == pred)
        return correct / len(y)

    @property
    def num_sv_(self):
        check_is_fitted(self)
        # EXACTLY the same as budget
        return self.budget
