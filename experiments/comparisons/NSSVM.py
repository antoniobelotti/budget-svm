import os
import pathlib

import matlab.engine
import numpy as np
import numpy.typing as npt
from matlab.engine import MatlabEngine
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted, check_X_y


class NSSVM(ClassifierMixin, BaseEstimator):
    """
    Wrapper class for the NSSVM solver https://github.com/ShenglongZhou/NSSVM
    The original repository is added as a submodule.
    """

    __SOLVER_PATH = (
        pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
        / "NSSVM/NSSVM/solver/NSSVM.m"
    )
    matlab_engine: MatlabEngine = matlab.engine.start_matlab()

    def __init__(
        self,
        C: float = 0.25,
        c: float = 0.125,
        budget: int = 9999,
        auto_tune: bool = False,
        max_iter: int = 2000,
    ):
        """
        :param matlab_engine: The matlab runtime/engine thingy
        :type matlab_engine: MatlabEngine from matlab.engine.matlabengine
        :param C: ?
        :type C: float in (0,1], default=0.25
        :param c: ?
        :type c: float in (0,1], default=0.125
        :param budget: maximum number of allowed support vectors
        :type budget: int, default=9999
        :param auto_tune: if set to True ignores "budget" and employs an automatic strategy to
            find a sufficiently sparse model without too much performance loss. If False, force
            at most "budget" support vectors
        :type auto_tune: bool, default=False
        :param max_iter: maximum number of iterations
        :type max_iter: int, default=2000

        """
        assert self.__SOLVER_PATH.exists() and self.__SOLVER_PATH.is_file()

        self.matlab_engine.cd(str(self.__SOLVER_PATH.parent), nargout=0)
        self.C = C
        self.c = c
        self.budget = budget
        self.auto_tune = auto_tune
        self.max_iter = max_iter

    def fit(self, X: npt.NDArray[float], y: npt.NDArray[int]) -> None:
        X, y = check_X_y(X, y)

        y = y.reshape(-1, 1)

        params = {
            "disp": 0,  # 1 verbose, 0 quiet
            "s0": float(self.budget),  # Breaks if not float
            "C": self.C,
            "c": self.c,
            "tune": 1 if self.auto_tune else 0,
            "maxit": self.max_iter,
        }
        res = self.matlab_engine.NSSVM(X, y, params)
        self.alpha_ = res["alpha"]
        self.num_sv_ = res["sv"]
        self.time_ = res["time"]
        self.w_ = res["w"][:-1]
        self.b_ = res["w"][-1]

    def predict(self, X: npt.NDArray[float]) -> npt.NDArray[int]:
        check_is_fitted(self)
        return np.sign(X.dot(self.w_) + self.b_).reshape(-1)

    def score(self, X: npt.NDArray[float], y: npt.NDArray[int], **kwargs) -> float:
        """Return the accuracy on given test dataset."""
        y_hat = self.predict(X)
        return (y_hat == y).sum() / len(y)
