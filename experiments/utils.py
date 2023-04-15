import json
import logging
import math
import time
import traceback
from uuid import UUID

import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, ParameterGrid

from budgetsvm.kernel import GaussianKernel, Kernel, PolynomialKernel, PrecomputedKernel, LinearKernel


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Kernel):
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def model_selection(model, X_train, X_test, y_train, y_test, cfg):
    """ Try multiple values of C and multiple kernels configurations. Return most accurate model"""

    c_values = cfg.get("C", [1.0])

    kernel_values = []
    for kernel_name, hp in cfg.get("kernels", ["linear"]).items():
        if kernel_name == "linear":
            kernel_values.append(LinearKernel())
        if kernel_name == "gaussian":
            for v in hp:
                kernel_values.append(GaussianKernel(v))
        if kernel_name == "polynomial":
            for v in hp:
                kernel_values.append(PolynomialKernel(v))

    grid_params = {'C': c_values, 'kernel': kernel_values}
    if model.budget:
        grid_params = {'budget': [model.budget], **grid_params}

    cvgrid = GridSearchCV(model, grid_params, refit=True, verbose=0, cv=cfg.get("cv", 5), n_jobs=-1)
    test_accuracy = 0.0
    try:
        num_params = math.prod(len(x) for x in cvgrid.param_grid.values())
        logging.debug(f"Launching GridSearcCV on {model} - {num_params} params, {cfg.get('cv', 5)}-folds, "
                      f"for a total of {num_params * cfg.get('cv', 5)} fit calls.")
        cvgrid.fit(X_train, y_train)
        test_accuracy = cvgrid.score(X_test, y_test)
    except FitFailedWarning:
        pass
    except Exception as e:
        logging.error(f"GridSearchCV failed with unexpected error.\n{grid_params}")
        logging.error(traceback.format_exc())
        return None, None, "Error while training"

    return cvgrid.best_estimator_, cvgrid.best_params_, test_accuracy


class Timer:
    def __enter__(self):
        self.__t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.__t
