import dataclasses
import json
import logging
import time
import traceback
from uuid import UUID

import numpy as np
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from budgetsvm.kernel import (
    GaussianKernel,
    PolynomialKernel,
    LinearKernel,
    PrecomputedKernel,
)

logger = logging.getLogger("experiments")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if "kernel" in type(obj).__name__.lower():
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if dataclasses.is_dataclass(obj):
            return dataclasses.asdict(obj)
        return super().default(obj)


def run_cvgrid(
    model, precomputed_X_train, precomputed_X_test, y_train, y_test, kernel, cfg
):
    skf = StratifiedKFold(
        n_splits=cfg.get("cv", 5), shuffle=True, random_state=cfg.get("seed", 42)
    )
    c_values = cfg.get("C", [1.0])

    cvgrid = GridSearchCV(
        model,
        {"C": c_values, "kernel": [kernel]},
        refit=True,
        verbose=0,
        cv=skf,
        n_jobs=10,
    )
    try:
        cvgrid.fit(precomputed_X_train, y_train)
        test_score = cvgrid.score(precomputed_X_test, y_test)
        return cvgrid.best_estimator_, cvgrid.best_params_, test_score
    except FitFailedWarning:
        pass
    except Exception as e:
        logger.error("GridSearchCV failed with unexpected error.")
        logger.error(traceback.format_exc())
    return None, None, 0


def model_selection(model, X_train, X_test, y_train, y_test, cfg):
    kernelcfg = cfg.get("kernels", {"linear"})
    num_params = len(cfg.get("C", [1.0])) * (
        len(kernelcfg.get("gaussian", []))
        + len(kernelcfg.get("polynomial", []))
        + (1 if "linear" in kernelcfg else 0)
    )

    logger.debug(
        f"Launching GridSearcCV on {model} - {num_params} params, {cfg.get('cv', 5)}-folds, "
        f"for a total of {num_params * cfg.get('cv', 5)} fit calls."
    )

    best_estimator = None
    best_score = 0.0
    best_params = None

    all_kernels = []
    for kernel_name, hp in cfg.get("kernels", {"linear"}).items():
        match kernel_name:
            case "linear":
                all_kernels.append(PrecomputedKernel(original_kernel=LinearKernel()))
            case "gaussian":
                for v in hp:
                    all_kernels.append(PrecomputedKernel(original_kernel=GaussianKernel(v)))
            case "polynomial":
                for v in hp:
                    all_kernels.append(PrecomputedKernel(original_kernel=PolynomialKernel(v)))

    for krn in all_kernels:
        precomputed_X_train = np.array(
            [[krn.compute(x, y) for y in X_train] for x in X_train]
        )
        precomputed_X_test = np.array(
            [[krn.compute(x, y) for y in X_train] for x in X_test]
        )
        est, par, score = run_cvgrid(
            model,
            precomputed_X_train,
            precomputed_X_test,
            y_train,
            y_test,
            krn,
            cfg,
        )
        if score > best_score:
            best_estimator = est
            best_params = par
            best_score = score

    return best_estimator, best_params, best_score


class Timer:
    def __enter__(self):
        self.__t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.__t
