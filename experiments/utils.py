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
import budgetsvm.kernel

logger = logging.getLogger("experiments")


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
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

        # FIXME: why isinstance(obj, Kernel) does not work?
        if "kernel" in type(obj).__name__.lower():
            return str(obj)
        return super().default(obj)


class Timer:
    def __enter__(self):
        self.__t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.__t
