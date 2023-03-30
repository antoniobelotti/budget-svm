import json
import time
from uuid import UUID

import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid

from budgetsvm.kernel import GaussianKernel, Kernel, PolynomialKernel


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Kernel):
            return str(obj)
        if isinstance(obj, UUID):
            return str(obj)
        return super().default(obj)


def model_selection(model, X_train, X_test, y_train, y_test, cv: int = None):
    """ Try multiple values of C and multiple kernels configurations. Return most accurate model"""
    c_values = np.logspace(-3, 3, 7)
    # c_values = np.logspace(-1, 1, 3)
    kernel_values = [GaussianKernel(s) for s in c_values]
    kernel_values.extend([PolynomialKernel(deg) for deg in range(2, 10)])
    grid_params = {'C': c_values, 'kernel': kernel_values}
    if model.budget:
        grid_params = {'budget': [model.budget], **grid_params}

    if cv:
        try:
            cvgrid = GridSearchCV(model, grid_params, refit=True, verbose=0, cv=cv)
            cvgrid.fit(X_train, y_train)
            test_accuracy = cvgrid.score(X_test, y_test)
            return cvgrid.best_estimator_, cvgrid.best_params_, test_accuracy
        except:
            return None, None, "Error while training"
    else:
        # no cross validation
        best_score = 0
        best_hp = None
        for hp in ParameterGrid(grid_params):
            model.set_params(**hp)
            try:
                model.fit(X_train, y_train)
            except:
                continue

            score = model.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_hp = hp

        if best_hp is None:
            return None, None, "Model selection failed"

        model.set_params(**best_hp)
        model.fit(X_train, y_train)

        return model, best_hp, best_score


class Timer:
    def __enter__(self):
        self.__t = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time = time.perf_counter() - self.__t
