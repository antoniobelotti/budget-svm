import time
import unittest

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC as SklearnSVC

from experiments.synthetic_datasets.pacman_ds import get_pacman_dataset
from experiments.utils import Timer
from kernel import GaussianKernel, LinearKernel
from optimization import ReusableGurobiSolver, GurobiSolver
from svm import SVC


class SVCTests(unittest.TestCase):
    def test_reusable_gurobi_solver_works_with_no_budget(self):
        ds = get_pacman_dataset(a=0.1)
        solver = ReusableGurobiSolver()
        model = SVC()
        model.fit(ds.X_train, ds.y_train, solver=solver)
        print(model.score(ds.X_test, ds.y_test))

    def test_multiple_runs_of_reusable_gurobi_solver(self):
        ds = get_pacman_dataset(a=0.1)

        solver = ReusableGurobiSolver()

        model = SVC()
        model.fit(ds.X_train, ds.y_train, solver=solver)
        print(model.score(ds.X_test, ds.y_test))
        print(f"unconstrained model found {len(model.alpha_)} support vectors")

        bmodel = SVC(budget=len(model.alpha_))
        bmodel.fit(ds.X_train, ds.y_train, solver=solver)
        print(bmodel.score(ds.X_test, ds.y_test))
        print(f"100% budget model found {len(bmodel.alpha_)} support vectors")

    def test_del(self):
        model = SklearnSVC()
        grid = GridSearchCV(
            model,
            {
                "C": [0.001, 0.01, 0.1, 1, 10],
                "kernel": ["rbf", "poly", "linear"]
                # [
                #    GaussianKernel(0.01),
                #    GaussianKernel(0.1),
                #    GaussianKernel(1),
                #    GaussianKernel(10),
                # ],
            },
            refit=True,
            verbose=0,
            n_jobs=1,
        )
        ds = get_pacman_dataset(dim=3, n=5000, a=0.5)
        grid.fit(ds.X_train, ds.y_train)
        print(grid.best_params_)

    def test_multiple_budgeted_runs_of_reusable_gurobi_solver_run_quicker(self):
        ds = get_pacman_dataset(dim=3, n=5000, a=0.1)

        # unconstrained to get a meaningful budget value
        model = SVC(C=1, kernel=LinearKernel())
        model.fit(ds.X_train, ds.y_train)
        print(model.score(ds.X_test, ds.y_test))

        solver = ReusableGurobiSolver()
        with Timer() as t:
            bmodel = SVC(
                C=1, budget=len(model.alpha_), kernel=LinearKernel()
            )
            bmodel.fit(ds.X_train, ds.y_train, solver=solver)
            print(bmodel.score(ds.X_test, ds.y_test))
        first_run_t = t.time

        with Timer() as t:
            bmodel = SVC(
                C=1, budget=len(model.alpha_), kernel=LinearKernel()
            )
            bmodel.fit(ds.X_train, ds.y_train, solver=solver)
            print(bmodel.score(ds.X_test, ds.y_test))
        second_run_t = t.time

        self.assertLess(second_run_t, first_run_t)
        print(f"first run took {first_run_t} seconds")
        print(f"second run took {second_run_t} seconds")
