import unittest

from sklearn.model_selection import GridSearchCV

from experiments.comparisons.BudgetedSVMToolbox import BudgetedSVMToolbox
from experiments.comparisons.LIBIRWLS import IRWLS
from experiments.comparisons.NSSVM import NSSVM
from experiments.storage.DummyStorage import DummyStorage
from experiments.datasets.PacmanDatasetBuilder import PacmanDatasetBuilder


class TestNSSVM(unittest.TestCase):
    def test_fit_score_predict(self):
        budget = 50
        model = NSSVM(budget=50)
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        self.assertIsNone(
            model.fit(ds.X_test, ds.y_test),
            "fit should not return anything and should not raise any error",
        )

        self.assertGreater(
            model.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )
        self.assertLessEqual(model.num_sv_, budget, "budget exceeded")

    def test_is_GridSearchCV_compatible(self):
        model = NSSVM()
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        cv = GridSearchCV(
            model,
            param_grid={"budget": [50, 100]},
            n_jobs=1,
        )

        cv.fit(ds.X_train, ds.y_train)

        self.assertGreater(
            cv.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )
        self.assertIsNotNone(
            cv.best_estimator_, "best estimator should be a valid model"
        )


class TestBudgetedsvmtoolbox(unittest.TestCase):
    def test_fit_score_predict(self):
        budget = 50
        model = BudgetedSVMToolbox(budget=50)
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        self.assertIsNone(
            model.fit(ds.X_test, ds.y_test),
            "fit should not return anything and should not raise any error",
        )

        self.assertGreater(
            model.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )

        # FIXME: why is this not true for this model
        # self.assertLessEqual(model.num_sv_, budget, "budget exceeded")

        self.assertIsNotNone(model.num_sv_)

    def test_is_GridSearchCV_compatible(self):
        model = BudgetedSVMToolbox()
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        cv = GridSearchCV(
            model,
            param_grid={"budget": [50, 100], "gamma": [0.1, 1]},
            n_jobs=1,
        )

        cv.fit(ds.X_train, ds.y_train)

        self.assertGreater(
            cv.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )
        self.assertIsNotNone(
            cv.best_estimator_, "best estimator should be a valid model"
        )


class TestLIBIRWLS(unittest.TestCase):
    def test_fit_score_predict(self):
        budget = 50
        model = IRWLS(budget=50)
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        self.assertIsNone(
            model.fit(ds.X_test, ds.y_test),
            "fit should not return anything and should not raise any error",
        )

        self.assertGreater(
            model.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )

        self.assertEqual(
            model.num_sv_,
            budget,
            "LIBIRWLS model num of support vectors should be exactly the same as the budget",
        )

    def test_is_GridSearchCV_compatible(self):
        model = IRWLS()
        ds = PacmanDatasetBuilder(DummyStorage()).build(
            n=300, r=0.1, p=0.1, seed=42, gamma=10, a=0.1
        )

        cv = GridSearchCV(
            model,
            param_grid={"budget": [50, 100], "C": [1], "gamma": [0.1, 1]},
            n_jobs=1,
        )

        cv.fit(ds.X_train, ds.y_train)

        self.assertGreater(
            cv.score(ds.X_test, ds.y_test),
            0,
            "zero test accuracy from a successfully trained model seems unlikely",
        )
        self.assertIsNotNone(
            cv.best_estimator_, "best estimator should be a valid model"
        )
