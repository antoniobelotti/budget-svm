import pprint
import unittest
from typing import Iterator

from experiments.BaseExperiment import BaseExperiment
from experiments.comparisons.NSSVM import NSSVM
from experiments.datasets.Base import Dataset
from experiments.datasets.PacmanDatasetBuilder import PacmanDatasetBuilder
from experiments.storage.LocalStorage import LocalStorage
from kernel import GaussianKernel
from budgetsvm.svm import SVC


class TestExperiment(unittest.TestCase):
    def test_experiment_with_our_method(self):
        storage = LocalStorage()

        class TExp(BaseExperiment):
            USE_PRECOMPUTED_KERNEL = True
            CROSS_VALIDATION = 2
            RANDOM_STATE = 7
            BUDGET_PERCENTAGES = [0.05, 0.1]

            def generate_datasets(self) -> Iterator[Dataset]:
                yield PacmanDatasetBuilder(storage=self.storage).build(n=100, a=1, gamma=10)
                yield PacmanDatasetBuilder(storage=self.storage).build(n=100, a=3, gamma=10)

            def get_cvgrid_parameter_dict(self) -> dict:
                return {"C": [1], "kernels": [GaussianKernel(2)]}

            def get_empty_model(self):
                return SVC()

        exp = TExp(storage)
        exp.run()

        # CHECKS
        n_produced_models = len(exp.BUDGET_PERCENTAGES) * 2  # 2 datasets

        # log is saved
        l = storage.get_log_file(exp.EXPERIMENT_ID)
        self.assertIsNotNone(l, "log file should have been saved")
        self.assertNotEqual("", l, "log should not be empty")

        # result file has been saved
        res = storage.get_results_json(exp.EXPERIMENT_ID)
        self.assertIsNotNone(res, "result file should have been saved")
        self.assertEqual(len(res), n_produced_models, "result file should contain n_datasets*budget_percentages rows")

        # all the experiment models have been saved
        for row in res:
            self.assertIsNotNone(
                storage.get_model(row["model_UUID"]), f"model {row['model_UUID']} should have been saved"
            )

        pprint.pprint(res)

    def test_experiment_with_NSSVM(self):
        storage = LocalStorage()

        class TExp(BaseExperiment):
            CROSS_VALIDATION = 2
            RANDOM_STATE = 7
            BUDGET_PERCENTAGES = [0.05, 0.1]

            def generate_datasets(self) -> Iterator[Dataset]:
                yield PacmanDatasetBuilder(storage=self.storage).build(n=100, a=1, gamma=10)
                yield PacmanDatasetBuilder(storage=self.storage).build(n=100, a=3, gamma=10)

            def get_cvgrid_parameter_dict(self) -> dict:
                return {}

            def get_empty_model(self):
                return NSSVM()

        exp = TExp(storage)
        exp.run()

        # CHECKS
        n_produced_models = len(exp.BUDGET_PERCENTAGES) * 2  # 2 datasets

        # log is saved
        l = storage.get_log_file(exp.EXPERIMENT_ID)
        self.assertIsNotNone(l, "log file should have been saved")
        self.assertNotEqual("", l, "log should not be empty")

        # result file has been saved
        res = storage.get_results_json(exp.EXPERIMENT_ID)
        self.assertIsNotNone(res, "result file should have been saved")
        self.assertEqual(len(res), n_produced_models, "result file should contain n_datasets*budget_percentages rows")

        # all the experiment models have been saved
        for row in res:
            self.assertIsNotNone(
                storage.get_model(row["model_UUID"]), f"model {row['model_UUID']} should have been saved"
            )

        pprint.pprint(res)
