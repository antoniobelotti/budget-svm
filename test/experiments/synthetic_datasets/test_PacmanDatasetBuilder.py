import unittest

from experiments.storage.DummyStorage import DummyStorage
from experiments.datasets.PacmanDatasetBuilder import PacmanDatasetBuilder


class TestPacmanDatasetBuilder(unittest.TestCase):
    def test_different_params_creates_different_hash(self):
        table = [
            {"n": 20, "a": 0.10, "gamma": 10},
            {"n": 50, "a": 0.10, "gamma": 10},
            {"n": 50, "a": 0.10, "gamma": 10, "scale_min_max": True},
        ]

        storage = DummyStorage()
        dsb = PacmanDatasetBuilder(storage)

        ids = [ds.id for ds in [dsb.build(**params) for params in table]]

        self.assertEqual(sorted(ids), sorted(list(set(ids))))

    def test_min_max_scaled_dataset_generation(self):
        not_scaled_params = {"n": 20, "a": 0.10, "gamma": 10}
        scaled_params = {"n": 20, "a": 0.10, "gamma": 10, "scale_min_max": True}

        storage = DummyStorage()
        dsb = PacmanDatasetBuilder(storage)

        not_scaled_ds = dsb.build(**not_scaled_params)
        scaled_ds = dsb.build(**scaled_params)

        # scaled range is [0,1]
        self.assertLessEqual(0.0, scaled_ds.X.min())
        self.assertGreaterEqual(1.0, scaled_ds.X.min())

        # not scaled range depends on gamma.
        self.assertGreater(not_scaled_ds.X.max() - not_scaled_ds.X.min(), 1)