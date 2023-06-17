import unittest

from experiments.storage.DummyStorage import DummyStorage
from experiments.synthetic_datasets.PacmanDatasetBuilder import PacmanDatasetBuilder


class TestPacmanDatasetBuilder(unittest.TestCase):
    def test_different_params_creates_different_hash(self):
        pl1 = {"n": 2000, "a": 0.10, "gamma": 10}
        pl2 = {"n":  500, "a": 0.10, "gamma": 10}
        storage = DummyStorage()

        dsb = PacmanDatasetBuilder(storage)
        ds1 = dsb.build(**pl1)
        ds2 = dsb.build(**pl2)

        self.assertNotEqual(ds1.id, ds2.id)
