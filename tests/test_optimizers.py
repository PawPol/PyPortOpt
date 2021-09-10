from PyPortOpt import Optimizers as o
import unittest


class TestOptimizer(unittest.TestCase):
    def test_testFunction(self):
        self.assertEqual(o.testFunction(), True)


if __name__ == "__main__":
    unittest.main()
