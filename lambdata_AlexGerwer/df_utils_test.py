"""
This is for unittesting the methods in the Stats class of df_utils
"""

import unittest
import statistics as st
from df_utils import Stats

class StatsTests(unittest.TestCase):
    # Input test data
    data = [3, 7, 2, 9, 8, 6, 9]

    """This is meant to test the methods within Stats class"""
    # Test stat_mean
    def test_stat_mean(self):
        a = Stats(self.data)
        mean_calculated = a.stat_mean()
        mean_expected = st.mean(self.data)
        self.assertAlmostEqual(mean_calculated, mean_expected)

    # Test stat_mode
    def test_stat_mode(self):
        a = Stats(self.data)
        mode_calculated = a.stat_mode()
        mode_expected = st.mode(self.data)
        self.assertAlmostEqual(mode_calculated, mode_expected)

    # Test stat_median
    def test_stat_median(self):
        a = Stats(self.data)
        median_calculated = a.stat_median()
        median_expected = st.median(self.data)
        self.assertAlmostEqual(median_calculated, median_expected)

    # Test stat_variance
    def test_stat_variance(self):
        a = Stats(self.data)
        variance_calculated = a.stat_variance()
        variance_expected = st.variance(self.data)
        self.assertAlmostEqual(variance_calculated, variance_expected)

    # Test stat_standard_deviation
    def test_stat_standard_deviation(self):
        a = Stats(self.data)
        standard_deviation_calculated = a.stat_standard_deviation()
        standard_deviation_expected = st.stdev(self.data)
        self.assertAlmostEqual(standard_deviation_calculated,
                               standard_deviation_expected)

if __name__ == '__main__':
    unittest.main()
