"""Continuous training must outlive schedules without negative learning rates."""
import itertools
import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from train import training_iterations, schedule_fraction



class TrainingControlChecks(unittest.TestCase):
    def test_finite_budget_preserved(self):
        assert list(training_iterations(8, 10)) == [9, 10]
        assert list(training_iterations(10, 10)) == []


    def test_continuation_beyond_horizon_and_resume(self):
        assert list(itertools.islice(training_iterations(9, 10, True), 4)) == [10, 11, 12, 13]
        assert list(itertools.islice(training_iterations(100, 10, True), 2)) == [101, 102]


    def test_schedule_floors_after_horizon(self):
        assert schedule_fraction(1, 10) == 0
        assert schedule_fraction(10, 10) == 1
        assert schedule_fraction(100000, 10) == 1
        for iteration in (1, 5, 10, 100000):
            frac = schedule_fraction(iteration, 10)
            assert 1e-4 / 3 <= 1e-4 * (1-frac) + 1e-4 / 3 * frac <= 1e-4
