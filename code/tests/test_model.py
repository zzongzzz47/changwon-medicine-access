import unittest
import numpy as np
from src.coverage_model import solve_maximum_coverage, radius_matrix, validate_records


class CoverageTests(unittest.TestCase):
    def test_existing_coverage_changes_choice(self):
        # 신규만 고려하면 100명 위치, 기존 포함하면 미커버 30명 위치를 선택해야 한다.
        a = np.array([[1, 0], [0, 1]])
        r = solve_maximum_coverage([100, 30], a, [True, False], 1)
        self.assertEqual(r["selected_indices"], [1])
        self.assertEqual(r["additional_population"], 30)

    def test_overlap_counted_once_and_minimum_facilities(self):
        r = solve_maximum_coverage([100, 30], [[1, 1], [1, 1]], [False, False], 2)
        self.assertEqual(r["final_covered_population"], 130)
        self.assertEqual(r["selected_count"], 1)

    def test_zero_budget_and_full_baseline(self):
        for baseline, budget in [([False, False], 0), ([True, True], 2)]:
            r = solve_maximum_coverage([100, 30], [[1], [1]], baseline, budget)
            self.assertEqual(r["additional_population"], 0)
            self.assertEqual(r["selected_count"], 0)

    def test_radius_boundary_and_empty_facilities(self):
        a = radius_matrix(np.array([[0, 0], [300, 400], [300, 400.1]]), np.array([[0, 0]]), 500)
        self.assertEqual(a.toarray().ravel().tolist(), [1, 1, 0])
        self.assertEqual(radius_matrix(np.array([[0, 0]]), np.empty((0, 2)), 500).shape, (1, 0))

    def test_missing_population_and_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError):
            validate_records([{"id": "a", "lon": 128.5, "lat": 35.2, "population": None}], True)
        with self.assertRaises(ValueError):
            validate_records([{"id": "a", "lon": 128.5, "lat": 35.2}] * 2)

    def test_invalid_weights_rejected(self):
        for weight in [[-1], [float("nan")], [0]]:
            with self.assertRaises(ValueError):
                solve_maximum_coverage(weight, [[1]], [False], 1)


if __name__ == "__main__":
    unittest.main()
