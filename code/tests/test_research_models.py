import itertools
import unittest
import numpy as np
from src.coverage_model import run_scenario
from src.research_models import dual_distance, expected_coverage


class ResearchModelTests(unittest.TestCase):
    def test_expected_matches_enumeration(self):
        w = np.array([100, 45, 20])
        a = np.array([[1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 1, 1]])
        q = .2
        r = expected_coverage(w, a, 2, q)
        brute = max(float(w @ (1 - q ** a[:, s].sum(axis=1)))
                    for s in itertools.combinations(range(4), 2))
        self.assertAlmostEqual(r["expected_population"], brute)
        floor = 160
        constrained = expected_coverage(w, a, 2, q, minimum_nominal_population=floor)
        permitted = [s for s in itertools.combinations(range(4), 2) if w[a[:, s].sum(axis=1)>0].sum() >= floor]
        self.assertAlmostEqual(constrained["expected_population"], max(float(w @ (1 - q ** a[:, s].sum(axis=1))) for s in permitted))

    def test_dual_distance_feasible_and_infeasible(self):
        a = np.eye(2)
        self.assertFalse(dual_distance([100, 10], a, a, 1)["feasible"])
        r = dual_distance([100, 10], a, a, 2)
        self.assertEqual(set(r["selected_indices"]), {0, 1})

    def test_standalone_needs_no_existing_facilities(self):
        demand = [{"id": "d", "lon": 128.6, "lat": 35.2, "population": 100}]
        candidates = [{"id": "c", "lon": 128.6, "lat": 35.2}]
        r = run_scenario(demand, candidates, [], 350, 1)
        self.assertEqual(r["baseline_coverage_pct"], 0)
        self.assertEqual(r["final_coverage_pct"], 100)
