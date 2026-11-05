import itertools
import unittest
import numpy as np
from src.fair_coverage_model import solve_fair_coverage


class FairCoverageTests(unittest.TestCase):
    def test_frontier_agrees_with_exhaustive_search(self):
        w = np.array([60., 30., 20., 10.])
        a = np.array([[1,0,0,1],[1,1,0,0],[0,1,1,0],[0,0,1,1]])
        groups = {"A":np.array([60.,30.,0,0]), "B":np.array([0.,0.,20.,10.])}
        for count in [0,1,2]:
            possibilities = []
            for k in range(count+1):
                for idx in itertools.combinations(range(4), k):
                    hit = a[:,idx].sum(axis=1)>0
                    possibilities.append((min(v[hit].sum()/v.sum() for v in groups.values()), w[hit].sum()))
            got = solve_fair_coverage(w,a,count,groups,maximize_minimum=True,relative_gap=0)
            expected = max(possibilities)
            self.assertAlmostEqual(got['minimum_group_coverage_rate'], expected[0])
            self.assertAlmostEqual(got['weighted_coverage'], expected[1])

    def test_impossible_floor_is_reported(self):
        r = solve_fair_coverage([10,20],[[1],[0]],1,{'unreachable':[0,20]}, {'unreachable':.1})
        self.assertFalse(r['feasible'])

    def test_coverage_never_double_counts(self):
        r = solve_fair_coverage([10,20],[[1,1],[0,1]],2)
        self.assertEqual(r['weighted_coverage'],30)
        self.assertEqual(r['coverage_rate'],1)

if __name__ == '__main__':
    unittest.main()
