import unittest
from itertools import combinations
import numpy as np
from src.supplement_model import solve_supplement


class SupplementTests(unittest.TestCase):
    def test_existing_offsets_in_fairness(self):
        # A구 기존 90/100, B구 기존 0/100. 신규 A 10명 또는 B 60명 중 B를 골라야 한다.
        w=np.array([90.,10.,40.,60.]);b=np.array([1,0,0,0],bool)
        a=np.array([[0,0],[1,0],[0,0],[0,1]])
        groups={'A':np.array([90,10,0,0]),'B':np.array([0,0,40,60])}
        s=solve_supplement(w,a,b,1,group_weights=groups,maximize_minimum=True)
        self.assertEqual(s['selected_indices'],[1])
        self.assertAlmostEqual(s['group_rates']['A'],.9)
        self.assertAlmostEqual(s['group_rates']['B'],.6)

    def test_bruteforce_efficiency_fairness_and_fixed(self):
        rng=np.random.default_rng(117)
        for _ in range(5):
            a=(rng.random((9,5))<.35).astype(int);w=rng.integers(1,25,9).astype(float)
            b=rng.random(9)<.25
            groups={'a':np.where(np.arange(9)<4,w,0),'b':np.where(np.arange(9)>=4,w,0)}
            for fair,fixed in [(False,()),(True,()),(False,(0,))]:
                exhaustive=[]
                for n in range(3):
                    for ids in combinations(range(5),n):
                        if not set(fixed)<=set(ids):continue
                        covered=b|(a[:,ids].sum(axis=1)>0)
                        score=w[covered].sum()
                        floor=min(v[covered].sum()/v.sum() for v in groups.values())
                        exhaustive.append((floor,score) if fair else (score,))
                optimum=max(exhaustive)
                s=solve_supplement(w,a,b,2,group_weights=groups,maximize_minimum=fair,fixed_indices=fixed)
                self.assertTrue(s['feasible'])
                if fair:self.assertAlmostEqual(min(s['group_rates'].values()),optimum[0])
                self.assertAlmostEqual(s['total_weight'],optimum[-1])

    def test_no_new_and_invalid_fixed_budget(self):
        s=solve_supplement([5,6],[[1],[1]],[True,False],0)
        self.assertEqual(s['selected_indices'],[])
        self.assertEqual(s['additional_weight'],0)
        with self.assertRaises(ValueError):solve_supplement([5,6],[[1],[1]],[True,False],0,fixed_indices=(0,))


if __name__=='__main__':unittest.main()
