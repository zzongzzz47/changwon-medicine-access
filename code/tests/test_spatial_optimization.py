import itertools
import unittest
import numpy as np
from scipy.sparse import csr_matrix
from run_spatial_optimization import compress_rows, evaluate
import pandas as pd


class SpatialOptimizationTests(unittest.TestCase):
    def test_compression_preserves_every_selection_and_group_denominator(self):
        a=csr_matrix([[1,0,1],[1,0,1],[0,1,0],[0,0,0],[0,1,0]])
        values={'population':np.array([10,20,30,40,50]),
                'district:A':np.array([10,0,30,0,50]),
                'district:B':np.array([0,20,0,40,0])}
        b,weights=compress_rows(a,values)
        self.assertEqual(b.shape[0],3)
        for n in range(4):
            for selected in itertools.combinations(range(3),n):
                before=np.asarray(a[:,list(selected)].sum(axis=1)).ravel()>0
                after=np.asarray(b[:,list(selected)].sum(axis=1)).ravel()>0
                for k,v in values.items():
                    self.assertAlmostEqual(v[before].sum(),weights[k][after].sum())
                    self.assertAlmostEqual(v.sum(),weights[k].sum())

    def test_population_and_households_have_separate_denominators(self):
        d=pd.DataFrame({'population':[100.,50.],'households':[0.,20.], 'district':['A','A']})
        m=evaluate(d,csr_matrix([[1],[0]]),[0])
        self.assertAlmostEqual(m['population_coverage_pct'],100*100/150)
        self.assertEqual(m['households_coverage_pct'],0)

if __name__=='__main__': unittest.main()
