import unittest
import geopandas as gpd
import numpy as np
from shapely.geometry import box
from src.spatial_join import allocate_grid_to_areas


class SpatialAllocationTests(unittest.TestCase):
    def setUp(self):
        self.grid = gpd.GeoDataFrame(dict(grid_id=["test"],population=[100.],households=[40.]),
                                    geometry=[box(1000000,1700000,1000100,1700100)],crs=5179)

    def test_clipped_city_uses_original_grid_denominator(self):
        area = gpd.GeoDataFrame(dict(adm_cd=["A"]), geometry=[box(1000000,1700000,1000050,1700100)],crs=5179)
        out = allocate_grid_to_areas(self.grid,area)
        self.assertAlmostEqual(out.population.sum(),50)
        self.assertAlmostEqual(out.households.sum(),20)
        self.assertAlmostEqual(out.x_m.iloc[0],1000050)

    def test_two_areas_preserve_total_mass(self):
        areas = gpd.GeoDataFrame(dict(adm_cd=["A","B"]),
                                geometry=[box(1000000,1700000,1000040,1700100),box(1000040,1700000,1000100,1700100)],crs=5179)
        out = allocate_grid_to_areas(self.grid,areas)
        self.assertAlmostEqual(out.population.sum(),100)
        self.assertEqual(sorted(out.population.round(6).tolist()),[40.,60.])

    def test_missing_households_are_not_zero(self):
        self.grid.loc[0,"households"] = np.nan
        area = gpd.GeoDataFrame(dict(adm_cd=["A"]),geometry=self.grid.geometry.copy(),crs=5179)
        out = allocate_grid_to_areas(self.grid,area)
        self.assertTrue(out.households.isna().all())
        self.assertFalse(out.households_observed.any())

    def test_overlapping_boundaries_are_rejected(self):
        areas = gpd.GeoDataFrame(dict(adm_cd=["A","B"]),geometry=[self.grid.geometry.iloc[0]]*2,crs=5179)
        with self.assertRaises(ValueError):
            allocate_grid_to_areas(self.grid,areas)

if __name__ == '__main__':
    unittest.main()
