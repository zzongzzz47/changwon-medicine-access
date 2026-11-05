import unittest
import numpy as np
from run_operational_demo import demo_input
from src.operational_model import solve_operational_location


class OperationalModelTests(unittest.TestCase):
    def test_capacity_frequency_equity_budget_tradeoffs(self):
        data = demo_input()
        group = [dict(name="priority", demand_indices=[2], minimum_rate=1)]
        basic = {**data, "plans": [p for p in data["plans"] if p["plan_id"] != "A_frequent"]}
        r0 = solve_operational_location(**basic, budget=6)
        r1 = solve_operational_location(**data, budget=6)
        r2 = solve_operational_location(**data, budget=6, service_groups=group)
        r3 = solve_operational_location(**data, budget=8, service_groups=group)
        for result, expected in zip([r0, r1, r2, r3], [100, 160, 100, 200]):
            self.assertAlmostEqual(result["potentially_served_population"], expected, places=5)
            allocation = np.array(result["allocation_fraction"])
            self.assertTrue((allocation.sum(axis=1) <= 1 + 1e-6).all())
            self.assertTrue((np.array(data["load_l_per_week"]) @ allocation <= np.array(result["selected_capacity_l_per_week_by_site"]) + 1e-6).all())
        self.assertEqual(r1["selected_plan_ids"], ["A_frequent"])
        self.assertEqual(r2["selected_plan_ids"], ["B_basic"])
        self.assertEqual(set(r3["selected_plan_ids"]), {"A_basic", "B_basic"})

    def test_impossible_group_and_zero_budget(self):
        data = demo_input()
        impossible = [dict(name="everyone", demand_indices=[0, 1, 2], minimum_rate=1)]
        self.assertFalse(solve_operational_location(**data, budget=6, service_groups=impossible)["feasible"])
        r = solve_operational_location(**data, budget=0)
        self.assertEqual(r["selected_plan_ids"], [])
        self.assertAlmostEqual(r["potentially_served_population"], 0)

    def test_ineligible_site_is_not_selected(self):
        data = demo_input()
        for plan in data["plans"]:
            if plan["site_index"] == 0:
                plan["eligible"] = False
        r = solve_operational_location(**data, budget=100)
        self.assertEqual(r["selected_site_indices"], [1])
        self.assertAlmostEqual(r["potentially_served_population"], 100, places=5)

    def test_quantity_unit_scaling_preserves_population(self):
        data = demo_input()
        r1 = solve_operational_location(**data, budget=6)
        data["load_l_per_week"] = np.array(data["load_l_per_week"]) * 1000
        for p in data["plans"]:
            p["capacity_l_per_week"] *= 1000
        r2 = solve_operational_location(**data, budget=6)
        self.assertEqual(r1["selected_plan_ids"], r2["selected_plan_ids"])
        self.assertAlmostEqual(r1["potentially_served_population"], r2["potentially_served_population"], places=5)

    def test_missing_load_rejected(self):
        data = demo_input()
        data["load_l_per_week"][0] = float("nan")
        with self.assertRaises(ValueError):
            solve_operational_location(**data, budget=6)
        data = demo_input()
        data["population"][0] = 0
        with self.assertRaises(ValueError):
            solve_operational_location(**data, budget=6)


if __name__ == "__main__":
    unittest.main()
