"""가상 예제로만 운영 모형의 작동을 확인한다. 창원시 결과가 아니다."""
import json
from pathlib import Path
from src.operational_model import solve_operational_location


def demo_input():
    return dict(population=[100, 60, 40], load_l_per_week=[10, 6, 4],
                reachable=[[1, 0], [1, 1], [0, 1]],
                travel_minutes=[[4, 0], [3, 3], [0, 4]],
                plans=[
                    dict(site_index=0, plan_id="A_basic", cost=4, capacity_l_per_week=10, eligible=True),
                    dict(site_index=0, plan_id="A_frequent", cost=6, capacity_l_per_week=20, eligible=True),
                    dict(site_index=1, plan_id="B_basic", cost=4, capacity_l_per_week=10, eligible=True)])


def main():
    inputs = demo_input()
    group = [dict(name="priority_group", demand_indices=[2], minimum_rate=1)]
    basic = {**inputs, "plans": [p for p in inputs["plans"] if p["plan_id"] != "A_frequent"]}
    scenarios = [
        ("budget6_basic_only", basic, dict(budget=6)),
        ("budget6_frequency_choice", inputs, dict(budget=6)),
        ("budget6_with_equity", inputs, dict(budget=6, service_groups=group)),
        ("budget8_with_equity", inputs, dict(budget=8, service_groups=group))]
    results = {name: solve_operational_location(**data, **options) for name, data, options in scenarios}
    result = {"data_basis": "synthetic", "not_changwon_results": True,
              "note": "가상 수요·비용·용량으로 수식의 작동을 확인한 예. 비용은 임의 단위.",
              "inputs": inputs, "scenarios": results}
    target = Path(__file__).parent / "results" / "operational_demo.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(result["note"])
    for name, r in results.items():
        print(name, r["selected_plan_ids"], round(r["potential_service_pct"], 4))


if __name__ == "__main__":
    main()
