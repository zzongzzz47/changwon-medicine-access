"""대전 제공 자료로 발전 방향을 검증. 창원 최종 분석과 구분하여 저장한다."""
from pathlib import Path
import json
import numpy as np
from src.coverage_model import project_points, radius_matrix, solve_maximum_coverage
from src.research_models import dual_distance, expected_coverage, evaluate_selection

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / "data/pilot/daejeon_supplied.json").read_text(encoding="utf-8"))
    demand, candidates = data["demand"], data["candidates"]
    w = np.array([d["population"] for d in demand])
    dxy, cxy = project_points(demand), project_points(candidates)
    filtered = data["prefilter_indices"]
    matrices = {r: radius_matrix(dxy, cxy, r) for r in [200, 250, 300, 350, 400, 500]}
    zero = np.zeros(len(w), dtype=bool)
    results = {"scope": data["region"], "limitations": data["source_note"],
               "existing_facilities_used": False, "population_weight_sum": float(w.sum()),
               "candidate_ablation": [], "benchmarks": [], "dual_distance": [], "availability": []}
    cache = {}
    for radius in [200, 250, 300, 350, 400]:
        for label, ids in [("prefilter_18", filtered), ("all_73", list(range(len(candidates))))]:
            a = matrices[radius][:, ids]
            all_potential = float(w[np.asarray(a.sum(axis=1)).ravel() > 0].sum() / w.sum() * 100)
            for p in range(1, 19):
                solved = solve_maximum_coverage(w, a, zero, p)
                selected = [ids[i] for i in solved["selected_indices"]]
                cache[(radius, label, p)] = selected
                results["candidate_ablation"].append({"radius_m": radius, "candidates": label,
                    "max_sites": p, "selected_count": len(selected),
                    "coverage_pct": solved["final_coverage_pct"], "all_candidates_ceiling_pct": all_potential,
                    "selected_ids": [candidates[i]["id"] for i in selected], "solver": solved["solver"]})
        print(f"후보 비교 완료: {radius}m", flush=True)
    # 사전 정의한 단순 대안: 주변인구 순위 / 합집합 인구의 순차 증가(탐욕법).
    for p in [6, 10, 11]:
        a = matrices[350].toarray()
        density = sorted(range(len(candidates)), key=lambda j: (-candidates[j]["pop500"], candidates[j]["id"]))[:p]
        selected = []
        current = np.zeros(len(w), dtype=bool)
        for _ in range(p):
            remaining = [j for j in range(a.shape[1]) if j not in selected]
            best = max(remaining, key=lambda j: float(w[(a[:, j] > 0) & ~current].sum()))
            selected.append(best)
            current |= a[:, best] > 0
        for label, chosen in [("population500_rank", density), ("greedy_incremental", selected),
                               ("all_73_mclp", cache[(350, "all_73", p)])]:
            results["benchmarks"].append({"max_sites": p, "method": label,
                **evaluate_selection(w, matrices[350], chosen),
                "selected_ids": [candidates[i]["id"] for i in chosen]})
    for short in [300, 350]:
        for p in [4, 6, 8, 10, 11, 12]:
            r = dual_distance(w, matrices[short], matrices[500], p)
            row = {"preferred_m": short, "maximum_m": 500, "max_sites": p, **r}
            standard = cache[(short, "all_73", p)]
            row["ordinary_preferred_pct"] = evaluate_selection(w, matrices[short], standard)["nominal_pct"]
            row["ordinary_maximum_distance_pct"] = evaluate_selection(w, matrices[500], standard)["nominal_pct"]
            if r["feasible"]:
                chosen = row.pop("selected_indices")
                row.update(evaluate_selection(w, matrices[short], chosen))
                row["maximum_distance_pct"] = evaluate_selection(w, matrices[500], chosen)["nominal_pct"]
                row["selected_ids"] = [candidates[i]["id"] for i in chosen]
            results["dual_distance"].append(row)
    a = matrices[350]
    for label, chosen in [("prefilter_18_mclp", cache[(350, "prefilter_18", 11)]),
                          ("all_73_mclp", cache[(350, "all_73", 11)])]:
        results["availability"].append({"method": label, "max_sites": 11,
             **evaluate_selection(w, a, chosen), "selected_ids": [candidates[i]["id"] for i in chosen]})
    for q in [.1, .2]:
        base = cache[(350, "all_73", 11)]
        optimum_population = float(w[np.asarray(a[:, base].sum(axis=1)).ravel() > 0].sum())
        tie = expected_coverage(w, a, 11, q, minimum_nominal_population=optimum_population)
        chosen = tie.pop("selected_indices")
        results["availability"].append({"method": f"nominal_optimum_best_expected_q{q}",
             "max_sites": 11, **tie, **evaluate_selection(w, a, chosen),
             "selected_ids": [candidates[i]["id"] for i in chosen]})
        r = expected_coverage(w, a, 11, q)
        chosen = r.pop("selected_indices")
        results["availability"].append({"method": f"expected_q{q}", "max_sites": 11, **r,
             **evaluate_selection(w, a, chosen), "selected_ids": [candidates[i]["id"] for i in chosen]})
    path = ROOT / "results/development_pilot.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", path)
    print(json.dumps({"availability": results["availability"], "dual_distance": results["dual_distance"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
