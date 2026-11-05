"""편의점 신규 배치와 기존 시설 보완을 각각 지원하는 최대 커버 모델."""
from __future__ import annotations

import numpy as np
from pyproj import Transformer
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack
from scipy.spatial import cKDTree


def project_points(records):
    """입력 경위도는 WGS84, 거리 계산은 EPSG:5179(m)."""
    if not records:
        return np.empty((0, 2), dtype=float)
    xy = np.asarray([[r["lon"], r["lat"]] for r in records], dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError("좌표에 결측값 또는 무한값이 있습니다.")
    if not (((124 < xy[:, 0]) & (xy[:, 0] < 132)) &
            ((33 < xy[:, 1]) & (xy[:, 1] < 39))).all():
        raise ValueError("WGS84 한국 범위 밖 좌표입니다. 좌표계와 위경도 순서를 확인하세요.")
    t = Transformer.from_crs(4326, 5179, always_xy=True)
    return np.column_stack(t.transform(xy[:, 0], xy[:, 1]))


def radius_matrix(demand_xy, facilities_xy, radius_m):
    """수요지×시설의 희소 0/1 행렬. 직선거리이며 보행거리가 아님."""
    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("반경은 양수(m)여야 합니다.")
    if len(facilities_xy) == 0:
        return csr_matrix((len(demand_xy), 0), dtype=float)
    hits = cKDTree(facilities_xy).query_ball_point(demand_xy, radius_m)
    rows = np.repeat(np.arange(len(hits)), [len(h) for h in hits])
    cols = np.array([j for h in hits for j in h], dtype=int)
    return csr_matrix((np.ones(len(cols)), (rows, cols)),
                      shape=(len(demand_xy), len(facilities_xy)))


def solve_maximum_coverage(population, candidate_coverage, existing_covered,
                           max_new, time_limit=60):
    """시설 수 상한 내 총 커버 인구 최대화 후, 같은 효과의 최소 시설 수 선택.

    모든 인구는 접근성의 가중치다. 폐의약품 발생량/수거량을 뜻하지 않는다.
    외부에서 만든 보행 네트워크 도달 행렬도 candidate_coverage에 넣을 수 있다.
    """
    w = np.asarray(population, dtype=float)
    a = csr_matrix(candidate_coverage, dtype=float)
    baseline = np.asarray(existing_covered, dtype=bool)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError("인구는 결측 없는 0 이상 1차원 배열이어야 합니다.")
    if w.sum() <= 0:
        raise ValueError("총인구가 0이므로 커버율을 계산할 수 없습니다.")
    if a.shape[0] != len(w) or baseline.shape != w.shape:
        raise ValueError("수요지와 도달 행렬의 행 개수가 다릅니다.")
    if not np.isfinite(a.data).all() or not np.isin(a.data, [0, 1]).all():
        raise ValueError("도달 행렬은 0/1이어야 합니다.")
    if isinstance(max_new, bool) or int(max_new) != max_new or max_new < 0:
        raise ValueError("설치 수 상한은 0 이상 정수여야 합니다.")
    n, m = a.shape
    selected = np.array([], dtype=int)
    statuses = []
    uncovered = ~baseline & (w > 0)
    if max_new and m and uncovered.any():
        au = a[uncovered]
        wu = w[uncovered]
        k = len(wu)
        # 변수: 시설 x(m), 기존 미커버 수요의 신규 커버 y(k).
        constraint = vstack([
            hstack([-au, eye(k, format="csr")]),
            csr_matrix(np.r_[np.ones(m), np.zeros(k)].reshape(1, -1))
        ], format="csr")
        limits = np.r_[np.zeros(k), min(int(max_new), m)]
        objective = np.r_[np.zeros(m), -wu]
        bounds = Bounds(np.zeros(m + k), np.ones(m + k))
        options = {"time_limit": time_limit, "mip_rel_gap": 0.0}
        result = milp(objective, integrality=np.ones(m + k), bounds=bounds,
                      constraints=LinearConstraint(constraint, -np.inf, limits),
                      options=options)
        if not result.success:
            raise RuntimeError(f"최대 커버 계산이 완료되지 않았습니다: {result.message}")
        statuses.append({"phase": "max_coverage", "status": int(result.status),
                         "mip_gap": float(result.mip_gap or 0)})
        optimum = float(-result.fun)
        # 같은 최적 효과를 내는 불필요한 추가 시설은 제거한다.
        tolerance = max(1e-7, optimum * 1e-10)
        second_constraint = vstack([constraint, csr_matrix(objective.reshape(1, -1))])
        result2 = milp(np.r_[np.ones(m), np.zeros(k)],
                       integrality=np.ones(m + k), bounds=bounds,
                       constraints=LinearConstraint(second_constraint, -np.inf,
                                                    np.r_[limits, -optimum + tolerance]),
                       options=options)
        if not result2.success:
            raise RuntimeError(f"최소 시설 수 계산이 완료되지 않았습니다: {result2.message}")
        statuses.append({"phase": "min_facilities", "status": int(result2.status),
                         "mip_gap": float(result2.mip_gap or 0)})
        selected = np.flatnonzero(result2.x[:m] > 0.5)
    new_covered = (np.asarray(a[:, selected].sum(axis=1)).ravel() > 0
                   if len(selected) else np.zeros(n, dtype=bool))
    final_covered = baseline | new_covered
    total = float(w.sum())
    before = float(w[baseline].sum())
    after = float(w[final_covered].sum())
    return {
        "selected_indices": selected.tolist(),
        "max_new": int(max_new), "selected_count": len(selected),
        "total_population": total, "baseline_covered_population": before,
        "final_covered_population": after, "additional_population": after - before,
        "baseline_coverage_pct": 100 * before / total,
        "final_coverage_pct": 100 * after / total,
        "improvement_percentage_points": 100 * (after - before) / total,
        "existing_covered": baseline.tolist(), "final_covered": final_covered.tolist(),
        "solver": statuses or [{"phase": "no_optimization_needed", "status": 0}],
    }


def validate_records(records, population=False):
    if not isinstance(records, list) or not records:
        raise ValueError("입력은 비어 있지 않은 JSON 레코드 목록이어야 합니다.")
    ids = [r.get("id") for r in records]
    if any(not isinstance(i, str) or not i.strip() for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("각 행에는 중복 없는 문자열 id가 필요합니다.")
    project_points(records)
    if population:
        for r in records:
            v = r.get("population")
            if v is None or isinstance(v, bool) or not np.isfinite(float(v)) or float(v) < 0:
                raise ValueError(f"{r['id']}: 인구 결측/음수는 자동으로 0 처리하지 않습니다.")


def run_scenario(demand, candidates, existing, radius_m, max_new):
    validate_records(demand, population=True)
    validate_records(candidates)
    if existing:
        validate_records(existing)
    xy = project_points(demand)
    a = radius_matrix(xy, project_points(candidates), radius_m)
    e = radius_matrix(xy, project_points(existing), radius_m)
    baseline = np.asarray(e.sum(axis=1)).ravel() > 0
    result = solve_maximum_coverage([r["population"] for r in demand], a, baseline, max_new)
    result["selected_ids"] = [candidates[i]["id"] for i in result.pop("selected_indices")]
    result["radius_m"] = radius_m
    result["distance_type"] = "EPSG:5179 Euclidean; demand representative points"
    result["demand_ids"] = [r["id"] for r in demand]
    result["group_metrics"] = {}
    groups = sorted({r.get("district") or "미분류" for r in demand})
    for group in groups:
        mask = np.array([(r.get("district") or "미분류") == group for r in demand])
        w = np.array([float(r["population"]) for r in demand])
        b, f = np.array(result["existing_covered"]), np.array(result["final_covered"])
        denominator = float(w[mask].sum())
        result["group_metrics"][group] = {
            "population": denominator,
            "before_covered": float(w[mask & b].sum()),
            "after_covered": float(w[mask & f].sum()),
            "after_pct": float(w[mask & f].sum() / denominator * 100) if denominator else None,
        }
    return result
