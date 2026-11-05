"""편의점별 운영안·예산·용량·집단별 서비스 수준을 반영한 입지 모형.

연속 배정은 집계 수요를 나눌 수 있다는 계획 가정이다. 반환 행동 예측이
아니며, 부하와 용량은 모두 L/주, 비용은 같은 사업기간 단위여야 한다.
"""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def solve_operational_location(population, load_l_per_week, reachable, plans,
                               budget, service_groups=None, travel_minutes=None,
                               max_sites=None, time_limit=60):
    """최대 서비스 인구 → 최소 비용 → 최소 총 보행시간 순으로 푼다.

    plans: site_index, plan_id, cost, capacity_l_per_week, eligible 필수.
    reachable: 서로 겹치지 않는 수요 단위 × 후보 점포의 0/1 행렬.
    service_groups: {name, demand_indices, minimum_rate} 목록. 집단끼리는
      겹쳐도 되지만, population의 각 행은 중복 없는 인구여야 한다.
    운영시간은 시간대별 reachable 시나리오로 먼저 반영한다.
    """
    w = np.asarray(population, dtype=float)
    d = np.asarray(load_l_per_week, dtype=float)
    a = np.asarray(reachable)
    if (w.ndim != 1 or not len(w) or not np.isfinite(w).all()
            or (w < 0).any() or w.sum() <= 0):
        raise ValueError("중복 없는 비음수 인구가 필요합니다.")
    if d.shape != w.shape or not np.isfinite(d).all() or (d < 0).any():
        raise ValueError("모든 수요의 계획 부하(L/주)가 필요합니다. 결측을 0으로 대체하지 마세요.")
    if ((w == 0) & (d > 0)).any():
        raise ValueError("인구가 0인 수요에 양의 부하를 배정할 수 없습니다. 수요 단위를 확인하세요.")
    if a.ndim != 2 or a.shape[0] != len(w) or not a.shape[1] or not np.isin(a, [0, 1]).all():
        raise ValueError("수요×점포의 0/1 도달 행렬이 필요합니다.")
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("예산은 0 이상이어야 합니다.")
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("계산 시간 제한은 양수여야 합니다.")
    n, m = a.shape
    if max_sites is not None and (isinstance(max_sites, bool) or not np.isfinite(max_sites)
            or int(max_sites) != max_sites or max_sites < 0):
        raise ValueError("설치 상한은 0 이상 정수여야 합니다.")
    if not plans:
        raise ValueError("최소 하나의 운영안이 필요합니다.")
    by_site = [[] for _ in range(m)]
    costs, capacities, eligible, ids = [], [], [], []
    for p, plan in enumerate(plans):
        j = plan["site_index"]
        if isinstance(j, bool) or not isinstance(j, (int, np.integer)) or not 0 <= j < m:
            raise ValueError("운영안의 점포 인덱스가 유효하지 않습니다.")
        if not isinstance(plan["eligible"], (bool, np.bool_)):
            raise ValueError("실행 가능 여부는 명시적인 참/거짓이어야 합니다. 미확인은 별도 시나리오로 처리하세요.")
        cost, cap = float(plan["cost"]), float(plan["capacity_l_per_week"])
        if not np.isfinite([cost, cap]).all() or min(cost, cap) < 0:
            raise ValueError("비용·용량은 결측 없는 비음수여야 합니다.")
        if not plan["plan_id"] or plan["plan_id"] in ids:
            raise ValueError("운영안 ID는 중복 없이 지정해야 합니다.")
        ids.append(plan["plan_id"])
        costs.append(cost)
        capacities.append(cap)
        eligible.append(int(plan["eligible"]))
        by_site[j].append(p)
    if any(not p for p in by_site):
        raise ValueError("모든 후보 점포에 운영안을 지정하세요.")
    if travel_minutes is not None:
        travel = np.asarray(travel_minutes, dtype=float)
        if travel.shape != a.shape or not np.isfinite(travel[a == 1]).all() or (travel[a == 1] < 0).any():
            raise ValueError("도달 가능한 모든 연결에 비음수 보행시간이 필요합니다.")
    else:
        travel = None
    groups = []
    for group in service_groups or []:
        idx = list(group["demand_indices"])
        rate = float(group["minimum_rate"])
        if (not idx or len(set(idx)) != len(idx) or any(
                isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < n for i in idx)
                or not np.isfinite(rate) or not 0 <= rate <= 1 or w[idx].sum() <= 0):
            raise ValueError("집단의 인덱스·인구·최소 서비스 비율을 확인하세요.")
        if group["name"] in [g["name"] for g in groups]:
            raise ValueError("집단 이름은 중복 없이 지정하세요.")
        groups.append({"name": group["name"], "indices": idx, "rate": rate})

    p_count = len(plans)
    edges = [(int(i), int(j)) for i, j in zip(*np.nonzero(a)) if w[i] > 0]
    size = p_count + len(edges)
    demand_edges = [[] for _ in range(n)]
    site_edges = [[] for _ in range(m)]
    for e, (i, j) in enumerate(edges, p_count):
        demand_edges[i].append(e)
        site_edges[j].append((e, i))
    rows, lower, upper = [], [], []

    def add(row, lo=-np.inf, hi=np.inf):
        rows.append(row)
        lower.append(lo)
        upper.append(hi)

    for j in range(m):
        add({p: 1 for p in by_site[j]}, hi=1)
    for i in range(n):
        add({e: 1 for e in demand_edges[i]}, hi=1)
    for e, (_, j) in enumerate(edges, p_count):
        add({e: 1, **{p: -1 for p in by_site[j]}}, hi=0)
    for j in range(m):
        add({**{e: d[i] for e, i in site_edges[j]},
             **{p: -capacities[p] for p in by_site[j]}}, hi=0)
    add({p: c for p, c in enumerate(costs)}, hi=budget)
    if max_sites is not None:
        add({p: 1 for p in range(p_count)}, hi=max_sites)
    for g in groups:
        add({e: w[i] for i in g["indices"] for e in demand_edges[i]},
            lo=g["rate"] * w[g["indices"]].sum())

    served_coeff = np.r_[np.zeros(p_count), [w[i] for i, _ in edges]]
    cost_coeff = np.r_[costs, np.zeros(len(edges))]
    bounds = Bounds(np.zeros(size), np.r_[eligible, np.ones(len(edges))])
    integrality = np.r_[np.ones(p_count), np.zeros(len(edges))]
    stages = []

    def optimize(objective, name):
        rr, cc, vv = [], [], []
        for r, row in enumerate(rows):
            for c, value in row.items():
                if value != 0:
                    rr.append(r)
                    cc.append(c)
                    vv.append(value)
        matrix = csr_matrix((vv, (rr, cc)), shape=(len(rows), size))
        result = milp(objective, integrality=integrality, bounds=bounds,
                      constraints=LinearConstraint(matrix, lower, upper),
                      options={"mip_rel_gap": 0.0, "time_limit": time_limit})
        stages.append({"stage": name, "status": int(result.status),
                       "message": result.message})
        if result.status == 2:
            return None
        if not result.success:
            raise RuntimeError(f"{name}: 계산이 완료되지 않았습니다: {result.message}")
        return result

    r = optimize(-served_coeff, "maximum_service")
    if r is None:
        return {"feasible": False, "stages": stages,
                "message": "예산·후보·용량 안에서 최소 서비스 조건을 만족할 수 없습니다."}
    optimum = float(served_coeff @ r.x)
    tolerance = max(1e-7, float(w.sum()) * 1e-9)
    add({i: v for i, v in enumerate(served_coeff) if v}, lo=max(0, optimum - tolerance))
    r = optimize(cost_coeff, "minimum_cost")
    if r is None:
        raise RuntimeError("동률 해 선택 단계에서 수치적 불일치가 발생했습니다.")
    minimum_cost = float(cost_coeff @ r.x)
    if travel is not None:
        add({p: c for p, c in enumerate(costs)}, hi=minimum_cost + max(1e-7, minimum_cost * 1e-9))
        objective = np.r_[np.zeros(p_count), [w[i] * travel[i, j] for i, j in edges]]
        r = optimize(objective, "minimum_walk_time")
        if r is None:
            raise RuntimeError("보행시간 최소화 단계에서 수치적 불일치가 발생했습니다.")

    selected = np.flatnonzero(r.x[:p_count] > .5).tolist()
    allocation = np.zeros((n, m))
    for e, (i, j) in enumerate(edges, p_count):
        allocation[i, j] = max(0.0, min(1.0, r.x[e]))
    fractions = allocation.sum(axis=1)
    served = float(w @ fractions)
    group_result = {g["name"]: {"minimum_rate": g["rate"],
                    "served_rate": float(w[g["indices"]] @ fractions[g["indices"]] / w[g["indices"]].sum())}
                    for g in groups}
    return {"feasible": True, "stages": stages,
            "selected_plan_ids": [ids[p] for p in selected],
            "selected_site_indices": sorted({int(plans[p]["site_index"]) for p in selected}),
            "total_cost": float(sum(costs[p] for p in selected)),
            "potentially_served_population": served,
            "potential_service_pct": 100 * served / float(w.sum()),
            "served_fraction_by_demand": fractions.tolist(),
            "allocation_fraction": allocation.tolist(),
            "assigned_load_l_per_week_by_site": (d @ allocation).tolist(),
            "selected_capacity_l_per_week_by_site": [float(sum(capacities[p] for p in by_site[j] if p in selected)) for j in range(m)],
            "groups": group_result,
            "mean_assigned_walk_minutes": float(sum(w[i] * travel[i, j] * allocation[i, j] for i, j in edges) / served) if travel is not None and served > 0 else None,
            "interpretation": "집계 수요의 배정에 따른 잠재 서비스 가능 인구. 실제 반환 인구·회수량 예측이 아님."}
