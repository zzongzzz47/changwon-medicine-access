"""총인구·가구 등 단일 지표 커버와 집단 간 최소 커버율을 비교한다.

기존 시설은 입력하지 않는다. 거리 행렬의 1은 주어진 서비스 기준에서
도달 가능하다는 뜻이며, 실제 반납이나 물량을 뜻하지 않는다.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack


def solve_fair_coverage(weights, reachable, max_sites, group_weights=None,
                        group_floors=None, maximize_minimum=False,
                        time_limit=60, relative_gap=0.001):
    w = np.asarray(weights, dtype=float)
    a = csr_matrix(reachable, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("커버 가중치는 양의 합을 갖는 결측 없는 비음수 배열이어야 합니다.")
    n, m = a.shape
    if n != len(w) or m == 0 or not np.isin(a.data, [0, 1]).all():
        raise ValueError("수요지×후보의 0/1 도달 행렬을 확인하세요.")
    if isinstance(max_sites, bool) or int(max_sites) != max_sites or max_sites < 0:
        raise ValueError("설치 수는 0 이상 정수여야 합니다.")
    groups = {str(k): np.asarray(v, dtype=float) for k, v in (group_weights or {}).items()}
    for k, v in groups.items():
        if v.shape != (n,) or not np.isfinite(v).all() or (v < 0).any() or v.sum() <= 0:
            raise ValueError(f"집단 {k}의 가중치를 확인하세요.")
    floors = group_floors or {}
    for k, f in floors.items():
        if k not in groups or not np.isfinite(f) or not 0 <= f <= 1:
            raise ValueError("집단별 최소 커버율은 정의된 집단과 0~1 범위여야 합니다.")
    if maximize_minimum and not groups:
        raise ValueError("최소 커버율 최대화에는 집단 정보가 필요합니다.")

    # 변수 x: 후보 선택, y: 수요 커버, t: 집단별 최소 커버율.
    # y는 연속 변수여도 x가 이진이고 가중치가 비음수이면 유효하다.
    size = m + n + int(maximize_minimum)
    z = csr_matrix((n, int(maximize_minimum)))
    matrix = [hstack([-a, eye(n, format="csr"), z], format="csr")]
    limits = [np.zeros(n)]
    row = np.zeros(size); row[:m] = 1
    matrix.append(csr_matrix(row[None, :])); limits.append([min(max_sites, m)])
    for k, v in groups.items():
        normalized = v / v.sum()
        if maximize_minimum:
            row = np.zeros(size); row[m:m+n] = -normalized; row[-1] = 1
            matrix.append(csr_matrix(row[None, :])); limits.append([0])
        if k in floors:
            row = np.zeros(size); row[m:m+n] = -normalized
            matrix.append(csr_matrix(row[None, :])); limits.append([-floors[k]])
    mat = vstack(matrix, format="csr"); ub = np.concatenate(limits)
    integrity = np.r_[np.ones(m), np.zeros(size-m)]
    options = dict(time_limit=time_limit, mip_rel_gap=relative_gap)
    objective = np.zeros(size)
    objective[-1] = -1 if maximize_minimum else 0
    if not maximize_minimum:
        objective[m:m+n] = -w/w.sum()
    result = milp(objective, integrality=integrity, bounds=Bounds(0, 1),
                  constraints=LinearConstraint(mat, -np.inf, ub), options=options)
    phases = []

    def status(r, phase):
        return dict(phase=phase, status=int(r.status), message=r.message,
                    proven_optimal=bool(r.status == 0 and (getattr(r, "mip_gap", 0) or 0) <= 1e-8),
                    mip_gap=float(r.mip_gap) if getattr(r, "mip_gap", None) is not None else None,
                    objective_upper_bound=float(-r.mip_dual_bound) if getattr(r, "mip_dual_bound", None) is not None else None)

    phases.append(status(result, "max_min_group" if maximize_minimum else "max_coverage"))
    if result.x is None:
        return dict(feasible=False, phases=phases, selected_indices=[])
    if maximize_minimum:
        # 1단계에서 달성한 최소 커버율을 유지하면서 전체 커버를 개선한다.
        threshold = max(0, float(result.x[-1]) - 1e-8)
        row = np.zeros(size); row[-1] = -1
        refined_mat = vstack([mat, csr_matrix(row[None, :])], format="csr")
        objective = np.zeros(size); objective[m:m+n] = -w/w.sum()
        refined = milp(objective, integrality=integrity, bounds=Bounds(0, 1),
                       constraints=LinearConstraint(refined_mat, -np.inf, np.r_[ub, -threshold]),
                       options=options)
        phases.append(status(refined, "max_coverage_given_achieved_floor"))
        if refined.x is not None:
            result = refined
    selected = np.flatnonzero(result.x[:m] > .5)
    covered = np.asarray(a[:, selected].sum(axis=1)).ravel() > 0
    rates = {k: float(v[covered].sum()/v.sum()) for k,v in groups.items()}
    assert len(selected) <= max_sites
    for k, f in floors.items():
        if rates[k] < f - 1e-6:
            raise RuntimeError("계산 결과가 요구한 집단별 최소 커버율을 충족하지 않습니다.")
    return dict(feasible=True, selected_indices=selected.tolist(), covered_indices=np.flatnonzero(covered).tolist(),
                weighted_coverage=float(w[covered].sum()), coverage_rate=float(w[covered].sum()/w.sum()),
                group_coverage_rates=rates, minimum_group_coverage_rate=min(rates.values()) if rates else None,
                phases=phases)
