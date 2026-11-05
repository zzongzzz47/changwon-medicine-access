"""선행연구의 개념을 이용한 편의점 전용 배치 확장 모형. 직선거리 파일럿용."""
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack, vstack, kron


def _inputs(population, coverage, budget):
    w = np.asarray(population, dtype=float)
    a = csr_matrix(coverage, dtype=float)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("결측 없는 비음수 인구가 필요합니다.")
    if a.shape[0] != len(w) or a.shape[1] == 0 or not np.isin(a.data, [0, 1]).all():
        raise ValueError("수요지×후보지의 0/1 도달 행렬이 필요합니다.")
    if isinstance(budget, bool) or int(budget) != budget or budget < 1:
        raise ValueError("설치 상한은 1 이상 정수여야 합니다.")
    return w, a, min(int(budget), a.shape[1])


def dual_distance(population, preferred, maximum, budget, time_limit=60):
    """근거리 인구 최대화 + 양의 인구 수요지 모두 최대거리 내 최소 1곳.

    근거: Rico, White & Albert (2026), CBLP의 기존 시설 없는 경우.
    입력 거리 기준 자체는 별도 지역 근거가 필요하다.
    """
    w, a, p = _inputs(population, preferred, budget)
    b = csr_matrix(maximum, dtype=float)
    if b.shape != a.shape or not np.isin(b.data, [0, 1]).all() or (a - b).max() > 0:
        raise ValueError("최대거리 도달 집합은 근거리 집합을 포함해야 합니다.")
    n, m = a.shape
    active = w > 0
    constraints = vstack([
        hstack([-a, eye(n)]),
        hstack([-b[active], csr_matrix((active.sum(), n))]),
        csr_matrix(np.r_[np.ones(m), np.zeros(n)].reshape(1, -1))
    ], format="csr")
    upper = np.r_[np.zeros(n), -np.ones(active.sum()), p]
    r = milp(np.r_[np.zeros(m), -w], integrality=np.ones(m + n),
             bounds=Bounds(np.zeros(m + n), np.ones(m + n)),
             constraints=LinearConstraint(constraints, -np.inf, upper),
             options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    if r.status == 2:
        return {"feasible": False, "status": 2, "message": "설치 상한 안에서 최대거리 조건을 만족할 수 없음"}
    if not r.success:
        raise RuntimeError(r.message)
    selected = np.flatnonzero(r.x[:m] > .5)
    return {"feasible": True, "selected_indices": selected.tolist(), "status": int(r.status),
            "mip_gap": float(r.mip_gap or 0)}


def expected_coverage(population, coverage, budget, unavailable_probability, time_limit=60,
                      minimum_nominal_population=None):
    """Σ 인구×(1-q^근처선정시설수)를 최대화. 동일·독립 미가용 확률 가정.

    Daskin(1983)의 기대 커버 개념을 점포 가용성 스트레스 시나리오에 적용.
    q는 실측 수치가 아니며, 운영시간/협약 확률의 추정값으로 해석하면 안 된다.
    """
    w, a, p = _inputs(population, coverage, budget)
    q = float(unavailable_probability)
    if not 0 < q < 1:
        raise ValueError("미가용 확률은 0과 1 사이여야 합니다.")
    n, m = a.shape
    k = min(p, int(np.asarray(a.sum(axis=1)).max()))
    if k == 0:
        if minimum_nominal_population is not None and minimum_nominal_population > 0:
            raise ValueError("커버 가능한 수요가 없어 최소 커버 인구를 만족할 수 없습니다.")
        return {"selected_indices": [], "expected_population": 0.0, "status": 0, "mip_gap": 0.0, "q": q}
    # z[i,l]: 수요지 i에 최소 l+1개 시설이 있다. 낮은 단계부터 채운다.
    counts = kron(eye(n), np.ones((1, k)), format="csr")
    matrices = [hstack([-a, counts]),
                csr_matrix(np.r_[np.ones(m), np.zeros(n * k)].reshape(1, -1))]
    upper = [np.zeros(n), np.array([p])]
    if minimum_nominal_population is not None:
        floor = float(minimum_nominal_population)
        if not np.isfinite(floor) or not 0 <= floor <= w.sum() + 1e-7:
            raise ValueError("최소 커버 인구가 유효하지 않습니다.")
        row = np.zeros(m + n * k)
        row[m::k] = -w
        matrices.append(csr_matrix(row.reshape(1, -1)))
        upper.append(np.array([-floor + max(1e-7, floor * 1e-10)]))
    if k > 1:
        diff = np.zeros((k - 1, k))
        for l in range(k - 1):
            diff[l, l] = -1
            diff[l, l + 1] = 1
        matrices.append(hstack([csr_matrix((n * (k - 1), m)), kron(eye(n), diff)]))
        upper.append(np.zeros(n * (k - 1)))
    increments = np.outer(w, (1 - q) * q ** np.arange(k)).ravel()
    r = milp(np.r_[np.zeros(m), -increments], integrality=np.ones(m + n * k),
             bounds=Bounds(np.zeros(m + n * k), np.ones(m + n * k)),
             constraints=LinearConstraint(vstack(matrices, format="csr"), -np.inf, np.concatenate(upper)),
             options={"mip_rel_gap": 0.0, "time_limit": time_limit})
    if not r.success:
        raise RuntimeError(r.message)
    selected = np.flatnonzero(r.x[:m] > .5)
    cnt = np.asarray(a[:, selected].sum(axis=1)).ravel()
    value = float(w @ (1 - q ** cnt))
    return {"selected_indices": selected.tolist(), "expected_population": value,
            "status": int(r.status), "mip_gap": float(r.mip_gap or 0), "q": q}


def evaluate_selection(population, coverage, selected):
    w = np.asarray(population, dtype=float)
    a = csr_matrix(coverage)
    selected = list(selected)
    sub = a[:, selected].toarray()
    counts = sub.sum(axis=1)
    total = float(w.sum())
    covered = float(w[counts > 0].sum())
    after_one_loss = [float(w[(counts - sub[:, j]) > 0].sum())
                      for j in range(len(selected))]
    return {"selected_count": len(selected), "nominal_pct": 100 * covered / total,
            "worst_single_site_unavailable_pct": 100 * min(after_one_loss) / total if after_one_loss else 0,
            "two_or_more_sites_pct": 100 * float(w[counts >= 2].sum()) / total,
            "expected_pct": {str(q): float(100 * w @ (1 - q ** counts) / total) for q in [.1, .2]},
            "note": "q=10%,20%는 동일·독립 미가용 가정의 스트레스 시나리오이며 실측치가 아님"}
