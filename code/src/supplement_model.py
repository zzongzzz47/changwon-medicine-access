"""기존 수거망을 고정한 인구 최대·지역 형평성·단계적 확장 모형."""
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import csr_matrix, eye, hstack, vstack


def solve_supplement(weights, reachable, existing_covered, max_new, *,
                     group_weights=None, maximize_minimum=False, fixed_indices=(),
                     time_limit=40, relative_gap=0):
    w=np.asarray(weights,dtype=float)
    a=csr_matrix(reachable,dtype=float);a.sort_indices()
    b=np.asarray(existing_covered,dtype=bool)
    if w.ndim!=1 or a.shape[0]!=len(w) or b.shape!=w.shape or not len(w):
        raise ValueError('수요와 행렬의 크기가 다릅니다.')
    if not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
        raise ValueError('수요는 합이 양수인 비음수여야 합니다.')
    if a.shape[1]==0 or not np.isfinite(a.data).all() or not np.isin(a.data,[0,1]).all():
        raise ValueError('후보와 0/1 도달 행렬이 필요합니다.')
    if isinstance(max_new,bool) or int(max_new)!=max_new or max_new<0:
        raise ValueError('설치 수 상한을 확인하세요.')
    fixed=sorted(set(fixed_indices))
    if any(not isinstance(k,(int,np.integer)) or not 0<=k<a.shape[1] for k in fixed) or len(fixed)>max_new:
        raise ValueError('고정 점포와 설치 수 상한이 맞지 않습니다.')
    groups={k:np.asarray(v,dtype=float) for k,v in (group_weights or {}).items()}
    for v in groups.values():
        if v.shape!=w.shape or not np.isfinite(v).all() or (v<0).any() or v.sum()<=0:
            raise ValueError('집단 수요를 확인하세요.')
    if maximize_minimum and not groups:
        raise ValueError('형평성 목적에는 집단이 필요합니다.')
    # 같은 후보 집합에 도달하는 수요를 묶되 기존망의 효과는 가중치에서 분리한다.
    mapping={};inverse=np.empty(len(w),int);signatures=[]
    for i in range(len(w)):
        sig=tuple(a.indices[a.indptr[i]:a.indptr[i+1]])
        if sig not in mapping:mapping[sig]=len(signatures);signatures.append(sig)
        inverse[i]=mapping[sig]
    rr=[];cc=[]
    for i,sig in enumerate(signatures):rr.extend([i]*len(sig));cc.extend(sig)
    ca=csr_matrix((np.ones(len(rr)),(rr,cc)),shape=(len(signatures),a.shape[1]))
    n,m=ca.shape
    residual=np.bincount(inverse,weights=w*(~b),minlength=n)
    group_res={k:np.bincount(inverse,weights=v*(~b),minlength=n)/v.sum() for k,v in groups.items()}
    group_base={k:float(v[b].sum()/v.sum()) for k,v in groups.items()}
    size=m+n+int(maximize_minimum)
    matrices=[hstack([-ca,eye(n,format='csr'),csr_matrix((n,int(maximize_minimum)))],format='csr')]
    limits=[np.zeros(n)]
    row=np.zeros(size);row[:m]=1
    matrices.append(csr_matrix(row[None]));limits.append([min(max_new,m)])
    if maximize_minimum:
        for k,v in group_res.items():
            row=np.zeros(size);row[m:m+n]=-v;row[-1]=1
            matrices.append(csr_matrix(row[None]));limits.append([group_base[k]])
    mat=vstack(matrices,format='csr');ub=np.concatenate(limits)
    lower=np.zeros(size);lower[fixed]=1
    bounds=Bounds(lower,np.ones(size));integ=np.r_[np.ones(m),np.zeros(size-m)]
    objective=np.zeros(size)
    if maximize_minimum:objective[-1]=-1
    else:objective[m:m+n]=-residual/w.sum()
    options={'time_limit':time_limit,'mip_rel_gap':relative_gap}
    result=milp(objective,integrality=integ,bounds=bounds,constraints=LinearConstraint(mat,-np.inf,ub),options=options)

    def status(r,phase):
        gap=getattr(r,'mip_gap',None);dual=getattr(r,'mip_dual_bound',None)
        return dict(phase=phase,status=int(r.status),message=r.message,
                    gap=float(gap) if gap is not None else None,
                    objective_upper_bound=float(-dual) if dual is not None else None,
                    proven_optimal=bool(r.status==0 and gap is not None and gap<=1e-8))
    phases=[status(result,'max_min_total_group_coverage' if maximize_minimum else 'max_additional_coverage')]
    if result.x is None:return dict(feasible=False,phases=phases)
    floor=None
    if maximize_minimum:
        first=np.flatnonzero(result.x[:m]>.5)
        mask=b|(np.asarray(a[:,first].sum(axis=1)).ravel()>0)
        floor=min(float(v[mask].sum()/v.sum()) for v in groups.values())
        row=np.zeros(size);row[-1]=-1
        nextmat=vstack([mat,csr_matrix(row[None])],format='csr')
        objective=np.zeros(size);objective[m:m+n]=-residual/w.sum()
        refined=milp(objective,integrality=integ,bounds=bounds,
                     constraints=LinearConstraint(nextmat,-np.inf,np.r_[ub,-floor+1e-8]),options=options)
        phases.append(status(refined,'max_additional_coverage_at_achieved_group_floor'))
        if refined.x is not None:result=refined
    selected=np.flatnonzero(result.x[:m]>.5)
    covered=b|(np.asarray(a[:,selected].sum(axis=1)).ravel()>0)
    rates={k:float(v[covered].sum()/v.sum()) for k,v in groups.items()}
    assert len(selected)<=max_new and set(fixed)<=set(selected)
    if floor is not None:assert min(rates.values())>=floor-1e-7
    return dict(feasible=True,selected_indices=selected.tolist(),phases=phases,
                achieved_group_floor=floor,group_rates=rates,
                total_weight=float(w[covered].sum()),additional_weight=float(w[covered&~b].sum()),
                compressed_rows=n)
