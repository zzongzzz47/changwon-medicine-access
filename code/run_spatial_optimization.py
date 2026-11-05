"""실제 창원 자료로 편의점 전용 잠재 접근성을 최적화한다."""
from pathlib import Path
import json
import time
import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.coverage_model import radius_matrix
from src.fair_coverage_model import solve_fair_coverage

ROOT=Path(__file__).resolve().parent

def compress_rows(a, values):
    """같은 편의점들로 커버되는 수요행을 합친다. 분모와 집단별 가중치는 보존한다."""
    a=a.tocsr(); a.sort_indices()
    groups={}; inverse=np.empty(a.shape[0],dtype=int); signatures=[]
    for i in range(a.shape[0]):
        ids=tuple(a.indices[a.indptr[i]:a.indptr[i+1]])
        if ids not in groups:
            groups[ids]=len(groups); signatures.append(ids)
        inverse[i]=groups[ids]
    rows=[];cols=[]
    for i,ids in enumerate(signatures): rows.extend([i]*len(ids));cols.extend(ids)
    compressed=csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(len(signatures),a.shape[1]))
    sums={k:np.bincount(inverse,weights=v,minlength=len(signatures)) for k,v in values.items()}
    for k,v in values.items(): assert np.isclose(np.sum(v),sums[k].sum())
    return compressed,sums

def greedy(a,w,p):
    covered=np.zeros(len(w),dtype=bool); chosen=[]
    for _ in range(p):
        gain=np.asarray(a.T @ (w*~covered)).ravel()
        gain[chosen]=-1
        j=int(gain.argmax())
        if gain[j]<=0:break
        chosen.append(j); covered |= np.asarray(a[:,[j]].sum(axis=1)).ravel()>0
    return chosen

def evaluate(d,a,selected):
    covered=np.asarray(a[:,selected].sum(axis=1)).ravel()>0
    result={}
    for col in ['population','households']:
        w=d[col].to_numpy()
        result[col+'_total']=float(w.sum())
        result[col+'_covered']=float(w[covered].sum())
        result[col+'_coverage_pct']=float(100*w[covered].sum()/w.sum())
    result['districts']=[]
    for name,g in d.groupby('district'):
        idx=g.index.to_numpy(); w=g.population.to_numpy(); local=covered[idx]
        result['districts'].append({'district':name,'population':float(w.sum()),'covered_population':float(w[local].sum()),'coverage_pct':float(100*w[local].sum()/w.sum())})
    result['minimum_district_pct']=min(g['coverage_pct'] for g in result['districts'])
    result['district_range_pp']=max(g['coverage_pct'] for g in result['districts'])-result['minimum_district_pct']
    return result

def main(time_limit):
    processed=ROOT/'data/processed';out=ROOT/'results'
    candidates=json.loads((processed/'convenience_candidates_spatial.json').read_text())
    candidates=[c for c in candidates if c['inside_city_boundary']]
    datasets={}
    for resolution in ['100m','500m']:
        d=pd.read_csv(processed/f'demand_{resolution}.csv',dtype={'adm_cd':str})
        # 결측은 실제 0명이 아니다. 공개값이 있는 부분만 각 지표의 분모에 포함한다.
        # 원본 CSV의 결측과 관측 여부 표시는 그대로 보존한다.
        d['population_missing'] = d.population.isna()
        d['households_missing'] = d.households.isna()
        d[['population','households']] = d[['population','households']].fillna(0)
        d=d.loc[(d.population>0)|(d.households>0)].reset_index(drop=True)
        datasets[resolution]=d
    results={'mode':'convenience_only','distance':'EPSG:5179 straight line from original grid centroid',
             'candidate_count':len(candidates),'main_resolution':'100m','baseline_all_candidates':[],
             'scenarios':[],'settings':{'time_limit_per_phase_seconds':time_limit,'relative_gap_target':.001}}
    cache={}
    def get_matrix(resolution,radius,eligibility):
        key=(resolution,radius,eligibility)
        if key not in cache:
            cs=[c for c in candidates if eligibility!='restricted' or '시설내점포_출입확인필요' not in c['flags']]
            d=datasets[resolution]
            a=radius_matrix(d[['x_m','y_m']].to_numpy(),np.array([[c['x_m'],c['y_m']] for c in cs]),radius)
            values={k:d[k].to_numpy() for k in ['population','households']}
            values.update({'district:'+g:np.where(d.district==g,d.population,0) for g in sorted(d.district.unique())})
            comp,sums=compress_rows(a,values)
            cache[key]=(cs,d,a,comp,sums)
        return cache[key]
    def save():
        (out/'spatial_optimization.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    for radius in [200,350,500]:
        cs,d,a,_,_=get_matrix('100m',radius,'all')
        v=evaluate(d,a,list(range(len(cs))))
        results['baseline_all_candidates'].append({'radius_m':radius,**v})
    save()
    scenarios=[('100m',350,p,'population','all') for p in [20,40,60,80,100,150]]
    scenarios += [('100m',350,60,'fair','all'),('100m',350,100,'fair','all'),
                  ('100m',350,60,'households','all'),('100m',200,60,'population','all'),
                  ('100m',500,60,'population','all'),('100m',350,60,'population','restricted'),
                  ('500m',350,60,'population','all')]
    for resolution,radius,p,objective,eligibility in scenarios:
        cs,d,a,comp,sums=get_matrix(resolution,radius,eligibility)
        groups={k.split(':',1)[1]:v for k,v in sums.items() if k.startswith('district:')}
        start=time.monotonic()
        solution=solve_fair_coverage(sums['households' if objective=='households' else 'population'],comp,p,
                    group_weights=groups,maximize_minimum=objective=='fair',time_limit=time_limit,relative_gap=.001)
        if not solution['feasible']:raise RuntimeError(f'실행 가능한 해를 얻지 못했습니다: {solution}')
        chosen=solution['selected_indices']
        assert len(chosen)<=p and len(set(chosen))==len(chosen)
        metric=evaluate(d,a,chosen)
        score_col='households' if objective=='households' else 'population'
        assert np.isclose(metric[score_col+'_covered'],solution['weighted_coverage'])
        chosen_ids=[cs[i]['id'] for i in chosen]
        row={'id':f'{resolution}_{radius}m_p{p}_{objective}_{eligibility}','resolution':resolution,
             'radius_m':radius,'max_sites':p,'objective':objective,'eligibility':eligibility,
             'candidate_count':len(cs),'demand_rows':len(d),'compressed_demand_rows':comp.shape[0],
             'selected_ids':chosen_ids,'selected_count':len(chosen),'metrics':metric,'solver_phases':solution['phases'],
             'elapsed_seconds':round(time.monotonic()-start,3)}
        if objective=='population':
            greedy_chosen=greedy(a,d.population.to_numpy(),p)
            row['greedy_population_coverage_pct']=evaluate(d,a,greedy_chosen)['population_coverage_pct']
        # 다른 해상도에서 고른 점포도 같은 100m 수요로 평가한다.
        base_cs,base_d,base_a,_,_=get_matrix('100m',radius,'all')
        idx={c['id']:i for i,c in enumerate(base_cs)}
        row['evaluation_on_100m']=evaluate(base_d,base_a,[idx[k] for k in chosen_ids])
        results['scenarios'].append(row);save()
        print(row['id'],f"coverage={metric['population_coverage_pct']:.3f}% min_district={metric['minimum_district_pct']:.3f}%",solution['phases'],flush=True)
    # 주 분석의 60곳을 고정하고 격자 내 위치 오차의 영향을 거리 상하한으로 평가한다.
    primary=next(s for s in results['scenarios'] if s['id']=='100m_350m_p60_population_all')
    cs,d,_,_,_=get_matrix('100m',350,'all');idx={c['id']:i for i,c in enumerate(cs)}
    selected=[idx[k] for k in primary['selected_ids']];xy=np.array([[c['x_m'],c['y_m']] for c in cs])
    bound=[]
    for radius in [350-100/np.sqrt(2),350+100/np.sqrt(2)]:
        a=radius_matrix(d[['x_m','y_m']].to_numpy(),xy,radius)
        bound.append({'evaluation_radius_m':radius,**evaluate(d,a,selected)})
    results['cell_position_bounds_fixed_p60']=bound
    # 각 설치 수의 해는 독립된 집합이며 순위나 순차 도입 계획이 아니다.
    rows=[]
    for s in results['scenarios']:
        for cid in s['selected_ids']:
            c=next(c for c in candidates if c['id']==cid)
            rows.append({'scenario':s['id'],'candidate_id':cid,'names':' / '.join(c['names']),
                         'address':' / '.join(c['addresses']),'district':c['spatial_district'],'dong':c['spatial_dong'],
                         'lon':c['lon'],'lat':c['lat'],'flags':' / '.join(c['flags'])})
    pd.DataFrame(rows).to_csv(out/'selected_candidates_by_scenario.csv',index=False,encoding='utf-8-sig')
    save()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--time-limit',type=float,default=30)
    main(parser.parse_args().time_limit)
