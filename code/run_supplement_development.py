"""기존 수거망의 공백 진단, 형평성, 단계 도입, 후보·거리 민감도, 점포 대체 분석."""
from pathlib import Path
import json
import time
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from shapely.geometry import Point
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from src.coverage_model import radius_matrix
from src.supplement_model import solve_supplement

P=Path(__file__).resolve().parent;R=P/'results/supplement_development';R.mkdir(parents=True,exist_ok=True)


def main():
    cached={}
    if '--reuse-results' in sys.argv and (R/'development.json').exists():
        cached={s['id']:s for s in json.loads((R/'development.json').read_text())['scenarios']}
    candidates=json.loads((P/'data/processed/convenience_candidates_spatial.json').read_text())
    bins=json.loads((P/'results/existing_network/existing_location_audit.json').read_text())
    exy=np.array([[b['x_m'],b['y_m']] for b in bins if b['inside_city']])
    d=pd.read_csv(P/'data/processed/demand_100m.csv',dtype={'adm_cd':str})
    d[['population','households']]=d[['population','households']].fillna(0)
    d=d.loc[(d.population>0)|(d.households>0)].reset_index(drop=True)
    xy=d[['x_m','y_m']].to_numpy();w=d.population.to_numpy();hh=d.households.to_numpy()
    edist=cKDTree(exy).query(xy)[0]
    groups={g:np.where(d.district==g,w,0) for g in sorted(d.district.unique())}
    admin=gpd.read_file(P/'data/processed/changwon_spatial.gpkg',layer='admin_dongs').to_crs(5179)
    transformer=Transformer.from_crs(4326,5179,always_xy=True)
    # 다른 업종의 브랜드명 기록은 확정 후보와 구분한 민감도에서만 사용한다.
    review=pd.read_csv(P/'results/candidate_category_review.csv')
    existing_coords={(c['lon'],c['lat']) for c in candidates}
    extras=[]
    for _,r in review.iterrows():
        lon,lat=float(r['경도']),float(r['위도'])
        if (lon,lat) in existing_coords:continue
        x,y=transformer.transform(lon,lat);pt=Point(x,y)
        areas=admin.loc[admin.geometry.covers(pt)]
        assert len(areas)==1
        area=areas.iloc[0]
        extras.append(dict(id='REVIEW_'+r['상가업소번호'],lon=lon,lat=lat,x_m=x,y_m=y,
                           names=[str(r['상호명'])+((' '+str(r['지점명'])) if pd.notna(r['지점명']) else '')],
                           addresses=[str(r['도로명주소'])],spatial_district=area.district,spatial_dong=area.dong,
                           flags=['다른업종_브랜드명후보_영업출입미검증']))
        existing_coords.add((lon,lat))
    assert len(extras)==9
    catalogs={'all':candidates,'access_review_excluded':[c for c in candidates if '시설내점포_출입확인필요' not in c['flags']],
              'review_added':candidates+extras}
    name_review=[c for c in candidates if any('더프레시' in n or '노브랜드' in n for n in c['names'])]
    name_review_ids={c['id'] for c in name_review}
    catalogs['screened']=[c for c in catalogs['access_review_excluded'] if c['id'] not in name_review_ids]
    assert len(name_review_ids)==2 and len(catalogs['screened'])==802
    meta={c['id']:c for c in candidates+extras}
    (R/'candidate_metadata.json').write_text(json.dumps(list(meta.values()),ensure_ascii=False,indent=2))
    matrices={}
    def get(radius=350,scope='all'):
        key=(radius,scope)
        if key not in matrices:
            cs=catalogs[scope];cxy=np.array([[c['x_m'],c['y_m']] for c in cs])
            matrices[key]=(cs,radius_matrix(xy,cxy,radius),edist<=radius)
        return matrices[key]

    def metrics(mask,base):
        districts={g:100*float(v[mask].sum()/v.sum()) for g,v in groups.items()}
        return dict(population_pct=100*float(w[mask].sum()/w.sum()),household_pct=100*float(hh[mask].sum()/hh.sum()),
                    population=float(w[mask].sum()),added_population=float(w[mask&~base].sum()),
                    added_pp=100*float(w[mask&~base].sum()/w.sum()),districts=districts,
                    minimum_district_pct=min(districts.values()),district_range_pp=max(districts.values())-min(districts.values()))

    cs,a,base=get();all_cover=base|(np.asarray(a.sum(axis=1)).ravel()>0)
    out=dict(problem='existing_network_gap_and_convenience_supplement',radius_m=350,
             existing_coordinates=len(exy),candidate_counts={k:len(v) for k,v in catalogs.items()},
             total_population=float(w.sum()),total_households=float(hh.sum()),baseline=metrics(base,base),
             candidate_union_ceiling=metrics(all_cover,base),scenarios=[],risk={},stability={},validation=[])
    out['name_review_excluded']=[dict(id=c['id'],names=c['names']) for c in name_review]
    out['primary_candidate_scope']='screened'
    out['screened_scope_rule']='시설내점포 출입확인 필요 59개 좌표와 명칭상 편의점 여부 확인 필요 2개 좌표를 제외한 조건; 남은 802개도 영업·출입·참여 확인 전'
    selected_masks={};rows=[]
    def save():
        (R/'development.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    def run(sid,p=60,radius=350,scope='all',objective='population',fixed_ids=()):
        start=time.monotonic();local_cs,ma,ba=get(radius,scope)
        idx={c['id']:i for i,c in enumerate(local_cs)}
        prev=cached.get(sid)
        if prev is not None:
            assert (prev['max_new'],prev['radius_m'],prev['candidate_scope'],prev['objective'],prev['fixed_ids'])==(p,radius,scope,objective,list(fixed_ids))
            chosen=[idx[cid] for cid in prev['selected_ids']]
            mask=ba|(np.asarray(ma[:,chosen].sum(axis=1)).ravel()>0)
            score=hh if objective=='households' else w
            solution=dict(feasible=True,selected_indices=chosen,phases=prev['phases'],achieved_group_floor=prev['achieved_group_floor'],
                          total_weight=float(score[mask].sum()),additional_weight=float(score[mask&~ba].sum()))
        else:
            solution=solve_supplement(hh if objective=='households' else w,ma,ba,p,
                        group_weights=groups,maximize_minimum=objective=='fair',
                        fixed_indices=[idx[k] for k in fixed_ids],time_limit=40,relative_gap=0)
        if not solution['feasible']:raise RuntimeError(solution)
        selected=solution['selected_indices'];ids=[local_cs[k]['id'] for k in selected]
        added=np.asarray(ma[:,selected].sum(axis=1)).ravel()>0
        mask=ba|added
        cxy=np.array([[local_cs[k]['x_m'],local_cs[k]['y_m']] for k in selected])
        assert np.array_equal(added,cdist(xy,cxy).min(axis=1)<=radius)
        score=hh if objective=='households' else w
        assert np.isclose(score[mask].sum(),solution['total_weight'])
        assert np.isclose(score[mask&~ba].sum(),solution['additional_weight'])
        result=dict(id=sid,max_new=p,radius_m=radius,candidate_scope=scope,objective=objective,
                    selected_count=len(ids),selected_ids=ids,fixed_ids=list(fixed_ids),metrics=metrics(mask,ba),
                    phases=solution['phases'],achieved_group_floor=solution['achieved_group_floor'],
                    elapsed_seconds=round(time.monotonic()-start,3))
        # 모든 안을 공통의 350m 기준으로도 재평가한다.
        common=(edist<=350)|(cdist(xy,cxy).min(axis=1)<=350)
        result['evaluation_350m']=metrics(common,base)
        selected_masks[sid]=common
        out['scenarios'].append(result)
        for rank,cid in enumerate(ids):
            c=meta[cid]
            rows.append(dict(scenario=sid,candidate_id=cid,names=' / '.join(c['names']),addresses=' / '.join(c['addresses']),
                             district=c['spatial_district'],dong=c['spatial_dong'],lon=c['lon'],lat=c['lat'],flags=' / '.join(c['flags'])))
        save();print(sid,round(result['metrics']['population_pct'],4),round(result['metrics']['minimum_district_pct'],4),solution['phases'],flush=True)
        return result

    efficiency={p:run(f'efficiency_{p}',p=p) for p in [20,40,60,80,100]}
    fair=run('fair_60',objective='fair')
    household=run('households_60',objective='households')
    run('distance_200_60',radius=200);run('distance_500_60',radius=500)
    run('access_review_excluded_60',scope='access_review_excluded')
    run('review_added_60',scope='review_added')
    stage40=run('staged_40',p=40,fixed_ids=efficiency[20]['selected_ids'])
    stage60=run('staged_60',p=60,fixed_ids=stage40['selected_ids'])
    screened_eff={p:run(f'screened_efficiency_{p}',p=p,scope='screened') for p in [20,40,60,80,100]}
    screened_fair=run('screened_fair_60',scope='screened',objective='fair')
    run('screened_households_60',scope='screened',objective='households')
    run('screened_distance_200_60',scope='screened',radius=200)
    run('screened_distance_500_60',scope='screened',radius=500)
    screened_stage40=run('screened_staged_40',p=40,scope='screened',fixed_ids=screened_eff[20]['selected_ids'])
    screened_stage60=run('screened_staged_60',p=60,scope='screened',fixed_ids=screened_stage40['selected_ids'])
    screened_cs,screened_a,_=get(350,'screened')
    screened_all=base|(np.asarray(screened_a.sum(axis=1)).ravel()>0)
    out['screened_union_ceiling']=metrics(screened_all,base)
    old=json.loads((P/'results/existing_network/comparison.json').read_text())
    for s in old['scenarios']:
        assert np.isclose(efficiency[s['max_new']]['metrics']['population_pct'],s['total']['population_pct'])
    for s in out['scenarios']:
        if s['max_new']==60:
            out['stability'][s['id']]=dict(shared_with_efficiency60=len(set(s['selected_ids'])&set(efficiency[60]['selected_ids'])),
                                           shared_with_fair60=len(set(s['selected_ids'])&set(fair['selected_ids'])),
                                           shared_with_screened_efficiency60=len(set(s['selected_ids'])&set(screened_eff[60]['selected_ids'])),
                                           shared_with_screened_fair60=len(set(s['selected_ids'])&set(screened_fair['selected_ids'])))
    out['review_added_selected_ids']=[cid for s in out['scenarios'] if s['id']=='review_added_60' for cid in s['selected_ids'] if cid.startswith('REVIEW_')]
    # 동별 공간 공백을 보완 가능 공백과 후보망 밖 공백으로 분리한다.
    dongrows=[]
    for (district,dong),g in d.groupby(['district','dong']):
        ids=g.index.to_numpy();local=w[ids];den=local.sum()
        row=dict(district=district,dong=dong,population=float(den),baseline_pct=float(100*local[base[ids]].sum()/den),
                 existing_gap_population=float(local[~base[ids]].sum()),
                 fillable_gap_population=float(local[(~base&all_cover)[ids]].sum()),
                 outside_candidate_network_population=float(local[(~all_cover)[ids]].sum()))
        row['screened_fillable_gap_population']=float(local[(~base&screened_all)[ids]].sum())
        row['screened_outside_candidate_network_population']=float(local[(~screened_all)[ids]].sum())
        for sid in ['efficiency_60','fair_60','screened_efficiency_60','screened_fair_60']:
            mask=selected_masks[sid][ids]
            row[sid+'_pct']=float(100*local[mask].sum()/den)
            row[sid+'_added_population']=float(local[mask&~base[ids]].sum())
            row[sid+'_remaining_population']=float(local[~mask].sum())
        assert np.isclose(row['existing_gap_population'],row['fillable_gap_population']+row['outside_candidate_network_population'])
        dongrows.append(row)
    dg=pd.DataFrame(dongrows)
    age=pd.read_csv(P/'results/dong_demographics.csv')[['district','dong','age65plus_share_pct']]
    dg=dg.merge(age,on=['district','dong'],how='left',validate='one_to_one')
    dg.to_csv(R/'dong_gap_diagnostics.csv',index=False,encoding='utf-8-sig')
    # 단일 점포 미참여: 나머지는 그대로 유지하고 하나의 대체 후보를 찾는다.
    index={c['id']:i for i,c in enumerate(cs)}
    riskrows=[];optionrows=[]
    for plan in [efficiency[60],fair,screened_eff[60],screened_fair]:
        chosen=[index[cid] for cid in plan['selected_ids']]
        counts=np.asarray(a[:,chosen].sum(axis=1)).ravel()
        initial=base|(counts>0);orig_min=plan['metrics']['minimum_district_pct']
        losses=[];restored_rates=[];floor_kept=[]
        standalone=np.asarray(a.T@(w*~base)).ravel()
        for removed in chosen:
            site_cover=np.asarray(a[:,[removed]].sum(axis=1)).ravel()>0
            after=base|((counts-site_cover)>0)
            loss=float(w[initial&~after].sum())
            gain=np.asarray(a.T@(w*~after)).ravel()
            district_rates=[]
            for name,v in groups.items():
                district_rates.append(100*(v[after].sum()+np.asarray(a.T@(v*~after)).ravel())/v.sum())
            mins=np.min(district_rates,axis=0)
            allowed=np.array([c['id'] in {k['id'] for k in catalogs[plan['candidate_scope']]} for c in cs]);allowed[chosen]=False
            preserve=allowed&(mins>=orig_min-1e-6)
            use=preserve if plan['objective']=='fair' and preserve.any() else allowed
            ordered=np.argsort(-np.where(use,gain,-np.inf),kind='stable')[:3]
            best=int(ordered[0]);restored=after|(np.asarray(a[:,[best]].sum(axis=1)).ravel()>0)
            restored_pct=100*float(w[restored].sum()/w.sum())
            if plan['objective']=='population':assert w[restored].sum()<=w[initial].sum()+1e-5
            remain=[k for k in chosen if k!=removed]+[best]
            direct=base|(cdist(xy,np.array([[cs[k]['x_m'],cs[k]['y_m']] for k in remain])).min(axis=1)<=350)
            assert np.array_equal(restored,direct)
            c=cs[removed];bc=cs[best]
            row=dict(plan=plan['id'],candidate_id=c['id'],name=' / '.join(c['names']),district=c['spatial_district'],dong=c['spatial_dong'],
                     standalone_added_population=float(standalone[removed]),exclusive_population=loss,
                     loss_pp=100*loss/w.sum(),coverage_after_failure_pct=100*float(w[after].sum()/w.sum()),
                     best_replacement_id=bc['id'],best_replacement_name=' / '.join(bc['names']),
                     replacement_gain_population=float(gain[best]),coverage_after_replacement_pct=restored_pct,
                     replacement_min_district_pct=float(mins[best]),fair_floor_preserved=bool(mins[best]>=orig_min-1e-6),
                     replacement_flags=' / '.join(bc['flags']),flags=' / '.join(c['flags']))
            riskrows.append(row);losses.append(loss);restored_rates.append(restored_pct);floor_kept.append(row['fair_floor_preserved'])
            for position,k in enumerate(ordered,1):
                optionrows.append(dict(plan=plan['id'],removed_id=c['id'],removed_name=row['name'],alternative_rank=position,
                    replacement_id=cs[k]['id'],replacement_name=' / '.join(cs[k]['names']),district=cs[k]['spatial_district'],dong=cs[k]['spatial_dong'],
                    new_population_after_failure=float(gain[k]),total_population_pct=100*float((w[after].sum()+gain[k])/w.sum()),
                    minimum_district_pct=float(mins[k]),preserves_original_minimum=bool(mins[k]>=orig_min-1e-6),flags=' / '.join(cs[k]['flags'])))
        out['risk'][plan['id']]=dict(worst_single_failure_population=max(losses),worst_single_failure_pp=100*max(losses)/w.sum(),
                                   minimum_coverage_after_failure_pct=plan['metrics']['population_pct']-100*max(losses)/w.sum(),
                                   minimum_coverage_after_one_replacement_pct=min(restored_rates),
                                   district_floor_preserved_cases=sum(floor_kept),cases=len(chosen))
        print('risk',plan['id'],out['risk'][plan['id']],flush=True)
    pd.DataFrame(riskrows).to_csv(R/'store_contributions_and_replacements.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(optionrows).to_csv(R/'replacement_options.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(rows).to_csv(R/'selected_stores_all_scenarios.csv',index=False,encoding='utf-8-sig')
    # 격자 내 위치 불확실성: 반대각선 길이만큼 판정 반경을 바꾼다.
    fixed_idx=[index[cid] for cid in screened_fair['selected_ids']]
    selected_dist=cdist(xy,np.array([[cs[k]['x_m'],cs[k]['y_m']] for k in fixed_idx])).min(axis=1)
    out['grid_position_sensitivity_screened_fair60']=[]
    for radius in [350-100/np.sqrt(2),350+100/np.sqrt(2)]:
        mask=(edist<=radius)|(selected_dist<=radius)
        out['grid_position_sensitivity_screened_fair60'].append(dict(radius_m=radius,metrics=metrics(mask,edist<=radius)))
    columns=d[['demand_id','grid_id','district','dong','population','households','x_m','y_m']].copy()
    columns['existing_covered']=base;columns['candidate_union_covered']=all_cover
    columns['screened_union_covered']=screened_all
    for sid in ['efficiency_60','fair_60','staged_60','screened_efficiency_60','screened_fair_60','screened_staged_60']:columns[sid+'_covered']=selected_masks[sid]
    columns.to_csv(R/'coverage_by_demand.csv',index=False,encoding='utf-8-sig')
    out['validation']=['new model validated against exhaustive enumeration on small cases',
                       'existing 20/40/60 population optima reproduced',
                       'every scenario checked using independent dense distance calculations',
                       'existing coverage offsets included in total district fairness',
                       'all 240 one-store replacement cases independently checked',
                       'existing gap equals candidate-fillable plus outside-network gap in all 55 dongs',
                       'staged plan fixed-site constraints verified by solver']
    save()
    print('COMPLETE',len(out['scenarios']),'scenarios',len(riskrows),'failure cases',len(optionrows),'replacement options',flush=True)


if __name__=='__main__':main()
