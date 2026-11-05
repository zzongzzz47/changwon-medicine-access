"""Compare fixed and reoptimized placements across explicit distance scenarios."""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from src.supplement_model import solve_supplement

P=Path(__file__).resolve().parent
R=P/'results/accessibility_extension'
RADII=[200,250,300,350,400,450,500]

def main():
    R.mkdir(parents=True,exist_ok=True)
    paths=[P/'data/processed/demand_100m.csv',P/'data/processed/convenience_candidates_spatial.json',P/'results/existing_network/existing_location_audit.json',P/'results/supplement_development/development.json']
    d=pd.read_csv(paths[0],dtype={'adm_cd':str})
    d[['population','households']]=d[['population','households']].fillna(0)
    d=d.loc[(d.population>0)|(d.households>0)].reset_index(drop=True)
    cs=[c for c in json.loads(paths[1].read_text()) if '시설내점포_출입확인필요' not in c['flags'] and not any('더프레시' in n or '노브랜드' in n for n in c['names'])]
    assert len(cs)==802
    bins=[b for b in json.loads(paths[2].read_text()) if b['inside_city']]
    old=json.loads(paths[3].read_text()); old={s['id']:s for s in old['scenarios']}
    xy=d[['x_m','y_m']].to_numpy(); w=d.population.to_numpy(); hh=d.households.to_numpy()
    cm=cdist(xy,np.array([[c['x_m'],c['y_m']] for c in cs]))
    em=cdist(xy,np.array([[b['x_m'],b['y_m']] for b in bins])).min(axis=1)
    groups={g:d.district.to_numpy()==g for g in sorted(d.district.unique())}
    idx={c['id']:i for i,c in enumerate(cs)}
    out={'radii_m':RADII,'max_new':60,'candidate_count':len(cs),'existing_coordinates':len(bins),'population':float(w.sum()),'input_sha256':{str(p.relative_to(P)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'plans':[], 'validation':[]}
    def metric(mask):
        rates={g:float(100*w[mask&v].sum()/w[v].sum()) for g,v in groups.items()}
        return {'population_pct':float(100*w[mask].sum()/w.sum()),'population':float(w[mask].sum()),'households_pct':float(100*hh[mask].sum()/hh.sum()),'minimum_district_pct':min(rates.values()),'districts':rates}
    def save(): (R/'radius_review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    baselines={str(r):metric(em<=r) for r in RADII};out['baselines']=baselines
    for radius in RADII:
        t=time.monotonic()
        oldkey={200:'screened_distance_200_60',350:'screened_efficiency_60',500:'screened_distance_500_60'}.get(radius)
        if oldkey:
            previous=old[oldkey]; chosen=[idx[c] for c in previous['selected_ids']]; phases=previous['phases']
            assert np.isclose(metric((em<=radius)|(cm[:,chosen].min(axis=1)<=radius))['population_pct'],previous['metrics']['population_pct'])
            origin='previous_exact_solution_recomputed'
        else:
            result=solve_supplement(w,csr_matrix(cm<=radius),em<=radius,60,time_limit=50,relative_gap=0)
            assert result['feasible'];chosen=result['selected_indices'];phases=result['phases'];origin='new_optimization'
        nearest=np.minimum(em,cm[:,chosen].min(axis=1))
        evaluations={str(r):metric(nearest<=r) for r in RADII}
        plan={'id':f'efficiency_{radius}','radius_m':radius,'objective':'population','selected_ids':[cs[i]['id'] for i in chosen],'evaluations':evaluations,'phases':phases,'origin':origin,'elapsed_seconds':time.monotonic()-t}
        assert len(chosen)<=60 and len(set(chosen))==len(chosen)
        assert all(evaluations[str(r)]['population_pct']>=baselines[str(r)]['population_pct']-1e-8 for r in RADII)
        out['plans'].append(plan);save()
        print(plan['id'],evaluations[str(radius)]['population_pct'],phases,flush=True)
    fair=old['screened_fair_60']; chosen=[idx[c] for c in fair['selected_ids']]
    nearest=np.minimum(em,cm[:,chosen].min(axis=1))
    out['fair350_fixed']={'selected_ids':fair['selected_ids'],'evaluations':{str(r):metric(nearest<=r) for r in RADII}}
    assert np.isclose(out['fair350_fixed']['evaluations']['350']['population_pct'],fair['metrics']['population_pct'])
    ef350=next(p for p in out['plans'] if p['radius_m']==350)
    selected350=set(ef350['selected_ids']); union=set().union(*(set(p['selected_ids']) for p in out['plans']))
    intersection=set.intersection(*(set(p['selected_ids']) for p in out['plans']))
    out['selection_stability']={'union_count':len(union),'intersection_count':len(intersection),'intersection_ids':sorted(intersection)}
    rows=[];cross=[]
    for plan in out['plans']:
        r=plan['radius_m'];own=plan['evaluations'][str(r)];fixed=ef350['evaluations'][str(r)]
        row={'radius_m':r,'baseline_pct':baselines[str(r)]['population_pct'],'reoptimized_pct':own['population_pct'],'fixed_eff350_pct':fixed['population_pct'],'fixed_fair350_pct':out['fair350_fixed']['evaluations'][str(r)]['population_pct'],'fixed_eff350_regret_pp':own['population_pct']-fixed['population_pct'],'added_population':own['population']-baselines[str(r)]['population'],'shared_with_eff350':len(set(plan['selected_ids'])&selected350),'optimal':all(p['proven_optimal'] for p in plan['phases'])}
        rows.append(row)
        for er in RADII:cross.append({'selection_radius_m':r,'evaluation_radius_m':er,'population_pct':plan['evaluations'][str(er)]['population_pct']})
        vals=[plan['evaluations'][str(er)]['population_pct'] for er in RADII]
        assert all(b>=a-1e-9 for a,b in zip(vals,vals[1:]))
    frequency=[]
    for c in cs:
        selected_r=[p['radius_m'] for p in out['plans'] if c['id'] in p['selected_ids']]
        if selected_r:frequency.append({'candidate_id':c['id'],'name':' / '.join(c['names']),'district':c['spatial_district'],'dong':c['spatial_dong'],'lon':c['lon'],'lat':c['lat'],'selected_count':len(selected_r),'radii_m':','.join(map(str,selected_r))})
    frequency.sort(key=lambda r:(-r['selected_count'],r['district'],r['name']))
    out['summary']=rows
    out['validation']=['previous 200/350/500m solutions independently reproduced','802 candidate scope verified','fixed-plan coverage monotone across radii','existing coverage never reduced','350m fairness coverage reproduced']
    out['interpretation']='Selected frequency is descriptive across one solution per radius; alternative optima are not enumerated. Regret at each radius compares plans at the same evaluation radius. Straight-line distance is not walking distance. Missing grid values excluded per original population denominator.'
    pd.DataFrame(rows).to_csv(R/'radius_summary.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(cross).to_csv(R/'radius_cross_evaluation.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(frequency).to_csv(R/'candidate_radius_frequency.csv',index=False,encoding='utf-8-sig')
    # A descriptive, area-level evaluation; these are not individual elderly coverage rates.
    dg=pd.read_csv(P/'results/supplement_development/dong_gap_diagnostics.csv')
    assert dg.age65plus_share_pct.notna().all()
    cut=float(dg.age65plus_share_pct.quantile(.75));top=dg.age65plus_share_pct>=cut
    out['older_area_diagnostics']={'upper_quartile_threshold_pct':cut,'dong_count':int(top.sum()),'district_dongs':dg.loc[top,['district','dong']].to_dict('records'),'denominator':'all publicly observed residents in selected dongs; not elderly persons','rates':{k:float(np.average(dg.loc[top,k],weights=dg.loc[top,'population'])) for k in ['baseline_pct','screened_efficiency_60_pct','screened_fair_60_pct']}}
    save();print('finished',out['selection_stability'],out['older_area_diagnostics'],flush=True)

if __name__=='__main__':main()
