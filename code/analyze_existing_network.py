"""기존 등록 수거망과 편의점 후보망의 공간적 보완 관계를 계산한다.

모든 등록 수거지는 수거 가능, 모든 편의점 후보는 설치 가능하다고 가정한다.
운영시간·협약·보행 경로는 확인하지 않았다. 기존 편의점 전용 결과는 보존한다.
"""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from pyproj import Transformer
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from src.coverage_model import radius_matrix
from src.fair_coverage_model import solve_fair_coverage
from run_spatial_optimization import compress_rows

P = Path(__file__).resolve().parent
R = P / 'results/existing_network'
R.mkdir(parents=True, exist_ok=True)


def main():
    cs = json.loads((P/'data/processed/convenience_candidates_spatial.json').read_text())
    assert all(c['inside_city_boundary'] for c in cs)
    bs = json.loads((P/'data/processed/existing_bins.json').read_text())
    admin = gpd.read_file(P/'data/processed/changwon_spatial.gpkg', layer='admin_dongs').to_crs(5179)
    city = admin.geometry.union_all()
    transformer = Transformer.from_crs(4326, 5179, always_xy=True)
    groups = {}
    for b in bs:
        key = (float(b['lon']), float(b['lat']))
        if not np.isfinite(key).all():
            raise ValueError('수거지 좌표 결측')
        groups.setdefault(key, []).append(b)
    unique = []
    for (lon, lat), records in groups.items():
        x, y = transformer.transform(lon, lat)
        inside = bool(city.covers(Point(x, y)))
        unique.append(dict(lon=lon, lat=lat, x_m=x, y_m=y, inside_city=inside,
                           ids=[b['id'] for b in records], names=[b['name'] for b in records],
                           addresses=[b['address'] for b in records]))
    (R/'existing_location_audit.json').write_text(json.dumps(unique, ensure_ascii=False, indent=2))
    valid = [v for v in unique if v['inside_city']]
    exy = np.array([[b['x_m'], b['y_m']] for b in valid])
    cxy = np.array([[c['x_m'], c['y_m']] for c in cs])
    d = pd.read_csv(P/'data/processed/demand_100m.csv', dtype={'adm_cd': str})
    d[['population','households']] = d[['population','households']].fillna(0)
    d = d.loc[(d.population>0)|(d.households>0)].reset_index(drop=True)
    xy = d[['x_m','y_m']].to_numpy()
    w = d.population.to_numpy(); hh = d.households.to_numpy()
    edist = cKDTree(exy).query(xy)[0]
    cdist_all = cKDTree(cxy).query(xy)[0]
    # 모든 시설까지의 거리를 별도로 계산해 최근접 거리 결과를 검증한다.
    for start in range(0, len(xy), 1200):
        stop = start+1200
        assert np.allclose(edist[start:stop], cdist(xy[start:stop], exy).min(axis=1))
        assert np.allclose(cdist_all[start:stop], cdist(xy[start:stop], cxy).min(axis=1))
    base = edist <= 350
    all_c = cdist_all <= 350
    a = radius_matrix(xy, cxy, 350)

    def metric(mask):
        return dict(population=float(w[mask].sum()), population_pct=float(100*w[mask].sum()/w.sum()),
                    households=float(hh[mask].sum()), households_pct=float(100*hh[mask].sum()/hh.sum()))

    masks = {'both':base & all_c, 'existing_only':base & ~all_c,
             'convenience_only':~base & all_c, 'neither':~base & ~all_c}
    assert np.all(np.stack(list(masks.values())).sum(axis=0)==1)
    out = dict(analysis='existing_network_supplement', radius_m=350,
               assumptions=['registered collection sites all available', 'candidate stores all eligible',
                            'straight distance from original 100m cell centroids', 'operating hours ignored'],
               existing_records=len(bs), existing_unique_coordinates=len(unique), existing_inside_city=len(valid),
               existing_duplicate_coordinate_groups=sum(len(v)>1 for v in groups.values()),
               existing_reference_date='2026-01-28', candidate_locations=len(cs),
               candidate_census_complete=False, total_published_population=float(w.sum()),
               existing_baseline=metric(base), all_candidates=metric(all_c), union_all=metric(base|all_c),
               partition={k:metric(v) for k,v in masks.items()}, scenarios=[], distance_sensitivity=[])
    out['existing_name_derived_types'] = dict(Counter('약국' if '약국' in b['name'] else '보건' if '보건' in b['name'] else '행정' if any(s in b['name'] for s in ['행정','주민센터','동주민','읍사무','면사무']) else '기타' for b in bs))
    for radius in [200,350,500]:
        out['distance_sensitivity'].append(dict(radius_m=radius, existing=metric(edist<=radius),
                                                   union_all=metric((edist<=radius)|(cdist_all<=radius))))
    old = json.loads((P/'results/spatial_optimization.json').read_text())
    by_id = {c['id']:i for i,c in enumerate(cs)}
    chosen_rows=[]
    d['existing_covered']=base; d['all_convenience_covered']=all_c
    d['category']=np.select(list(masks.values()), list(masks), default='invalid')
    # 기존 수거망에서 이미 커버되는 수요의 가중치를 0으로 두어 새 접근 기회만 최대화한다.
    cw, values = compress_rows(a, {'additional_population':w*(~base)})
    for p in [20,40,60]:
        solution = solve_fair_coverage(values['additional_population'], cw, p, time_limit=30, relative_gap=0)
        assert solution['feasible'], solution
        selected=solution['selected_indices']
        addition=np.asarray(a[:,selected].sum(axis=1)).ravel()>0
        # 선정 좌표에 대한 별도 거리 계산과 합집합·증분 목적함수를 검증한다.
        direct=cdist(xy,cxy[selected]).min(axis=1)<=350
        assert np.array_equal(addition,direct)
        new=addition & ~base; union=base|addition
        assert np.isclose(w[new].sum(),solution['weighted_coverage'])
        assert np.isclose(w[union].sum(),w[base].sum()+w[new].sum())
        original=next(s for s in old['scenarios'] if s['id']==f'100m_350m_p{p}_population_all')
        oi=[by_id[cid] for cid in original['selected_ids']]
        orig=np.asarray(a[:,oi].sum(axis=1)).ravel()>0
        standalone=metric(orig)
        assert np.isclose(standalone['population_pct'],original['metrics']['population_coverage_pct'])
        if solution['phases'][0]['proven_optimal']:
            assert w[union].sum()>=w[base|orig].sum()-1e-6
        row=dict(max_new=p, selected_count=len(selected), selected_ids=[cs[j]['id'] for j in selected],
                 total=metric(union), additional=metric(new), existing_gap_closed_pct=float(100*w[new].sum()/w[~base].sum()),
                 phases=solution['phases'], convenience_only_plan=standalone,
                 existing_plus_convenience_only_plan=metric(base|orig),
                 shared_sites_with_convenience_only_plan=len(set(selected)&set(oi)))
        out['scenarios'].append(row)
        d[f'combined_p{p}']=union
        d[f'added_p{p}']=new
        for j in selected:
            c=cs[j]
            chosen_rows.append(dict(max_new=p,candidate_id=c['id'],names=' / '.join(c['names']),
                                   addresses=' / '.join(c['addresses']),district=c['spatial_district'],dong=c['spatial_dong'],
                                   lon=c['lon'],lat=c['lat'],flags=' / '.join(c['flags'])))
        print('new',p,'total',row['total']['population_pct'],'additional_pp',row['additional']['population_pct'],
              'optimal',row['phases'][0]['proven_optimal'],flush=True)
    districts=[]
    for name, group in d.groupby('district'):
        ids=group.index.to_numpy(); total=w[ids].sum()
        rec=dict(district=name,population=float(total),existing_pct=float(100*w[ids][base[ids]].sum()/total))
        for p in [20,40,60]:
            rec[f'combined_p{p}_pct']=float(100*group.loc[group[f'combined_p{p}'],'population'].sum()/total)
        districts.append(rec)
    out['districts']=districts
    out['validation']=['all existing coordinates checked against official city boundary',
                       'KD-tree distances equal chunked exhaustive distances',
                       'four spatial categories partition all demand rows',
                       'each selected network checked by exhaustive distances',
                       'union equals existing plus additional with no double count',
                       'original convenience-only results reproduced']
    (R/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
    d.to_csv(R/'coverage_by_demand.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(chosen_rows).to_csv(R/'selected_additional_stores.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(districts).to_csv(R/'district_coverage.csv',index=False,encoding='utf-8-sig')
    print('baseline',out['existing_baseline'],'partition',out['partition'],flush=True)


if __name__=='__main__':
    main()
