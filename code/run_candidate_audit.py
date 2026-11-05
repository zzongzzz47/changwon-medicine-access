from pathlib import Path
import json, csv, collections, hashlib, sys, re
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from src.coverage_model import project_points
P=Path(__file__).resolve().parent
raw=json.loads((P/'data/raw/changwon_convenience_20260630.json').read_text())
shops=json.loads((P/'data/raw/changwon_health_businesses_20260630.json').read_text())
age={'by_dong':{},'by_district':{}}
with (P/'data/raw/changwon_age_20250531.csv').open(encoding='utf-8-sig',newline='') as f:
    for r in csv.DictReader(f):
        area=r['행정기관']; a=int(re.search(r'\d+',r['연령(만)']).group()); n=int(r['합계'])
        assert n==int(r['남'])+int(r['여'])
        for target,key in [(age['by_dong'],area),(age['by_district'],area.split()[0])]:
            z=target.setdefault(key,dict(population=0,age65plus=0,age75plus=0,age0to14=0))
            z['population']+=n;z['age65plus']+=n if a>=65 else 0;z['age75plus']+=n if a>=75 else 0;z['age0to14']+=n if a<=14 else 0
age['by_dong']['마산합포구 반월중앙동']=age['by_dong'].pop('마산합포구 반월중앙')
clusters=collections.defaultdict(list)
for r in raw: clusters[(float(r['경도']),float(r['위도']))].append(r)
locations=[]
for (lon,lat),rows in sorted(clusters.items()):
    names=sorted(set((r['상호명']+' '+r['지점명']).strip() for r in rows))
    addresses=sorted(set(r['도로명주소'] for r in rows))
    districts=sorted(set(r['시군구명'].replace('창원시 ','') for r in rows)); dongs=sorted(set(r['행정동명'] for r in rows))
    text=' '.join(names+addresses+[r['건물명'] for r in rows])
    flags=[]
    if len(rows)>1: flags.append('동일좌표_복수업소기록')
    if re.search(r'대학교|대학로|경남대|창원대|해군|군부대|병원|의료원|NC파크|엔씨파크|군항|기지사령|부대내',text,re.I): flags.append('시설내점포_출입확인필요')
    if len(districts)>1 or len(dongs)>1: flags.append('동일좌표_행정구역불일치')
    if any(r['표준산업분류코드']!='G47122' for r in rows): flags.append('산업분류_교차확인필요')
    if any(len(r['층정보']) and r['층정보'] not in ['1','1층'] for r in rows): flags.append('층정보_접근확인필요')
    locations.append(dict(id='CWCVS_'+hashlib.sha256('|'.join(sorted(r['상가업소번호'] for r in rows)).encode()).hexdigest()[:12],lon=lon,lat=lat,district=districts[0],dong=dongs[0],districts=districts,dongs=dongs,names=names,addresses=addresses,raw_count=len(rows),raw_ids=sorted(r['상가업소번호'] for r in rows),flags=flags,operating_hours=None,participation_agreement=None,public_access_verified=False,capacity_l=None))
assert len(raw)==950 and len(locations)==863
assert sum(s['raw_count'] for s in locations)==len(raw)
assert len(set(r['상가업소번호'] for r in raw))==len(raw)
xy=project_points(locations)
nearest=cKDTree(xy).query(xy,k=2)[0][:,1]
bus=json.loads((P/'data/processed/bus_stops.json').read_text()); bxy=project_points(bus)
health=[dict(id=r['상가업소번호'],name=r['상호명'],type=r['상권업종소분류명'],lon=float(r['경도']),lat=float(r['위도'])) for r in shops if r['상권업종소분류코드'].startswith(('Q101','Q102')) or r['상권업종소분류코드']=='G21501']
hxy=project_points(health)
busdist=cKDTree(bxy).query(xy)[0]; buscounts=cKDTree(bxy).query_ball_point(xy,350)
healthcounts=cKDTree(hxy).query_ball_point(xy,350)
for i,s in enumerate(locations):
    s.update(nearest_other_candidate_m=round(float(nearest[i]),1),nearest_bus_stop_m=round(float(busdist[i]),1),bus_stop_records_within_350m=len(buscounts[i]),health_business_records_within_350m=len(healthcounts[i]))
rows=[]
for area,stats in age['by_dong'].items():
    dist,dong=area.split(' ',1)
    candidates=[s for s in locations if s['district']==dist and s['dong']==dong]
    out=dict(area=area,district=dist,dong=dong,**stats,candidate_locations=len(candidates),raw_store_records=sum(s['raw_count'] for s in candidates))
    out['age65plus_share_pct']=100*stats['age65plus']/stats['population']
    out['population_per_candidate']=stats['population']/len(candidates) if candidates else None
    out['age65plus_per_candidate']=stats['age65plus']/len(candidates) if candidates else None
    rows.append(out)
assert sum(r['candidate_locations'] for r in rows)==len(locations), 'Unmatched candidate administrative area'
assert sum(r['population'] for r in rows)==995372
bydistrict=[]
for dist,stats in age['by_district'].items():
    cs=[s for s in locations if s['district']==dist]; n=len(cs)
    bydistrict.append(dict(district=dist,**stats,candidate_locations=n,raw_store_records=sum(s['raw_count'] for s in cs),age65plus_share_pct=100*stats['age65plus']/stats['population'],population_per_candidate=stats['population']/n,age65plus_per_candidate=stats['age65plus']/n))
quality=dict(raw_records=len(raw),unique_raw_ids=len(raw),candidate_coordinates=len(locations),coordinate_groups_with_multiple_records=sum(s['raw_count']>1 for s in locations),raw_records_in_multi_groups=sum(s['raw_count'] for s in locations if s['raw_count']>1),flags=dict(collections.Counter(f for s in locations for f in s['flags'])),nearest_neighbor_m_percentiles=dict(zip(['p10','p25','p50','p75','p90'],np.percentile(nearest,[10,25,50,75,90]).tolist())),candidates_with_bus_within_350m=int(sum(busdist<=350)),coordinates_unverified=True,city_boundary_check_pending=True,age_reference='2025-05-31',shops_reference='2026-06-30',mode='convenience_only',counts_are_not_population_coverage=True)
(P/'data/processed/convenience_candidates.json').write_text(json.dumps(locations,ensure_ascii=False,indent=2))
(P/'data/processed/health_business_anchors.json').write_text(json.dumps(health,ensure_ascii=False,indent=2))
(P/'results/changwon_supply_demographics.json').write_text(json.dumps(dict(quality=quality,by_district=bydistrict,by_dong=rows),ensure_ascii=False,indent=2))
(P/'data/processed/convenience_candidates.geojson').write_text(json.dumps(dict(type='FeatureCollection',features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[s['lon'],s['lat']]),properties=s) for s in locations]),ensure_ascii=False))
for name,data in [('candidate_locations',locations),('district_demographics',bydistrict),('dong_demographics',rows)]:
    dest=P/'results'/f'{name}.csv'
    with dest.open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0]));wr.writeheader();wr.writerows([{k:' | '.join(map(str,v)) if isinstance(v,list) else v for k,v in r.items()} for r in data])
print(json.dumps(dict(quality=quality,districts=bydistrict,top_elderly_per_candidate=sorted(rows,key=lambda r:r['age65plus_per_candidate'] or 1e9,reverse=True)[:8]),ensure_ascii=False,indent=2))
