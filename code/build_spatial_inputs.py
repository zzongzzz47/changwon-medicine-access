"""SGIS 원본 ZIP에서 창원 55개 읍면동·100m/500m 통계를 공간 연결한다."""
from pathlib import Path
import argparse
import io
import json
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
from src.spatial_join import allocate_grid_to_areas

ROOT = Path(__file__).resolve().parent

def main(cache):
    cache.mkdir(parents=True, exist_ok=True)
    raw = ROOT / 'data/raw'
    out = ROOT / 'data/processed'
    files = [*raw.glob('*grid_border*.zip'), *raw.glob('bnd_dong*.zip')]
    if len(files) != 5:
        raise ValueError('SGIS 경계 ZIP 5개가 필요합니다.')
    for path in files:
        with zipfile.ZipFile(path) as z:
            if z.testzip() is not None:
                raise ValueError(f'압축 손상: {path}')
            for f in z.infolist():
                if Path(f.filename).is_absolute() or '..' in Path(f.filename).parts:
                    raise ValueError('안전하지 않은 압축 경로')
            z.extractall(cache/path.stem)
    codes = pd.DataFrame(json.loads((out/'changwon_admin_codes_2025.json').read_text())['records'])
    codes['district'] = codes.district.str.replace('창원시 ', '', regex=False)
    areas = gpd.read_file(next(cache.rglob('bnd_dong*.shp'))).rename(columns={'ADM_CD':'adm_cd'})
    areas = areas.merge(codes, on='adm_cd', how='inner')[['adm_cd','district','dong','geometry']].to_crs(5179)
    assert len(areas) == 55 and areas.adm_cd.is_unique
    assert areas.geometry.is_valid.all()
    city = areas.geometry.union_all()
    areas.to_file(out/'changwon_spatial.gpkg',layer='admin_dongs',driver='GPKG')
    audit = {'admin_dongs':len(areas),'city_area_km2':city.area/1e6,'crs':'EPSG:5179',
             'statistics_year':2024,'boundary_date':'2025-06-30','by_resolution':{}}
    with zipfile.ZipFile(raw/'sgis_grid_population_households_2024.zip') as z:
        tables = {}
        for info in z.infolist():
            if not info.filename.endswith('.csv'): continue
            name = info.filename
            if not (info.flag_bits & 0x800):
                try: name = name.encode('cp437').decode('cp949')
                except (UnicodeError,LookupError): pass
            tables[name] = pd.read_csv(io.BytesIO(z.read(info)),encoding='cp949',header=None,
                                      names=['year','grid_id','stat_code','value'],dtype={'grid_id':str})
    for resolution in ['100M','500M']:
        geoms = pd.concat([gpd.read_file(p,bbox=city.bounds) for p in cache.rglob(f'*_{resolution}.shp')],ignore_index=True)
        grids = gpd.GeoDataFrame(geoms,crs=5179).rename(columns={'GRID_CD':'grid_id'})
        grids = grids.loc[grids.intersects(city)].copy()
        assert grids.grid_id.is_unique
        for word, stat, col in [('인구','to_in_001','population'),('가구','to_ga_001','households')]:
            t = pd.concat([v for k,v in tables.items() if word in k and f'_{resolution}.csv' in k],ignore_index=True)
            t = t.loc[t.stat_code == stat,['grid_id','value']]
            assert t.grid_id.is_unique
            t['value'] = pd.to_numeric(t.value,errors='coerce')
            grids = grids.merge(t.rename(columns={'value':col}),on='grid_id',how='left')
        joined = allocate_grid_to_areas(grids,areas)
        joined.to_file(out/'changwon_spatial.gpkg',layer=f'demand_{resolution.lower()}',driver='GPKG')
        joined.drop(columns='geometry').to_csv(out/f'demand_{resolution.lower()}.csv',index=False,encoding='utf-8-sig')
        # 시 경계의 면적 배분 효과를 원래 공개값과 비교해 기록한다.
        fraction = joined.groupby('grid_id').area_fraction.sum()
        city_edge = fraction.index[fraction < 1 - 1e-6]
        known_pop = grids.population.notna()
        centers = gpd.GeoSeries(grids.centroid,crs=5179)
        check = {'intersecting_grid_count':len(grids),'allocation_rows':len(joined),
                 'population_observed_grids':int(known_pop.sum()),
                 'population_missing_grids':int((~known_pop).sum()),
                 'households_missing_grids':int(grids.households.isna().sum()),
                 'population_positive_grids':int((grids.population>0).sum()),
                 'population_area_allocated':float(joined.population.sum()),
                 'households_area_allocated':float(joined.households.sum()),
                 'population_center_inside_total':float(grids.loc[centers.within(city),'population'].sum()),
                 'city_edge_grid_count':len(city_edge),
                 'city_edge_population_allocated':float(joined.loc[joined.grid_id.isin(city_edge),'population'].sum()),
                 'by_district':joined.groupby('district')[['population','households']].sum().reset_index().to_dict('records')}
        audit['by_resolution'][resolution] = check
        print(resolution, json.dumps(check,ensure_ascii=False),flush=True)
    candidates = json.loads((out/'convenience_candidates.json').read_text())
    points = gpd.GeoDataFrame({'id':[c['id'] for c in candidates]},
            geometry=gpd.points_from_xy([c['lon'] for c in candidates],[c['lat'] for c in candidates]),crs=4326).to_crs(5179)
    mapped = gpd.sjoin(points,areas,predicate='within',how='left')
    assert mapped.id.is_unique
    for c, row in zip(candidates,mapped.itertuples()):
        c['inside_city_boundary'] = bool(pd.notna(row.adm_cd))
        c['spatial_adm_cd'] = str(row.adm_cd) if pd.notna(row.adm_cd) else None
        c['spatial_district'] = row.district if pd.notna(row.district) else None
        c['spatial_dong'] = row.dong if pd.notna(row.dong) else None
        c['x_m'] = float(row.geometry.x); c['y_m'] = float(row.geometry.y)
        c['distance_to_city_m'] = float(row.geometry.distance(city))
    (out/'convenience_candidates_spatial.json').write_text(json.dumps(candidates,ensure_ascii=False,indent=2))
    audit['candidate_count'] = len(candidates)
    audit['outside_candidates'] = [c for c in candidates if not c['inside_city_boundary']]
    audit['district_mismatch_ids'] = [c['id'] for c in candidates if c['inside_city_boundary'] and c['district']!=c['spatial_district']]
    audit['dong_mismatch_ids'] = [c['id'] for c in candidates if c['inside_city_boundary'] and c['dong']!=c['spatial_dong']]
    (ROOT/'results/spatial_join_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
    print('Spatial join complete; outside candidates:',len(audit['outside_candidates']),flush=True)

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache-dir',type=Path,default=ROOT/'.cache/boundaries')
    main(parser.parse_args().cache_dir)
