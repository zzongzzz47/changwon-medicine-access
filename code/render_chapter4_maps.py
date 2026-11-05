"""Export report-ready Chapter 4 maps from the saved 350 m analysis."""
from pathlib import Path
import json
import shutil
import zipfile

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
R = P / 'results/supplement_development'
O = P / 'results/chapter4_visuals'
O.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'font.size': 12,
                     'axes.unicode_minus': False, 'figure.facecolor': 'white',
                     'savefig.facecolor': 'white', 'svg.fonttype': 'path',
                     'text.color': '#203C43', 'axes.labelcolor': '#203C43'})
INK, TEAL, PURPLE, GREY = '#203C43', '#16868A', '#AE4C65', '#C1C9CC'
LAND, LINE = '#F0F3F3', '#CDD5D7'
SOURCE = '자료: 전국폐의약품수거함표준데이터(2026.1.28), 상가(상권)정보(2026.6.30), SGIS 인구(2024)·경계(2025) / 본 연구 분석'

summary = json.loads((R / 'development.json').read_text())
admin = gpd.read_file(P / 'data/processed/changwon_spatial.gpkg', layer='admin_dongs').to_crs(5179)
dg = pd.read_csv(R / 'dong_gap_diagnostics.csv')
admin = admin.merge(dg, on=['district', 'dong'], validate='one_to_one')
admin['all_candidates_pct'] = 100 * (1 - admin.screened_outside_candidate_network_population / admin.population)
districts = admin[['district', 'geometry']].dissolve(by='district')
grid = gpd.read_file(P / 'data/processed/changwon_spatial.gpkg', layer='demand_100m').to_crs(5179)
grid = grid.loc[grid.population > 0].copy()
coverage = pd.read_csv(R / 'coverage_by_demand.csv')
grid = grid.merge(coverage[['demand_id', 'existing_covered', 'screened_union_covered']], on='demand_id', validate='one_to_one')
grid['status'] = np.select([grid.existing_covered, grid.screened_union_covered],
                           ['기존망 커버', '편의점으로 보완 가능'], default='모든 후보 활용 후에도 남는 공백')

existing = pd.DataFrame(json.loads((P / 'results/existing_network/existing_location_audit.json').read_text()))
existing = existing.loc[existing.inside_city].copy()
existing = gpd.GeoDataFrame(existing, geometry=gpd.points_from_xy(existing.x_m, existing.y_m), crs=5179)
cs = json.loads((P / 'data/processed/convenience_candidates_spatial.json').read_text())
cs = [c for c in cs if '시설내점포_출입확인필요' not in c['flags']
      and not any('더프레시' in n or '노브랜드' in n for n in c['names'])]
candidates = pd.DataFrame(cs)
candidates = gpd.GeoDataFrame(candidates, geometry=gpd.points_from_xy(candidates.x_m, candidates.y_m), crs=5179)

# Validate displayed classes against the same coordinates and radius as the report.
assert len(admin) == 55 and len(existing) == 439 and len(candidates) == 802
xy = grid[['x_m', 'y_m']].to_numpy()
base = cKDTree(existing[['x_m', 'y_m']]).query(xy)[0] <= 350
union = base | (cKDTree(candidates[['x_m', 'y_m']]).query(xy)[0] <= 350)
assert np.array_equal(base, grid.existing_covered.to_numpy())
assert np.array_equal(union, grid.screened_union_covered.to_numpy())
assert abs(grid.population.sum() - summary['total_population']) < 1e-6
assert abs(grid.loc[base, 'population'].sum() - summary['baseline']['population']) < 1e-6
assert abs(grid.loc[union, 'population'].sum() - summary['screened_union_ceiling']['population']) < 1e-6

def head(fig, title, subtitle):
    fig.text(.045, .956, title, fontsize=23, weight='bold', va='top')
    fig.text(.045, .903, subtitle, fontsize=12.5, color='#61757C', va='top')

def footer(fig, note):
    fig.text(.045, .060, SOURCE, fontsize=8.7, color='#61757C', va='top')
    fig.text(.045, .035, note, fontsize=8.7, color='#61757C', va='top')

def limits(ax, bounds=None, scale=5000):
    x0, y0, x1, y1 = admin.total_bounds if bounds is None else bounds
    pad = max(x1-x0, y1-y0)*.035
    ax.set_xlim(x0-pad, x1+pad); ax.set_ylim(y0-pad, y1+pad)
    ax.set_aspect('equal'); ax.set_axis_off()
    sx = x0 + (x1-x0)*.04; sy = y0 + (y1-y0)*.025
    ax.plot([sx, sx+scale], [sy, sy], c=INK, lw=1.6, zorder=20)
    ax.text(sx+scale/2, sy+(y1-y0)*.025, f'{scale/1000:g} km', ha='center', fontsize=9, zorder=21)
    ax.annotate('N', xy=(.965,.94), xytext=(.965,.86), xycoords='axes fraction',
                ha='center', fontsize=10, arrowprops=dict(arrowstyle='-|>', lw=1.1, color=INK))

def borders(ax, labels=True):
    districts.boundary.plot(ax=ax, color='#667F85', linewidth=.8, zorder=7)
    if labels:
        for name, row in districts.iterrows():
            p = row.geometry.representative_point()
            ax.text(p.x, p.y, name, ha='center', va='center', fontsize=11.3, weight='bold', zorder=12,
                    path_effects=[patheffects.withStroke(linewidth=3.5, foreground='white')])

def save(fig, stem):
    fig.savefig(O / f'{stem}.png', dpi=300)
    fig.savefig(O / f'{stem}.svg')
    plt.close(fig)

def side_text(ax, y, label, value, color=INK):
    ax.text(0, y, label, fontsize=12, color='#61757C', va='top')
    ax.text(0, y-.048, value, fontsize=25, color=color, weight='bold', va='top')

# 1. Population and existing collection locations.
fig = plt.figure(figsize=(13, 9.2))
ax = fig.add_axes([.035,.125,.66,.735]); side = fig.add_axes([.735,.15,.24,.66]); side.axis('off')
admin.plot(ax=ax, color=LAND, edgecolor=LINE, linewidth=.25)
pop_colors = ['#FBE2A2','#F5BA57','#E88335','#BB4D1D']
pop_class = np.digitize(grid.population_reported, [50,200,500])
for k,c in enumerate(pop_colors): grid.loc[pop_class==k].plot(ax=ax, color=c, edgecolor='none', zorder=2)
existing.plot(ax=ax, marker='o', facecolor='none', edgecolor='#305C8B', markersize=13, linewidth=.65, zorder=5)
borders(ax); limits(ax)
head(fig, '기존 수거지와 인구는 어디에 분포하는가?', '기존 수거지 439개 지점 · 인구가 공개된 100m 격자')
side_text(side, .97, '분석 대상 인구', '약 100.7만 명')
side_text(side, .80, '기존망 350m 커버율', '69.15%', TEAL)
side.text(0,.60,'100m 원격자 인구 (명)',fontsize=13,weight='bold')
handles=[Patch(facecolor=c,label=t) for c,t in zip(pop_colors,['1–49','50–199','200–499','500 이상'])]
handles += [Line2D([],[],linestyle='',marker='o',markerfacecolor='none',markeredgecolor='#305C8B',label='기존 수거지')]
side.legend(handles=handles,loc='upper left',bbox_to_anchor=(-.025,.58),frameon=False,labelspacing=.85,fontsize=11)
side.text(0,.12,'경계선: 창원시 5개 구·55개 읍면동\n밝은 배경: 양의 공개 인구 격자 외 영역',fontsize=10,color='#61757C',linespacing=1.8)
footer(fig, '인구 색상은 경계 배분 전 원격자 인구 기준. 밝은 배경에는 인구 비공개·결측 격자도 포함된다. 좌표계 EPSG:5179.')
save(fig,'01_existing_population')

# 2. Dong coverage choropleth and gu comparison.
fig = plt.figure(figsize=(13,9.2)); ax=fig.add_axes([.035,.18,.66,.67]); bars=fig.add_axes([.775,.44,.18,.31])
colors=['#B34860','#D89493','#F0CFAC','#9BC8C0','#347F80']
bins=[0,25,50,70,85,100.0001]
admin.plot(column='baseline_pct',ax=ax,cmap=ListedColormap(colors),norm=BoundaryNorm(bins,5),edgecolor='white',linewidth=.45)
borders(ax); limits(ax)
order=['의창구','마산합포구','진해구','성산구','마산회원구']
vals=[summary['baseline']['districts'][g] for g in order]
bars.barh(order,vals,color=TEAL,height=.56); bars.invert_yaxis();bars.set_xlim(0,100)
bars.axvline(summary['baseline']['population_pct'],c='#7C9095',lw=1,ls='--')
for y,v in enumerate(vals): bars.text(v+1.1,y,f'{v:.2f}',va='center',fontsize=10)
bars.set_xlabel('구별 커버율 (%)',fontsize=10);bars.tick_params(axis='both',labelsize=10,length=0)
for sp in bars.spines.values():sp.set_visible(False)
bars.set_title('전체 69.15%',loc='left',fontsize=16,pad=16,weight='bold')
fig.text(.72,.34,'커버율 50% 미만',fontsize=12,color='#61757C')
fig.text(.72,.288,'13개 읍면동',fontsize=22,weight='bold',color=PURPLE)
fig.text(.72,.215,'구별 최고–최저 차이 10.71%p\n점선은 창원시 전체 커버율',fontsize=10.5,color='#61757C',linespacing=1.8)
head(fig,'기존 수거망의 접근성은 지역마다 얼마나 다른가?', '읍면동별 인구 커버율 · 기존 수거지까지 직선거리 350m 기준')
fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(colors,['25% 미만','25–50% 미만','50–70% 미만','70–85% 미만','85% 이상'])],
           loc='lower center',bbox_to_anchor=(.36,.10),ncol=5,frameon=False,fontsize=10)
footer(fig,'커버율 = 해당 읍면동의 기존 수거지 350m 이내 인구 ÷ 해당 읍면동의 공개 인구. 통계는 인구 기준, 지도 면적은 행정경계 기준.')
save(fig,'02_dong_coverage')

# 3. Existing gaps, with two comparable local views.
fig=plt.figure(figsize=(13,10.0)); ax=fig.add_axes([.025,.23,.62,.56])
admin.plot(ax=ax,color=LAND,edgecolor=LINE,linewidth=.25)
for mask,col in [(grid.existing_covered,GREY),(~grid.existing_covered & grid.screened_union_covered,TEAL),(~grid.screened_union_covered,PURPLE)]:
    grid.loc[mask].plot(ax=ax,color=col,edgecolor='none')
borders(ax);limits(ax)
for district,dong,rect in [('마산합포구','가포동',[.70,.51,.265,.275]),('마산합포구','진전면',[.70,.19,.265,.275])]:
    ar=admin.loc[(admin.district==district)&(admin.dong==dong)]
    row=ar.iloc[0]; xx=fig.add_axes(rect)
    bounds=ar.total_bounds; pad=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*.05
    local_grid=grid.cx[bounds[0]-pad:bounds[2]+pad,bounds[1]-pad:bounds[3]+pad]
    admin.plot(ax=xx,color=LAND,edgecolor=LINE,linewidth=.35)
    for mask,col in [(local_grid.existing_covered,GREY),(~local_grid.existing_covered & local_grid.screened_union_covered,TEAL),(~local_grid.screened_union_covered,PURPLE)]:
        subset=local_grid.loc[mask]
        if len(subset): subset.plot(ax=xx,color=col,edgecolor='none')
    near=candidates.cx[bounds[0]-pad:bounds[2]+pad,bounds[1]-pad:bounds[3]+pad]
    if len(near):near.plot(ax=xx,marker='o',facecolor='none',edgecolor=INK,markersize=19,linewidth=.75,zorder=7)
    ar.boundary.plot(ax=xx,color=INK,linewidth=1.0,zorder=6)
    limits(xx,bounds,scale=1000)
    xx.set_title(f'{dong}   {row.baseline_pct:.2f}% → {row.all_candidates_pct:.2f}%',loc='left',fontsize=12.5,pad=8,weight='bold')
    p=row.geometry.representative_point()
    ax.plot(p.x,p.y,marker='o',mfc='white',mec=INK,ms=4,zorder=20)
    ax.annotate(dong,(p.x,p.y),xytext=(8,-14 if dong=='가포동' else 9),textcoords='offset points',fontsize=10.5,zorder=21,
                path_effects=[patheffects.withStroke(linewidth=3,foreground='white')])
head(fig,'기존 접근 공백 중 어디를 편의점으로 보완할 수 있는가?', '주 분석 후보 802곳을 모두 활용한 공간적 보완 가능성 · 직선거리 350m')
fig.text(.045,.832,'보완 가능  약 24.2만 명',fontsize=17,weight='bold',color=TEAL)
fig.text(.365,.832,'남는 공백  약 6.8만 명',fontsize=17,weight='bold',color=PURPLE)
fig.legend(handles=[Patch(facecolor=GREY,label='기존망 커버'),Patch(facecolor=TEAL,label='편의점으로 보완 가능'),
                    Patch(facecolor=PURPLE,label='모든 후보 활용 후에도 남는 공백'),
                    Line2D([],[],linestyle='',marker='o',mfc='none',mec=INK,label='편의점 후보 (확대도)')],
           loc='lower center',bbox_to_anchor=(.5,.112),ncol=2,frameon=False,fontsize=11,columnspacing=2.5,labelspacing=.8)
footer(fig,'색상은 양의 공개 인구가 있는 격자에 표시. 확대도의 수치는 기존 커버율 → 전체 후보 활용 시 커버율. 60곳 최적화 결과와 구분.')
save(fig,'03_fillable_gaps')

# Reusable GIS / table data. Attribute definitions are in the guide.
keep=['adm_cd','district','dong','population','baseline_pct','existing_gap_population','screened_fillable_gap_population',
      'screened_outside_candidate_network_population','all_candidates_pct','geometry']
admin[keep].to_crs(4326).to_file(O/'dong_coverage.geojson',driver='GeoJSON')
def point_export(frame, name):
    out=frame[['lon','lat','names','addresses','geometry']].copy()
    for col in ['names','addresses']:out[col]=out[col].apply(lambda x:' / '.join(x))
    out.to_crs(4326).to_file(O/name,driver='GeoJSON')
point_export(existing,'existing_439.geojson');point_export(candidates,'candidates_802.geojson')
grid[['demand_id','grid_id','district','dong','population','population_reported','existing_covered','screened_union_covered','status','geometry']].to_file(O/'population_coverage.gpkg',layer='population_100m',driver='GPKG')
labels={'adm_cd':'행정동코드','district':'구','dong':'읍면동','population':'분석인구_명','baseline_pct':'기존커버율_pct',
        'existing_gap_population':'기존공백인구_명','screened_fillable_gap_population':'편의점보완가능인구_명',
        'screened_outside_candidate_network_population':'모든후보활용후공백인구_명','all_candidates_pct':'모든후보활용시커버율_pct'}
admin[keep[:-1]].rename(columns=labels).sort_values(['구','읍면동']).to_csv(O/'읍면동별_시각화자료.csv',index=False,encoding='utf-8-sig')
shutil.copyfile(R/'figures/01_기존망의_공백과_보완가능성.png',O/'04_gap_population_chart.png')

guide='''# 4장 시각화 자료

## 보고서 배치
- 4.1: 01_existing_population — 기존 수거지 439개 지점과 100m 격자 인구 분포.
- 4.2: 02_dong_coverage — 55개 읍면동 커버율 지도와 5개 구 비교.
- 4.3~4.4: 03_fillable_gaps — 모든 후보 802곳을 활용했을 때 보완 가능한 공백과 남는 공백. 가포동·진전면 확대 비교.
- 공백 인구의 크기 비교가 필요하면 04_gap_population_chart를 함께 활용.

PNG는 300dpi(새 지도 3장), SVG는 확대 가능한 벡터 그림이다. 지도 문자는 폰트가 바뀌지 않도록 경로로 저장했다.

## 자료 파일
- 읍면동별_시각화자료.csv: 55개 읍면동의 공개 인구, 커버율, 공백 인구. UTF-8 BOM.
- dong_coverage.geojson: 행정경계에 같은 통계를 연결한 파일. EPSG:4326.
- existing_439.geojson: 같은 좌표를 합친 기존 수거지. EPSG:4326.
- candidates_802.geojson: 주 분석에 사용한 편의점 후보. EPSG:4326.
- population_coverage.gpkg: 양의 공개 인구가 있는 100m 격자 경계 조각과 접근성 분류. EPSG:5179.

인구 격자는 읍면동 경계에서 분할되어 demand_id로 식별된다. population은 중첩 면적에 따라 배분한 인구이고 population_reported는 원격자의 공개 인구다. 합계는 population을 사용한다. 동일 원격자의 population_reported를 중복 합산하면 안 된다.

## 해석 기준
350m는 원격자 중심점에서 시설까지의 직선거리다. 기존 수거망을 유지하며 후보 802곳을 모두 활용한 범위를 그렸다. 점포 수를 제한한 최적화 결과와 구분한다. 밝은 배경에는 인구 비공개·결측 영역도 포함된다. 색칠된 면적 비율은 인구 비율이 아니다.

## 출처
- 창원시 기존 수거지, 2026.1.28: https://www.data.go.kr/data/15129445/standard.do
- 소상공인시장진흥공단 상가(상권)정보, 2026.6.30: https://www.data.go.kr/data/15083033/fileData.do
- SGIS 2024년 인구·2025년 격자 및 행정경계: https://sgis.mods.go.kr/view/pss/openDataIntrcn
- 커버율과 접근 공백: 본 연구 분석.

기존 수거지·편의점 명부의 영업 및 실제 참여 여부는 공개자료 취득 이후 현장 확인하지 않았다. SGIS 자료의 이용·결과물 제출 조건은 원자료 신청 조건을 따른다.
'''
(O/'자료_사용안내.md').write_text(guide)
captions=[('01_existing_population','4.1 기존 수거지와 인구 분포','수거지 위치와 주민 분포를 함께 보여주는 지도'),
          ('02_dong_coverage','4.2 읍면동별 커버율','커버율이 낮은 지역과 구별 차이를 보여주는 지도'),
          ('03_fillable_gaps','4.3–4.4 보완 가능한 공백과 남는 공백','802곳 전체 후보망의 잠재 범위와 가포동·진전면 비교')]
cards=''.join(f'<section><h2>{title}</h2><p>{desc}</p><a href="{stem}.png"><img src="{stem}.png" alt="{title}"></a><p><a href="{stem}.png" download>PNG 다운로드</a> · <a href="{stem}.svg" download>SVG 다운로드</a></p></section>' for stem,title,desc in captions)
(O/'4장_지도와_자료.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>4장 지도와 자료</title><style>body{margin:0;background:#f3f6f5;color:#203c43;font:16px/1.7 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}main{max-width:1100px;margin:40px auto;padding:0 24px}h1{font-size:32px}section{background:white;padding:24px;margin:24px 0;border-radius:14px}img{width:100%;display:block}a{color:#137b82}p{color:#5b7076}li{margin:8px 0}</style><main><h1>4장 · 기존 수거망의 접근 공백</h1><p>보고서용 지도 3장과 재사용 가능한 시각화 자료입니다. 기존 수거지 439개·편의점 후보 802개·직선거리 350m 기준.</p>'''+cards+'''<section><h2>읍면동별 공백 인구 비교</h2><a href="04_gap_population_chart.png"><img src="04_gap_population_chart.png" alt="기존 공백 인구 상위 10개 읍면동"></a></section><section><h2>자료 내려받기</h2><ul><li><a href="읍면동별_시각화자료.csv" download>55개 읍면동 통계표 (CSV)</a></li><li><a href="dong_coverage.geojson" download>통계가 연결된 읍면동 지도 (GeoJSON)</a></li><li><a href="existing_439.geojson" download>기존 수거지 439개 (GeoJSON)</a></li><li><a href="candidates_802.geojson" download>편의점 후보 802개 (GeoJSON)</a></li><li><a href="population_coverage.gpkg" download>100m 인구 격자와 공백 분류 (GeoPackage)</a></li><li><a href="자료_사용안내.md">자료 정의와 출처</a></li></ul></section></main></html>''')
validation={'existing_points':len(existing),'candidate_points':len(candidates),'admin_dongs':len(admin),'positive_population_fragments':len(grid),
            'population':float(grid.population.sum()),'existing_pct':float(100*grid.loc[base,'population'].sum()/grid.population.sum()),
            'all_candidates_pct':float(100*grid.loc[union,'population'].sum()/grid.population.sum()),'distance_recheck':'all rows match saved 350m classifications'}
(O/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
with zipfile.ZipFile(O.parent/'4장_지도와_시각화자료.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(O.iterdir()):
        if f.is_file():z.write(f,arcname=f'4장_지도와_시각화자료/{f.name}')
print(json.dumps(validation,ensure_ascii=False));print('OUTPUT',O)
