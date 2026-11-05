"""Export a report figure comparing the two verified 60-store selections."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
R = P / 'results/supplement_development'
O = P / 'results/chapter5_figures_only'
O.mkdir(exist_ok=True)

eff = pd.read_csv(R / '효율안_검토편의점60곳.csv', encoding='utf-8-sig')
fair = pd.read_csv(R / '형평안_검토편의점60곳.csv', encoding='utf-8-sig')
eid, fid = set(eff['후보ID']), set(fair['후보ID'])
assert len(eff) == len(eid) == len(fair) == len(fid) == 60
assert (len(eid & fid), len(eid - fid), len(fid - eid)) == (45, 15, 15)
shared_a = eff.set_index('후보ID').loc[sorted(eid & fid), ['경도', '위도']]
shared_b = fair.set_index('후보ID').loc[sorted(eid & fid), ['경도', '위도']]
assert np.allclose(shared_a, shared_b, atol=1e-10, rtol=0)

cols = ['후보ID', '점포명_원자료', '구', '읍면동', '도로명주소', '경도', '위도']
union = pd.concat([eff[cols], fair[cols]], ignore_index=True).drop_duplicates('후보ID').copy()
union['선정구분'] = union['후보ID'].map(
    lambda k: '공통 선정' if k in eid & fid else '효율안만 선정' if k in eid else '형평안만 선정')
points = gpd.GeoDataFrame(union, geometry=gpd.points_from_xy(union['경도'], union['위도']), crs=4326).to_crs(5179)
admin = gpd.read_file(P / 'results/chapter4_visuals/dong_coverage.geojson').to_crs(5179)
assert len(admin) == 55
districts = admin[['district', 'geometry']].dissolve(by='district')
assert len(districts) == 5
assert points.geometry.covered_by(admin.geometry.union_all().buffer(1)).all()

font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'font.size': 12,
                     'axes.unicode_minus': False, 'svg.fonttype': 'path',
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white',
                     'text.color': '#203C43'})
INK = '#203C43'
styles = [
    ('공통 선정', '#74868B', 'o', 31, '공통 선정 45곳'),
    ('효율안만 선정', '#167EAD', '^', 63, '효율안만 선정 15곳'),
    ('형평안만 선정', '#C77325', 'D', 47, '형평안만 선정 15곳'),
]
fig = plt.figure(figsize=(9.2, 8.2))
ax = fig.add_axes([.015, .09, .97, .90])
admin.plot(ax=ax, facecolor='#F1F4F4', edgecolor='#D5DDDF', linewidth=.35, zorder=1)
districts.boundary.plot(ax=ax, color='#819397', linewidth=.9, zorder=2)
for i, (category, color, marker, size, label) in enumerate(styles):
    sub = points.loc[points['선정구분'] == category]
    ax.scatter(sub.geometry.x, sub.geometry.y, s=size, marker=marker,
               facecolor=color, edgecolor='white', linewidth=.65, zorder=5+i)
for name, row in districts.iterrows():
    center = row.geometry.representative_point()
    ax.text(center.x, center.y, name, fontsize=12, weight='bold', ha='center', va='center',
            zorder=9, path_effects=[patheffects.withStroke(linewidth=3.5, foreground='white')])
x0, y0, x1, y1 = admin.total_bounds
pad = max(x1-x0, y1-y0) * .025
ax.set(xlim=(x0-pad, x1+pad), ylim=(y0-pad, y1+pad), aspect='equal')
ax.set_axis_off()
sx, sy = x0+(x1-x0)*.055, y0+(y1-y0)*.025
ax.plot([sx, sx+5000], [sy, sy], c=INK, lw=1.6, zorder=10)
for x in [sx, sx+5000]:
    ax.plot([x, x], [sy-200, sy+200], c=INK, lw=1.2, zorder=10)
ax.text(sx+2500, sy+750, '5 km', ha='center', fontsize=10)
ax.annotate('N', xy=(.95,.95), xytext=(.95,.87), xycoords='axes fraction',
            ha='center', fontsize=11, arrowprops=dict(arrowstyle='-|>', lw=1.2, color=INK))
handles = [Line2D([], [], linestyle='', marker=marker, markersize=8,
                  markerfacecolor=color, markeredgecolor='white', label=label)
           for _, color, marker, _, label in styles]
fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.018), ncol=3,
           frameon=False, fontsize=12, handletextpad=.35, columnspacing=1.5)
stem = '그림5-1_효율안과_형평안_선정편의점_비교'
for ext in ['png', 'svg']:
    fig.savefig(O / f'{stem}.{ext}', dpi=350, bbox_inches='tight', pad_inches=.1)
plt.close(fig)
union.sort_values(['선정구분', '구', '후보ID']).to_csv(O / '그림5-1_선정편의점75곳.csv', index=False, encoding='utf-8-sig')
caption = '''그림 5-1. 효율안과 형평안의 선정 편의점 비교

주: 직선거리 350m, 기존 수거지 439곳을 유지하고 편의점 60곳을 추가하는 조건이다. 두 안에 공통으로 선정된 편의점은 45곳이며, 각 안에서만 선정된 편의점은 15곳씩이다.
자료: 소상공인시장진흥공단 상가(상권)정보(2026.6.30), 전국폐의약품수거함표준데이터(2026.1.28), SGIS 인구(2024)·경계(2025)를 활용한 본 연구 분석.

삽입 위치: 5.2의 “두 안에서 공통으로 선정된 점포는 45곳이었다.”로 시작하는 문단 다음.
'''
(O / '그림5-1_캡션과_출처.txt').write_text(caption, encoding='utf-8')
print(json.dumps({'output': str(O / f'{stem}.png'), 'counts': union['선정구분'].value_counts().to_dict(),
                  'district_counts': pd.concat([eff.assign(안='효율안'), fair.assign(안='형평안')], ignore_index=True).groupby(['안', '구']).size().unstack().to_dict()}, ensure_ascii=False))
