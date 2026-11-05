"""Redraw Chapter 4 plots with only map/chart labels and legends."""
from pathlib import Path
import json
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
S = P / 'results/chapter4_visuals'
O = P / 'results/chapter4_figures_only'
O.mkdir(exist_ok=True)
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family':'Apple SD Gothic Neo','font.size':12,'axes.unicode_minus':False,
                     'figure.facecolor':'white','savefig.facecolor':'white','svg.fonttype':'path',
                     'text.color':'#203C43','axes.labelcolor':'#203C43'})
INK,TEAL,PURPLE,GREY,LAND,LINE='#203C43','#16868A','#AE4C65','#C1C9CC','#F0F3F3','#CDD5D7'
admin=gpd.read_file(S/'dong_coverage.geojson').to_crs(5179)
grid=gpd.read_file(S/'population_coverage.gpkg',layer='population_100m')
existing=gpd.read_file(S/'existing_439.geojson').to_crs(5179)
candidates=gpd.read_file(S/'candidates_802.geojson').to_crs(5179)
districts=admin[['district','geometry']].dissolve(by='district')
summary=json.loads((P/'results/supplement_development/development.json').read_text())
assert len(existing)==439 and len(candidates)==802 and len(admin)==55

def limits(ax,bounds=None,scale=5000):
    x0,y0,x1,y1=admin.total_bounds if bounds is None else bounds
    pad=max(x1-x0,y1-y0)*.025
    ax.set(xlim=(x0-pad,x1+pad),ylim=(y0-pad,y1+pad),aspect='equal')
    ax.set_axis_off()
    sx=x0+(x1-x0)*.045;sy=y0+(y1-y0)*.018
    ax.plot([sx,sx+scale],[sy,sy],c=INK,lw=1.5,zorder=20)
    ax.text(sx+scale/2,sy+(y1-y0)*.024,f'{scale/1000:g} km',ha='center',fontsize=10,zorder=21)
    ax.annotate('N',xy=(.965,.955),xytext=(.965,.875),xycoords='axes fraction',ha='center',fontsize=10,
                arrowprops=dict(arrowstyle='-|>',lw=1.1,color=INK))

def borders(ax):
    districts.boundary.plot(ax=ax,color='#667F85',linewidth=.8,zorder=7)
    for name,row in districts.iterrows():
        p=row.geometry.representative_point()
        ax.text(p.x,p.y,name,ha='center',va='center',fontsize=12,weight='bold',zorder=12,
                path_effects=[patheffects.withStroke(linewidth=3.5,foreground='white')])

def save(fig,stem):
    fig.savefig(O/f'{stem}.png',dpi=300,bbox_inches='tight',pad_inches=.07)
    fig.savefig(O/f'{stem}.svg',bbox_inches='tight',pad_inches=.07)
    plt.close(fig)

# Map 1: population and collection points.
fig=plt.figure(figsize=(10,9));ax=fig.add_axes([.02,.10,.96,.88])
admin.plot(ax=ax,color=LAND,edgecolor=LINE,linewidth=.25)
pop_colors=['#FBE2A2','#F5BA57','#E88335','#BB4D1D']
pop_class=np.digitize(grid.population_reported,[50,200,500])
for k,c in enumerate(pop_colors):grid.loc[pop_class==k].plot(ax=ax,color=c,edgecolor='none',zorder=2)
existing.plot(ax=ax,facecolor='none',edgecolor='#305C8B',markersize=14,linewidth=.7,zorder=5)
borders(ax);limits(ax)
fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(pop_colors,['1–49명','50–199명','200–499명','500명 이상'])]
           +[Line2D([],[],linestyle='',marker='o',mfc='none',mec='#305C8B',label='기존 수거지')],
           loc='lower center',bbox_to_anchor=(.5,.02),frameon=False,ncol=5,fontsize=11)
save(fig,'01_existing_population')

# Map 2: dong coverage. The gu chart is exported separately.
fig=plt.figure(figsize=(10,9));ax=fig.add_axes([.02,.10,.96,.88])
colors=['#B34860','#D89493','#F0CFAC','#9BC8C0','#347F80']
admin.plot(column='baseline_pct',ax=ax,cmap=ListedColormap(colors),
           norm=BoundaryNorm([0,25,50,70,85,100.0001],5),edgecolor='white',linewidth=.5)
borders(ax);limits(ax)
fig.legend(handles=[Patch(facecolor=c,label=t) for c,t in zip(colors,
           ['25% 미만','25–50% 미만','50–70% 미만','70–85% 미만','85% 이상'])],
           loc='lower center',bbox_to_anchor=(.5,.02),frameon=False,ncol=5,fontsize=10.5)
save(fig,'02_dong_coverage')

def draw_classes(ax,grids):
    for mask,col in [(grids.existing_covered.astype(bool),GREY),
                     (~grids.existing_covered.astype(bool)&grids.screened_union_covered.astype(bool),TEAL),
                     (~grids.screened_union_covered.astype(bool),PURPLE)]:
        subset=grids.loc[mask]
        if len(subset):subset.plot(ax=ax,color=col,edgecolor='none',zorder=2)

# Map 3: city view and local insets, without summary numbers or captions.
fig=plt.figure(figsize=(12,9));ax=fig.add_axes([.01,.15,.65,.82])
admin.plot(ax=ax,color=LAND,edgecolor=LINE,linewidth=.25)
draw_classes(ax,grid);borders(ax);limits(ax)
for dong,rect in [('가포동',[.71,.56,.28,.40]),('진전면',[.71,.12,.28,.39])]:
    ar=admin.loc[(admin.district=='마산합포구')&(admin.dong==dong)]
    bounds=ar.total_bounds;pad=max(bounds[2]-bounds[0],bounds[3]-bounds[1])*.03
    sub=grid.cx[bounds[0]-pad:bounds[2]+pad,bounds[1]-pad:bounds[3]+pad]
    xx=fig.add_axes(rect);admin.plot(ax=xx,color=LAND,edgecolor=LINE,linewidth=.3)
    draw_classes(xx,sub)
    near=candidates.cx[bounds[0]-pad:bounds[2]+pad,bounds[1]-pad:bounds[3]+pad]
    if len(near):near.plot(ax=xx,facecolor='none',edgecolor=INK,markersize=19,linewidth=.75,zorder=7)
    ar.boundary.plot(ax=xx,color=INK,linewidth=1,zorder=6);limits(xx,bounds,1000)
    xx.set_title(dong,loc='left',fontsize=12,pad=6)
    p=ar.iloc[0].geometry.representative_point()
    ax.plot(p.x,p.y,'o',mfc='white',mec=INK,ms=4,zorder=20)
    ax.annotate(dong,(p.x,p.y),xytext=(7,-13 if dong=='가포동' else 9),textcoords='offset points',fontsize=11,zorder=21,
                path_effects=[patheffects.withStroke(linewidth=3,foreground='white')])
fig.legend(handles=[Patch(facecolor=GREY,label='기존망 커버'),Patch(facecolor=TEAL,label='편의점으로 보완 가능'),
                    Patch(facecolor=PURPLE,label='남는 공백'),
                    Line2D([],[],linestyle='',marker='o',mfc='none',mec=INK,label='편의점 후보 (확대도)')],
           loc='lower center',bbox_to_anchor=(.5,.015),ncol=4,frameon=False,fontsize=11)
save(fig,'03_fillable_gaps')

# Chart 4: gap population. Keep the axis, legend and values only.
top=admin.nlargest(10,'existing_gap_population').iloc[::-1]
fig,ax=plt.subplots(figsize=(10.5,6.3));fig.subplots_adjust(left=.21,right=.98,bottom=.13,top=.89)
y=np.arange(len(top));fill=top.screened_fillable_gap_population/1000;outside=top.screened_outside_candidate_network_population/1000
ax.barh(y,fill,color=TEAL,height=.65,label='편의점으로 보완 가능')
ax.barh(y,outside,left=fill,color=GREY,height=.65,label='남는 공백')
ax.set_yticks(y,[f'{r.district} {r.dong}' for r in top.itertuples()])
for yy,total in zip(y,top.existing_gap_population):ax.text(total/1000+.20,yy,f'{total:,.0f}명',va='center',fontsize=10.5)
ax.set(xlim=(0,27),xlabel='기존 수거지 350m 밖의 인구 (천 명)')
ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,ncol=2,fontsize=11)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for edge in ['top','right']:ax.spines[edge].set_visible(False)
save(fig,'04_gap_population')

# Chart 5: gu comparison, as its own figure.
order=['의창구','마산합포구','진해구','성산구','마산회원구']
vals=[summary['baseline']['districts'][g] for g in order]
fig,ax=plt.subplots(figsize=(7.0,3.8));fig.subplots_adjust(left=.2,right=.97,bottom=.18,top=.95)
ax.barh(order,vals,color=TEAL,height=.58);ax.invert_yaxis()
for y,v in enumerate(vals):ax.text(v+1.1,y,f'{v:.2f}%',va='center',fontsize=11)
ax.set(xlim=(0,100),xlabel='기존 수거망 커버율 (%)')
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
for edge in ['top','right','left']:ax.spines[edge].set_visible(False)
ax.tick_params(axis='y',length=0)
save(fig,'05_gu_coverage')

(O/'그림_캡션과_출처.txt').write_text('''그림 1. 기존 수거지와 100m 격자 인구 분포.
그림 2. 기존 수거망의 읍면동별 커버율.
그림 3. 편의점으로 보완 가능한 공백과 모든 후보 활용 후에도 남는 공백.
그림 4. 기존 접근 공백 인구가 많은 10개 읍면동의 보완 가능성.
그림 5. 기존 수거망의 구별 커버율.

공통 기준: 기존 수거지 439개, 주 분석 편의점 후보 802개, 원격자 중심점에서 직선거리 350m. 보완 가능성은 후보 802곳을 모두 활용한 조건이다.
지도 1 색상: 100m 원격자의 공개 인구. 지도 3 색상: 양의 공개 인구가 있는 격자의 접근성 분류. 밝은 배경에는 비공개·결측 영역도 포함된다.
인구 합계는 행정경계 중첩 면적에 따라 배분한 값을 사용한다.

자료: 전국폐의약품수거함표준데이터(2026.1.28), 소상공인시장진흥공단 상가(상권)정보(2026.6.30), SGIS 인구(2024)·경계(2025)를 활용한 본 연구 분석.
https://www.data.go.kr/data/15129445/standard.do
https://www.data.go.kr/data/15083033/fileData.do
https://sgis.mods.go.kr/view/pss/openDataIntrcn
''')
with zipfile.ZipFile(O.parent/'4장_그림만_PNG_SVG.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(O.iterdir()):
        if f.is_file():z.write(f,arcname=f.name)
print('Saved 5 figures, PNG + SVG, with captions in a separate file:',O)
