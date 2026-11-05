"""기존 수거망과 편의점 보완안의 예비 결과를 그림과 보고서로 만든다."""
from pathlib import Path
import json
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parent
R=P/'results/existing_network'
F=R/'figures';F.mkdir(exist_ok=True)
j=json.loads((R/'comparison.json').read_text())
data=pd.read_csv(R/'coverage_by_demand.csv')
grids=gpd.read_file(P/'data/processed/changwon_spatial.gpkg',layer='demand_100m').to_crs(5179)
extra=['demand_id','category','existing_covered','added_p60','combined_p60']
grids=grids.loc[grids.population>0].merge(data[extra],on='demand_id',validate='one_to_one')
admin=gpd.read_file(P/'data/processed/changwon_spatial.gpkg',layer='admin_dongs').to_crs(5179)
districts=admin.dissolve(by='district')
selected=pd.read_csv(R/'selected_additional_stores.csv')
selected=selected.loc[selected.max_new==60]
points=gpd.GeoDataFrame(selected,geometry=gpd.points_from_xy(selected.lon,selected.lat),crs=4326).to_crs(5179)
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family':'Apple SD Gothic Neo','axes.unicode_minus':False,'font.size':12,
                    'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
colors={'both':'#317B83','existing_only':'#8C73AA','convenience_only':'#E49634','neither':'#AB514B'}
labels={'both':'두 망 모두 접근 가능','existing_only':'기존 수거망만 접근 가능',
        'convenience_only':'편의점 후보망만 접근 가능','neither':'두 망 모두 접근 어려움'}
oldcol='#317B83';newcol='#E49634';gapcol='#B6BDC1'
base=j['existing_baseline']['population_pct'];s=j['scenarios'][-1];total=s['total']['population_pct']
footer=('SGIS 인구 2024·경계 2025 / 등록 수거지 2026.1 / 편의점 후보 2026.6\n'
        '100m 격자 중심점·직선 350m·운영시간 미반영 / 공개 인구 기준 / 빈 공간은 인구 0명을 뜻하지 않음')

def background(ax):
    admin.plot(ax=ax,facecolor='#F4F5F4',edgecolor='#DEE2E2',linewidth=.35,zorder=1)
    ax.set_axis_off()

def finish_map(ax,labelsize=12):
    districts.boundary.plot(ax=ax,color='#879598',linewidth=.75,zorder=4)
    for name,row in districts.iterrows():
        pt=row.geometry.representative_point()
        ax.text(pt.x,pt.y,name,ha='center',va='center',fontsize=labelsize,color='#233943',
                path_effects=[patheffects.withStroke(linewidth=3,foreground='white')],zorder=7)
    xmin,ymin,xmax,ymax=admin.total_bounds
    ax.set_xlim(xmin-1200,xmax+1200);ax.set_ylim(ymin-1200,ymax+1200)
    sx=xmax-6500;sy=ymin+200
    ax.plot([sx,sx+5000],[sy,sy],color='#51616A',lw=1.8,zorder=8)
    ax.text(sx+2500,sy+500,'5 km',fontsize=9,ha='center',color='#51616A')

fig=plt.figure(figsize=(10.5,10.5))
gs=fig.add_gridspec(2,1,height_ratios=[12,1],left=.035,right=.97,bottom=.17,top=.86,hspace=.06)
ax=fig.add_subplot(gs[0]);background(ax)
for key,color in colors.items():
    sub=grids.loc[grids.category==key]
    if len(sub):sub.plot(ax=ax,color=color,edgecolor='none',zorder=3)
finish_map(ax)
fig.text(.055,.953,'기존 수거망과 편의점 후보망의 보완 관계',fontsize=21,fontweight='bold',color='#213D45')
fig.text(.055,.913,'등록 수거지 439개 좌표와 확보한 편의점 후보 863개 좌표를 모두 이용한다고 가정',fontsize=12,color='#56656C')
bar=fig.add_subplot(gs[1]);left=0
for key,color in colors.items():
    v=j['partition'][key]['population_pct'];bar.barh(0,v,left=left,color=color,height=.5)
    left+=v
bar.set_xlim(0,100);bar.set_ylim(-.45,.45);bar.set_axis_off()
handles=[Patch(facecolor=colors[k],label=f"{labels[k]}  {j['partition'][k]['population_pct']:.2f}%") for k in colors]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.082),ncol=2,frameon=False,fontsize=11)
fig.text(.055,.055,footer+'\n색칠한 면적 비율과 인구 비율은 다름 / 863개 후보는 전체 영업 편의점 명부로 검증되지 않음',fontsize=9.7,color='#607078',va='top')
fig.savefig(F/'01_두_수거망의_보완관계.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(14,8.9))
fig.subplots_adjust(left=.025,right=.985,bottom=.19,top=.79,wspace=.06)
for ax in axes:
    background(ax);grids.plot(ax=ax,color=gapcol,edgecolor='none',zorder=2)
    grids.loc[grids.existing_covered].plot(ax=ax,color=oldcol,edgecolor='none',zorder=3)
grids.loc[grids.added_p60].plot(ax=axes[1],color=newcol,edgecolor='none',zorder=3)
points.plot(ax=axes[1],marker='o',facecolor='none',edgecolor='#9B5308',markersize=20,linewidth=.8,zorder=5)
for ax in axes:finish_map(ax,11)
fig.text(.045,.845,f'기존 등록 수거망\n공개 인구 {base:.2f}%',fontsize=16,va='top',color='#213D45')
fig.text(.535,.845,f'기존망 + 보완 편의점 60곳\n공개 인구 {total:.2f}%',fontsize=16,va='top',color='#213D45')
fig.text(.045,.946,'편의점 60곳을 어디에 더하면 접근 공백을 줄일 수 있을까?',fontsize=22,fontweight='bold',color='#213D45')
fig.text(.045,.897,f"기존망에서 닿지 않던 공개 인구 약 {s['additional']['population']/10000:.1f}만 명이 추가 접근 범위에 포함  (+{s['additional']['population_pct']:.2f}%p)",fontsize=13,color='#56656C')
fig.legend(handles=[Patch(facecolor=oldcol,label='기존망 접근 영역'),Patch(facecolor=newcol,label='새로 접근 가능한 영역'),
                    Patch(facecolor=gapcol,label='남는 접근 공백'),Line2D([],[],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='#9B5308',label='보완 편의점 60곳')],
           loc='lower center',bbox_to_anchor=(.5,.11),ncol=4,frameon=False,fontsize=11)
fig.text(.045,.077,footer+'\n기존 수거지는 모두 수거 가능, 편의점은 모두 설치 가능 가정 / 60곳은 비교용 설치 수이며 권고 수량이 아님',fontsize=10,color='#607078',va='top')
fig.savefig(F/'02_기존망과_60곳_보완안.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(10.7,6.5))
fig.subplots_adjust(left=.09,right=.94,top=.76,bottom=.20)
xx=[0]+[s['max_new'] for s in j['scenarios']]
yy=[base]+[s['total']['population_pct'] for s in j['scenarios']]
zz=[base]+[s['existing_plus_convenience_only_plan']['population_pct'] for s in j['scenarios']]
ax.plot(xx,yy,'o-',color=oldcol,lw=2.8,ms=7,label='기존망의 접근 공백을 고려해 선정')
ax.plot(xx,zz,'s--',color='#A17E59',lw=2,ms=6,label='편의점 단독 기준으로 선정한 곳을 추가')
for x,y in zip(xx,yy):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,10),textcoords='offset points',ha='center',color='#244E57',fontsize=12)
for x,y in zip(xx[1:],zz[1:]):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,-20),textcoords='offset points',ha='center',color='#765836',fontsize=11)
ax.set(xlim=(-3,65),ylim=(60,96),xticks=xx,yticks=[60,70,80,90],xlabel='추가하는 편의점 수 (곳)',ylabel='기존망과 합친 공개 인구 커버율 (%)')
ax.grid(axis='y',alpha=.17);ax.legend(loc='upper left',frameon=False,fontsize=11)
fig.text(.075,.94,'같은 60곳을 더해도, 선정 목적에 따라 효과가 달라집니다',fontsize=20,fontweight='bold',color='#213D45')
fig.text(.075,.873,f"기존망을 고려한 보완안은 기존 편의점 단독 선정안보다 {total-s['existing_plus_convenience_only_plan']['population_pct']:.2f}%p 높음",fontsize=13,color='#56656C')
fig.text(.075,.07,'두 선 모두 기존 수거망을 유지한 결과 / 각 설치 수는 독립적으로 최적화한 집합\n100m 격자·직선 350m·시간 미반영 / 수거량·야간 이용 효과·같은 예산에서의 비교를 뜻하지 않음',fontsize=10,color='#607078',va='bottom')
fig.savefig(F/'03_설치규모와_선정목적_비교.png',dpi=180);plt.close(fig)

rows='\n'.join(f"| {r['max_new']}곳 | {r['total']['population_pct']:.2f}% | +{r['additional']['population_pct']:.2f}%p | {r['existing_plus_convenience_only_plan']['population_pct']:.2f}% |" for r in j['scenarios'])
md=f'''# 기존 수거망을 고려한 편의점 보완안 — 예비 공간 분석

2026-09-28. 기존 편의점 전용 분석을 보존하고, 기존 등록 수거지를 유지하는 보완안을 추가했다.

## 연구 질문

기존 폐의약품 수거망에서 접근이 어려운 지역을 편의점으로 얼마나 보완할 수 있는가? 같은 설치 수에서 편의점만을 기준으로 고른 입지와 기존망의 공백을 고려한 입지가 어떻게 달라지는가?

## 자료와 전제

- [창원 등록 수거함 자료](https://www.data.go.kr/data/15129445/standard.do): 2026-01-28 기준 472건, 동일 좌표 통합 후 439개 위치. 모두 공식 창원 경계 안에 위치한다. 좌표 일치는 실제 시설 중복을 뜻하지 않으므로 원본과 통합 내역을 보존했다.
- 명칭상 약국 438건, 보건 관련 20건, 행정 관련 13건, 기타 1건이다. 명칭에 따른 잠정 분류이며 시설 유형을 공적으로 검증한 결과는 아니다.
- 편의점은 2026-06 상가정보에서 확보한 863개 후보 좌표. 전체 영업 점포 명부의 완전성은 미검증이다. 다른 업종에 등록된 후보 등은 별도 재점검 대상으로 남겨 두었다.
- SGIS 2024년 100m 공개 인구·가구, 2025년 행정경계. 기존 분석과 같은 경계 면적 배분 및 결측 처리 기준을 유지했다. 인구의 분모는 약 {j['total_published_population']/10000:.2f}만 명에 해당하는 공개 인구 배분값이다.
- 등록 수거지는 모두 수거 가능하고, 후보 편의점은 모두 설치 가능하다고 가정한다. 직선 350m·격자 중심점 기준이며 보행 경로, 영업·수거 시간, 실제 참여, 용량·비용은 미반영이다.

## 그림 1: 두 망의 공간적 보완 관계

![두 망의 보완 관계](figures/01_두_수거망의_보완관계.png)

기존망만 접근 가능한 인구 {j['partition']['existing_only']['population_pct']:.2f}%, 확보한 편의점 후보망만 접근 가능한 인구 {j['partition']['convenience_only']['population_pct']:.2f}%, 두 망 모두 접근 가능한 인구 {j['partition']['both']['population_pct']:.2f}%, 어느 망에도 접근하기 어려운 인구 {j['partition']['neither']['population_pct']:.2f}%다. 후보망 전체를 사용하는 이 비교는 공간적 역할을 진단하며 같은 시설 수나 같은 비용의 성과 비교가 아니다.

## 그림 2: 기존망에 60곳을 보완했을 때

![기존망과 보완안](figures/02_기존망과_60곳_보완안.png)

등록 수거망의 잠재 커버율은 {base:.2f}%. 기존에 커버되지 않던 인구를 최대화하도록 편의점 60곳을 더하면 {total:.2f}%로 증가한다. 약 {s['additional']['population']/10000:.1f}만 명에 해당하는 공개 인구가 추가 범위에 들어간다. 이 결과는 실제 반환 인원이나 회수량 증가를 뜻하지 않는다. 60곳은 비교용 조건이다.

## 그림 3: 설치 수와 선정 목적에 따른 차이

![설치 수 비교](figures/03_설치규모와_선정목적_비교.png)

| 추가 편의점 수 | 공백을 고려한 보완안의 총 커버율 | 기존망 대비 증가 | 편의점 단독 선정안을 기존망에 합친 커버율 |
|---|---:|---:|---:|
{rows}

세 보완안은 이번 입력과 조건에서 최적성 간격 0으로 계산됐다. 현실의 최적 설치 위치가 검증됐다는 의미는 아니다. 각 설치 수의 해는 독립된 집합으로, 20곳을 먼저 설치한 뒤 20곳씩 붙이는 단계별 계획과 같지 않다. 60곳 보완안과 앞선 편의점 단독 60곳 선정안의 공통 점포는 {s['shared_sites_with_convenience_only_plan']}곳이다.

## 최적화와 검증

기존 수거지를 고정하고, 이미 접근 가능한 격자의 인구는 추가 효과를 계산할 때 가중치 0으로 둔다. 남은 인구를 추가로 커버하는 편의점 조합을 선택한다. 평가에서는 기존망과 신규망의 합집합을 사용해 중복 인원을 세지 않는다.

모든 수요점에서 기존·편의점 후보까지의 최근접 거리를 별도의 전체 거리 계산으로 검산했다. 네 공간 유형의 완전 분할, 추가 인구와 총인구의 합집합 관계, 선정 점포의 거리 판정과 기존 편의점 단독 결과의 재현을 확인했다. 비교 반경 200m·350m·500m에서 기존망과 전체 후보망 합집합의 수치도 comparison.json에 저장했다.

## 원래 아이디어와의 연결

이 그림들은 ‘편의점이 기존 수거망의 공간적 공백을 얼마나 채울 수 있는가’에 답한다. 두 망이 겹치는 지역에서도 편의점이 다른 시간대의 배출 기회를 제공할 가능성은 있다. 이를 판단하려면 기존 수거지와 편의점의 실제 수거 시간을 확인해야 한다. 공간적 중복만으로 편의점의 효용이 없다고 단정하지 않으며 야간 효과가 입증됐다고 표현하지도 않는다.

현재망 진단 → 편의점 보완 위치 선정 → 새로 접근 가능한 인구와 남는 공백 → 야간 운영·참여 조건을 포함한 후속 운영 제안 순서로 공모전 이야기를 구성할 수 있다.
'''
(R/'기존수거망_보완분석.md').write_text(md)
parts=[]
for name in ['01_두_수거망의_보완관계.png','02_기존망과_60곳_보완안.png','03_설치규모와_선정목적_비교.png']:
    b64=base64.b64encode((F/name).read_bytes()).decode()
    parts.append(f'<img src="data:image/png;base64,{b64}" alt="{name}">')
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>기존 수거망과 편의점 보완안</title><style>body{max-width:1100px;margin:40px auto;padding:0 20px;color:#263f48;font:17px/1.7 -apple-system,BlinkMacSystemFont,sans-serif}img{width:100%;height:auto}h1{font-size:30px}p{max-width:950px}.note{color:#596b73}a{color:#176b75}</style><h1>기존 수거망과 편의점의 보완 관계</h1>'''
html+=f'<p>직선 350m 기준 등록 수거망의 잠재 커버율은 <b>{base:.2f}%</b>입니다. 기존 공백을 고려해 편의점 60곳을 더하면 <b>{total:.2f}%</b>로 증가합니다. 실제 이용률이나 회수량을 계산한 결과는 아닙니다.</p>'
html+='<p class="note">등록 수거지 472건·439개 좌표가 모두 수거 가능하고, 확보한 편의점 후보 863곳이 모두 설치 가능하다고 가정했습니다. 전체 편의점 명부의 완전성, 실제 보행 경로, 운영시간과 협약은 미검증입니다. 60곳은 비교 조건입니다.</p>'
html+=''.join(parts)
html+='<p>기존망에서 이미 커버되는 인구를 중복해서 추가 효과로 세지 않았습니다. 20·40·60곳 보완안 모두 이번 모형에서 최적성 간격 0으로 계산됐습니다. 공간적으로 겹치는 지역의 야간 수거 효과는 별도 운영시간 자료가 필요합니다.</p><p><a href="기존수거망_보완분석.md">자료·방법·한계 상세</a> · <a href="selected_additional_stores.csv">추가 후보 목록</a> · <a href="https://www.data.go.kr/data/15129445/standard.do">등록 수거함 자료 출처</a></p></html>'
(R/'기존수거망_보완분석.html').write_text(html)
print('saved figures and report',R)
