"""기존 수거망 보완 분석을 오프라인 보고서, 그림, 검토 명부로 정리한다."""
from pathlib import Path
import base64
import html
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

P = Path(__file__).resolve().parent
R = P / 'results/supplement_development'
F = R / 'figures'
F.mkdir(exist_ok=True)
j = json.loads((R / 'development.json').read_text())
sc = {s['id']: s for s in j['scenarios']}
base = j['baseline']
eff = sc['screened_efficiency_60']['metrics']
fair = sc['screened_fair_60']['metrics']
dg = pd.read_csv(R / 'dong_gap_diagnostics.csv')
selected = pd.read_csv(R / 'selected_stores_all_scenarios.csv').fillna('')
risk = pd.read_csv(R / 'store_contributions_and_replacements.csv').fillna('')
fr = risk.loc[risk.plan == 'screened_fair_60'].sort_values('exclusive_population', ascending=False)
raw = json.loads((P / 'data/raw/changwon_existing_bins_20260928.json').read_text())
audit = {'record_count': len(raw), 'field_nonempty_counts': {
    k: sum(str(x.get(k, '')).strip() != '' for x in raw)
    for k in ['INSTL_PLC_NM', 'LCTN_ROAD_NM', 'LAT', 'LOT', 'ACTL_PSTN', 'MNG_INST_TELNO', 'CRTR_YMD']},
    'raw_fields': list(raw[0]),
    'interpretation': '확보한 공개 파일의 기재 여부. 다른 안내 채널의 정보 존재 여부나 시민 인지도를 뜻하지 않는다.'}
(R / 'existing_information_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2))

font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family': 'Apple SD Gothic Neo', 'axes.unicode_minus': False,
                     'font.size': 12, 'figure.facecolor': 'white', 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.labelcolor': '#31444B', 'text.color': '#203C43',
                     'xtick.color': '#566971', 'ytick.color': '#31444B', 'savefig.facecolor': 'white'})
TEAL, ORANGE, GREY, RED = '#237F85', '#D68B2D', '#94A6AC', '#A85551'
INK, LIGHT = '#203C43', '#F3F5F5'
common = 'SGIS 공개 인구 2024·경계 2025 / 기존 수거지 2026.1·편의점 후보 2026.6'
assumption = '100m 격자 중심점·직선 350m / 기존 439개 좌표 유지·검토 후보 802곳 / 실제 수거 시간·참여 미확인'
files = []


def heading(fig, title, subtitle):
    fig.text(.055, .95, title, fontsize=21, fontweight='bold', va='top')
    fig.text(.055, .889, subtitle, fontsize=12, color='#5A6D74', va='top')


def save(fig, name, note=assumption):
    fig.text(.055, .069, common + '\n' + note, fontsize=9.4, color='#62737A', va='top')
    fig.savefig(F / name, dpi=175)
    plt.close(fig)
    files.append(name)


# 1. 기존망 공백이 큰 동의 보완 가능성과 편의점망의 한계.
top = dg.nlargest(10, 'existing_gap_population').iloc[::-1]
fig, ax = plt.subplots(figsize=(12.2, 7.5))
fig.subplots_adjust(left=.19, right=.91, bottom=.2, top=.77)
y = np.arange(len(top))
fillable = top.screened_fillable_gap_population / 1000
outside = top.screened_outside_candidate_network_population / 1000
ax.barh(y, fillable, color=TEAL, height=.68, label='802곳 후보망으로 보완 가능한 공백')
ax.barh(y, outside, left=fillable, color=GREY, height=.68, label='802곳을 모두 더해도 남는 공백')
for yy, total in zip(y, top.existing_gap_population):
    ax.text(total / 1000 + .28, yy, f'{total:,.0f}명', va='center', fontsize=10.5)
ax.set_yticks(y, [f'{r.district} {r.dong}' for r in top.itertuples()])
ax.set_xlim(0, 27)
ax.set_xlabel('기존 수거지 350m 밖의 공개 인구 (천 명)')
ax.grid(axis='x', alpha=.14)
ax.set_axisbelow(True)
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.14), frameon=False, ncol=2, fontsize=11)
heading(fig, '어디의 접근 공백부터 살펴봐야 할까?',
        '기존 수거망 밖 인구가 많은 10개 읍면동 · 북면과 내서읍은 각각 약 2만 명 이상의 공백')
save(fig, '01_기존망의_공백과_보완가능성.png', assumption + '\n편의점 60곳 선정 결과가 아닌 전체 802개 후보망의 공간적 가능성. 인구는 경계 면적 배분값을 반올림.')

# 2. 전체 효율과 구 단위 형평성.
district_order = ['의창구', '성산구', '마산합포구', '마산회원구', '진해구']
fig = plt.figure(figsize=(12.2, 7.3))
ax = fig.add_axes([.12, .23, .58, .5])
for yy, g in enumerate(district_order):
    ax.plot([base['districts'][g], max(eff['districts'][g], fair['districts'][g])], [yy, yy], color='#D8E0E1', lw=2)
ax.scatter([base['districts'][g] for g in district_order], range(5), c=GREY, s=85, label='기존망')
ax.scatter([eff['districts'][g] for g in district_order], np.arange(5)-.12, c=TEAL, marker='s', s=85, label='효율안 60곳')
ax.scatter([fair['districts'][g] for g in district_order], np.arange(5)+.12, c=ORANGE, marker='D', s=85, label='형평안 60곳')
for yy, g in enumerate(district_order):
    ax.text(base['districts'][g]-.8, yy, f"{base['districts'][g]:.1f}", ha='right', va='center', fontsize=10, color='#6C818A')
    ax.text(eff['districts'][g]+.55, yy-.17, f"{eff['districts'][g]:.1f}", va='center', fontsize=10, color=TEAL)
    ax.text(fair['districts'][g]+.55, yy+.18, f"{fair['districts'][g]:.1f}", va='center', fontsize=10, color='#9B621E')
ax.set(yticks=range(5), yticklabels=district_order, xlim=(59,96), ylim=(-.6,4.6), xlabel='구별 공개 인구 커버율 (%)')
ax.invert_yaxis()
ax.grid(axis='x', alpha=.14)
ax.legend(loc='upper left', bbox_to_anchor=(0, 1.18), ncol=3, frameon=False, fontsize=11)
side = fig.add_axes([.755, .23, .23, .53]); side.set_axis_off()
side.text(0, .93, '전체 인구 커버율', fontsize=13)
side.text(0, .80, f"효율  {eff['population_pct']:.2f}%", color=TEAL, fontsize=20, fontweight='bold')
side.text(0, .68, f"형평  {fair['population_pct']:.2f}%", color='#A76B22', fontsize=20, fontweight='bold')
side.text(0, .45, '가장 낮은 구의 커버율', fontsize=13)
side.text(0, .32, f"효율  {eff['minimum_district_pct']:.2f}%", color=TEAL, fontsize=20, fontweight='bold')
side.text(0, .20, f"형평  {fair['minimum_district_pct']:.2f}%", color='#A76B22', fontsize=20, fontweight='bold')
heading(fig, '60곳을 더할 때, 전체 인구와 지역 격차 사이의 선택',
        '형평안은 전체 커버율 1.16%p를 낮추는 대신 최저 구 커버율을 3.54%p 높임')
save(fig, '02_효율과_형평성의_비교.png', assumption + '\n형평안은 가장 낮은 구 커버율을 최대화한 뒤 전체 커버율을 최대화. 읍면동별 형평성을 보장하지 않음.')

# 3. 지리적 변화. 전체 시 경계와 인구 격자를 사용한다.
d = pd.read_csv(R / 'coverage_by_demand.csv')
grid = gpd.read_file(P / 'data/processed/changwon_spatial.gpkg', layer='demand_100m').to_crs(5179)
grid = grid.loc[grid.population > 0].merge(d[['demand_id', 'existing_covered', 'screened_fair_60_covered']], on='demand_id', validate='one_to_one')
admin = gpd.read_file(P / 'data/processed/changwon_spatial.gpkg', layer='admin_dongs').to_crs(5179)
districts = admin.dissolve(by='district')
fs = selected.loc[selected.scenario == 'screened_fair_60']
points = gpd.GeoDataFrame(fs, geometry=gpd.points_from_xy(fs.lon, fs.lat), crs=4326).to_crs(5179)
fig, axes = plt.subplots(1, 2, figsize=(13.8, 9.4))
fig.subplots_adjust(left=.025, right=.985, bottom=.20, top=.77, wspace=.03)
for ax in axes:
    admin.plot(ax=ax, facecolor=LIGHT, edgecolor='#DEE3E4', linewidth=.3)
    grid.plot(ax=ax, color='#AAB5BA', edgecolor='none')
    grid.loc[grid.existing_covered].plot(ax=ax, color=TEAL, edgecolor='none')
grid.loc[grid.screened_fair_60_covered & ~grid.existing_covered].plot(ax=axes[1], color=ORANGE, edgecolor='none')
points.plot(ax=axes[1], facecolor='none', edgecolor='#904D10', markersize=23, linewidth=.8)
for ax in axes:
    districts.boundary.plot(ax=ax, color='#879A9E', linewidth=.7)
    for name, row in districts.iterrows():
        pt = row.geometry.representative_point()
        ax.text(pt.x, pt.y, name, ha='center', fontsize=11, path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
    xmin,ymin,xmax,ymax = admin.total_bounds
    ax.set(xlim=(xmin-1200,xmax+1200), ylim=(ymin-1000,ymax+1000))
    ax.set_axis_off()
    sx,sy=xmax-6500,ymin+200
    ax.plot([sx,sx+5000],[sy,sy],color='#62737A',lw=1.7)
    ax.text(sx+2500,sy+500,'5 km',ha='center',fontsize=9)
heading(fig, '기존 수거망을 유지하고, 닿지 않던 생활권을 보완',
        f"형평안 60곳으로 약 {fair['added_population']/10000:.1f}만 명에 해당하는 공개 인구가 추가 접근 범위에 포함")
fig.text(.07,.831,f"기존 등록 수거망  {base['population_pct']:.2f}%",fontsize=16,fontweight='bold',va='top')
fig.text(.535,.831,f"기존망 + 형평안 60곳  {fair['population_pct']:.2f}%",fontsize=16,fontweight='bold',va='top')
fig.legend(handles=[Patch(facecolor=TEAL,label='기존망으로 접근 가능'),Patch(facecolor=ORANGE,label='새로 접근 가능'),
                    Patch(facecolor='#AAB5BA',label='남는 접근 공백'),Line2D([],[],linestyle='none',marker='o',markerfacecolor='none',markeredgecolor='#904D10',label='추가 검토 편의점')],
           loc='lower center',bbox_to_anchor=(.5,.12),ncol=4,frameon=False,fontsize=11)
save(fig, '03_형평안60곳_공간변화.png', assumption + '\n빈 공간은 인구 0명을 뜻하지 않음. 색칠한 면적 비율과 인구 비율은 다름. 편의점은 설치 확정지가 아님.')

# 4. 선정점의 역할: 중복되지 않는 접근 인구.
ft = fr.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(12.2, 7.4))
fig.subplots_adjust(left=.34, right=.93, bottom=.2, top=.77)
y=np.arange(len(ft))
ax.barh(y, ft.exclusive_population, color=ORANGE, height=.62)
for yy, n in zip(y,ft.exclusive_population): ax.text(n+100, yy, f'{n:,.0f}명', va='center', fontsize=11)
ax.set_yticks(y,[f'{r.name}\n{r.district} {r.dong}' for r in ft.itertuples()])
ax.tick_params(axis='y',labelsize=11)
ax.set_xlim(0,8500)
ax.set_xlabel('해당 점포가 빠지면 접근 범위에서 벗어나는 공개 인구 (명)')
ax.grid(axis='x',alpha=.14);ax.set_axisbelow(True)
heading(fig, '어떤 편의점이 특히 도움이 되는가?',
        '형평안에서 다른 수거지가 대신 맡지 못하는 인구가 큰 8곳 · 점포 1곳씩 제외해 계산')
save(fig, '04_점포별_고유기여.png', assumption + '\n60곳 조합 안에서의 고유 기여도. 수거량 예측·설치 순위가 아니며, 각 값의 합은 전체 추가 인구와 다름.')

# 5. 규모와 실제로 기존 선정점을 유지하는 단계 도입 비교.
fig, axes=plt.subplots(1,2,figsize=(12.8,7.0),gridspec_kw={'width_ratios':[1.55,1]})
fig.subplots_adjust(left=.075,right=.96,bottom=.22,top=.76,wspace=.27)
x=[0,20,40,60,80,100]
rates=[base['population_pct']]+[sc[f'screened_efficiency_{p}']['metrics']['population_pct'] for p in x[1:]]
axes[0].plot(x,rates,'o-',color=TEAL,lw=2.5,ms=7)
for xx,yy in zip(x,rates): axes[0].annotate(f'{yy:.2f}%',(xx,yy),xytext=(0,11),textcoords='offset points',ha='center',fontsize=11)
axes[0].axhline(j['screened_union_ceiling']['population_pct'],color=GREY,ls='--',lw=1)
axes[0].text(2,94.2,f"802곳을 모두 더한 범위  {j['screened_union_ceiling']['population_pct']:.2f}%",fontsize=10,color='#71848A')
axes[0].set(xlim=(-6,106),ylim=(66,98),xticks=x,xlabel='추가 편의점 수 (곳)',ylabel='기존망을 포함한 커버율 (%)')
axes[0].grid(axis='y',alpha=.14)
axes[1].set_axis_off()
for yy,label,pid in [(.95,'1단계 · 20곳','screened_efficiency_20'),(.66,'2단계 · 20곳 더해 총 40곳','screened_staged_40'),(.37,'3단계 · 20곳 더해 총 60곳','screened_staged_60')]:
    axes[1].text(0,yy,label,fontsize=13,va='top')
    axes[1].text(0,yy-.10,f"{sc[pid]['metrics']['population_pct']:.2f}%",fontsize=25,fontweight='bold',color=TEAL,va='top')
heading(fig,'설치 수가 늘수록, 20곳당 추가 효과는 줄어듭니다',
        '오른쪽 단계안은 앞서 고른 점포를 유지하고 확장 · 60곳 일괄 효율안과 차이는 0.0084%p')
save(fig,'05_설치규모와_단계도입.png',assumption+'\n왼쪽은 각 규모별 독립 최적화. 오른쪽은 효율 목적의 단계안으로, 앞선 형평안 60곳과 다른 배치. 비용 미반영.')

# 6. 주 가정인 반경 변화와 재선정 여부.
fig,axes=plt.subplots(1,2,figsize=(12.5,7.0))
fig.subplots_adjust(left=.075,right=.96,bottom=.23,top=.75,wspace=.25)
rad=[200,350,500]
ids=['screened_distance_200_60','screened_efficiency_60','screened_distance_500_60']
own=[sc[s]['metrics']['population_pct'] for s in ids]
common350=[sc[s]['evaluation_350m']['population_pct'] for s in ids]
for ax, vals, color, title in zip(axes,[own,common350],[GREY,TEAL],['선정에 쓴 반경으로 평가','모든 배치를 같은 350m로 평가']):
    ax.bar(range(3),vals,color=color,width=.56)
    for xx, yy in enumerate(vals):ax.text(xx,yy+2,f'{yy:.2f}%',ha='center',fontsize=12)
    ax.set(xticks=range(3),xticklabels=[f'{n}m 선정' for n in rad],ylim=(0,110),ylabel='기존망과 합친 커버율 (%)')
    ax.set_title(title,pad=15,fontsize=14)
    ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
heading(fig,'거리 가정을 바꾸면, 선정하는 편의점도 달라집니다',
        '350m 효율안과 같은 점포는 200m 선정안 25곳, 500m 선정안 23곳 · 각 안은 모두 60곳')
save(fig,'06_거리와_입지민감도.png','기존 439개 좌표 유지·검토 후보 802곳 / 거리별로 기존망과 신규망 모두 같은 기준 적용'+'\n직선거리 기준. 넓은 반경에서의 높은 커버율을 실제 보행 접근성 향상으로 해석하지 않음.')

# 검토 명부: 고유 기여도 순으로 정렬하되 설치 순위로 부르지 않는다.
list60=fs.merge(fr.drop(columns=['name','district','dong','flags']),on='candidate_id',validate='one_to_one')
list60=list60.sort_values('exclusive_population',ascending=False)
columns={'candidate_id':'후보ID','names':'점포명_원자료','district':'구','dong':'읍면동','addresses':'도로명주소',
         'lon':'경도','lat':'위도','exclusive_population':'고유기여_공개인구_명','standalone_added_population':'단독추가_공개인구_명',
         'flags':'추가확인사항','best_replacement_name':'한곳제외시_대안점포',
         'replacement_flags':'대안점포_추가확인사항','coverage_after_failure_pct':'제외후_전체커버율_pct',
         'coverage_after_replacement_pct':'대체후_전체커버율_pct','replacement_min_district_pct':'대체후_최저구커버율_pct',
         'fair_floor_preserved':'대체후_원래형평하한_유지'}
list60[list(columns)].rename(columns=columns).to_csv(R/'형평안_검토편의점60곳.csv',index=False,encoding='utf-8-sig')
effs=selected.loc[selected.scenario=='screened_efficiency_60'].merge(risk.loc[risk.plan=='screened_efficiency_60'].drop(columns=['name','district','dong','flags']),on='candidate_id',validate='one_to_one')
effs.sort_values('exclusive_population',ascending=False)[list(columns)].rename(columns=columns).to_csv(R/'효율안_검토편의점60곳.csv',index=False,encoding='utf-8-sig')
stageids={1:set(sc['screened_efficiency_20']['selected_ids'])}
stageids[2]=set(sc['screened_staged_40']['selected_ids'])-stageids[1]
stageids[3]=set(sc['screened_staged_60']['selected_ids'])-stageids[1]-stageids[2]
assert all(len(x)==20 for x in stageids.values())
stages=selected.loc[selected.scenario=='screened_staged_60'].copy()
stages['도입단계']=stages.candidate_id.map({cid:k for k,v in stageids.items() for cid in v})
stages.rename(columns=columns)[['도입단계','후보ID','점포명_원자료','구','읍면동','도로명주소','경도','위도','추가확인사항']].sort_values(['도입단계','구','읍면동']).to_csv(R/'효율목적_단계별20곳_도입검토.csv',index=False,encoding='utf-8-sig')


def table(headers, rows):
    return '<div class="tablewrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in rows)+'</tbody></table></div>'


def figure(index, caption):
    fn=files[index-1]
    uri=base64.b64encode((F/fn).read_bytes()).decode()
    return f'<figure><img src="data:image/png;base64,{uri}" alt="{html.escape(fn)}"><figcaption>{caption}</figcaption></figure>'


paper='https://link.springer.com/article/10.1007/s11081-026-10080-x'
minutes='https://clik.nanet.go.kr/minutes/viewer.do?DOCID=CLIKC1328922813173639&collection=minutes'
egan='https://pubmed.ncbi.nlm.nih.gov/37940096/'
sections=[]
sections.append(f'''<header><p class="eyebrow">창원 폐의약품 수거망 · 공공데이터 기반 보완 분석 · 2026.09.28</p>
<h1>기존 수거망에서 닿지 않는 곳,<br>어떤 편의점으로 채울 수 있을까?</h1>
<p class="lead">기존 수거지를 유지한 채, 접근 공백을 줄이는 편의점을 찾았습니다. <b>60곳 효율안은 87.40%, 형평안은 86.24%</b>의 공개 인구를 직선 350m 범위에 포함합니다. 형평안에서 가장 낮은 구의 커버율은 85.87%입니다.</p>
<div class="facts"><div><strong>439</strong><span>기존 등록 수거지 좌표</span></div><div><strong>802</strong><span>이번 주 분석의 후보 좌표</span></div><div><strong>24</strong><span>계산한 배치 시나리오</span></div><div><strong>240</strong><span>점포 1곳 제외·대체 사례</span></div></div>
<p class="note">이 수치는 실제 회수율이 아닌 잠재 접근성입니다. 60곳은 비교 조건이며 예산으로 결정한 권고 수량이 아닙니다. 기존 수거지의 수거 가능 상태와 후보의 영업·출입·참여를 가정했습니다.</p></header>
<nav><a href="#purpose">문제 정의</a><a href="#gaps">기존망의 공백</a><a href="#plans">효율과 형평성</a><a href="#stores">도움이 되는 점포</a><a href="#stages">단계 도입</a><a href="#limits">남는 공백</a><a href="#method">분석 근거</a><a href="#files">결과 파일</a></nav>''')
sections.append(f'''<section id="purpose"><p class="eyebrow">01 · 문제 정의</p><h2>수거지가 있어도, 생활 속에서 배출하기 쉬운가?</h2>
<p>출발점은 “따로 찾아가고 시간을 맞춰야 하는 배출을, 평소 들르는 편의점에서 할 수 없을까?”입니다. 창원에서는 이 생각을 <b>기존 수거망을 진단하고 필요한 곳에 편의점을 보완하는 문제</b>로 구체화했습니다.</p>
<p>확보한 등록 자료 472건 중 명칭에 ‘약국’이 들어간 기록은 438건입니다. 행정복지센터만을 기존망으로 보면 창원의 실제 등록망과 차이가 납니다. 약국·보건시설·행정시설을 모두 포함하고, 기존망이 이미 담당하는 인구는 신규 효과에서 빼야 합니다. 시설 유형은 명칭에 따른 잠정 분류입니다.</p>
<p>창원시의회 2025년 행정사무감사에서는 시민에게 배출 장소와 방법을 알리는 문제와 홍보 강화가 논의됐습니다. 이는 지역 문제의 근거이며 시민 전체의 인지도를 측정한 조사 결과는 아닙니다. <a href="{minutes}">회의록</a></p>
<div class="callout"><b>연구 질문</b><p>기존 폐의약품 수거망의 공간적 공백과 공개 안내정보의 부족을 진단하고, 제한된 추가 설치 수에서 편의점이 얼마나 많은 공백을 보완할 수 있는지, 그 효과가 지역별로 어떻게 나뉘는지 평가한다.</p></div>
<h3>이번에 확인한 것은 두 가지입니다</h3>
{table(['진단 대상','이번에 확인한 내용','해석'],[
['공간적 공백',f"기존망 커버율 {base['population_pct']:.2f}% · 범위 밖 약 31.1만 명",'새 입지로 줄일 공백을 계산할 수 있음'],
['공개 파일의 안내정보','도로명주소·좌표 472/472건, 상세 위치·관리기관 전화 0/472건, 운영시간 항목 없음','안내정보 보완 필요성을 제안할 근거. 시민 인지도나 실제 안내 상태는 별개'],
['시간과 일상 동선','실제 수거 시간·개인 이동 자료 없음','편의점의 야간·동선상 이점은 후속 검증할 가설']])}
<p class="note">예를 들어 건물 안 수거함 위치, 수거 가능한 시간, 수거 대상, 문의처를 함께 안내하는 개선안을 제안할 수 있습니다. 안내 개선에 따른 이용 증가율은 이번 자료로 계산하지 않았습니다.</p></section>''')
sections.append(f'''<section id="gaps"><p class="eyebrow">02 · 기존망의 공백</p><h2>북면·내서읍처럼 공백이 큰 생활권부터 구분합니다</h2>
<p>기존망의 범위 밖 인구 약 <b>31.1만 명</b> 중 약 <b>24.2만 명</b>은 확보한 802개 후보망으로 공간상 보완할 수 있습니다. 약 <b>6.8만 명</b>은 이 후보를 모두 이용해도 남습니다. ‘편의점을 추가하면 해결되는 곳’과 ‘다른 방식이 필요한 곳’을 나누는 이유입니다.</p>
{figure(1,'막대 전체가 기존 수거망의 공백입니다. 초록 부분은 후보 편의점의 범위에 들어가고, 회색은 후보망 밖에 남습니다. 모든 후보가 아니라 60곳만 고르면 실제로 줄어드는 공백은 더 작습니다.')}
<h3>후보 명부도 다시 점검했습니다</h3><p>편의점 업종으로 확보한 950개 기록은 동일 좌표를 통합하면 863곳입니다. 군부대·시설 내부 등 출입 확인 표시가 있는 59곳과 명칭상 편의점 여부를 재확인할 2곳을 제외한 <b>802곳을 주 분석</b>에 썼습니다. 두 명칭은 ‘지에스더프레시진해 남문점’, ‘노브랜드진해풍호점’입니다. <a href="https://www.gsretail.com/gsretail/ko/media/gsr-magazine-view?magazineId=8798517943753">GS리테일의 브랜드 설명</a>도 GS25와 GS더프레시를 구분합니다.</p>
<p class="note">이 제외는 보수적인 비교 조건입니다. 59곳을 모두 이용 불가라고 확정한 것이 아니며, 남은 802곳도 영업 중인 전체 편의점 목록으로 검증된 것은 아닙니다. 중복 기록·산업분류·층 정보 관련 표시를 검토 명부에 유지했습니다.</p></section>''')
planrows=[]
for label,m in [('기존망만',base),('효율안 · 60곳 추가',eff),('형평안 · 60곳 추가',fair)]:
    planrows.append([label,f"{m['population_pct']:.2f}%",f"+{m['added_pp']:.2f}%p",f"{m['minimum_district_pct']:.2f}%",f"{m['district_range_pp']:.2f}%p"])
sections.append(f'''<section id="plans"><p class="eyebrow">03 · 두 가지 선정 목적</p><h2>전체 효과를 높일지, 지역 간 격차를 줄일지 선택할 수 있습니다</h2>
{table(['배치','전체 인구 커버율','기존망 대비','가장 낮은 구','구 간 최대 격차'],planrows)}
<p><b>효율안</b>은 기존망에서 벗어난 인구를 가장 많이 추가로 포함합니다. <b>형평안</b>은 가장 낮은 구의 커버율을 먼저 최대화하고, 그 수준을 유지하는 조합 중 전체 인구를 가장 많이 포함합니다. 5개 구에 같은 수를 배분한 안은 아닙니다.</p>
{figure(2,'형평안은 전체 커버율 약 1.16%p, 공개 인구 약 1.17만 명의 차이를 감수하고 최저 구 커버율을 높입니다. 두 안에 공통으로 들어간 점포는 45곳, 나머지 15곳씩은 다릅니다.')}
<p>형평안의 배분은 의창구 19곳, 마산합포구 15곳, 진해구 10곳, 성산구 9곳, 마산회원구 7곳입니다. 기존망의 출발 수준과 새 후보의 위치가 함께 반영된 결과입니다.</p>
{figure(3,'주황색은 기존망에 없던 접근 범위를 편의점이 새로 제공하는 인구 격자입니다. 기존 범위와 겹치는 인구는 추가 효과로 중복 계산하지 않았습니다.')}
<div class="callout"><b>공모전에서의 제안</b><p>형평안을 지역 균형을 고려한 검토안으로 제시하고, 효율안을 비교 기준으로 함께 보여주는 구성이 좋습니다. 추가 설치가 도시 전체에 얼마나 도움이 되는지와 그 혜택이 어디로 가는지를 한 번에 설명할 수 있습니다. 어느 목적을 우선할지는 정책 판단입니다.</p></div></section>''')
toprows=[[r.name, f'{r.district} {r.dong}', f'{r.exclusive_population:,.0f}명',r.best_replacement_name,'유지' if r.fair_floor_preserved else '미달'] for r in fr.head(5).itertuples()]
sections.append(f'''<section id="stores"><p class="eyebrow">04 · 점포의 역할과 대체 가능성</p><h2>단순히 인구가 많은 곳보다, 다른 수거지가 맡지 못하는 곳</h2>
<p>주변 인구가 많아도 이미 기존 수거지가 잘 담당하면 추가 효과가 작을 수 있습니다. 선정 점포를 하나씩 빼 보고, 기존망과 나머지 59곳으로는 접근할 수 없게 되는 인구를 계산했습니다. 이 값을 <b>해당 조합에서의 고유 기여</b>로 표시했습니다.</p>
{figure(4,'고유 기여가 크면 배치의 효과가 해당 점포에 많이 의존한다는 뜻도 됩니다. 한 점포씩 제외하는 분석이며 여러 점포가 동시에 빠지는 상황의 결과는 아닙니다.')}
{table(['형평안의 검토 점포','지역','고유 기여','한 곳 제외 시 대안','구별 하한 유지'],toprows)}
<p>예를 들어 씨유창원반림현대점은 이 조합에서 7,590명에 해당하는 공개 인구를 고유하게 담당합니다. 이 점포가 빠지면 전체 커버율은 86.24%에서 85.49%로 내려갑니다. 지에스25창원일동점으로 바꾸면 86.18%까지 회복하지만, 원래의 최저 구 기준 85.87%에는 조금 못 미칩니다.</p>
<p>형평안 60건 중 <b>24건은 한 곳 교체로 원래 구별 하한을 유지</b>할 수 있었습니다. 나머지 36건은 같은 조건의 한 곳 교체만으로 하한을 지키지 못했습니다. 전체 커버율은 대체 후 각 사례에서 최소 86.01%였지만, 지역 균형까지 자동으로 회복되는 것은 아닙니다.</p>
<p class="note">대안은 확보한 후보 중 계산상 가능한 위치입니다. 실제 협약이나 대기 점포가 아닙니다. 전체 커버율을 기준으로 골라 원래 점포와 멀리 떨어져 있을 수 있습니다. 형평 하한을 유지하는 대안이 있으면 그 중 전체 효과가 큰 곳을 우선했습니다. 없으면 전체 효과를 최대화하고 하한 미달을 표시했습니다.</p></section>''')
sections.append(f'''<section id="stages"><p class="eyebrow">05 · 설치 규모와 단계 도입</p><h2>20곳부터 시작하고, 이미 고른 곳을 유지한 채 늘릴 수 있습니다</h2>
{figure(5,'20→40→60곳 단계안은 앞선 점포를 고정한 조건에서 다시 최적화했습니다. 효율 목적의 단계안이며 형평안 60곳의 설치 순서를 정한 것은 아닙니다.')}
<p>처음 20곳은 기존망 대비 약 9.7만 명을 추가로 포함합니다. 독립 효율안에서 40→60곳의 추가 효과는 약 3.5만 명, 80→100곳은 약 1.3만 명으로 줄어듭니다. 이 곡선에 실제 점포별 비용을 연결하면 설치 수를 정할 근거가 됩니다.</p>
<p>단계안의 60곳 커버율은 87.39%로, 한 번에 고른 효율안 87.40%와 차이는 약 85명입니다. 현재 조건에서는 큰 손실 없이 단계적으로 검토할 수 있습니다. 실제 비용이나 참여 가능성을 입력하면 다시 선정해야 합니다.</p></section>''')
worst=dg.nsmallest(5,'screened_fair_60_pct')
sections.append(f'''<section id="limits"><p class="eyebrow">06 · 남는 문제와 가정의 영향</p><h2>구의 평균이 좋아져도, 모든 읍면동이 좋아지는 것은 아닙니다</h2>
{table(['지역','형평안 60곳 후 커버율','802곳을 모두 더해도 남는 인구'],[[f'{r.district} {r.dong}',f'{r.screened_fair_60_pct:.2f}%',f'{r.screened_outside_candidate_network_population:,.0f}명'] for r in worst.itertuples()])}
<p>진전면·구산면·진북면·대산면 등은 후보 편의점망으로도 해결하기 어려운 인구가 남습니다. 이 결과는 편의점 확대가 효과적인 생활권과, 다른 공공 거점이나 방문·이동 수거 같은 방식을 별도로 비교할 지역을 나눠 줍니다. 후자의 운영 효과는 이번 분석에서 계산하지 않았습니다.</p>
{figure(6,'오른쪽은 서로 다른 반경으로 선정한 60곳을 모두 같은 350m로 평가한 결과입니다. 점포 조합 자체가 거리 가정에 민감하므로, 현재 위치를 보행 기준의 확정 추천으로 부르면 안 됩니다.')}
<h3>무엇을 바꿔도 비슷했고, 무엇을 바꾸면 달라졌나?</h3>
{table(['변경 조건','전체 인구 결과','해석'],[
['검토 전 863곳 → 보수적 802곳','60곳 효율안 87.47% → 87.40%','전체 효과는 유사하나 4곳이 다른 조합'],
['인구 최대 → 가구 최대','87.40% → 87.17% · 공통 점포 56곳','가구 커버율은 87.56% → 87.66%; 수요 단위를 바꿔 비교'],
['다른 업종의 브랜드명 후보 9개 추가','검토 전 863곳 기준 87.47% 유지','이 9곳으로 결과가 바뀌지 않음. 후보망의 완전성 입증은 아님'],
['200m·350m·500m 각각 재선정','55.25% · 87.40% · 94.88%','각자의 반경에서 평가한 값. 반경이 달라 직접적인 성과 우열 비교 불가']])}
<p class="note">100m 격자 안에서의 실제 거주 위치도 알 수 없습니다. 형평안 60곳을 고정하고 격자 중심에서 모서리까지의 거리 약 70.7m를 반경에 가감하면 74.30~90.61%로 달라집니다. 이는 격자 내 위치를 극단적으로 본 기하학적 민감도이며 통계적 신뢰구간이 아닙니다.</p></section>''')
sections.append(f'''<section id="method"><p class="eyebrow">07 · 변수와 최적화의 근거</p><h2>변수를 점수 하나에 더하지 않고, 역할에 맞게 사용했습니다</h2>
<p>Rico·White·Albert(2026)는 기존 폐의약품 수거망의 접근 공백을 진단하고 신규 수거지를 보완하는 입지 모형을 제시했습니다. 이 구조를 참고해 기존망을 고정하고 추가 커버리지를 계산했습니다. 5개 구의 최소 커버율 최대화, 단계 도입, 점포 제외 분석은 이 프로젝트에 맞게 확장한 설계입니다. <a href="{paper}">논문</a></p>
<p>Egan 등(2024)의 약국 연구에서는 유지 비용과 조직의 의사결정이 수거함 도입에 영향을 주는 조건으로 다뤄졌습니다. 창원 편의점의 참여율을 이 연구에서 가져오지는 않았습니다. 대신 특정 점포에 참여를 가정한 결과가 얼마나 의존하는지 제외·대체 분석으로 확인했습니다. <a href="{egan}">연구 초록</a></p>
{table(['변수·조건','역할','이번 처리'],[
['기존 수거지','유지할 기본망','439개 좌표를 고정하고 중복 효과 제거'],
['100m 인구','주 목적함수의 수요','기존망 밖 공개 인구의 추가 커버리지 최대화'],
['100m 가구','대체 수요 기준','인구와 임의 가중 합산 없이 별도 최적화'],
['구별 인구·기존 커버율','형평성 기준','최저 구의 기존+신규 총 커버율 최대화'],
['편의점 위치·접근 검토 표시','후보 범위','출입·업종 검토 대상 제외 전후 비교'],
['설치 수·이전 선정점','실행 제약','20~100곳 규모 비교와 20→40→60곳 고정 확장'],
['거리·격자 위치','불확실성 점검','200/350/500m 재선정 및 위치 민감도'],
['읍면동 고령인구','결과 해석용 배경','격자별 고령인구로 임의 배분하지 않음'],
['정류소 접근성','후보 현황 점검','863곳 중 849곳이 350m 이내여서 유동인구 점수로 사용하지 않음'],
['시간·회수량·용량·비용','향후 운영 조건','실측값이 없어 이번 창원 수치에 넣지 않음']])}
<details><summary>최적화식과 검증 방법 보기</summary><p>격자 i의 공개 인구를 wᵢ, 기존망 커버 여부를 bᵢ, 편의점 j의 선택 여부를 xⱼ, 거리 내 연결 여부를 aᵢⱼ, 추가 커버 여부를 zᵢ로 둡니다.</p>
<pre>효율 목적: 최대화 Σᵢ wᵢ(1−bᵢ)zᵢ
도달 조건: zᵢ ≤ Σⱼ aᵢⱼxⱼ,  0 ≤ zᵢ ≤ 1
설치 조건: Σⱼ xⱼ ≤ p,  xⱼ ∈ {0,1}
구 g의 총 커버율 Cg = Σᵢ∈g wᵢ[bᵢ+(1−bᵢ)zᵢ] / Σᵢ∈g wᵢ
형평 목적: 1단계 max t, Cg ≥ t (모든 구)
           2단계 달성한 t를 유지하며 전체 추가 커버 인구 최대화
단계 도입: 이전 단계에서 선정한 점포의 xⱼ = 1</pre>
<p>주 분석인 802개 후보의 11개 시나리오는 각 단계에서 최적성 간격 0으로 계산됐습니다. 이는 정해진 자료·모형에서의 최적성을 뜻합니다. 24개 전체 시나리오 중 검토 전 863개 후보 형평안의 1단계는 시간 제한으로 최적성을 확정하지 못했으며 계산상 간격은 약 0.205%입니다. 주 분석의 형평안과 구분해 저장했습니다.</p>
<p>작은 문제의 모든 조합 비교, 기존망 기준 결과 재현, 모든 배치의 독립 거리 검산, 240건의 대체 후 거리 검산을 수행했습니다. 55개 읍면동에서 기존 공백이 ‘후보망으로 보완 가능’과 ‘후보망 밖’의 합과 일치하는지도 확인했습니다. 자세한 상태는 development.json에 있습니다.</p></details></section>''')
allrows=[[r.names,f'{r.district} {r.dong}',r.addresses,f'{r.exclusive_population:,.0f}',r.flags or '별도 표시 없음'] for r in list60.itertuples()]
sections.append(f'''<section id="files"><p class="eyebrow">08 · 결과와 활용</p><h2>공모전의 이야기는 이 순서로 이어갈 수 있습니다</h2>
<div class="story">배출 장소를 쉽게 찾고 생활 속에서 이용하도록 하자 → 기존 수거망의 공백과 안내정보를 진단 → 공백을 채우는 편의점 선정 → 전체 효과와 지역 균형 비교 → 참여가 어려운 점포의 대안과 단계별 도입 제시 → 편의점만으로 남는 지역에는 다른 수단 검토</div>
<p>본 연구는 창원의 기존 수거망과 생활권을 연결해 <b>‘기존망을 얼마나, 어디서, 어떤 조건으로 개선할 것인가’</b>를 분석했습니다. 야간 이용과 인지도 향상은 이 취지를 설명하는 가설로 두고, 수치 결과는 이번에 확인한 공간적 접근성에 한정하면 주장이 더 분명해집니다.</p>
<div class="downloads"><a href="형평안_검토편의점60곳.csv">형평안 60곳 · 주소와 대체 후보</a><a href="효율안_검토편의점60곳.csv">효율안 60곳</a><a href="효율목적_단계별20곳_도입검토.csv">효율 목적의 단계별 명부</a><a href="dong_gap_diagnostics.csv">55개 읍면동 진단</a><a href="replacement_options.csv">사례별 대안 3곳</a><a href="분석방법과_해석.md">방법과 재현 안내</a></div>
<details><summary>형평안 60곳 전체 명부 펼치기</summary><p class="note">고유 기여도 내림차순입니다. 설치 우선순위가 아니며, 별도 표시가 없어도 운영·출입·참여가 확인된 것은 아닙니다. 원자료 명칭과 주소를 보존했습니다.</p>{table(['점포명','지역','도로명주소','고유 기여(명)','검토 표시'],allrows)}</details>
<h3>자료의 기준과 출처</h3>
<p>SGIS 2024년 공개 인구·가구, 2025년 경계, 2026년 1월 28일 등록 수거지, 2026년 6월 상가정보를 연결했습니다. 분모는 창원 경계에 면적 비례로 배분한 공개 인구 약 1,007,328명입니다. 서로 다른 자료의 기준 시점이 같지 않습니다. 실제 주민등록 총인구와 동일한 분모로 읽지 않습니다.</p>
<p>경계에 걸친 격자는 교차 면적 비율로 배분하고 거리는 원래 격자 중심점으로 계산했습니다. 결측 원본은 보존하며 공개된 값만 지표별 분모·분자에 포함했습니다. 통계행이 없는 격자를 실제 인구 0명으로 해석하지 않습니다.</p>
<ul><li><a href="https://www.data.go.kr/data/15129445/standard.do">전국 폐의약품 수거함 표준데이터</a> — 창원 472건, 동일 좌표 통합 후 439곳. 동일 좌표가 실제 시설 중복을 뜻하지는 않습니다.</li>
<li><a href="https://www.data.go.kr/data/15083033/fileData.do">소상공인시장진흥공단 상가(상권)정보</a> — 업종코드 G20405, 후보 검토 표시 보존.</li>
<li><a href="https://sgis.mods.go.kr/view/pss/openDataIntrcn">SGIS 자료제공</a> — 인구·가구 및 공간 경계. SGIS 출처와 신청 시 동의한 이용조건을 최종 결과물에도 유지해야 합니다.</li>
<li><a href="https://www.data.go.kr/data/15004972/fileData.do">창원 연령별 인구</a> — 2025.05.31, 읍면동별 고령인구 비율을 해석용으로 연결.</li></ul>
<p class="note">문헌은 모형 구조와 검토 항목의 근거입니다. 해외 연구의 참여율·회수량이나 거리 기준을 창원의 실측값으로 옮기지 않았습니다. 350m는 비교를 위한 가정이며 창원 시민의 허용 보행거리로 검증된 값이 아닙니다.</p></section>''')
css='''*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f3f5f5;color:#263f47;font:17px/1.8 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif}main{max-width:1200px;margin:auto;background:white;padding:52px 55px}h1{font-size:40px;line-height:1.32;letter-spacing:-1.4px;margin:15px 0 23px}h2{font-size:27px;line-height:1.45;letter-spacing:-.6px}h3{font-size:21px;margin-top:30px}.eyebrow{font-size:13px;letter-spacing:1px;color:#317b83;font-weight:700}.lead{font-size:20px;max-width:980px}.note,figcaption{font-size:14px;color:#63757c}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:15px;margin:28px 0}.facts div{padding:20px;background:#edf5f4;border-radius:8px}.facts strong{display:block;font-size:33px;line-height:1.35;color:#237f85}.facts span{font-size:14px}nav{display:flex;flex-wrap:wrap;gap:7px 20px;border-block:1px solid #dce5e5;padding:20px 0;margin:30px 0 10px;font-size:14px}a{color:#176e78;text-underline-offset:3px}section{padding:42px 0;border-bottom:1px solid #dce5e5}p{margin:17px 0}figure{margin:27px -15px}img{width:100%;height:auto;display:block}figcaption{padding:0 20px}.tablewrap{overflow-x:auto;margin:22px 0}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}th,td{border-bottom:1px solid #dce4e5;padding:12px 11px;text-align:left;vertical-align:top}th{background:#edf3f3;color:#203c43}td:first-child{font-weight:600}.callout{padding:22px 28px;border-left:4px solid #d68b2d;background:#fff8ec;margin:26px 0}.callout p{margin-bottom:0}.story{padding:26px;border-radius:8px;background:#edf5f4;font-size:20px;line-height:1.9}.downloads{display:flex;flex-wrap:wrap;gap:13px;margin:25px 0}.downloads a{background:#edf3f3;border:1px solid #cfdddd;border-radius:6px;padding:10px 15px;font-size:14px}details{background:#f7f9f9;padding:18px 22px;margin:25px 0}summary{cursor:pointer;font-weight:700}pre{overflow:auto;background:#edf2f3;padding:20px;font:14px/1.8 monospace}footer{padding-top:25px;font-size:13px;color:#6c7c82}@media(max-width:700px){main{padding:25px 18px}h1{font-size:29px}h2{font-size:23px}.facts{grid-template-columns:repeat(2,1fr)}figure{margin:20px -8px}section{padding:25px 0}}@media print{body{background:white}main{padding:0}nav,.downloads{display:none}figure,table,.callout{break-inside:avoid}h2,h3{break-after:avoid}details{display:block}}'''
doc='<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>창원 기존 수거망과 편의점 보완 분석</title><style>'+css+'</style></head><body><main>'+''.join(sections)+'<footer>2026-09-28 분석 · 사용자 검토용. 그림은 본문에 포함되어 인터넷 없이 열립니다. 원자료와 이전 분석은 프로젝트 안에 보존했습니다.</footer></main></body></html>'
(R/'창원_기존망진단과_편의점보완_보고서.html').write_text(doc)

# 재현과 해석을 위한 짧은 독립 문서.
phase_rows=[]
for s in j['scenarios']:
    states=' / '.join(f"{p['phase']}: gap={p.get('gap')}, optimal={p['proven_optimal']}" for p in s['phases'])
    phase_rows.append(f"| {s['id']} | {s['candidate_scope']} | {s['max_new']} | {s['radius_m']} | {s['metrics']['population_pct']:.6f} | {states} |")
md=f'''# 기존망 진단과 편의점 보완: 분석 방법과 해석

2026-09-28. 최신 보고서는 [그림 중심 보고서](창원_기존망진단과_편의점보완_보고서.html)다.

## 목적과 자료

기존 수거망을 유지하고 접근 공백을 줄이는 편의점을 찾는다. 등록 472건을 좌표 단위로 통합한 439곳, SGIS 2024 공개 인구·가구, 2025 경계, 2026년 6월 상가정보를 썼다. 863개 후보 중 출입 확인 표시 59곳과 명칭상 업종 재확인 2곳을 제외한 802개 좌표를 주 분석에 썼다. 영업·출입·참여가 확인된 명부가 아니며 동일 좌표 통합이 실제 시설 중복을 증명하지 않는다.

기존망은 모두 이용 가능하다고 가정한다. 범위는 직선 350m, 대표점은 원래 100m 격자 중심이다. 경계 인구는 교차 면적으로 배분했다. 공개 인구 분모는 {j['total_population']:.6f}, 공개 가구 분모는 {j['total_households']:.6f}이다. 지표별 결측값은 공개 합계 기여에서 제외했고 실제 0으로 해석하지 않는다. 원본 결측값은 보존했다.

## 결과

- 기존망 커버율 {base['population_pct']:.6f}%, 최저 구 {base['minimum_district_pct']:.6f}%.
- 효율 60곳: 전체 {eff['population_pct']:.6f}%, 추가 {eff['added_population']:.6f}명, 최저 구 {eff['minimum_district_pct']:.6f}%.
- 형평 60곳: 전체 {fair['population_pct']:.6f}%, 추가 {fair['added_population']:.6f}명, 최저 구 {fair['minimum_district_pct']:.6f}%.
- 802곳 전체 후보망의 기존망 포함 범위 {j['screened_union_ceiling']['population_pct']:.6f}%. 남는 인구 약 6.8만 명.
- 효율 목적 20→40→60곳 단계안: 78.816641% → 83.923120% → 87.391596%. 형평안의 도입 순서와 다르다.
- 형평안 한 곳 제외 60사례 중 24사례만 한 곳 교체로 원래의 최저 구 하한 유지. 대체 후보는 실제 협약 점포가 아니며, 원래 점포 인근이라는 조건도 부과하지 않았다.

## 모형

기존 커버 여부 b_i, 수요 w_i, 후보 선택 x_j, 후보 도달 a_ij에 대해 추가 수요 w_i(1-b_i)를 커버하는 집합을 고른다. z_i ≤ Σ_j a_ij x_j, 0 ≤ z_i ≤ 1, Σ_j x_j ≤ p, x_j∈{{0,1}}. 효율 목적은 Σ_i w_i(1-b_i)z_i의 최대화다. 비음수 가중치와 이진 선택으로 z는 도달 여부에 맞춰 최대로 채울 수 있다. 같은 후보 집합에 도달하는 격자는 수요를 합쳐 계산한다.

형평 목적은 구별 전체 인구를 분모로, 기존과 추가 커버 인구를 분자로 둔 비율의 최솟값을 먼저 최대화한다. 두 번째 단계는 이 하한을 유지하며 전체 추가 커버 인구를 최대화한다. 하한 허용오차 1e-8(비율 단위)을 사용했다. 5개 구 기준이며 55개 읍면동 기준이 아니다.

점포 제외 분석은 나머지 선정점과 기존망을 고정한다. 한 곳 교체 시 같은 후보 범위의 미선정점만 사용하고 제외점은 다시 고르지 않는다. 형평 하한을 유지할 수 있는 대안이 있으면 그 안에서 추가 인구 최대, 없으면 전체 대안 중 추가 인구 최대를 찾는다. 사례별 3개 대안을 저장했다. 순차·동시 다중 제외, 확률적 참여율, 수거용량은 분석하지 않았다.

## 검증과 계산 상태

작은 사례 전수조합 비교, 기존 결과 재현, 모든 배치와 240건 대체 결과의 독립 거리 검산, 55개 읍면동 공백 분할 검산을 수행했다. 최신 자동검사 결과는 validation_summary.json에 저장한다. 802개 후보의 11개 시나리오는 각 단계에서 최적성 간격 0이다. 검토 전 863개 형평안은 1단계 시간제한이 있어 전역 최적이라 부르지 않는다.

| 시나리오 | 후보 범위 | 추가 수 | 거리(m) | 전체 인구(%) | 단계별 상태 |
|---|---|---:|---:|---:|---|
{chr(10).join(phase_rows)}

## 변수 근거

[Rico 등(2026)]({paper})의 기존 수거망 진단과 신규 입지 보완 구조를 참고했다. 구별 max-min, 단계 도입, 한 점포 제외·대체는 본 프로젝트의 확장이다. [Egan 등(2024)]({egan})의 조직적 도입 조건을 참고해 참여 불확실성을 검토하되 해외 비율을 창원에 대입하지 않았다. [창원시의회 회의록]({minutes})의 안내·홍보 논의는 문제 배경이며 인지도 조사 결과가 아니다.

인구·가구는 별도 목적함수, 기존망은 고정된 기본 서비스, 구는 형평성 제약, 설치 수는 규모 조건, 후보 표시와 거리는 민감도에 쓴다. 읍면동 고령 비율을 인구 격자에 균등 배분하거나 정류소 개수를 유동인구로 치환하지 않았다. 시간·비용·회수량·용량의 가짜 값을 넣지 않았다.

## 재현

프로젝트의 requirements.txt 환경에서 다음 순서로 실행한다. 현재 결과 입력은 프로젝트에 포함했다.

```sh
python -m unittest discover -s tests -v
python run_supplement_development.py
python render_supplement_development.py
```

기본 실행은 모든 시나리오를 새로 푼다. --reuse-results는 같은 입력을 계속 사용하는 개발 중에만 사용한다. 이 옵션은 설정을 비교하지만 입력파일 해시를 대조하는 완전한 캐시 검증을 하지 않는다. 입력을 바꿨다면 옵션 없이 다시 실행한다. 시간제한 해와 동률 위치는 환경에 따라 달라질 수 있다. 그림은 Apple SD Gothic Neo를 사용한다.

development.json은 결과·계산 상태, candidate_metadata.json은 후보 명부, coverage_by_demand.csv는 격자별 커버 여부, dong_gap_diagnostics.csv는 읍면동 진단, store_contributions_and_replacements.csv는 240개 제외 사례, replacement_options.csv는 720개 대안이다. 이름이 한글인 명부는 사용자 검토용으로 핵심 필드를 정리한 파일이다. 인구값은 면적 배분으로 소수가 될 수 있다.

## 해석 한계와 출처

실제 보행경로·산이나 하천의 장애, 수거 시간, 참여, 수거량과 비용이 반영되지 않았다. 350m는 시민의 허용거리를 조사해 얻은 값이 아니다. 수치는 잠재 접근성으로, 이용률·반환 인원·회수량으로 바꾸어 쓰지 않는다. 802개를 창원 전체 영업 편의점 수로 부르지 않는다.

기존 정보 감사는 확보한 파일에서만 주소·좌표 472/472, 상세 위치·관리기관 전화 0/472, 시간 항목 없음을 확인했다. 다른 채널의 정보 존재 여부와 시민의 인지도를 알 수는 없다. 자세한 필드는 existing_information_audit.json에 보존했다.

[기존 수거지](https://www.data.go.kr/data/15129445/standard.do), [상가정보](https://www.data.go.kr/data/15083033/fileData.do), [SGIS](https://sgis.mods.go.kr/view/pss/openDataIntrcn), [연령별 인구](https://www.data.go.kr/data/15004972/fileData.do). 원본 기준 시점이 서로 다르다. SGIS 출처와 신청 시 동의한 이용조건을 최종 결과물에 유지한다.
'''
(R/'분석방법과_해석.md').write_text(md)
print('완료:',len(files),'개 그림, 오프라인 보고서, 검토 명부 3개, 방법 문서')
