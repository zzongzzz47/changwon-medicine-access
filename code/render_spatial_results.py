"""창원 공간 분석 결과의 보고서·그림·지도·읍면동 표를 만든다."""
from pathlib import Path
import json
import html
import base64
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import folium
from src.coverage_model import radius_matrix

P=Path(__file__).resolve().parent; R=P/'results'; F=R/'figures'; F.mkdir(exist_ok=True)
data=json.loads((R/'spatial_optimization.json').read_text())
audit=json.loads((R/'spatial_join_audit.json').read_text())
sc={s['id']:s for s in data['scenarios']}
main=sc['100m_350m_p60_population_all'];fair=sc['100m_350m_p60_fair_all']
hh=sc['100m_350m_p60_households_all']; coarse=sc['500m_350m_p60_population_all']
restrict=sc['100m_350m_p60_population_restricted']
base350=next(s for s in data['baseline_all_candidates'] if s['radius_m']==350)
font=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if font.exists():font_manager.fontManager.addfont(str(font))
plt.rcParams.update({'font.family':'Apple SD Gothic Neo' if font.exists() else 'sans-serif',
    'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':'white','axes.facecolor':'white'})
front=[sc[f'100m_350m_p{p}_population_all'] for p in [20,40,60,80,100,150]]
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
xs=[s['max_sites'] for s in front];ys=[s['metrics']['population_coverage_pct'] for s in front]
ax.plot(xs,ys,'o-',color='#176b75',lw=2.5,ms=7)
ax.axhline(base350['population_coverage_pct'],color='#b5663d',ls='--',lw=1.5)
ax.text(21,base350['population_coverage_pct']+2,f"전체 후보 863곳을 모두 쓸 때 {base350['population_coverage_pct']:.2f}%",color='#a2522e')
for x,y in zip(xs,ys):ax.annotate(f'{y:.2f}%',(x,y),xytext=(0,10),textcoords='offset points',ha='center')
ax.set(ylim=(0,103),xlim=(12,160),xticks=xs,xlabel='설치 수 상한 (곳)',ylabel='공개 인구 커버율 (%)')
ax.grid(axis='y',alpha=.15);ax.set_title('설치 수에 따른 잠재 접근성',loc='left',fontsize=19,fontweight='bold',pad=18)
fig.supxlabel('100m 인구 격자 · 직선 350m · 편의점 전용 · 모든 후보의 설치 가능성을 가정',fontsize=10,color='#666')
fig.savefig(F/'coverage_frontier.png',dpi=180);plt.close(fig)

names=['의창구','성산구','마산합포구','마산회원구','진해구']
def district_values(s):
    d={r['district']:r['coverage_pct'] for r in s['metrics']['districts']}
    return [d[n] for n in names]
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained'); y=np.arange(5)
for offset,s,c,label in [(-.19,main,'#176b75','전체 인구 커버 최대'),(.19,fair,'#cc7949','구별 최소 커버율 우선')]:
    vals=district_values(s);bars=ax.barh(y+offset,vals,height=.34,color=c,label=label)
    for bar,v in zip(bars,vals):ax.text(v+.6,bar.get_y()+bar.get_height()/2,f'{v:.2f}%',va='center',fontsize=11)
ax.set(yticks=y,yticklabels=names,xlim=(0,72),xlabel='구별 공개 인구 커버율 (%)');ax.invert_yaxis();ax.grid(axis='x',alpha=.12);ax.set_axisbelow(True)
ax.set_title('같은 60곳, 다른 배치 목표',loc='left',fontsize=19,fontweight='bold',pad=18);ax.legend(loc='lower right',frameon=False)
fig.supxlabel('350m 직선거리 · 형평성 안은 시간 제한 내 실행 가능해이며 전역 최적성 미확정',fontsize=10,color='#666')
fig.savefig(F/'district_equity_comparison.png',dpi=180);plt.close(fig)

fig,ax=plt.subplots(figsize=(10,4.7),layout='constrained')
labels=['100m 격자로 선정\n100m에서 평가','500m 격자로 선정\n500m에서 평가','500m 격자로 선정\n100m에서 재평가']
vals=[main['metrics']['population_coverage_pct'],coarse['metrics']['population_coverage_pct'],coarse['evaluation_on_100m']['population_coverage_pct']]
bars=ax.bar(labels,vals,color=['#176b75','#a8b4b9','#cc7949'],width=.55)
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+.7,f'{v:.2f}%',ha='center',fontweight='bold')
ax.set(ylim=(0,62),ylabel='공개 인구 커버율 (%)');ax.grid(axis='y',alpha=.12);ax.set_axisbelow(True)
ax.set_title('격자 크기에 따라 선정 결과가 크게 달라진다',loc='left',fontsize=18,fontweight='bold',pad=18)
fig.supxlabel('60곳 · 직선 350m · 해상도별 통계·대표점·경계 배분이 함께 달라지는 비교',fontsize=10,color='#666')
fig.savefig(F/'grid_resolution_sensitivity.png',dpi=180);plt.close(fig)

cs=json.loads((P/'data/processed/convenience_candidates_spatial.json').read_text())
cs=[c for c in cs if c['inside_city_boundary']]; index={c['id']:i for i,c in enumerate(cs)}
d=pd.read_csv(P/'data/processed/demand_100m.csv',dtype={'adm_cd':str})
a=radius_matrix(d[['x_m','y_m']].to_numpy(),np.array([[c['x_m'],c['y_m']] for c in cs]),350)
pub=d.population.fillna(0).to_numpy()
def coverage(ids):return np.asarray(a[:,[index[i] for i in ids]].sum(axis=1)).ravel()>0
d['all_covered']=np.asarray(a.sum(axis=1)).ravel()>0
d['population60_covered']=coverage(main['selected_ids'])
d['fair60_covered']=coverage(fair['selected_ids'])
d['population100_covered']=coverage(sc['100m_350m_p100_population_all']['selected_ids'])
d['fair100_covered']=coverage(sc['100m_350m_p100_fair_all']['selected_ids'])
rows=[]
for (code,district,dong),g in d.groupby(['adm_cd','district','dong']):
    total=float(g.population.sum());row={'adm_cd':code,'district':district,'dong':dong,'sgis_population':total}
    for col in ['all','population60','fair60','population100','fair100']:
        covered=float(g.loc[g[col+'_covered'],'population'].sum())
        row[col+'_coverage_pct']=100*covered/total if total else None
        row[col+'_uncovered_population']=total-covered
    rows.append(row)
dong=pd.DataFrame(rows)
age=pd.read_csv(R/'dong_demographics.csv')[['district','dong','population','age65plus','age65plus_share_pct']]
age=age.rename(columns={'population':'mois_population_20250531'})
dong=dong.merge(age,on=['district','dong'],validate='one_to_one',how='left');assert dong.age65plus.notna().all()
dong.to_csv(R/'dong_coverage_and_age_context.csv',index=False,encoding='utf-8-sig')

# 지도에는 공개 인구가 양수인 격자만 표시한다. 빈 땅의 인구 0을 단정하지 않는다.
geo=gpd.read_file(P/'data/processed/changwon_spatial.gpkg',layer='demand_100m')
geo=geo.merge(d[['demand_id','population60_covered','fair60_covered']],on='demand_id',validate='one_to_one')
geo=geo.loc[geo.population>0].copy()
areas=gpd.read_file(P/'data/processed/changwon_spatial.gpkg',layer='admin_dongs')
mapobj=folium.Map(location=[35.205,128.59],zoom_start=11,tiles='OpenStreetMap',control_scale=True)
ag=areas.copy();ag.geometry=ag.geometry.simplify(20,preserve_topology=True)
folium.GeoJson(ag.to_crs(4326),name='읍면동 경계',show=False,
    style_function=lambda f:{'color':'#526070','weight':1,'fillOpacity':0},
    tooltip=folium.GeoJsonTooltip(fields=['district','dong'],aliases=['구','읍면동'])).add_to(mapobj)
for s,label,col,show in [(main,'인구 최대 60곳','#176b75',True),(fair,'형평성 우선 60곳','#cc7949',False)]:
    group=folium.FeatureGroup(name=label,show=show).add_to(mapobj)
    for cid in s['selected_ids']:
        c=cs[index[cid]]
        popup=f"<b>{html.escape(' / '.join(c['names']))}</b><br>{html.escape(' / '.join(c['addresses']))}<hr>{label}<br>{html.escape(c['spatial_district']+' '+c['spatial_dong'])}<br>점검: {html.escape(' / '.join(c['flags']) or '자동 표시 없음')}<br><small>영업·출입·협약·수거시간 현장 확인 전</small>"
        folium.Circle([c['lat'],c['lon']],radius=350,color=col,weight=1,fill=True,fill_opacity=.05,opacity=.4).add_to(group)
        folium.CircleMarker([c['lat'],c['lon']],radius=5,color='white',weight=1,fill=True,fill_color=col,fill_opacity=1,
           tooltip=c['names'][0],popup=folium.Popup(popup,max_width=340)).add_to(group)
for col,label,show in [('population60_covered','인구 최대 60곳: 미커버 격자',True),('fair60_covered','형평성 60곳: 미커버 격자',False)]:
    sub=geo.loc[~geo[col],['grid_id','district','dong','population','geometry']].copy()
    sub['population']=sub.population.round(1);sub.geometry=sub.geometry.simplify(3,preserve_topology=True)
    folium.GeoJson(sub.to_crs(4326),name=label,show=show,
        style_function=lambda f:{'fillColor':'#b34f45','color':'#b34f45','weight':.15,'fillOpacity':.45},
        tooltip=folium.GeoJsonTooltip(fields=['grid_id','district','dong','population'],aliases=['격자','구','읍면동','공개 인구 배분값'])).add_to(mapobj)
folium.LayerControl(collapsed=False).add_to(mapobj)
mapobj.get_root().html.add_child(folium.Element(f'''<div style="position:fixed;bottom:24px;left:16px;z-index:9999;background:white;padding:16px;max-width:330px;border-radius:10px;box-shadow:0 2px 12px #aaa;font-family:sans-serif;font-size:13px;line-height:1.6"><b style="font-size:16px">창원 편의점 수거망 · 60곳 비교</b><br><span style="color:#176b75">● 인구 최대</span> {main['metrics']['population_coverage_pct']:.2f}%　<span style="color:#cc7949">● 형평성 우선</span> {fair['metrics']['population_coverage_pct']:.2f}%<br>원: 직선 350m / 붉은 격자: 해당 안의 미커버 인구<br>오른쪽에서 선택안과 미커버 격자를 함께 전환하세요.<hr>100m 격자 중심점 기준 · 실제 보행 경로 아님<br>빈 공간은 인구 0명을 뜻하지 않습니다.<br>편의점만 사용하며 기존 수거함은 포함하지 않음.<br><small>인구 2024 / 경계 2025 / 업소 2026.6<br>형평성 안은 시간 제한 내 해. 배경지도·기능은 인터넷 필요.</small></div>'''))
mapobj.save(str(R/'창원_최적화_비교지도.html'))

def table(headers,rows):
    return '<div class="table-wrap"><table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table></div>'
def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
sections=[]
def section(title,paras=(),headers=None,rows=None,figure=None):
    sections.append({'title':title,'paras':list(paras),'headers':headers,'rows':rows,'figure':figure})
percent=lambda s:s['metrics']['population_coverage_pct']
minpct=lambda s:s['metrics']['minimum_district_pct']
ratehh=lambda s:s['metrics']['households_coverage_pct']
section('후보 범위 재점검', ['후보 범위 재점검(2026-09-28): 863은 업종코드 G20405에서 얻은 좌표 수이며 창원 전체의 실제 영업 편의점 수가 아니다. 다른 업종에 편의점 브랜드명으로 등록된 10개 기록(기존에 없던 좌표 9개)을 추가 검토 대상으로 확인했다. 영업 상태와 일반인 출입 가능성은 미확인이다. 91.38%는 기존 863개 후보집합에서의 커버율이며, 창원 전체 편의점망의 확정 상한으로 해석하지 않는다.'])
section('이번 분석에서 확인한 것',[
    '창원시 편의점 후보 863곳과 SGIS 100m 인구·가구 격자를 연결해 13개 입지 시나리오를 계산했다. 기존 수거함은 주 분석에 포함하지 않았다. 결과는 공개 통계와 직선거리로 계산한 잠재 접근성이며 실제 방문·반납률이나 수거량이 아니다.',
    f"직선 350m 기준, 60곳의 인구 최대안은 {percent(main):.2f}%를 커버한다. 형평성 우선안은 전체 커버율 {percent(fair):.2f}%로, 최저 구의 커버율을 {minpct(main):.2f}%에서 {minpct(fair):.2f}%로 높인다. 60곳은 비교를 위한 예시이며 예산에 근거한 권고 설치 수는 아니다.",
    f"전체 후보를 모두 사용해도 같은 기준의 커버율 상한은 {base350['population_coverage_pct']:.2f}%다. 편의점 네트워크 자체의 공간 공백이 있으므로, 무조건 100%를 요구하는 모형은 실행 불가능해질 수 있다.",
    f"500m 격자에서 고른 60곳은 해당 해상도에서 {percent(coarse):.2f}%이지만, 동일 점포를 100m 격자에서 재평가하면 {coarse['evaluation_on_100m']['population_coverage_pct']:.2f}%다. 짧은 접근 반경을 분석할 때 격자 크기를 검토하는 것이 핵심이다."
])
section('1. 자료와 공간 연결',[
    '거리 계산 좌표계는 EPSG:5179다. 2025년 2분기 경계에서 창원 55개 읍면동을 골랐고, 격자 코드로 2024년 인구·가구 통계를 연결했다. 경계에 걸친 격자는 원래 격자 면적에 대한 교차 면적 비율로 배분했다. 이는 격자 안 인구의 균일 분포를 가정한 근사다.',
    '통계행이 없는 격자는 원자료에서 결측으로 남겼다. 커버율의 분자·분모에는 해당 지표의 공개값만 포함하며, 계산 배열에서 결측의 기여를 0으로 두는 것은 실제 인구 0명이라는 대체가 아니다. 인구와 가구의 관측 범위·분모는 각각 다르다.',
    '상가정보의 동일 좌표 업소기록은 한 후보로 묶었다. 863곳 모두 창원 경계 안에 있었고 구 단위 불일치는 없었다. 원자료의 동명과 공간 연결 결과가 다른 4곳은 검토 대상으로 남겼으며 분석에는 공간 연결 동을 썼다. 자료 시점·주소 좌표·행정경계 차이를 확인해야 한다.'
],['자료','범위·규모','기준'],[
    ['편의점','950개 업소기록 → 863개 후보 위치','2026-06-30'],
    ['100m 인구','통계행 있는 격자 11,076개 / 양수 10,190개','2024'],
    ['100m 인구 배분합',f"{audit['by_resolution']['100M']['population_area_allocated']:,.1f}명 상당 공개값",'실제 주민등록 합계와 별개'],
    ['100m 가구 배분합',f"{audit['by_resolution']['100M']['households_area_allocated']:,.1f}가구 상당 공개값",'총가구이며 1인 가구 아님'],
    ['경계','55개 읍면동 / 4개 격자 구획','2025'],
    ['연령별 주민등록인구','55개 읍면동, 995,372명','2025-05-31'],
    ['정류소','3,526개 기록','2025-12-31']])
section('2. 설치 개수별 커버율',[
    '인구 최대화는 격자 인구를 중복 없이 한 번만 센다. 반경에 겹쳐 들어가는 인구를 점포별로 단순 합산하지 않는다. 아래 여섯 결과는 모형의 최적성 간격 0으로 확인했다. 각 설치 수에서 독립적으로 고른 조합이므로 순차 설치 순위로 읽으면 안 된다.'
],['설치 수','인구 커버율','가구 커버율','가장 낮은 구의 인구 커버율'],[
    [s['max_sites'],f"{percent(s):.2f}%",f"{ratehh(s):.2f}%",f"{minpct(s):.2f}%"] for s in front],figure='coverage_frontier.png')
section('3. 총 커버와 지역 형평성의 교환관계',[
    '형평성 안은 5개 구의 커버율 중 최솟값을 먼저 높이고, 그 수준을 유지하면서 총 커버율을 개선하는 두 단계 모형이다. 모든 구에 같은 개수를 나눠 주는 규칙과는 다르다. 구 단위 형평성은 읍면동·고령층의 형평성을 보장하지 않는다.',
    f"60곳에서 총 커버율 차이는 {percent(main)-percent(fair):.2f}%p이며 최저 구의 개선은 {minpct(fair)-minpct(main):.2f}%p다. 두 안의 공통 후보는 {len(set(main['selected_ids'])&set(fair['selected_ids']))}곳이다. 지역 간 격차를 얼마나 줄일지는 정책 판단이며 임의 점수 가중치로 감추지 않았다.",
    f"형평성 계산에는 단계마다 {data['settings']['time_limit_per_phase_seconds']:g}초 제한을 두었다. 60곳의 최소 커버율은 {minpct(fair):.2f}%, 계산된 최적 가능 상한은 {100*fair['solver_phases'][0]['objective_upper_bound']:.2f}%다. 100곳의 최소 커버율은 {minpct(sc['100m_350m_p100_fair_all']):.2f}%, 상한은 {100*sc['100m_350m_p100_fair_all']['solver_phases'][0]['objective_upper_bound']:.2f}%다. 단계별 최적성 확정 여부와 시간 제한 상태는 결과 JSON에 기록했다."
],['구','인구 최대 60곳','형평성 우선 60곳'],[[n,f'{v:.2f}%',f'{w:.2f}%'] for n,v,w in zip(names,district_values(main),district_values(fair))],figure='district_equity_comparison.png')
section('4. 변수를 바꾸면 무엇이 달라지나',[
    f"가구를 수요로 쓰면 60곳의 가구 커버율은 {ratehh(main):.2f}%에서 {ratehh(hh):.2f}%로 높아지고, 인구 커버율은 {percent(main):.2f}%에서 {percent(hh):.2f}%로 바뀐다. 인구 최대안과 공통 후보는 {len(set(main['selected_ids'])&set(hh['selected_ids']))}곳이다. 두 수요를 임의 가중합하지 않고 별도 목적함수로 비교했다.",
    f"시설 내부 점포로 의심되는 59곳을 제외하는 가정에서는 후보가 804곳이 되고, 60곳의 인구 커버율은 {percent(restrict):.2f}%다. 이는 출입이 실제 불가능하다는 판정이 아니라 후보 적격성 민감도다.",
    '정류소 350m 이내 후보가 849/863곳이라 정류소 유무만으로는 후보를 잘 구별하기 어렵다. 정류소 수·의료업종 수는 보조 정보로 남겼고 인구에 더해 수요 점수를 만들지 않았다. 동 단위 고령인구 비율을 100m 격자에 복사해 고령자 커버율로 부르지도 않았다.'
],['변수','이번 분석 역할','근거·판단'],[
    ['총인구','접근 가능한 인구 최대화','커버형 입지모형의 직접 성과'],
    ['총가구','가구 접근성 최대화의 대안 목적','김경은 외(2025)의 가구 단위 수요 참고'],
    ['구별 커버율','최저 구의 접근률 우선','접근 공백을 다룬 Rico 외(2026)에서 착안한 별도 제안'],
    ['서비스 거리','200·350·500m 민감도','창원에서 검증된 행동 임계값은 아님'],
    ['점포 출입 조건','59곳 제외 가정','시설명 자동 표시, 현장 검증 전'],
    ['격자 크기','100m·500m 선정 및 동일 기준 재평가','공간 집계 민감도'],
    ['65세 이상 인구','읍면동 배경 설명','100m 관측값 미확보, 형평성 성과 계산에서 제외'],
    ['비용·용량·수거주기','운영 모형 설계 완료','실측 미확보로 이번 수치에 미적용']])
section('5. 거리와 격자 크기에 대한 민감도',[
    '200m·350m·500m는 비교용 서비스 기준이다. 넓은 반경에서 커버율이 높다는 이유로 그 반경을 최적이라고 고를 수 없다. 도로 단절·횡단보도·경사·출입구를 반영한 보행거리도 아니다.',
    f"500m 격자에서 선정한 조합은 100m 최적 조합과 {len(set(main['selected_ids'])&set(coarse['selected_ids']))}곳만 겹친다. 500m 격자의 반대각선은 약 353.6m로 서비스 반경 350m와 비슷하다. 통계의 해상도별 비밀보호·대표점·경계 배분 차이가 함께 작용하므로 하나의 원인 효과로 단정하지 않는다.",
    f"100m 격자도 중심점 오차가 남는다. 60곳 인구 최대안을 고정하고 반경을 350m±70.7m로 평가하면 커버율은 {data['cell_position_bounds_fixed_p60'][0]['population_coverage_pct']:.2f}~{data['cell_position_bounds_fixed_p60'][1]['population_coverage_pct']:.2f}%다. 격자 내부 위치만을 고려한 기하학적 보수 범위이며 통계적 신뢰구간이 아니다. 편의점 좌표 오차·보행망·미공개 인구의 불확실성은 이 범위에 들어 있지 않다."
],['직선 반경','60곳 인구 최대 커버율','전체 863곳의 커버 상한'],[
    [f'{rad}m',f"{percent(sc[f'100m_{rad}m_p60_population_all']):.2f}%",f"{next(b for b in data['baseline_all_candidates'] if b['radius_m']==rad)['population_coverage_pct']:.2f}%"] for rad in [200,350,500]],figure='grid_resolution_sensitivity.png')
gaps=dong.sort_values('all_uncovered_population',ascending=False).head(8)
section('6. 모든 편의점을 활용해도 남는 공백',[
    '아래 표는 전체 후보 863곳을 모두 가용하다고 가정해도 350m 밖에 남는 공개 인구가 많은 읍면동이다. 기존 수거함과 비교한 부족량이 아니다. 편의점 전용 수거망의 범위를 보여주며, 이 공백을 함께 보고 정책 적용 범위를 정하는 것이 좋다.',
    '고령 비율은 다른 시점의 동 단위 주민등록인구로 계산한 배경 정보다. 미커버 인구 중 고령자의 비율이나 고령자 미커버 인원으로 해석하면 안 된다.'
],['읍면동','전체 후보의 인구 커버율','남는 공개 인구','동의 65세 이상 비율'],[
    [r.district+' '+r.dong,f'{r.all_coverage_pct:.2f}%',f'{r.all_uncovered_population:,.0f}명 상당',f'{r.age65plus_share_pct:.2f}%'] for r in gaps.itertuples()])
section('7. 재현 가능한 모형과 현실화 순서',[
    '입지 변수 xⱼ는 후보 선택(0/1), 커버 변수 yᵢ는 수요 격자 커버(0~1), Aᵢⱼ는 거리 기준 충족(0/1), Nᵢ는 공개 인구 배분값이다. 기본식은 max ΣᵢNᵢyᵢ, 제약은 yᵢ ≤ ΣⱼAᵢⱼxⱼ, Σⱼxⱼ ≤ p, 0 ≤ yᵢ ≤ 1, xⱼ ∈ {0,1}이다. x가 이진이고 가중치가 비음수이므로 y를 연속으로 두어도 인구 최대화의 유효한 커버 값을 얻는다.',
    '형평성 식은 각 구 g에 대해 Σᵢ∈gNᵢyᵢ ≥ tΣᵢ∈gNᵢ를 두고 t를 최대화한다. 달성한 t를 유지하는 두 번째 계산에서 총 인구 커버를 최대화한다. 시 경계를 걸친 격자의 분할행은 같은 중심점으로 거리를 계산하되 각 구에 배분된 인구만 집계한다.',
    '다음 단계는 선택 후보의 실제 출입구와 보행 경로를 확인하고, 점주·본사 동의 및 수거 가능 시간을 조사하는 것이다. 그 뒤 같은 사업기간의 설치·관리·운반 비용, L/주 단위 유입량, 유효 용량, 최대 수거 간격을 연결하면 기존 운영 모형의 예산·용량 제약을 실제 값으로 적용할 수 있다.',
    '이번 결과만으로 최적 예산, 실제 회수량, 24시간 수거 효과, 수거차량 경로가 계산됐다고 쓰면 안 된다. 인구를 의약품 부피로 바꾸는 계수도 임의로 만들지 않았다. 운영식·근거표는 docs/변수_근거와_현실적_최적화_설계.md에 따로 수록했다.'
])
section('8. 검증과 제출 시 표현',[
    '23개 자동 검사를 통과했다. 같은 접근 후보를 가진 수요행을 합치는 계산 축약이 모든 작은 후보 조합에서 인구·집단별 분모를 보존하는지 확인했다. 경계 절단의 면적 배분, 경계 중첩 거절, 결측 유지, 인구 중복 제거, 작은 모형의 전수 조합 대조를 포함한다.',
    '100m 인구표가 있는 격자는 11,076개이며 창원 경계와 겹치는 전체 78,300개 격자 중 통계행이 없는 격자는 67,224개다. 이 개수는 누락 주민 수나 무인 격자 수가 아니다. 총가구가 없는 격자에 공개 인구 배분값 약 13,301명이 포함되어 있어 인구·가구 분모도 같지 않다.',
    'SGIS 격자 통계는 iLBA 비밀보호를 거친 값이다. 공식 이용안내는 5명 미만 값의 0 또는 5 치환과 더 큰 값에 대한 최대 ±7 조정을 설명한다. 0을 정확한 무인 상태로, 총인구와 남녀·해상도 간 합계 차이를 단순 오류로 해석하지 않았다. 개인정보를 역추정하지 않고 공개값 그대로 분석했다.',
    '시 경계에 걸쳐 배분된 100m 격자의 공개 인구는 약 5,267명 상당(전체의 약 0.52%)이다. 중심점이 시 안에 있는 격자의 원값을 합치면 1,007,827명으로 면적 배분합과 약 499명 차이가 난다. 읍면동 간 배분에는 격자 내 균일 분포 가정이 별도로 남는다.',
    f'권장 결과 표현: “창원 편의점 후보와 공개 인구 격자를 이용한 잠재 접근성 분석에서, 60곳·직선 350m의 인구 최대안은 공개 인구의 {percent(main):.2f}%를 커버했다. 지역별 최소 접근률을 우선한 실행 가능안은 {percent(fair):.2f}%의 전체 커버율을 유지하면서 최저 구 커버율을 {minpct(fair):.2f}%로 높였다.”',
    '프로젝트 ZIP에는 재현을 위한 원본과 결과가 포함돼 있다. SGIS 출처와 신청 시 동의한 이용조건을 최종 산출물에도 유지해야 한다. 이번 작업에서 보고서를 기관이나 공모전에 제출하지는 않았다.'
])
sources=[
    ('SGIS 신청자료·경계·공식 이용안내','https://sgis.mods.go.kr/view/pss/downloadList'),
    ('소상공인시장진흥공단 상가정보','https://www.data.go.kr/data/15083033/fileData.do'),
    ('창원시 연령별 인구','https://www.data.go.kr/data/15004972/fileData.do'),
    ('창원시 버스정류소 위치','https://www.data.go.kr/data/15037805/fileData.do'),
    ('Rico·White·Albert(2026), On locating prescription drug collection boxes','https://link.springer.com/article/10.1007/s11081-026-10080-x'),
    ('김경은 외(2025), 우체통의 위치를 반영한 갈현동 폐의약품 수거함 입지 — 초록','https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12140132'),
    ('Hasan 외(2023) — 용량·비용·이동을 반영한 설계 참고','https://aliciasassermodestino.com/wp-content/uploads/2025/05/Socio-Economic-Planning-Sciences.pdf')]
md='# 창원 편의점 폐의약품 수거망: 공간 연결과 최적화 결과\n\n2026-09-28 · 실제 자료 분석 · 직선거리 예비 입지안\n\n'
body=''
for sec in sections:
    md+='## '+sec['title']+'\n\n';body+='<section><h2>'+html.escape(sec['title'])+'</h2>'
    for p in sec['paras']:md+=p+'\n\n';body+='<p>'+html.escape(p)+'</p>'
    if sec['headers']:
        md+=mdtable(sec['headers'],sec['rows'])+'\n';body+=table(sec['headers'],sec['rows'])
    if sec['figure']:
        md+=f"![{sec['title']}](figures/{sec['figure']})\n\n"
        img=base64.b64encode((F/sec['figure']).read_bytes()).decode();body+=f'<img class="chart" alt="{html.escape(sec["title"])}" src="data:image/png;base64,{img}">'
    body+='</section>'
md+='## 출처\n\n'+''.join(f'- [{name}]({url})\n' for name,url in sources)
body+='<section><h2>출처</h2><ul>'+''.join(f'<li><a href="{url}">{html.escape(name)}</a></li>' for name,url in sources)+'</ul><p class="small">주요 데이터의 수집일·원본 해시와 계산 상태는 프로젝트의 JSON 파일에 기록했습니다. 선행연구에서 가져온 설계 원리와 이번 창원 계산 결과를 구분했습니다.</p></section>'
for phrase,url in [('김경은 외(2025)의 가구 단위 수요 참고','https://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE12140132'),('접근 공백을 다룬 Rico 외(2026)에서 착안한 별도 제안','https://link.springer.com/article/10.1007/s11081-026-10080-x')]:
    md=md.replace(phrase,f'[{phrase}]({url})')
    body=body.replace(phrase,f'<a href="{url}">{phrase}</a>')
(R/'창원_공간분석_최적화_결과.md').write_text(md,encoding='utf-8')
css='''*{box-sizing:border-box}body{margin:0;background:#eef2f3;color:#172b36;font:16px/1.8 -apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo","Malgun Gothic",sans-serif}main{max-width:1080px;margin:40px auto;padding:52px 60px;background:white;border-radius:16px;box-shadow:0 8px 40px #23394710}header{border-bottom:2px solid #176b75;padding-bottom:26px;margin-bottom:34px}.eyebrow{color:#176b75;font-size:13px;letter-spacing:.08em}h1{font-size:32px;line-height:1.4;margin:12px 0}h2{font-size:23px;margin:0 0 20px;color:#176b75}section{padding:28px 0;border-bottom:1px solid #e5eaec}p{margin:0 0 16px;word-break:keep-all}a{color:#176b75}table{border-collapse:collapse;width:100%;font-size:14px;line-height:1.6}th,td{padding:11px 12px;border-bottom:1px solid #dae3e6;text-align:left;vertical-align:top}th{background:#eaf3f3;color:#124a52}tr:nth-child(even){background:#f8fafb}.table-wrap{overflow:auto;margin:20px 0}.chart{width:100%;height:auto;margin:24px 0 8px}.links{display:flex;gap:10px;flex-wrap:wrap}.links a{padding:8px 14px;border:1px solid #176b75;border-radius:6px;text-decoration:none;font-size:14px}.small{font-size:13px;color:#60717a}@media(max-width:720px){main{margin:0;padding:25px 20px;border-radius:0}h1{font-size:26px}h2{font-size:21px}}@media print{body{background:white}main{box-shadow:none;margin:0;padding:20px;max-width:none}section{break-inside:avoid}.links{display:none}}'''
page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>창원 편의점 수거망 분석 결과</title><style>{css}</style></head><body><main><header><div class="eyebrow">CHANGWON · SPATIAL ANALYSIS · 2026.09.28</div><h1>편의점 수거망을 어디에 둘 것인가</h1><p>창원시 공개 자료의 공간 연결과 13개 최적화 시나리오</p><div class="links"><a href="창원_최적화_비교지도.html">60곳 배치 비교 지도</a><a href="selected_candidates_by_scenario.csv">선정 편의점 전체 목록</a><a href="dong_coverage_and_age_context.csv">55개 읍면동 결과표</a></div><p class="small" style="margin-top:16px">편의점 전용 · 기존 수거함 제외 · 공개 인구·가구 기준 · 직선거리 예비 입지안</p></header>{body}</main></body></html>'''
(R/'창원_공간분석_최적화_결과.html').write_text(page,encoding='utf-8')
print('Charts, map, report, and 55-dong results saved.')
