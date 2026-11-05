"""Render the completed distance analysis and newly acquired data audit."""
from pathlib import Path
import base64, html, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

P=Path(__file__).resolve().parent
R=P/'results/accessibility_extension'; F=R/'figures'; F.mkdir(exist_ok=True)
radius=json.loads((R/'radius_review.json').read_text())
walking=json.loads((R/'walking_data_audit.json').read_text())
variables=json.loads((R/'variable_data_audit.json').read_text())
rows=radius['summary']; bus=variables['bus']; age=radius['older_area_diagnostics']
font_manager.fontManager.addfont('/System/Library/Fonts/AppleSDGothicNeo.ttc')
plt.rcParams.update({'font.family':'Apple SD Gothic Neo','axes.unicode_minus':False,
    'font.size':12,'figure.facecolor':'white','savefig.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'text.color':'#183b40',
    'axes.labelcolor':'#183b40','xtick.color':'#52676c','ytick.color':'#52676c'})
TEAL='#147d85'; ORANGE='#d38229'; GREY='#8a9da3'; DARK='#183b40'; BLUE='#527ac0'
common='SGIS 공개 인구 2024 / 기존 439개 좌표·편의점 후보 802곳 / 최대 60곳 추가·직선거리'

def heading(fig,title,subtitle):
    fig.text(.055,.955,title,fontsize=21,weight='bold',va='top')
    fig.text(.055,.89,subtitle,fontsize=12,color='#5b6e73',va='top')

def save(fig,name,note):
    fig.text(.055,.065,note,fontsize=9.5,color='#61757a',va='top')
    fig.savefig(F/name,dpi=165);plt.close(fig)

r=np.array([a['radius_m'] for a in rows])
fig,ax=plt.subplots(figsize=(12,7.3));fig.subplots_adjust(left=.10,right=.94,bottom=.21,top=.77)
heading(fig,'반경을 바꾸면 접근성 추정도 달라진다','같은 반경 안에서 기존망과 추가 배치를 비교해야 한다.')
ax.plot(r,[a['baseline_pct'] for a in rows],'o-',color=GREY,lw=2.3,label='기존 수거망')
ax.plot(r,[a['reoptimized_pct'] for a in rows],'o-',color=TEAL,lw=2.7,label='반경마다 다시 선정한 효율안')
ax.plot(r,[a['fixed_eff350_pct'] for a in rows],'s--',color=ORANGE,lw=2,label='350m 효율안 60곳을 그대로 평가')
ax.axvline(350,color=DARK,lw=1,alpha=.2)
ax.annotate('87.40%',(350,87.40003),xytext=(327,96),fontsize=12,color=TEAL,weight='bold',arrowprops={'arrowstyle':'-','color':TEAL})
ax.annotate('기존 69.15%',(350,69.15177),xytext=(364,62),fontsize=11,color='#60767e',arrowprops={'arrowstyle':'-','color':GREY})
ax.set(xticks=r,xlabel='서비스 범위로 가정한 직선 반경 (m)',ylabel='공개 인구 커버율 (%)',ylim=(30,101))
ax.grid(axis='y',alpha=.17);ax.legend(loc='lower right',frameon=False,fontsize=11)
save(fig,'01_radius_coverage.png',common+'\n반경이 커져 커버율이 높아지는 현상은 해당 거리가 적절하다는 근거가 아니다.')

fig,(a,b)=plt.subplots(1,2,figsize=(12,7.1));fig.subplots_adjust(left=.08,right=.95,bottom=.23,top=.74,wspace=.3)
heading(fig,'전체 효과는 비슷해도, 점포 명단은 달라질 수 있다','300·400m에서 기존 350m 효율안의 손실은 1%p 미만이다. 점포 교집합은 각각 47곳·40곳이다.')
x=np.arange(len(rows));vals=np.array([v['fixed_eff350_regret_pp'] for v in rows])
bars=a.bar(x,vals,color=[ORANGE if v in [200,250] else TEAL for v in r],width=.65)
a.bar_label(bars,labels=[f'{v:.2f}' for v in vals],padding=5,fontsize=10)
a.set(xticks=x,xticklabels=r,ylim=(0,5.3),ylabel='동일 반경 최적안 대비 커버율 손실 (%p)',xlabel='평가 반경 (m)');a.grid(axis='y',alpha=.15)
bars=b.bar(x,[v['shared_with_eff350'] for v in rows],color=[ORANGE if v==350 else BLUE for v in r],width=.65)
b.bar_label(bars,padding=5,fontsize=11);b.set(xticks=x,xticklabels=r,ylim=(0,72),ylabel='350m 효율안과 공통으로 선정된 점포 (곳)',xlabel='다시 선정한 반경 (m)');b.grid(axis='y',alpha=.15)
save(fig,'02_selection_stability.png',common+'\n각 반경에서 계산된 하나의 최적해를 비교했다. 동률인 다른 최적해까지 모두 조사한 결과는 아니다.')

fig,ax=plt.subplots(figsize=(12,6.8));fig.subplots_adjust(left=.10,right=.94,bottom=.23,top=.75)
heading(fig,'구의 균형을 높여도, 고령인구 비중이 높은 동에는 공백이 남는다','65세 이상 비중이 상위 25%인 14개 읍면동을 묶어, 그 지역 전체 주민의 접근성을 평가했다.')
x=np.arange(3);area=list(age['rates'].values());total=[rows[3]['baseline_pct'],rows[3]['reoptimized_pct'],rows[3]['fixed_fair350_pct']]
bars1=ax.bar(x-.18,total,.34,color=GREY,label='창원 전체 공개 인구');bars2=ax.bar(x+.18,area,.34,color=TEAL,label='고령인구 비중 상위 14개 동의 전체 공개 인구')
for bars in [bars1,bars2]:ax.bar_label(bars,labels=[f'{p.get_height():.2f}%' for p in bars],padding=5,fontsize=12)
ax.set(xticks=x,xticklabels=['기존망','효율안 60곳 추가','5개 구 형평안 60곳 추가'],ylabel='인구 커버율 (%)',ylim=(0,105));ax.legend(frameon=False,loc='upper left',fontsize=11);ax.grid(axis='y',alpha=.12)
save(fig,'03_older_area_gap.png','직선 350m / 연령 구성 2025.5·격자 전체 인구 2024 / 상위 25% 경계: 65세 이상 비중 30.08%\n고령자 개인의 커버율·폐의약품 발생량을 계산한 결과가 아니다. 연령 구성은 배치 후 지역 진단에 사용했다.')

fig,ax=plt.subplots(figsize=(12,6.8));fig.subplots_adjust(left=.17,right=.93,bottom=.23,top=.74)
heading(fig,'공개 도로 자료는 확보했다. 실제 통행 가능성은 더 확인해야 한다','가장 가까운 지도 선까지의 거리로 연결 가능성을 점검했다. 아래 값은 수거함 커버율이 아니다.')
keys=['population','candidates','existing'];labels=['인구 격자 중심점\n(공개 인구 가중)','편의점 후보 802곳\n(좌표 수 기준)','기존 수거지 439곳\n(좌표 수 기준)']
p50=np.array([walking[k]['within_segment_distance_pct']['50'] for k in keys]);p100=np.array([walking[k]['within_segment_distance_pct']['100'] for k in keys]);y=np.arange(3)
ax.barh(y,p50,color=TEAL,label='지도 선에서 50m 이내');ax.barh(y,p100-p50,left=p50,color='#a4d5d7',label='50m 초과~100m 이내');ax.barh(y,100-p100,left=p100,color='#e0e6e8',label='100m 초과')
for i,v in enumerate(p50):ax.text(v/2,i,f'{v:.1f}%',ha='center',va='center',color='white',weight='bold',fontsize=13)
ax.set(yticks=y,yticklabels=labels,xlim=(0,100),xlabel='각 집단에서 차지하는 비율 (%)');ax.invert_yaxis();ax.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=3,frameon=False,fontsize=10.5)
save(fig,'04_road_linkage_audit.png','출처: © OpenStreetMap contributors, ODbL / 창원과 주변 영역, 2026.9.29 확보\n출입구·횡단보도·차단시설·계단·경사 검증 전이다. 도로 중심선에 가깝다고 안전하게 걸어갈 수 있다는 뜻은 아니다.')

def image(name,alt):
    data=base64.b64encode((F/name).read_bytes()).decode()
    return f'<img class="chart" alt="{html.escape(alt)}" src="data:image/png;base64,{data}">'

table=''.join(f"<tr><td>{v['radius_m']}m</td><td>{v['baseline_pct']:.2f}%</td><td>{v['reoptimized_pct']:.2f}%</td><td>{v['fixed_eff350_pct']:.2f}%</td><td>{v['fixed_fair350_pct']:.2f}%</td><td>{v['fixed_eff350_regret_pp']:.2f}%p</td></tr>" for v in rows)
health_table=''.join(f"<tr><td>{v['district']}</td><td>{v['hypertension_patients']:,}</td><td>{v['diabetes_patients']:,}</td></tr>" for v in variables['health']['rows'])
older_dongs=' · '.join(v['district']+' '+v['dong'] for v in age['district_dongs'])
source_health='https://www.nhis.or.kr/nhis/together/wbhaec06900m01.do?articleNo=11006399&amp;mode=view'
source_bus='https://www.data.go.kr/data/15143794/fileData.do'
source_paper='https://journals.sagepub.com/doi/10.1177/0734242X221105416'

page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>창원 수거망 확장 분석 · 거리 기준과 데이터 확보</title>
<style>
:root{{--ink:#183b40;--muted:#597078;--teal:#147d85;--line:#dce5e6;--bg:#f4f7f7}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'Apple SD Gothic Neo','Noto Sans KR',sans-serif;line-height:1.7}}main{{max-width:1130px;margin:auto;padding:36px 28px 80px}}header{{background:#163e43;color:white;border-radius:20px;padding:40px 42px}}header .eyebrow{{color:#a9dadc;font-size:13px;letter-spacing:.09em}}h1{{font-size:36px;line-height:1.3;letter-spacing:-.04em;margin:14px 0 20px}}header p{{max-width:850px;color:#dce9eb;margin:0}}nav{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}a{{color:#116b77;text-underline-offset:3px}}nav a{{background:white;padding:8px 15px;border:1px solid var(--line);border-radius:30px;text-decoration:none;font-size:14px}}section{{background:white;border:1px solid var(--line);border-radius:16px;margin:22px 0;padding:30px}}h2{{font-size:25px;letter-spacing:-.02em;margin:0 0 16px}}h3{{font-size:19px;margin:24px 0 10px}}p{{margin:10px 0 16px}}.lead{{font-size:19px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{padding:22px;background:#f0f6f5;border-radius:13px}}.card b{{display:block;font-size:26px;line-height:1.35;margin:8px 0;color:#116f77}}.small,.note{{font-size:14px;color:var(--muted)}}.callout{{border-left:4px solid var(--teal);background:#eef7f6;padding:16px 20px;margin:20px 0}}.amber{{border-left-color:#c88634;background:#fff7eb}}.chart{{display:block;width:100%;height:auto;border:1px solid #e6eeee;border-radius:10px;margin:22px 0}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:12px 13px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}th{{background:#eff5f5;font-weight:600}}.scroll{{overflow-x:auto}}.pill{{display:inline-block;font-size:12px;border-radius:20px;padding:2px 9px;background:#def0ec;color:#166856;white-space:nowrap}}.pending{{background:#fff0d9;color:#8d5d21}}details{{border-top:1px solid var(--line);margin-top:18px;padding-top:15px}}summary{{cursor:pointer;font-weight:600}}li{{margin:7px 0}}.flow{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}}.flow div{{border-top:3px solid #329098;background:#f4f8f8;padding:17px}}.flow strong{{display:block;margin-bottom:8px}}footer{{color:#61757a;font-size:13px}}@media(max-width:750px){{main{{padding:16px 12px 50px}}header{{padding:28px 22px}}h1{{font-size:28px}}section{{padding:22px 17px}}.grid,.flow{{grid-template-columns:1fr}}h2{{font-size:22px}}}}@media print{{body{{background:white}}main{{padding:0}}nav{{display:none}}section{{break-inside:avoid}}a{{color:inherit}}}}
</style></head><body><main>
<header><div class="eyebrow">창원 폐의약품 수거망 · 확장 분석 / 2026.09.29</div><h1>350m는 얼마나 타당할까?<br>새 변수를 넣을 자료는 확보할 수 있을까?</h1><p>기존 수거망을 유지하면서 생활 가까이에 편의점 수거 지점을 더한다는 방향을 바탕으로, 거리 기준을 다시 검토하고 추가 자료를 직접 내려받았다. 이번에는 효과의 민감도와 실제 분석에 사용할 수 있는 데이터 범위를 확인했다.</p></header>
<nav><a href="#findings">핵심 결과</a><a href="#radius">350m 검토</a><a href="#data">데이터 확보</a><a href="#equity">고령 지역 공백</a><a href="#next">분석 방향</a><a href="#methods">계산·원본</a></nav>
<section id="findings"><h2>거리 민감도 분석은 완료했고, 자료 3종을 새로 확보했다</h2><div class="grid"><div class="card">직선 반경 비교<b>200~500m · 7개</b><span class="small">효율안을 반경별로 계산하고, 350m 선정안도 각 반경에서 재평가했다.</span></div><div class="card">350m 효율안의 성능 손실<b>0.73 / 0.38%p</b><span class="small">각각 300m·400m 기준 최적안 대비. 200m에서는 4.22%p로 커진다.</span></div><div class="card">새로 내려받은 원본<b>도로 · 진료 · 승차량</b><span class="small">자료를 확보한 단계와 최적화에 바로 사용할 수 있는 단계를 구분했다.</span></div></div>
<div class="callout"><strong>판단:</strong> 350m를 당장 다른 숫자로 바꿀 근거는 충분하지 않다. 현재 비교 기준으로 표시하고, 200~500m 결과를 함께 제시하는 편이 타당하다. 실제 보행 경로로 검증한 뒤 거리·시간 기준을 다시 판단해야 한다.</div>
<p>현재 결과는 <strong>기존 439개 좌표 + 검토 후보 802곳 중 최대 60곳 추가</strong>를 전제로 한다. 60곳은 비교를 위한 설치 수이며, 예산과 점포 참여가 확정된 사업 규모는 아니다. 커버율은 공개 인구 격자 중심점에서 범위 안에 시설이 있는 비율이다. 실제 배출 참여율이나 회수량을 뜻하지 않는다.</p>
<div class="flow"><div><strong>질문 1 · 어디가 불편한가?</strong>기존망 밖의 인구와 지역별 접근 공백을 찾는다.</div><div><strong>질문 2 · 어디를 더하면 도움이 되나?</strong>편의점을 추가할 때 인구 접근성과 지역 균형을 비교한다.</div><div><strong>질문 3 · 조건이 달라도 도움이 되나?</strong>거리·보행·이용 시간·참여 조건을 바꾸어 결론을 점검한다.</div></div></section>
<section id="radius"><h2>350m의 근거와 이번 검증을 나누어 봐야 한다</h2>
<p>Veiga 등의 포르투갈 가구 조사(n=454)는 약국까지의 거리와 폐의약품 처리 방식 사이의 관련성을 보고했다. 200~500m에 사는 집단이 약국에 반납하는 경향도 보고했지만, 이는 <strong>설문 거리 구간에 대한 결과</strong>다. 350m를 최적 기준으로 추정하거나, 창원의 편의점·직선거리 커버리지를 검증한 연구는 아니다. 표본도 고학력층과 특정 지역에 편중되어 있다. <a href="{source_paper}">Veiga et al., 2023 (온라인 공개 2022)</a></p>
<p>따라서 이 논문은 <strong>거리 변수를 검토할 이유</strong>를 뒷받침한다. 이번 문헌 확인을 과거 350m 설정의 근거로 소급하지 않았다. 확인한 자료에서는 폐의약품 수거의 표준 반경을 350m로 정한 근거를 찾지 못했다.</p>
<div class="callout amber">직선 350m를 ‘도보 5분’으로 표현하면 안 된다. 보행속도 1.2m/s와 우회·신호대기가 전혀 없다는 가정에서만 약 4.9분이다. 이 계산은 설명용 가정이며 창원의 실제 이동시간을 측정한 값이 아니다.</div>
{image('01_radius_coverage.png','기존망, 반경별 효율안, 고정 350m 효율안의 인구 커버율 비교')}
<p>반경별 효율안 7개는 모두 계산된 모형에서 최적성 간격 0으로 확인됐다. 250·300·400·450m는 이번에 새로 최적화했고, 200·350·500m는 기존 최적해를 불러와 별도로 커버율을 검산했다.</p>
<div class="scroll"><table><thead><tr><th>평가 반경</th><th>기존망</th><th>반경별 효율안</th><th>350m 효율안 고정</th><th>350m 형평안 고정</th><th>고정 효율안 손실</th></tr></thead><tbody>{table}</tbody></table></div>
<p class="note">모든 비율은 기존망과 추가 점포를 합친 총 커버율이다. 형평안은 기존 350m 명부를 고정해 평가했으며, 다른 반경에서 형평 목적을 다시 최적화하지 않았다. 손실은 같은 반경의 효율안끼리 비교한 %p 차이다.</p>
{image('02_selection_stability.png','350m 효율안의 반경별 성능 손실과 공통 점포 수')}
<p>7개 해에서 한 번이라도 선정된 점포는 <strong>151곳</strong>, 모두에서 공통 선정된 점포는 <strong>5곳</strong>이었다. 다만 반경별로 가능한 모든 동률 최적해를 탐색하지 않았으므로, 이 5곳을 반드시 설치해야 할 점포로 해석할 수는 없다. 전체 효과의 안정성과 개별 점포 명단의 안정성을 구분해 검토해야 한다.</p>
<p><a href="candidate_radius_frequency.csv">151개 후보의 반경별 선정 빈도</a> · <a href="radius_cross_evaluation.csv">49개 배치·평가 반경 조합 결과</a></p></section>
<section id="data"><h2>데이터 확보 결과: 구할 수 있지만 쓰임새가 다르다</h2>
<div class="scroll"><table><thead><tr><th>변수</th><th>확보 상태</th><th>공간·시간 단위</th><th>이번에 확인한 활용 범위</th></tr></thead><tbody>
<tr><td>실제 보행 접근성</td><td><span class="pill">도로 원본 확보</span></td><td>OpenStreetMap 선·노드 / 2026.9.29</td><td>지도 선과 좌표의 근접성 점검 완료. 실제 보행 경로 최적화는 아직 완료하지 않음.</td></tr>
<tr><td>의약품 관련 수요</td><td><span class="pill">진료 통계 확보</span></td><td>2024 지역별 의료이용 통계 / 창원 5개 구</td><td>고혈압·당뇨 진료실인원 추출. 구 단위 배경 지표로 사용 가능. 동·격자별 수요를 직접 만들 수는 없음.</td></tr>
<tr><td>일상 이동과 정류장 이용</td><td><span class="pill">승차량 원본 확보</span><br><span class="pill pending">코드 연결 필요</span></td><td>2025.5.19~25 / 정류장별 합계, 노선별 시간대</td><td>기존 위치 자료와 코드 불일치. 이름의 일대일 일치만으로는 일부만 연결되어 점포 선정 변수로 사용 보류.</td></tr>
<tr><td>고령 지역의 접근성</td><td><span class="pill">기존 자료 활용</span></td><td>55개 읍면동 / 2025.5.31</td><td>고령인구 비중이 높은 지역을 구분해 기존 선정안의 공백을 추가 진단.</td></tr>
<tr><td>실제 수거 가능 시간</td><td><span class="pill pending">확인 자료 부족</span></td><td>시설·요일·시간대별 필요</td><td>확보한 수거지 파일에 시간 항목이 없음. 이번 확인에서 일괄 공개 자료를 확보하지 못함.</td></tr>
<tr><td>참여·용량·회수비용</td><td><span class="pill pending">운영 자료 미확보</span></td><td>점포·회수 방식별 필요</td><td>협약 참여율, 수거함 용량, 회수비용은 실제 최적화 입력으로 넣지 않음.</td></tr>
</tbody></table></div>
<h3>도로: 공간 연결을 진행할 기반은 확보했다</h3>
<p>창원과 주변 영역의 OSM 도로 자료를 내려받아 통행 제한 표기를 일부 제외한 뒤 <strong>{walking['included_ways']:,}개 선 객체</strong>를 점검했다. 다운로드 영역 안에 분석 인구 격자 중심점이 모두 들어간다. OSM은 공공기관 자료와 구분되는 공개지도이며, 공식 보행망과의 대조 검증은 남아 있다.</p>
{image('04_road_linkage_audit.png','공개지도 선에서 50m 및 100m 이내에 있는 인구와 시설 좌표의 비율')}
<p>다음 보행 계산에서는 출입구 연결, 횡단 위치, 통행 제한과 단절 구간부터 검증해야 한다. 이번 원본의 노드는 좌표 중심으로 받았으므로 차단시설·문 개방 시간 등의 추가 정보도 필요하다. 지도에 길이 빠진 지역을 실제로 이동할 수 없는 지역으로 단정하지 않는다. <a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors, ODbL</a></p>
<h3>진료: 원본은 확보했지만 세밀한 수요 가중치로 쓰기에는 거칠다</h3>
<p><a href="{source_health}">국민건강보험공단 2024 지역별 의료이용 통계연보</a>의 경남 포함 엑셀에서 창원 5개 구를 추출했다. 아래는 질환별 진료실인원이며, 질환 간 중복이 있으므로 두 열을 합쳐 인원수로 쓰면 안 된다.</p>
<table><thead><tr><th>지역</th><th>고혈압 진료실인원</th><th>당뇨 진료실인원</th></tr></thead><tbody>{health_table}</tbody></table>
<p class="note">폐의약품 발생량·미사용 의약품 수량은 아니다. 구별 인구·연령 구조가 달라 단순 인원만으로 우선순위를 판단할 수 없다. 이번 추출에서 거주지 귀속 기준과 연령 표준화는 별도로 검증하지 않았으며, 100m 격자로 배분하거나 최적화 가중치로 사용하지 않았다. 원본 시트·행 번호는 <a href="nhis_district_context_2024.csv">추출표</a>에 보존했다.</p>
<h3>버스: 위치 자료와의 연결이 현재 가장 큰 제약이다</h3>
<p><a href="{source_bus}">창원 시내버스 탑승인원 원본</a>에는 정류장 {bus['stop_rows']:,}개의 일주일 합계와 {bus['route_sheets']}개 노선의 시간대별 표가 있다. 기존 정류소 위치 {bus['location_rows']:,}건과 코드가 바로 맞는 기록은 없었다. 양쪽에서 이름이 중복 없이 일치하는 기록은 <strong>270개({bus['unique_name_match_pct']:.2f}%)</strong>이며, 승차 건수의 {bus['matched_boarding_event_pct']:.2f}%에 해당한다. 이 연결도 기준 시점이 달라 잠정적이다.</p>
<div class="callout amber">현재 일부만 연결된 승차량을 점포 점수에 넣으면 누락 지역이 불리해진다. 정류장 코드 대응표를 먼저 확보해야 한다. 정류장 합계에는 시간대가 없고, 시간대 자료는 노선 단위이므로 ‘정류장별 야간 유동인구’를 직접 만들 수도 없다.</div>
<p class="note">정류장 표의 합계는 {bus['stop_boarding_events']:,}건, 노선 시간대 표의 합계는 {bus['route_hour_boarding_events']:,}건으로 일치하지 않았다. 범위 차이의 원인은 원본 제공기관 확인 전이며 두 표를 합산하지 않았다. 승차 건수는 반복 이용·환승을 포함해 실제 사람 수, 의약품 수요, 야간 배출 수요와 다르다. 원본은 일주일 관측으로 계절 대표성이 확인되지 않았다.</p>
</section>
<section id="equity"><h2>지금 가진 변수로도 지역 공백을 더 구체적으로 볼 수 있다</h2>
{image('03_older_area_gap.png','창원 전체와 고령인구 비중 상위 14개 동의 전체 주민 커버율 비교')}
<p>65세 이상 비중이 상위 25%인 14개 읍면동의 전체 주민 커버율은 기존망 <strong>{area[0]:.2f}%</strong>, 효율안 추가 후 <strong>{area[1]:.2f}%</strong>, 기존 형평안 추가 후 <strong>{area[2]:.2f}%</strong>였다. 5개 구의 최저 커버율을 높이는 것만으로는 동 단위의 공백이 충분히 해소되지 않을 수 있다.</p>
<p>이 분석에서 연령 변수는 <strong>선정 결과를 평가할 지역 집단을 정하는 데</strong> 사용했다. 고령자일수록 폐의약품이 일정 비율 더 생긴다는 가중치를 부여하지 않았다. 상위 25%는 이번 진단을 위한 구분이며 검증된 위험 기준은 아니다.</p>
<details><summary>14개 지역과 해석 범위</summary><p>{older_dongs}</p><p class="note">2025년 5월 연령 구성으로 집단을 정하고 2024년 공개 전체 인구 격자로 커버율을 계산했다. 고령자 개인의 거주 격자를 알 수 없으므로 고령자 커버율로 바꾸어 말할 수 없다. 고령층의 이동속도·계단 이용·실제 방문도 측정하지 않았다.</p></details></section>
<section id="next"><h2>추천하는 확장: 변수를 질문에 맞게 배치한다</h2>
<p class="lead">현재의 ‘기존망 진단 → 편의점 보완’ 구성을 유지하면서, 보행 접근성과 동 단위의 공백을 우선 보강하는 방향이 적절하다.</p>
<div class="scroll"><table><thead><tr><th>연구 질문</th><th>변수의 역할</th><th>구체적인 분석</th></tr></thead><tbody>
<tr><td>기존 수거지에 쉽게 갈 수 있는가?</td><td>인구는 수요, 거리는 도달 조건</td><td>직선 200~500m 비교는 완료. 다음은 검증된 보행망으로 5·10분 등 계획 시나리오를 비교한다. 시간 기준 자체도 정책 가정으로 표시한다.</td></tr>
<tr><td>어떤 지역의 공백을 먼저 메울까?</td><td>연령 구성은 형평성 평가 집단</td><td>5개 구 균형과 고령인구 비중이 높은 동의 공백을 함께 보고, 전체 커버율과 지역 개선 사이의 교환관계를 비교한다.</td></tr>
<tr><td>평소 이동 중에 들를 수 있을까?</td><td>승차량은 이동 활동의 보조 지표</td><td>코드 연결과 표별 합계 차이를 확인한 뒤 활용한다. 거주 인구와 승차 건수를 더해 하나의 수요량으로 만들지 않는다.</td></tr>
<tr><td>퇴근 후에도 배출할 수 있을까?</td><td>수거 시간은 시설 이용 조건</td><td>실제 수거 가능 시간이 확인된 시설만 시간대별 모형에 넣는다. 편의점 영업시간을 수거 가능 시간으로 그대로 간주하지 않는다.</td></tr>
<tr><td>설치 후 계속 운영할 수 있을까?</td><td>참여·예산·용량은 제약 조건</td><td>기존 점포 제외·대체 분석을 유지하고, 실제 비용·용량이 확보되면 설치 수 제한을 예산·운영 제약으로 확장한다.</td></tr>
</tbody></table></div>
<p>핵심은 변수마다 다른 역할을 주는 것이다. 인구·거리·연령·승차량을 임의의 비율로 더한 점수보다, <strong>누구의 어떤 불편을 줄이는지</strong>에 따라 목적함수, 이용 조건, 형평성 기준, 결과 평가를 나누는 편이 해석하기 쉽다. 이번에는 새 진료·승차량 자료를 기존 최적화에 넣지 않았다.</p>
<div class="callout"><strong>이번 단계에서 확정할 수 있는 결론:</strong> 거리 가정을 바꾸어도 편의점 추가의 잠재적 접근성 개선은 확인된다. 350m 근처에서는 효율안의 전체 성능 차이가 작지만, 개별 점포와 취약 지역의 결과는 별도로 살펴봐야 한다. 실제 수거 참여나 야간 이용 개선은 아직 검증한 결과가 아니다.</div></section>
<section id="methods"><h2>계산 범위와 확인 자료</h2><p>공개 인구 합계는 {radius['population']:,.2f}명이다. 원래 분석과 같은 100m 격자 중심점, 창원 경계 교차 면적 배분, EPSG:5179 거리 계산을 유지했다. SGIS 결측은 실제 인구 0명으로 단정하지 않았다. 후보 802곳은 확보·검토한 좌표 수이며 전체 영업 편의점의 완전한 명부로 검증된 수는 아니다.</p>
<ul><li>효율안은 기존망 밖에서 추가로 커버하는 공개 인구를 최대화한다. 여러 시설에 포함되는 인구는 한 번만 센다.</li><li>반경 외 조건을 같게 유지했다. 실제 수거 가능 시간·점포 참여·회수량·예산·용량은 미반영이다.</li><li>기존 350m 형평안은 5개 구의 최저 총 커버율을 우선 최대화한 결과다. 이번 반경 민감도에서 다른 거리의 형평 최적해를 새로 구하지 않았다.</li><li>결과에 입력 해시, 해의 상태, 선정 점포 ID를 보존했다. 기존 반경 해의 재계산 일치, 기존망 유지, 선정 수, 고정 명부의 반경별 단조성을 확인했다.</li></ul>
<p><a href="radius_summary.csv">반경별 요약</a> · <a href="radius_review.json">상세 계산 결과</a> · <a href="walking_data_audit.json">도로 연결 점검</a> · <a href="variable_data_audit.json">추가 변수 점검</a> · <a href="boarding_stop_linkage_audit.csv">정류장 연결 내역</a></p>
<details><summary>다운로드한 원본과 출처</summary><ul><li><a href="../../data/raw/changwon_osm_highways_20260929.json">OSM 도로 원본</a> / <a href="../../data/raw/changwon_osm_highways_20260929.metadata.json">쿼리·시각·해시</a>. <a href="https://www.openstreetmap.org/copyright">© OpenStreetMap contributors, ODbL</a>.</li><li><a href="../../data/raw/nhis_gyeongnam_medical_use_2024.xlsx">2024 의료이용 통계 엑셀</a> / <a href="{source_health}">국민건강보험공단 원문 게시물</a>.</li><li><a href="../../data/raw/changwon_boardings_20250519_0525.xlsx">버스 탑승인원 엑셀</a> / <a href="{source_bus}">공공데이터포털 원문</a>.</li><li><a href="{source_paper}">Veiga et al., End-of-life management strategies of pharmaceuticals in Portuguese households</a>, Waste Management & Research 41(1), 235–247. DOI:10.1177/0734242X221105416.</li></ul></details>
<p><a href="../supplement_development/창원_기존망진단과_편의점보완_보고서.html">기존망 진단과 편의점 보완 본 보고서로 이동</a></p></section>
<footer>2026.09.29 분석 확장본 · 원본 파일은 보존했다. SGIS 출처 명시와 신청 시 동의한 이용조건을 유지한다. 이 파일의 그림은 인터넷 연결 없이 표시된다.</footer>
</main></body></html>'''
target=R/'창원_거리기준과_데이터확보_확장분석.html'
target.write_text(page)
print(target)
