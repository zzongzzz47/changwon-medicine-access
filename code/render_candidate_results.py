from pathlib import Path
import json,html,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import folium
from folium.plugins import MarkerCluster
P=Path(__file__).resolve().parent;R=P/'results';F=R/'figures'
font_path=Path('/System/Library/Fonts/AppleSDGothicNeo.ttc')
if font_path.exists(): font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({'font.family':('Apple SD Gothic Neo' if font_path.exists() else 'sans-serif'),'axes.unicode_minus':False,'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6f8fb','axes.facecolor':'#f6f8fb'})
d=json.loads((R/'changwon_supply_demographics.json').read_text());ds=d['by_district'];names=[r['district'] for r in ds]
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
colors=['#416d9d','#416d9d','#bf663e','#bf663e','#416d9d']
for ax,field,title,fmt in [(axes[0],'age65plus_share_pct','구별 65세 이상 인구 비율','{:.1f}%'),(axes[1],'age65plus_per_candidate','후보 위치 1곳당 65세 이상 인구','{:.0f}명')]:
    val=[r[field] for r in ds]; bars=ax.barh(names,val,color=colors,height=.6);ax.invert_yaxis();ax.set_xlim(0,max(val)*1.28);ax.set_title(title,loc='left',fontweight='bold',pad=16);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    for bar,v in zip(bars,val): ax.text(v+max(val)*.03,bar.get_y()+bar.get_height()/2,fmt.format(v),va='center')
    ax.tick_params(axis='both',length=0)
fig.suptitle('창원시 편의점 후보와 고령인구 분포',ha='left',x=.04,fontweight='bold',fontsize=18)
fig.supxlabel('인구: 2025.5.31 / 편의점: 2026.6.30 · 동일 좌표는 한 후보 위치로 통합 · 인구 커버율과 수거량을 뜻하지 않음',fontsize=10,color='#555',y=-.03)
fig.savefig(F/'district_supply_age.png',dpi=180,bbox_inches='tight');plt.close(fig)
cs=json.loads((P/'data/processed/convenience_candidates.json').read_text())
m=folium.Map(location=[35.19,128.6],zoom_start=11,tiles='OpenStreetMap',control_scale=True)
groups={name:folium.FeatureGroup(name=name,show=True).add_to(m) for name in names}
clusters={name:MarkerCluster().add_to(group) for name,group in groups.items()}
for s in cs:
    flags='<br>'.join(html.escape(x) for x in s['flags']) or '자동 점검 특이사항 없음 (현장 확인 전)'
    popup=f"<b>{'<br>'.join(html.escape(x) for x in s['names'])}</b><br>{'<br>'.join(html.escape(x) for x in s['addresses'])}<hr>원자료 {s['raw_count']}건을 같은 좌표로 묶음<br>가까운 정류소: 직선 {s['nearest_bus_stop_m']:.0f}m<br>350m 내 정류소: {s['bus_stop_records_within_350m']}개 기록<br>350m 내 의료업종·약국: {s['health_business_records_within_350m']}개 기록<hr>{flags}<br><small>영업시간·설치동의·정확한 출입구 미확인</small>"
    folium.Marker([s['lat'],s['lon']],tooltip=s['names'][0],popup=folium.Popup(popup,max_width=400),icon=folium.Icon(color='orange' if '시설내점포_출입확인필요' in s['flags'] else 'blue',icon='info-sign')).add_to(clusters[s['district']])
folium.LayerControl(collapsed=False).add_to(m)
m.get_root().html.add_child(folium.Element('''<div style="position:fixed;bottom:25px;left:15px;z-index:9999;background:white;padding:14px;max-width:310px;border-radius:8px;box-shadow:0 2px 10px #aaa;font-family:sans-serif;font-size:13px"><b>창원 편의점 후보 위치 863곳</b><br>편의점 분류 업소기록 950건을 좌표별로 정리.<br>주황색 핀은 시설 내부 점포 등 출입 확인 대상.<br>숫자가 있는 원은 가까운 후보를 묶은 표시.<br><b>최종 선정 위치·인구 커버 지도는 아직 아닙니다.</b><br><small>상가정보 2026.6.30 / 정류소 2025.12.31<br>배경지도 및 지도 기능은 인터넷 연결 필요.</small></div>'''))
m.save(str(R/'창원_편의점_후보지도.html'))
print('Saved chart and interactive candidate map')
