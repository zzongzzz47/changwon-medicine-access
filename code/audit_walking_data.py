"""Audit mapping proximity and connectivity without claiming verified walkability."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import linestrings,points,STRtree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

P=Path(__file__).resolve().parent;R=P/'results/accessibility_extension'
def main():
 raw=json.loads((P/'data/raw/changwon_osm_highways_20260929.json').read_text())
 nodes={e['id']:(e['lon'],e['lat']) for e in raw['elements'] if e['type']=='node'}
 ways=[e for e in raw['elements'] if e['type']=='way'];keep=[];excluded=Counter();classes=Counter()
 for w in ways:
  t=w.get('tags',{});h=t.get('highway','');foot=t.get('foot');access=t.get('access')
  if h in ['motorway','motorway_link','construction','proposed','raceway'] or t.get('area')=='yes':excluded['nonwalk_or_area']+=1;continue
  if h in ['trunk','trunk_link'] and foot not in ['yes','designated','permissive']:excluded['trunk_without_explicit_foot']+=1;continue
  if foot in ['no','private'] or (access in ['no','private'] and foot not in ['yes','designated','permissive']):excluded['restricted']+=1;continue
  if t.get('motorroad')=='yes' and foot not in ['yes','designated','permissive']:excluded['motorroad']+=1;continue
  keep.append(w);classes[h]+=1
 nodeids=sorted(set(n for w in keep for n in w['nodes']));ix={n:i for i,n in enumerate(nodeids)}
 ll=np.array([nodes[n] for n in nodeids]);tr=Transformer.from_crs(4326,5179,always_xy=True);xx,yy=tr.transform(ll[:,0],ll[:,1]);xy=np.c_[xx,yy]
 edges=set()
 for w in keep:
  for a,b in zip(w['nodes'],w['nodes'][1:]):
   if a!=b:edges.add(tuple(sorted((ix[a],ix[b]))))
 uv=np.array(sorted(edges));length=np.linalg.norm(xy[uv[:,0]]-xy[uv[:,1]],axis=1);nz=length>0;uv=uv[nz];length=length[nz]
 matrix=coo_matrix((np.r_[length,length],(np.r_[uv[:,0],uv[:,1]],np.r_[uv[:,1],uv[:,0]])),shape=(len(xy),len(xy))).tocsr()
 ncomp,labels=connected_components(matrix,directed=False);sizes=np.bincount(labels);largest=int(sizes.argmax())
 geom=linestrings(xy[uv]);tree=STRtree(geom)
 d=pd.read_csv(P/'data/processed/demand_100m.csv');d=d.loc[d.population.fillna(0)>0].reset_index(drop=True)
 cs=[c for c in json.loads((P/'data/processed/convenience_candidates_spatial.json').read_text()) if '시설내점포_출입확인필요' not in c['flags'] and not any('더프레시' in n or '노브랜드' in n for n in c['names'])]
 bins=[b for b in json.loads((P/'results/existing_network/existing_location_audit.json').read_text()) if b['inside_city']]
 result={'source':'OpenStreetMap via Overpass, retrieved 2026-09-29 KST','bbox':[35.02,128.33,35.45,128.90],'bbox_raw_ways':len(ways),'included_ways':len(keep),'included_segments':len(uv),'graph_nodes':len(xy),'components':int(ncomp),'largest_component_node_pct':float(100*sizes[largest]/len(xy)),'classes':dict(classes),'excluded':dict(excluded),'status':'mapping feasibility audit, not verified pedestrian accessibility'}
 for name,coords,weights in [('population',d[['x_m','y_m']].to_numpy(),d.population.to_numpy()),('candidates',np.array([[c['x_m'],c['y_m']] for c in cs]),np.ones(len(cs))),('existing',np.array([[b['x_m'],b['y_m']] for b in bins]),np.ones(len(bins)))]:
  pairs,dist=tree.query_nearest(points(coords),return_distance=True,all_matches=False);order=np.argsort(pairs[0]);dist=dist[order];edges_i=pairs[1][order]
  assert len(dist)==len(coords)
  result[name]={'count':len(coords),'weight_sum':float(weights.sum()),'within_segment_distance_pct':{str(k):float(100*weights[dist<=k].sum()/weights.sum()) for k in [25,50,100]},'nearest_segment_in_largest_component_pct':float(100*weights[labels[uv[edges_i,0]]==largest].sum()/weights.sum()),'maximum_distance_m':float(dist.max())}
  if name=='population':
   d['nearest_mapped_segment_m']=dist;d['nearest_component']=labels[uv[edges_i,0]]
   d[['demand_id','district','dong','population','nearest_mapped_segment_m','nearest_component']].to_csv(R/'walking_mapping_demand_audit.csv',index=False,encoding='utf-8-sig')
   lon,lat=Transformer.from_crs(5179,4326,always_xy=True).transform(coords[:,0],coords[:,1]);inside=(lon>=128.33)&(lon<=128.90)&(lat>=35.02)&(lat<=35.45)
   result['population']['inside_download_bbox_pct']=float(100*weights[inside].sum()/weights.sum())
 result['limitations']=['Proximity to a mapped line does not verify a legal/safe entrance connection.','OSM road centerlines do not establish sidewalks or permissible crossing locations.','Node barrier tags, gate opening hours, stairs and slope constraints are not evaluated in this acquisition.','Disconnected components may be real islands or missing mapped connections; do not equate to physical inaccessibility.','Data coverage has not been independently validated against an official pedestrian network.']
 (R/'walking_data_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False),flush=True)
 np.savez_compressed(R/'osm_graph_audit.npz',node_ids=np.array(nodeids),xy=xy,edges=uv,length_m=length,components=labels)
if __name__=='__main__':main()
