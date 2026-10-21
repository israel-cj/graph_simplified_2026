"""Historical LTS 1-4, plus 2022 car-road planning layer. No Matplotlib imports.
Requires existing grenoble_lts_graph.py and grenoble_lts_historical.py.
"""
from pathlib import Path
from contextlib import contextmanager
import pickle
import math
import networkx as nx
import pandas as pd
import osmnx as ox

from grenoble_lts_graph import _values, infer_speed_kmh, infer_lanes, normalize_polygon
from grenoble_lts_historical import build_historical_lts_graph, normalize_snapshot_date

ROOT = Path('grenoble_lts_output/historical')

def save_pickle(G, filepath):
    filepath=Path(filepath); filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open('wb') as f: pickle.dump(G,f,protocol=pickle.HIGHEST_PROTOCOL)

def load_pickle(filepath):
    with Path(filepath).open('rb') as f: return pickle.load(f)

def classify_integer_lts(raw):
    if raw is None or float(raw)<0:return None
    return min(4, 1 if raw<=1 else 2 if raw<=2 else 3 if raw<=3 else 4)

def prepare_bike(G):
    H=G.copy()
    drop=[]
    for u,v,k,d in H.edges(keys=True,data=True):
        lts=classify_integer_lts(d.get('lts_score',-1))
        if lts is None: drop.append((u,v,k));continue
        d['key']=lts;d['lts']=lts;d['network_layer']='cycling'
        d['distance']=float(d.get('distance',d.get('length',0)))
        d['mandatory_upgrade_to_lts1']=False
    H.remove_edges_from(drop); H.remove_nodes_from(list(nx.isolates(H)))
    if H.number_of_nodes():
        biggest=max(nx.weakly_connected_components(H),key=len)
        H=H.subgraph(biggest).copy()
    return H

def simplify_bike(G):
    """Follow user's original simplification rules, preserve segment shape for plotting.
    LTS5 excluded BEFORE simplification for metric graph.
    """
    H=nx.Graph();H.add_nodes_from((n,d.copy()) for n,d in G.nodes(data=True));H.graph.update(G.graph)
    eds=G.edges(keys=True,data=True) if G.is_multigraph() else ((u,v,0,d) for u,v,d in G.edges(data=True))
    for u,v,k,d in eds:
        if u==v or int(d.get('lts',d.get('key',5)))>4: continue
        lts=int(d['key']);dist=float(d.get('distance',d.get('length',0)))
        # user original rule: worst LTS wins; equal LTS shortest parallel
        if (not H.has_edge(u,v) or lts>H[u][v]['key'] or (lts==H[u][v]['key'] and dist<H[u][v]['distance'])):
            H.add_edge(u,v,key=lts,lts=lts,distance=dist,geometry=d.get('geometry'))
    H.remove_nodes_from(list(nx.isolates(H)))
    if H.number_of_nodes(): H=H.subgraph(max(nx.connected_components(H),key=len)).copy()
    # Avoid original O(E*N) scan; queue degree-2 nodes
    from collections import deque
    q=deque(n for n in H if H.degree(n)==2)
    while q:
        n=q.popleft()
        if n not in H or H.degree(n)!=2:continue
        a,b=list(H.neighbors(n))
        if a==b or H[n][a]['key']!=H[n][b]['key']:continue
        lts=H[n][a]['key']; dist=H[n][a]['distance']+H[n][b]['distance']
        # Preserve shape of two connected geometries in direction a -> n -> b
        def segment_points(x,y,d):
            g=d.get('geometry')
            if g is not None and getattr(g,'geom_type','')=='LineString':
                pts=list(g.coords)
                p=(H.nodes[x]['x'],H.nodes[x]['y'])
                if (pts[-1][0]-p[0])**2+(pts[-1][1]-p[1])**2 < (pts[0][0]-p[0])**2+(pts[0][1]-p[1])**2: pts.reverse()
                return pts
            return [(H.nodes[x]['x'],H.nodes[x]['y']),(H.nodes[y]['x'],H.nodes[y]['y'])]
        from shapely.geometry import LineString
        points=segment_points(a,n,H[a][n])+segment_points(n,b,H[n][b])[1:]
        geom=LineString(points) if len(points)>=2 else None
        H.remove_node(n)
        if (not H.has_edge(a,b) or lts>H[a][b]['key'] or (lts==H[a][b]['key'] and dist<H[a][b]['distance'])):
            H.add_edge(a,b,key=lts,lts=lts,distance=dist,geometry=geom)
        for t in (a,b):
            if H.degree(t)==2:q.append(t)
    return H

@contextmanager
def historical_osm(year,timeout=600,cache_root='cache_historical_osm',overpass_url=None):
    old={k:getattr(ox.settings,k) for k in ['overpass_settings','requests_timeout','cache_folder','use_cache','overpass_url']}
    date=normalize_snapshot_date(str(year))
    try:
        ox.settings.overpass_settings=f'[out:json][timeout:{{timeout}}][date:"{date}"]{{maxsize}}'
        ox.settings.requests_timeout=int(timeout);ox.settings.cache_folder=str(Path(cache_root)/date[:10]);ox.settings.use_cache=True
        if overpass_url is not None:ox.settings.overpass_url=overpass_url.rstrip('/')
        yield
    finally:
        for k,v in old.items():setattr(ox.settings,k,v)

def osm_way_ids(d):
    x=d.get('osmid',[])
    if not isinstance(x,(list,tuple,set)):x=[x]
    return set(str(i) for i in x)

def same_segment(a,b):return (a[0]==b[0] and a[1]==b[1]) or (a[0]==b[1] and a[1]==b[0])

def add_2022_car_layer(bike,poly,timeout=600,cache_root='cache_historical_osm',overpass_url=None,match='endpoints'):
    """Add driving network not represented in biking network; LTS5 is planning-only.
    OSMnx network extraction semantics can differ, so inspect overlap metrics.
    """
    from overpass_robust import download_graph, OVERPASS_ENDPOINTS
    from grenoble_lts_graph import configure_osmnx
    configure_osmnx()  # keep bicycle/maxspeed/lanes tags even when the bike graph came from a pickle
    date=normalize_snapshot_date('2022')
    endpoints=([overpass_url] if overpass_url else [])+[e for e in OVERPASS_ENDPOINTS if e!=(overpass_url or '').rstrip('/')]
    cars=download_graph(normalize_polygon(poly),'drive',date=date,server_timeout=timeout,
                        endpoints=endpoints,cache_folder=Path(cache_root)/date[:10])
    result=bike.copy()
    # Matching: match same OSM way and same endpoints, OR same endpoints.
    bike_pairs={frozenset((u,v)) for u,v,_,_ in bike.edges(keys=True,data=True)}
    bike_way_edges=set()
    bike_way_ids=set()
    for _,_,_,d in bike.edges(keys=True,data=True): bike_way_ids|=osm_way_ids(d)
    for u,v,k,d in bike.edges(keys=True,data=True):
        for w in osm_way_ids(d):bike_way_edges.add((w,frozenset((u,v))))
    added=0;skipped=0
    for u,v,k,d in cars.edges(keys=True,data=True):
        pair=frozenset((u,v));ways=osm_way_ids(d)
        if match=='way' and ways and ways<=bike_way_ids:
            skipped+=1;continue
        if pair in bike_pairs or any((w,pair) in bike_way_edges for w in ways):
            skipped+=1;continue
        h=set(_values(d,'highway'))
        speed=float(infer_speed_kmh(d));lanes=int(infer_lanes(d))
        prohibited=bool(h & {'motorway','motorway_link','trunk','trunk_link'}) or bool(_values(d,'bicycle') & {'no','private'})
        dangerous=speed>=50 or lanes>=4 or bool(h & {'primary','primary_link','secondary','secondary_link','tertiary_link'})
        mandatory=bool(dangerous and not prohibited)
        for node in (u,v):
            if node not in result:result.add_node(node,**dict(cars.nodes[node]))
        e=d.copy()
        e.update(key=5,lts=5,network_layer='missing_car',distance=float(d.get('length',0)),
                 estimated_speed_kmh=speed,estimated_lanes=lanes,
                 bicycle_prohibited=prohibited,high_stress_car_road=dangerous,
                 mandatory_upgrade_to_lts1=mandatory,target_lts=1 if mandatory else None,
                 optimization_candidate=not prohibited,
                 upgrade_reason='prohibited: independent facility required' if prohibited else ('high speed/lanes/class' if mandatory else 'no mandatory upgrade under rule'))
        e.pop('key',None)
        added_key=result.add_edge(u,v,**e)
        result[u][v][added_key]['key']=5
        added+=1
    result.graph['year']=2022;result.graph['lts5_meaning']='missing drivable roads (planning category)'
    print(f'Added {added:,} directed car edges; skipped {skipped:,} matching directed car edges')
    print('Note: matching by endpoints is approximate where OSMnx splits ways differently.')
    return result

def get_bike_year(year,poly,rebuild=False,root=ROOT,timeout=600,cache_root='cache_historical_osm'):
    p=Path(root)/str(year)/f'grenoble_lts_{year}_bike_prepared.pkl'
    # Reuse original saved graph if available; only suitable for metrics/plot, not adding 2022 cars pre-simplification.
    if p.exists() and not rebuild:return load_pickle(p)
    original=Path(root)/str(year)/f'grenoble_lts_{year}_simplified.pkl'
    if original.exists() and not rebuild and year!=2022:
        print(f'[{year}] original simplified graph exists; no redownload needed')
        return None
    print(f'[{year}] downloading via ORIGINAL historical builder/cache')
    # Prevent Matplotlib kernel crash without changing original classification logic.
    import grenoble_lts_graph as orig
    old_plot=orig.plot_lts_graph
    orig.plot_lts_graph=lambda *a,**k: None
    try:
        full, main, _, _=build_historical_lts_graph(poly=poly,snapshot_date=str(year),output_dir=Path(root)/str(year),requests_timeout=timeout,cache_root=cache_root)
    finally:orig.plot_lts_graph=old_plot
    bike=prepare_bike(main);bike.graph['year']=year
    save_pickle(bike,p)
    return bike

def build_year(year,poly,rebuild=False,root=ROOT,timeout=600,cache_root='cache_historical_osm'):
    year=int(year);folder=Path(root)/str(year)
    target=folder/f'grenoble_lts_{year}_simplified.pkl'
    full_target=folder/'grenoble_lts5_2022_pre_simplification.pkl'
    if target.exists() and not rebuild and (year!=2022 or full_target.exists()):
        return load_pickle(target)
    bike=get_bike_year(year,poly,rebuild,root,timeout,cache_root)
    if bike is None: return load_pickle(target)
    # 2022 first create and SAVE the full graph, then simplify ONLY the cycling graph.
    if year==2022:
        if full_target.exists() and not rebuild:full=load_pickle(full_target)
        else:
            full=add_2022_car_layer(bike,poly,timeout,cache_root)
            save_pickle(full,full_target)
    simplified=simplify_bike(bike); simplified.graph['year']=year
    save_pickle(simplified,target)
    print(f'[{year}] saved {target}')
    return simplified

def load_year(year,root=ROOT,full_2022=False):
    p=Path(root)/str(year)/('grenoble_lts5_2022_pre_simplification.pkl' if year==2022 and full_2022 else f'grenoble_lts_{year}_simplified.pkl')
    return load_pickle(p)

def _geometry_segments(G,u,v,d):
    g=d.get('geometry')
    if g is not None and getattr(g,'geom_type','')=='LineString':return [list(g.coords)]
    if g is not None and getattr(g,'geom_type','')=='MultiLineString':return [list(part.coords) for part in g.geoms]
    return [[(G.nodes[u]['x'],G.nodes[u]['y']),(G.nodes[v]['x'],G.nodes[v]['y'])]]

def plot_year(G,year=None):
    """Six batched traces, original geometry, inline Plotly. Never saves and never imports Matplotlib."""
    import plotly.graph_objects as go
    import numpy as np
    palette={1:'#2ca25f',2:'#99d8c9',3:'#fdae61',4:'#d73027',5:'#808080',6:'#8e44ad'}
    coords={i:([],[]) for i in palette}
    eds=G.edges(keys=True,data=True) if G.is_multigraph() else ((u,v,0,d) for u,v,d in G.edges(data=True))
    for u,v,k,d in eds:
        cat=6 if d.get('network_layer')=='missing_car' and d.get('mandatory_upgrade_to_lts1') else 5 if d.get('network_layer')=='missing_car' else int(d.get('lts',d.get('key',4)))
        if cat not in coords:continue
        xs,ys=coords[cat]
        for seq in _geometry_segments(G,u,v,d):
            for x,y in seq:xs.append(x);ys.append(y)
            xs.append(None);ys.append(None)
    fig=go.Figure()
    names={1:'LTS 1',2:'LTS 2',3:'LTS 3',4:'LTS 4',5:'LTS 5: missing car',6:'LTS 5: mandatory upgrade'}
    for cat in [5,6,4,3,2,1]:
        if not coords[cat][0]:continue
        fig.add_trace(go.Scattergl(x=coords[cat][0],y=coords[cat][1],mode='lines',line=dict(color=palette[cat],width=1.6 if cat<=4 else 1.2),name=names[cat],hoverinfo='skip'))
    meanlat=float(np.mean([d['y'] for _,d in G.nodes(data=True)]))
    fig.update_layout(title=f'Grenoble LTS {year or G.graph.get("year", "")}',width=1000,height=800,template='plotly_white',margin=dict(l=20,r=20,t=60,b=20),yaxis=dict(scaleanchor='x',scaleratio=1/max(math.cos(math.radians(meanlat)),.01)))
    fig.show();return fig

def edge_length(d):return float(d.get('distance',d.get('length',0)))

def annual_metrics(graphs):
    """Non-demand length+LCC measures, LTS thresholds 1-4. All exclude LTS5."""
    rows=[]
    for year,G in sorted(graphs.items()):
        all_length=sum(edge_length(d) for _,_,d in G.edges(data=True) if int(d.get('lts',d.get('key',5)))<=4)
        row={'year':year,'network_length_km':all_length/1000}
        for threshold in [1,2,3,4]:
            H=nx.Graph()
            for u,v,d in G.edges(data=True):
                if int(d.get('lts',d.get('key',5)))<=threshold:
                    H.add_edge(u,v,**d)
            if H.number_of_nodes():
                largest=max(nx.connected_components(H),key=len)
                lcc=H.subgraph(largest)
                length=sum(edge_length(d) for _,_,d in lcc.edges(data=True))
            else:length=0
            row[f'lcc_lts{threshold}_km']=length/1000
            row[f'lcc_lts{threshold}_share']=length/all_length if all_length else 0
        rows.append(row)
    df=pd.DataFrame(rows)
    reference=max(df.network_length_km.max(),1e-12)
    weights={1:1,2:0.75,3:0.5,4:0.25}; total=sum(weights.values())
    df['weighted_lcc_score']=sum(weights[t]*df[f'lcc_lts{t}_km']/reference for t in weights)/total
    df['length_normalized_reference']=df.network_length_km/reference
    for col in ['network_length_km','weighted_lcc_score']:
        mn,mx=df[col].min(),df[col].max()
        df[col+'_minmax']=(df[col]-mn)/(mx-mn) if mx>mn else 0.0
    df['combined_infrastructure_index']=0.5*df['network_length_km_minmax']+0.5*df['weighted_lcc_score_minmax']
    return df
