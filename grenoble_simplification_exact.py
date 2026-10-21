"""Exact-style same-LTS degree-2 graph contraction for Grenoble.

- nx.Graph, worst parallel LTS (shortest if equal)
- largest connected component (undirected)
- degree-2 contraction iff incident LTS values equal
- summed distance
- LTS 5 merged mandatory-upgrade flag is logical OR
- keep geometry for optional geographic plotting; topology view ignores it
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import pickle
import networkx as nx
from shapely.geometry import LineString


def _coordinates(G, u, v, data):
    geom = data.get('geometry')
    if getattr(geom, 'geom_type', None) == 'LineString':
        points = list(geom.coords)
        if points:
            start = (float(G.nodes[u]['x']), float(G.nodes[u]['y']))
            if sum((points[-1][i]-start[i])**2 for i in (0,1)) < sum((points[0][i]-start[i])**2 for i in (0,1)):
                points.reverse()
            return points
    return [(float(G.nodes[u]['x']),float(G.nodes[u]['y'])),(float(G.nodes[v]['x']),float(G.nodes[v]['y']))]


def _edge_data(d):
    e = dict(d)
    level = int(e.get('lts', e.get('key')))
    e['key'] = e['lts'] = level
    e['distance'] = float(e.get('distance', e.get('length', 0.0)))
    e['length'] = e['distance']
    if level == 5:
        e['mandatory_upgrade_to_lts1'] = bool(e.get('mandatory_upgrade_to_lts1', False))
    else:
        e['mandatory_upgrade_to_lts1'] = False
    return e


def _choose_parallel(a, b):
    # Exactly user's worst LTS / shortest same-LTS representative rule.
    chosen = dict(b if (b['key'] > a['key'] or (b['key'] == a['key'] and b['distance'] < a['distance'])) else a)
    if a['key'] == b['key'] == 5:
        # Do not lose a mandatory flag when dropping a parallel alternative.
        chosen['mandatory_upgrade_to_lts1'] = bool(a.get('mandatory_upgrade_to_lts1') or b.get('mandatory_upgrade_to_lts1'))
        chosen['bicycle_prohibited'] = bool(a.get('bicycle_prohibited') or b.get('bicycle_prohibited'))
        chosen['high_stress_car_road'] = bool(a.get('high_stress_car_road') or b.get('high_stress_car_road'))
    return chosen


def simplify_lts_graph(G, lts_attr='key'):
    """Return a connected nx.Graph with all possible same-LTS degree-2 nodes removed.

    This keeps the no-parallel-edge property. An existing a--b edge is resolved
    by the original 'worst LTS, then shortest distance' rule after contracting n.
    Such a collision may change total retained length: see graph metadata.
    """
    H = nx.Graph()
    H.add_nodes_from((n, dict(attrs)) for n, attrs in G.nodes(data=True))
    H.graph.update(G.graph)
    edges = G.edges(keys=True, data=True) if G.is_multigraph() else ((u,v,None,d) for u,v,d in G.edges(data=True))
    parallel_collisions = 0
    for u,v,_,d in edges:
        if u == v: continue
        e = _edge_data(d)
        if e['key'] not in (1,2,3,4,5): continue
        if H.has_edge(u,v):
            parallel_collisions += 1
            H[u][v].update(_choose_parallel(dict(H[u][v]),e))
        else: H.add_edge(u,v,**e)
    H.remove_nodes_from(list(nx.isolates(H)))
    if H.number_of_nodes() and not nx.is_connected(H):
        H = H.subgraph(max(nx.connected_components(H),key=len)).copy()
    before_nodes, before_edges = H.number_of_nodes(), H.number_of_edges()
    q = deque(n for n in H.nodes if H.degree(n)==2)
    contracted = 0
    created_parallel = 0
    while q:
        n = q.popleft()
        if n not in H or H.degree(n)!=2: continue
        a,b = list(H.neighbors(n))
        if a == b: continue
        e1, e2 = dict(H[a][n]),dict(H[n][b])
        if e1['key'] != e2['key']:continue
        merged = dict(e1)
        merged['distance'] = merged['length'] = e1['distance'] + e2['distance']
        merged['lts'] = merged['key'] = e1['key']
        if merged['key'] == 5:
            merged['mandatory_upgrade_to_lts1'] = bool(e1.get('mandatory_upgrade_to_lts1') or e2.get('mandatory_upgrade_to_lts1'))
            merged['bicycle_prohibited'] = bool(e1.get('bicycle_prohibited') or e2.get('bicycle_prohibited'))
            merged['high_stress_car_road'] = bool(e1.get('high_stress_car_road') or e2.get('high_stress_car_road'))
            merged['optimization_candidate'] = bool(e1.get('optimization_candidate',True) and e2.get('optimization_candidate',True)) and not merged['bicycle_prohibited']
            merged['target_lts'] = 1 if merged['mandatory_upgrade_to_lts1'] else None
        else: merged['mandatory_upgrade_to_lts1'] = False
        merged['geometry'] = LineString(_coordinates(H,a,n,e1)+_coordinates(H,n,b,e2)[1:])
        prior = dict(H[a][b]) if H.has_edge(a,b) else None
        H.remove_node(n)
        if prior is not None:
            created_parallel += 1
            merged = _choose_parallel(prior, merged)
        H.add_edge(a,b,**merged)
        contracted += 1
        for candidate in (a,b):
            if H.degree(candidate)==2:q.append(candidate)
    H.graph.update(degree2_contracted=contracted, nodes_before_contraction=before_nodes,
                   edges_before_contraction=before_edges, parallel_collisions=parallel_collisions,
                   contractions_with_existing_direct_edge=created_parallel,
                   simplification_rule='worst parallel, equal-LTS degree2, summed distance, largest connected component')
    assert type(H) is nx.Graph
    assert H.number_of_nodes()==0 or nx.is_connected(H)
    assert not any(H.degree(n)==2 and len({H[n][m]['key'] for m in H.neighbors(n)})==1 for n in H)
    return H


def load_combined_2022_and_simplify(root='grenoble_lts_output/historical', overwrite=True):
    folder=Path(root)/'2022'
    source=folder/'grenoble_lts5_2022_pre_simplification.pkl'
    destination=folder/'grenoble_lts5_2022_SIMPLE_optimization.pkl'
    if not source.exists():raise FileNotFoundError(f'No cached 2022 combined graph: {source}')
    if destination.exists() and not overwrite:
        with destination.open('rb') as fh:return pickle.load(fh)
    with source.open('rb') as fh:G=pickle.load(fh)
    H=simplify_lts_graph(G)
    with destination.open('wb') as fh:pickle.dump(H,fh,pickle.HIGHEST_PROTOCOL)
    print('Saved', destination)
    return H


def check_graph(H):
    from collections import Counter
    same_lts_degree2=sum(H.degree(n)==2 and len({H[n][m]['key'] for m in H.neighbors(n)})==1 for n in H)
    result={'nodes':H.number_of_nodes(),'edges':H.number_of_edges(),'multigraph':H.is_multigraph(),
            'connected':nx.is_connected(H) if len(H) else False,
            'remaining_contractible_degree2_nodes':same_lts_degree2,
            'lts_counts':dict(Counter(d['key'] for _,_,d in H.edges(data=True))),
            'mandatory_lts5_edges':sum(d['key']==5 and bool(d.get('mandatory_upgrade_to_lts1')) for _,_,d in H.edges(data=True))}
    print(result)
    return result


def plot_topology(H, show_nodes=False):
    """Plot simplified topology INLINE, ignoring detailed edge geometry."""
    import plotly.graph_objects as go
    from math import cos, radians
    colors={1:'#2ca25f',2:'#99d8c9',3:'#fdae61',4:'#d73027',5:'#808080',6:'#8e44ad'}
    groups={i:([],[]) for i in colors}
    for u,v,d in H.edges(data=True):
        level=int(d.get('key',d.get('lts')))
        category=6 if level==5 and d.get('mandatory_upgrade_to_lts1') else level
        xx,yy=groups[category]
        xx.extend([H.nodes[u]['x'],H.nodes[v]['x'],None])
        yy.extend([H.nodes[u]['y'],H.nodes[v]['y'],None])
    fig=go.Figure()
    for category in [5,6,4,3,2,1]:
        x,y=groups[category]
        if x:
            fig.add_trace(go.Scattergl(x=x,y=y,mode='lines',line=dict(color=colors[category],width=1.6),name={5:'LTS5',6:'LTS5 mandatory'}.get(category,f'LTS {category}')))
    if show_nodes:
        fig.add_trace(go.Scattergl(x=[d['x'] for _,d in H.nodes(data=True)],y=[d['y'] for _,d in H.nodes(data=True)],mode='markers',marker=dict(size=2,color='#202020'),name='Remaining nodes'))
    lat=sum(float(d['y']) for _,d in H.nodes(data=True))/max(len(H),1)
    fig.update_layout(width=1000,height=800,template='plotly_white',title='Simplified topology (endpoint edges; not road geometry)',yaxis=dict(scaleanchor='x',scaleratio=1/max(cos(radians(lat)),0.01)))
    fig.show()
    return fig


def lcc_metrics(H):
    """Lengths of node-largest connected components for thresholds 1,2,3,4.
    Never counts LTS 5. H should be a simplified cycling-only nx.Graph.
    """
    out={}
    total=sum(float(d['distance']) for _,_,d in H.edges(data=True) if int(d['key'])<=4)
    out['cycling_length_km']=total/1000
    for threshold in (1,2,3,4):
        sub=nx.Graph((u,v,dict(d)) for u,v,d in H.edges(data=True) if int(d['key'])<=threshold)
        if sub.number_of_nodes():
            largest=max(nx.connected_components(sub),key=len)
            edges=sub.subgraph(largest).edges(data=True)
            length=sum(float(d['distance']) for _,_,d in edges)
        else:length=0.0
        out[f'lcc_lts{threshold}_km']=length/1000
        out[f'lcc_lts{threshold}_share']=length/total if total else 0.0
    return out
