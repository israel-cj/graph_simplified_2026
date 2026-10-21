"""Small repair layer for saved Grenoble graphs. Does not download OSM data."""
from pathlib import Path
import pickle
import networkx as nx
from grenoble_simplification_exact import simplify_lts_graph, check_graph, plot_topology

ROOT = Path('grenoble_lts_output/historical')

def cycling_lts_integer(data):
    """Original classifier's raw LTS scores 0,1,2,2.5,3,4,5 -> 1,1,2,3,3,4,4.

    Existing normalized integer `key` is used if `lts_score` is absent.
    """
    raw = data.get('lts_score', None)
    if raw is not None:
        raw = float(raw)
        if raw < 0: return None
        return 1 if raw <= 1 else 2 if raw <= 2 else 3 if raw <= 3 else 4
    val = data.get('key', data.get('lts', None))
    if val is None: return None
    raw = float(val)
    if raw < 0: return None
    return 1 if raw <= 1 else 2 if raw <= 2 else 3 if raw <= 3 else 4

def prepare_metrics_source(source):
    """Copy only bicycle edges, map original raw scores to LTS1..4 before simplification."""
    H = nx.MultiGraph() if source.is_multigraph() else nx.Graph()
    H.add_nodes_from((n, dict(d)) for n,d in source.nodes(data=True))
    H.graph.update(source.graph)
    edges = source.edges(keys=True,data=True) if source.is_multigraph() else ((u,v,None,d) for u,v,d in source.edges(data=True))
    kept=0
    for u,v,k,d in edges:
        if d.get('network_layer') == 'missing_car': continue
        lts=cycling_lts_integer(d)
        if lts is None: continue
        a=dict(d);a.update(key=lts,lts=lts,network_layer='cycling',mandatory_upgrade_to_lts1=False)
        a['distance']=float(a.get('distance',a.get('length',0)))
        if source.is_multigraph(): H.add_edge(u,v,**a)
        else: H.add_edge(u,v,**a)
        kept += 1
    H.remove_nodes_from(list(nx.isolates(H)))
    if not kept: raise ValueError('No usable cycling edges in saved graph')
    return H

def build_saved_metrics(year, root=ROOT):
    folder=Path(root)/str(year)
    prepared=folder/f'grenoble_lts_{year}_bike_prepared.pkl'
    original=folder/f'grenoble_lts_{year}_simplified.pkl'
    if prepared.exists(): src=prepared
    elif original.exists(): src=original
    else: raise FileNotFoundError(f'No cached cycling source for {year}: {prepared} or {original}')
    with src.open('rb') as f: G_source=pickle.load(f)
    bike=prepare_metrics_source(G_source)
    H=simplify_lts_graph(bike)
    assert type(H) is nx.Graph and all(1 <= int(d['key']) <= 4 for _,_,d in H.edges(data=True))
    out=folder/f'grenoble_lts_{year}_SIMPLE_metrics_exact.pkl'
    with out.open('wb') as f:pickle.dump(H,f,pickle.HIGHEST_PROTOCOL)
    print(f'{year}: from {src.name}; {bike.number_of_nodes()} -> {H.number_of_nodes()} nodes; '
          f'{bike.number_of_edges()} -> {H.number_of_edges()} edges; contracted={H.graph["degree2_contracted"]}')
    return H

def build_cached_2022_optimization(root=ROOT):
    folder=Path(root)/'2022'
    source=folder/'grenoble_lts5_2022_pre_simplification.pkl'
    if not source.exists(): raise FileNotFoundError(f'Combined 2022 cache missing: {source}. Run your existing 2022 car builder once.')
    with source.open('rb') as f:G=pickle.load(f)
    # Combined 2022 prepared file already contains bike LTS 1..4, cars LTS 5.
    H=simplify_lts_graph(G)
    destination=folder/'grenoble_lts5_2022_SIMPLE_optimization.pkl'
    with destination.open('wb') as f:pickle.dump(H,f,pickle.HIGHEST_PROTOCOL)
    print(f'2022 optimization: {G.number_of_nodes()} -> {H.number_of_nodes()} nodes, '
          f'{G.number_of_edges()} -> {H.number_of_edges()} edges; saved {destination}')
    return H
