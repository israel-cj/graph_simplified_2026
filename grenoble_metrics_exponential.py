"""Metrics from saved, ALREADY SIMPLIFIED nx.Graph objects; no downloads or plotting."""
from __future__ import annotations
from pathlib import Path
import math
import networkx as nx
import pandas as pd

THRESHOLDS=(1,2,3,4)

def exponential_weights(kappa=1.0):
    if kappa < 0: raise ValueError('kappa must be nonnegative')
    terms={t:math.exp(-kappa*(t-1)) for t in THRESHOLDS}
    denominator=sum(terms.values())
    return {t:value/denominator for t,value in terms.items()}

def _level(data):
    return int(data.get('key',data.get('lts')))

def _length(data):
    return float(data.get('distance',data.get('length',0.0)))

def graph_metrics(G):
    """Compute ONLY bike LTS1-4 length and node-largest LCC by threshold.

    G may contain LTS5; it is never counted or used for connectivity.
    No re-simplification/subgraph simplification occurs: all distances come
    directly from G's already contracted edges.
    """
    if type(G) is not nx.Graph or not nx.is_connected(G):
        raise ValueError('Expected nonempty connected, simplified undirected nx.Graph')
    levels={_level(d) for _,_,d in G.edges(data=True)}
    if not levels.issubset({1,2,3,4,5}):
        raise ValueError(f'Unexpected levels: {levels}')
    total=sum(_length(d) for _,_,d in G.edges(data=True) if _level(d)<=4)
    row={'cycling_length_km':total/1000,
         'graph_nodes':G.number_of_nodes(), 'graph_edges':G.number_of_edges(),
         'lts5_edges':sum(_level(d)==5 for _,_,d in G.edges(data=True)),
         'lts5_mandatory_edges':sum(_level(d)==5 and bool(d.get('mandatory_upgrade_to_lts1',False)) for _,_,d in G.edges(data=True))}
    for t in THRESHOLDS:
        sub=nx.Graph()
        sub.add_edges_from((u,v,{'distance':_length(d)}) for u,v,d in G.edges(data=True) if _level(d)<=t)
        if sub.number_of_edges():
            largest=max(nx.connected_components(sub),key=len)
            L=sum(_length(d) for _,_,d in sub.subgraph(largest).edges(data=True))
        else:
            L=0.0
        row[f'lcc_lts{t}_km']=L/1000
        row[f'lcc_lts{t}_share_of_year']=L/total if total else 0.0
    return row

def annual_metrics(graphs, kappa=1.0, reference_length_km=None):
    """Fixed reference denominator for every year (default: max six-year cycling length).

    For 2022, pass combined LTS1-5 graph; metrics filter LTS5 only after
    simplification. For earlier years, pass cycling LTS1-4 simplified graphs.
    """
    rows=[]
    for year,G in sorted(graphs.items()):
        rec=graph_metrics(G)
        rec['year']=int(year)
        rows.append(rec)
    df=pd.DataFrame(rows)
    if not len(df): raise ValueError('No annual graphs')
    ref=float(df['cycling_length_km'].max() if reference_length_km is None else reference_length_km)
    if ref <= 0: raise ValueError('Reference cycling length must be positive')
    w=exponential_weights(kappa)
    df['reference_length_km']=ref
    df['normalized_network_length']=df['cycling_length_km']/ref
    for t in THRESHOLDS:
        df[f'normalized_lcc_lts{t}']=df[f'lcc_lts{t}_km']/ref
        df[f'weight_lts{t}']=w[t]
    df['weighted_lcc_normalized']=sum(w[t]*df[f'normalized_lcc_lts{t}'] for t in THRESHOLDS)
    df['kappa']=kappa
    return df

def save_txt(df, filename):
    path=Path(filename)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        f.write('Grenoble infrastructure metrics: SIMPLIFIED network only\n')
        f.write('2017–2021: simplified cycling LTS1–4 graphs\n')
        f.write('2022: simplified combined LTS1–5 graph, excluding LTS5 only for metrics\n')
        f.write('LCC: component with most nodes at each LTS threshold, measured by summed edge distances\n')
        f.write('Normalized length: cycling length / fixed reference cycling length\n')
        f.write('Weighted LCC: sum(exp(-kappa*(t-1))/sum weights * LCC_t/reference_length), t=1,...,4\n')
        f.write('No demand metric. Lengths from saved simplified graph edges.\n\n')
        f.write(df.to_string(index=False, float_format=lambda x:f'{x:.6f}'))
        f.write('\n')
    return path
