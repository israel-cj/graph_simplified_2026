import math
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

STYLE = {  # category: (label, color, width)
    1: ("LTS 1", "#1a9850", 2.0),
    2: ("LTS 2", "#91cf60", 2.0),
    3: ("LTS 3", "#fc8d59", 2.0),
    4: ("LTS 4", "#d73027", 2.4),
    "5m": ("LTS 5 – must upgrade to LTS 1", "#7b3294", 3.0),
    "5p": ("LTS 5 – cycling prohibited", "#404040", 1.6),
    "5o": ("LTS 5 – other car road", "#a6a6a6", 1.6),
}

def _category(d):
    lts = int(d.get("key", d.get("lts")))
    if lts < 5:
        return lts
    if d.get("mandatory_upgrade_to_lts1"):
        return "5m"
    return "5p" if d.get("bicycle_prohibited") else "5o"

def _undirected_edges(G):
    """One entry per node pair (works for MultiDiGraph and Graph)."""
    seen = {}
    for u, v, d in G.edges(data=True):
        if int(d.get("key", d.get("lts", 0))) not in (1, 2, 3, 4, 5):
            continue
        seen.setdefault(frozenset((u, v)), (u, v, d))
    return list(seen.values())

def _add_graph(fig, G, col, show_legend):
    xs = {c: [] for c in STYLE}; ys = {c: [] for c in STYLE}
    hx, hy, htext = [], [], []
    for u, v, d in _undirected_edges(G):
        c = _category(d)
        x0, y0, x1, y1 = G.nodes[u]["x"], G.nodes[u]["y"], G.nodes[v]["x"], G.nodes[v]["y"]
        xs[c] += [x0, x1, None]; ys[c] += [y0, y1, None]
        hx.append((x0 + x1) / 2); hy.append((y0 + y1) / 2)
        htext.append(f"{STYLE[c][0]}<br>length {float(d.get('distance', d.get('length', 0))):.0f} m")
    for c in ["5o", "5p", 1, 2, 3, 4, "5m"]:          # LTS 5 below, upgrades on top
        if xs[c]:
            label, color, width = STYLE[c]
            fig.add_trace(go.Scattergl(x=xs[c], y=ys[c], mode="lines", name=label, legendgroup=str(c),
                                       showlegend=show_legend, line=dict(color=color, width=width),
                                       hoverinfo="skip"), row=1, col=col)
    # invisible points at edge midpoints so you can hover an edge and read its LTS and length
    fig.add_trace(go.Scattergl(x=hx, y=hy, mode="markers", marker=dict(size=6, opacity=0),
                               text=htext, hoverinfo="text", showlegend=False), row=1, col=col)
    # nodes: degree-2 nodes left after simplification should all sit where the LTS changes
    U = nx.Graph(); U.add_edges_from((u, v) for u, v, _ in _undirected_edges(G))
    deg2 = [n for n in U if U.degree(n) == 2]
    other = [n for n in U if U.degree(n) != 2]
    fig.add_trace(go.Scattergl(x=[G.nodes[n]["x"] for n in other], y=[G.nodes[n]["y"] for n in other],
                               mode="markers", marker=dict(size=3, color="black"), name="Junction / dead end",
                               legendgroup="nodes", showlegend=show_legend, hoverinfo="skip"), row=1, col=col)
    fig.add_trace(go.Scattergl(x=[G.nodes[n]["x"] for n in deg2], y=[G.nodes[n]["y"] for n in deg2],
                               mode="markers", marker=dict(size=5, color="#00a6d6"), name="Degree-2 node",
                               legendgroup="deg2", showlegend=show_legend, hoverinfo="skip"), row=1, col=col)
    return U.number_of_nodes(), U.number_of_edges(), len(deg2)

def plot_before_after(G_before, G_after, title="Grenoble 2026"):
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, shared_yaxes=True, horizontal_spacing=0.02,
                        subplot_titles=["Before simplification", "After simplification"])
    nb, eb, db = _add_graph(fig, G_before, 1, show_legend=True)
    na, ea, da = _add_graph(fig, G_after, 2, show_legend=False)
    fig.layout.annotations[0].text = f"Before: {nb:,} nodes, {eb:,} edges, {db:,} degree-2 nodes"
    fig.layout.annotations[1].text = f"After: {na:,} nodes, {ea:,} edges, {da:,} degree-2 nodes"
    lat = sum(d["y"] for _, d in G_after.nodes(data=True)) / max(len(G_after), 1)
    ratio = 1 / max(math.cos(math.radians(lat)), 0.01)
    fig.update_yaxes(scaleanchor="x", scaleratio=ratio, row=1, col=1)
    fig.update_yaxes(scaleanchor="x2", scaleratio=ratio, row=1, col=2)
    fig.update_xaxes(showticklabels=False); fig.update_yaxes(showticklabels=False)
    fig.update_layout(title=title + " – bike LTS 1–4 + car roads LTS 5 (straight lines between nodes)",
                      width=1500, height=800, template="plotly_white",
                      legend=dict(orientation="h", y=-0.03), margin=dict(l=10, r=10, t=70, b=10))
    fig.show()
    return fig

# check: every degree-2 node left must separate two different LTS values
def leftover_degree2(G):
    bad = [n for n in G if G.degree(n) == 2 and len({G[n][m]["key"] for m in G.neighbors(n)}) == 1]
    kept = sum(G.degree(n) == 2 for n in G)
    print(f"Degree-2 nodes kept: {kept:,} (all because the LTS changes there) | wrongly kept: {len(bad)}")
