"""graph/export.py — L2 Analysis.

nodes/edges CSV + GraphML 직렬화.
Neo4j 이식 가능 형식으로 출력.
"""
from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def to_csv(g: nx.MultiDiGraph, out_dir: Path) -> tuple[Path, Path]:
    """MultiDiGraph → nodes.csv + edges.csv.

    Args:
        g: 직렬화할 그래프.
        out_dir: 출력 디렉토리 (없으면 생성).

    Returns:
        (nodes_path, edges_path)

    Schema 준수:
        nodes: [node_id, node_type, label, time_layer, case_id, attrs_json]
        edges: [src, dst, edge_type, case_id, attrs_json]
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # nodes
    nodes_rows = []
    for nid, ndata in g.nodes(data=True):
        nodes_rows.append({
            "node_id": nid,
            "node_type": ndata.get("node_type", ""),
            "label": ndata.get("label", ""),
            "time_layer": ndata.get("time_layer", ""),
            "case_id": ndata.get("case_id", ""),
            "attrs_json": ndata.get("attrs_json", "{}"),
        })
    nodes_df = pd.DataFrame(nodes_rows)

    # edges
    edges_rows = []
    for src, dst, edata in g.edges(data=True):
        edges_rows.append({
            "src": src,
            "dst": dst,
            "edge_type": edata.get("edge_type", ""),
            "case_id": edata.get("case_id", ""),
            "attrs_json": edata.get("attrs_json", "{}"),
        })
    edges_df = pd.DataFrame(edges_rows)

    nodes_path = out_dir / "nodes.csv"
    edges_path = out_dir / "edges.csv"
    nodes_df.to_csv(nodes_path, index=False)
    edges_df.to_csv(edges_path, index=False)

    logger.info("CSV export: %s (%d), %s (%d)", nodes_path, len(nodes_df), edges_path, len(edges_df))
    return nodes_path, edges_path


def to_graphml(g: nx.MultiDiGraph, out_path: Path) -> Path:
    """MultiDiGraph → GraphML.

    attrs_json은 평탄화하여 직렬화(nx.read_graphml로 무손실 재로딩 보장).

    Args:
        g: 직렬화할 그래프.
        out_path: 출력 파일 경로.

    Returns:
        out_path (Path).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # GraphML은 dict 속성을 지원하지 않으므로 모든 속성을 string으로 강제 변환
    g_copy = nx.MultiDiGraph()
    for nid, ndata in g.nodes(data=True):
        str_attrs = {k: str(v) for k, v in ndata.items()}
        g_copy.add_node(nid, **str_attrs)
    for src, dst, edata in g.edges(data=True):
        str_attrs = {k: str(v) for k, v in edata.items()}
        g_copy.add_edge(src, dst, **str_attrs)

    nx.write_graphml(g_copy, str(out_path))
    logger.info("GraphML export: %s", out_path)
    return out_path
