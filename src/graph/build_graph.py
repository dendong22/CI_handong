"""graph/build_graph.py — L2 Analysis.

더미 CSV → networkx MultiDiGraph 적재.
PRECEDES 엣지는 time_layer 순서에서 자동 생성.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

from src.schema import TIME_LAYERS, validate_edges, validate_nodes

logger = logging.getLogger(__name__)


def load_case(cfg, case_id: str) -> nx.MultiDiGraph:
    """단일 케이스 CSV → MultiDiGraph.

    Args:
        cfg: Config 인스턴스.
        case_id: 케이스 디렉토리명.

    Returns:
        노드 속성(node_type, label, time_layer, case_id) +
        엣지 속성(edge_type) 부착된 MultiDiGraph.

    Raises:
        FileNotFoundError: 필수 CSV 파일 없을 때.
    """
    case_dir = cfg.paths.data_raw / case_id
    nodes_path = case_dir / "nodes.csv"
    edges_path = case_dir / "edges.csv"

    if not nodes_path.exists():
        raise FileNotFoundError(f"nodes.csv 없음: {nodes_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"edges.csv 없음: {edges_path}")

    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)

    validate_nodes(nodes_df)
    validate_edges(edges_df)

    g = nx.MultiDiGraph()

    # ── 노드 적재 ──────────────────────────────────────────────────────────────
    for _, row in nodes_df.iterrows():
        attrs = {
            "node_type": row["node_type"],
            "label": row["label"],
            "time_layer": row["time_layer"],
            "case_id": row["case_id"],
            "attrs_json": row.get("attrs_json", "{}"),
        }
        g.add_node(row["node_id"], **attrs)

    # ── 엣지 적재 ──────────────────────────────────────────────────────────────
    for _, row in edges_df.iterrows():
        g.add_edge(
            row["src"],
            row["dst"],
            edge_type=row["edge_type"],
            case_id=row["case_id"],
            attrs_json=row.get("attrs_json", "{}"),
        )

    # PRECEDES 자동 생성: 동일 case 내 인접 time_layer Event 노드 간 ──────────
    layer_event_nodes: dict[str, list[str]] = {tl: [] for tl in TIME_LAYERS}
    for nid, ndata in g.nodes(data=True):
        if ndata.get("case_id") == case_id and ndata.get("node_type") == "Event":
            tl = ndata.get("time_layer")
            if tl in layer_event_nodes:
                layer_event_nodes[tl].append(nid)

    for i in range(len(TIME_LAYERS) - 1):
        srcs = layer_event_nodes[TIME_LAYERS[i]]
        dsts = layer_event_nodes[TIME_LAYERS[i + 1]]
        for s in srcs[:1]:
            for d in dsts[:1]:
                g.add_edge(s, d, edge_type="PRECEDES", case_id=case_id, attrs_json="{}")

    logger.info("케이스 '%s' 로드: %d 노드, %d 엣지", case_id, g.number_of_nodes(), g.number_of_edges())
    return g


def load_all(cfg) -> nx.MultiDiGraph:
    """모든 케이스 CSV → case_id 구획된 통합 MultiDiGraph.

    Args:
        cfg: Config 인스턴스.

    Returns:
        통합 MultiDiGraph. 노드·엣지에 case_id 속성 보존.
    """
    combined = nx.MultiDiGraph()

    case_dirs = [d for d in cfg.paths.data_raw.iterdir() if d.is_dir()]
    if not case_dirs:
        raise FileNotFoundError(f"케이스 디렉토리 없음: {cfg.paths.data_raw}")

    for case_dir in sorted(case_dirs):
        case_id = case_dir.name
        nodes_path = case_dir / "nodes.csv"
        if not nodes_path.exists():
            logger.warning("nodes.csv 없음, 건너뜀: %s", case_dir)
            continue
        g = load_case(cfg, case_id)
        combined = nx.compose(combined, g)

    logger.info("통합 그래프: %d 노드, %d 엣지", combined.number_of_nodes(), combined.number_of_edges())
    return combined
