"""graph/queries.py — L2 Analysis.

쿼리 ①: 인과 모티프 빈도 집계
쿼리 ②: Y→W 백트레이싱 (최장 인과 사슬)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def causal_motifs(g: nx.MultiDiGraph, top_k: int = 5, data_raw_dir: "Path | None" = None) -> pd.DataFrame:
    """assertions CSV에서 인과 모티프(cause_concept→effect_concept) 빈도 집계.

    data_raw_dir가 None이면 그래프 노드 attrs_json에서 case_id를 추출해
    cfg.paths.data_raw를 유추하지 않으므로, 직접 경로를 전달할 것.
    그래프만 있을 때의 폴백으로 CAUSE 엣지 label 쌍을 사용한다.

    Args:
        g: 분석 대상 MultiDiGraph.
        top_k: 반환할 상위 모티프 수.
        data_raw_dir: data/raw/ 경로. None이면 CAUSE 엣지 기반 폴백.

    Returns:
        DataFrame[cause_concept, effect_concept, count] (count 내림차순 정렬).
    """
    from collections import Counter

    counter: Counter = Counter()

    # 1순위: assertions.csv 직접 파싱
    if data_raw_dir is not None:
        data_raw_dir = Path(data_raw_dir)
        for case_dir in sorted(data_raw_dir.iterdir()):
            apath = case_dir / "assertions.csv"
            if apath.exists():
                df = pd.read_csv(apath)
                for _, row in df.iterrows():
                    counter[(row["cause_concept"], row["effect_concept"])] += 1

    # 폴백: CAUSE 엣지 label 쌍
    if not counter:
        for src, dst, edata in g.edges(data=True):
            if edata.get("edge_type") == "CAUSE":
                src_label = g.nodes[src].get("label", src) if src in g.nodes else src
                dst_label = g.nodes[dst].get("label", dst) if dst in g.nodes else dst
                counter[(src_label, dst_label)] += 1

    if not counter:
        logger.warning("인과 모티프를 찾을 수 없음")
        return pd.DataFrame(columns=["cause_concept", "effect_concept", "count"])

    most_common = counter.most_common(top_k)
    rows = [
        {"cause_concept": c, "effect_concept": e, "count": cnt}
        for (c, e), cnt in most_common
    ]
    return pd.DataFrame(rows)


def backtrace(g: nx.MultiDiGraph, target_label: str = "Y") -> dict:
    """target_label 노드에서 W(t0) 노드까지 역방향 최장 인과 사슬 탐색.

    CAUSE 엣지만 사용하여 역방향 BFS로 탐색.

    Args:
        g: 분석 대상 MultiDiGraph.
        target_label: 탐색 시작 노드 레이블 (기본값 "Y").

    Returns:
        {
            "longest_chain": [node_id, ...],  # Y부터 최장 경로
            "depth": int,
            "reaches_w_at_t0": bool           # True가 수용 기준
        }
    """
    # target 노드 탐색 (label 또는 node_id가 "Y"인 노드)
    target_nodes = [
        nid for nid, nd in g.nodes(data=True)
        if nd.get("label") == target_label or nid.endswith("_Y_t5")
    ]

    if not target_nodes:
        logger.warning("target 노드 '%s' 없음", target_label)
        return {"longest_chain": [], "depth": 0, "reaches_w_at_t0": False}

    # CAUSE 엣지 역방향 그래프 구성
    reverse_cause = nx.MultiDiGraph()
    for src, dst, edata in g.edges(data=True):
        if edata.get("edge_type") == "CAUSE":
            reverse_cause.add_edge(dst, src, **edata)

    # 모든 target 노드에서 최장 경로 탐색
    best_chain: list[str] = []

    for target in target_nodes:
        if target not in reverse_cause:
            continue
        # BFS로 도달 가능한 모든 노드 탐색 후 최장 단순 경로 선택
    for target in target_nodes:
        if target not in reverse_cause:
            continue
        try:
            # reverse_cause에서 target → 모든 도달 노드로의 단순 경로 탐색
            # (원 그래프에서는 ancestor → target 방향)
            reachable = list(nx.single_source_shortest_path(reverse_cause, target, cutoff=10).keys())
            for dest in reachable:
                try:
                    paths = list(nx.all_simple_paths(reverse_cause, source=target, target=dest, cutoff=10))
                    for path in paths:
                        if len(path) > len(best_chain):
                            best_chain = path
                except (nx.NetworkXError, nx.NetworkXNoPath):
                    continue
        except nx.NetworkXError:
            continue

    # W(t0) 도달 여부: 경로 내 node_id가 *_W_t0 패턴이거나 label이 W인 노드
    reaches_w_at_t0 = any(
        (nid.endswith("_W_t0") or g.nodes.get(nid, {}).get("time_layer") == "t0")
        and g.nodes.get(nid, {}).get("label", "").startswith("외")  # 풍요유입 == W
        or nid.endswith("_W_t0")
        for nid in best_chain
    )

    # 보다 직접적인 검사: _W_t0 suffix를 가진 노드가 경로에 있는지
    reaches_w_at_t0 = any(nid.endswith("_W_t0") for nid in best_chain) or reaches_w_at_t0

    result = {
        "longest_chain": best_chain,
        "depth": len(best_chain),
        "reaches_w_at_t0": reaches_w_at_t0,
    }
    logger.info("백트레이싱 결과: depth=%d, reaches_w_at_t0=%s", result["depth"], result["reaches_w_at_t0"])
    return result
