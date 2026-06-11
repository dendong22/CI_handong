"""datagen/corpus_dummy.py — L1 Data.

3사례(A·B·C) + 대조사례의 추출 완료형 더미 CSV 생성.
schema.py의 컬럼 계약을 반드시 준수한다.

설계 제약 (하드 보장):
    (a) assertions의 (cause→effect) 모티프: '외부충격→붕괴' 유형 최빈
    (b) CAUSE/EFFECT 사슬이 Y 노드에서 t0의 W 노드까지 연결 경로 최소 1개
    (c) A·B는 t0–t5 전 구간, C는 희소(층위당 1–3 노드)
    (d) controls(Y=0)는 붕괴 Event 부재 + 동일 W 유입 Event 보유
"""
from __future__ import annotations

import json
import logging
from itertools import product
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from src.schema import (
    ASSERTIONS_COLUMNS,
    EDGES_COLUMNS,
    NODES_COLUMNS,
    TIME_LAYERS,
    validate_assertions,
    validate_nodes,
)

logger = logging.getLogger(__name__)

# ── 케이스 메타데이터 ─────────────────────────────────────────────────────────
CASE_META: Dict[str, dict] = {
    "case_a_korea1997": {
        "label": "한국 외환위기 1997",
        "w_event": "외자 유입 급증(chaebol 확장)",
        "collapse_event": "IMF 구제금융 신청",
        "nodes_per_layer": 9,  # A·B: 8–10 노드/층위
        "sparse": False,
        "is_control": False,
        "y_value": 1,
    },
    "case_b_dotcom": {
        "label": "닷컴 버블 붕괴 2001",
        "w_event": "VC/IPO 자금 폭증",
        "collapse_event": "NASDAQ 80% 하락 및 기업 파산",
        "nodes_per_layer": 8,
        "sparse": False,
        "is_control": False,
        "y_value": 1,
    },
    "case_c_sparta": {
        "label": "스파르타 쇠퇴 BC371",
        "w_event": "헬로트 노동 전유",
        "collapse_event": "레욱트라 전투 패배",
        "nodes_per_layer": 2,  # C: 희소 (1–3)
        "sparse": True,
        "is_control": False,
        "y_value": 1,
    },
    "controls": {
        "label": "대조 사례 (Y=0)",
        "w_event": "외자 유입(대만 1997)",
        "collapse_event": None,  # Y=0 → 붕괴 Event 없음
        "nodes_per_layer": 6,
        "sparse": False,
        "is_control": True,
        "y_value": 0,
    },
}

# ── 모티프 분포 (제약 a): 외부충격→붕괴가 최빈이 되도록 가중치 설정 ───────────
MOTIF_TEMPLATES = [
    # (cause_concept, effect_concept, weight)
    ("외부충격",      "붕괴",         5),   # 최빈 모티프 (제약 a 보장)
    ("풍요유입",      "구조적엔트로피", 3),
    ("엘리트과잉생산", "제도마비",      2),
    ("담론사일로화",   "취약성증가",    2),
    ("유동성급증",     "레버리지확대",  2),
    ("감시약화",      "도덕적해이",    1),
    ("외부충격",      "취약성노출",    1),
    ("구조적엔트로피", "붕괴",         2),
]


def _motif_pool(rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    """가중치 기반 모티프 샘플링. 제약 (a) 보장."""
    templates = [(c, e) for c, e, _ in MOTIF_TEMPLATES]
    weights = np.array([w for _, _, w in MOTIF_TEMPLATES], dtype=float)
    weights /= weights.sum()
    indices = rng.choice(len(templates), size=n, p=weights, replace=True)
    return [templates[i] for i in indices]


def generate_case(cfg, case_id: str) -> Dict[str, pd.DataFrame]:
    """단일 케이스의 nodes·assertions·corpus_meta DataFrame 생성.

    Args:
        cfg: Config 인스턴스.
        case_id: CASE_META 키 중 하나.

    Returns:
        {"nodes": df, "assertions": df, "corpus_meta": df}

    Raises:
        KeyError: 알 수 없는 case_id.
        AssertionError: 제약 (b) 위반 시 (Y→W(t0) 경로 부재).
    """
    if case_id not in CASE_META:
        raise KeyError(f"알 수 없는 case_id: {case_id}. 허용값: {list(CASE_META)}")

    meta = CASE_META[case_id]
    rng = np.random.default_rng(cfg.seed_dgp + hash(case_id) % (2**31))
    layers = TIME_LAYERS  # ["t0","t1","t2","t3","t4","t5"]
    npl = meta["nodes_per_layer"]  # nodes per layer

    # ── 1. 노드 생성 ──────────────────────────────────────────────────────────
    nodes_rows = []
    node_counter = 0

    # 제약 (b) 보장: W 노드(t0)와 Y 노드 반드시 포함
    w_node_id = f"{case_id}_W_t0"
    y_node_id = f"{case_id}_Y_t5"

    for tl in layers:
        if meta["sparse"] and tl not in ("t0", "t2", "t5"):
            # C 사례: 희소 — t0, t2, t5만 노드 생성
            continue

        n_this_layer = npl if not meta["sparse"] else rng.integers(1, 3 + 1)
        for i in range(n_this_layer):
            node_id = f"{case_id}_N{node_counter:03d}_{tl}"
            node_type = rng.choice(["Event", "Concept", "Organization"])

            if tl == "t0" and i == 0:
                # W 노드 (풍요 유입) — 제약 (b) 앵커
                node_id = w_node_id
                node_type = "Event"
                label = meta["w_event"]
            elif tl == "t5" and i == 0:
                # Y 노드 (붕괴) — 제약 (b) 앵커
                node_id = y_node_id
                node_type = "Event"
                label = meta["collapse_event"] if meta["collapse_event"] else "안정유지"
            else:
                label_candidates = [
                    "엔트로피지표상승", "담론분열심화", "감독기관약화",
                    "레버리지확대", "엘리트파벌형성", "정보비대칭증가",
                    "정책대응지연", "외부압력", "신호무시",
                    "제도이완", "과잉투자", "거버넌스공백",
                ]
                label = label_candidates[node_counter % len(label_candidates)]

            nodes_rows.append({
                "node_id": node_id,
                "node_type": node_type,
                "label": label,
                "time_layer": tl,
                "case_id": case_id,
                "attrs_json": json.dumps({"y_value": meta["y_value"]}),
            })
            node_counter += 1

    nodes_df = pd.DataFrame(nodes_rows, columns=NODES_COLUMNS)

    # ── 2. assertions 생성 ─────────────────────────────────────────────────────
    # 제약 (a): 외부충격→붕괴 최빈 (모티프 풀로 보장)
    # 제약 (b): Y→...→W(t0) CAUSE 사슬 최소 1개 하드 삽입
    # 희소 사례도 최빈 모티프 보장을 위해 최소 15개 보장
    n_assertions = max(15, npl * 3)
    motifs = _motif_pool(rng, n_assertions)

    assertions_rows = []
    speakers = ["analyst", "media", "official", "academic"]

    for idx, (cause, effect) in enumerate(motifs):
        tl = rng.choice(layers)
        assertions_rows.append({
            "assertion_id": f"{case_id}_A{idx:04d}",
            "cause_concept": cause,
            "effect_concept": effect,
            "speaker": rng.choice(speakers),
            "confidence": round(float(rng.uniform(0.5, 1.0)), 3),
            "polarity": "negative" if "붕괴" in effect or "마비" in effect else "positive",
            "source_id": f"{case_id}_SRC{idx // 3:03d}",
            "source_span": f"문서_{case_id}_{tl}_p{idx + 1}",  # 제약: 비어 있으면 안 됨
            "time_layer": tl,
            "case_id": case_id,
        })

    # 제약 (a) 강제 보장: 외부충격→붕괴 레코드를 풀 크기의 40% 이상으로 명시 삽입
    n_forced = max(4, int(n_assertions * 0.40))
    for ki in range(n_forced):
        tl = rng.choice(layers)
        assertions_rows.append({
            "assertion_id": f"{case_id}_FORCED{ki:03d}",
            "cause_concept": "외부충격",
            "effect_concept": "붕괴",
            "speaker": rng.choice(speakers),
            "confidence": round(float(rng.uniform(0.7, 1.0)), 3),
            "polarity": "negative",
            "source_id": f"{case_id}_SRC_FORCED{ki:02d}",
            "source_span": f"문서_{case_id}_{tl}_forced{ki}",
            "time_layer": tl,
            "case_id": case_id,
        })

    # 제약 (b) 하드 보장: 풍요유입(t0) → 엔트로피 → 붕괴(t5) CAUSE 체인 명시
    chain_assertions = [
        ("풍요유입", "구조적엔트로피", "t0"),
        ("구조적엔트로피", "취약성증가", "t2"),
        ("취약성증가", "붕괴", "t4"),
    ]
    for ci, (cause, effect, tl) in enumerate(chain_assertions):
        assertions_rows.append({
            "assertion_id": f"{case_id}_CHAIN{ci:02d}",
            "cause_concept": cause,
            "effect_concept": effect,
            "speaker": "analyst",
            "confidence": 0.95,
            "polarity": "negative",
            "source_id": f"{case_id}_SRC_CHAIN",
            "source_span": f"문서_{case_id}_{tl}_chain{ci}",
            "time_layer": tl,
            "case_id": case_id,
        })

    assertions_df = pd.DataFrame(assertions_rows, columns=ASSERTIONS_COLUMNS)

    # ── 3. corpus_meta (Source·Mention 링크) ─────────────────────────────────
    source_ids = assertions_df["source_id"].unique()
    corpus_rows = []
    for sid in source_ids:
        tl_sample = assertions_df[assertions_df["source_id"] == sid]["time_layer"].iloc[0]
        corpus_rows.append({
            "source_id": sid,
            "title": f"[더미] {meta['label']} {sid}",
            "time_layer": tl_sample,
            "case_id": case_id,
            "doc_type": rng.choice(["news", "report", "speech"]),
        })
    corpus_meta_df = pd.DataFrame(corpus_rows)

    # ── 4. 엣지 생성 (CAUSE/EFFECT/PRECEDES/SUPPORTED_BY) ────────────────────
    edges_rows = []
    node_ids_by_layer: Dict[str, list[str]] = {tl: [] for tl in TIME_LAYERS}
    for _, row in nodes_df.iterrows():
        node_ids_by_layer[row["time_layer"]].append(row["node_id"])

    # PRECEDES: 인접 층위 Event 노드 간 자동 연결
    for i in range(len(TIME_LAYERS) - 1):
        tl_cur = TIME_LAYERS[i]
        tl_nxt = TIME_LAYERS[i + 1]
        src_ids = node_ids_by_layer[tl_cur]
        dst_ids = node_ids_by_layer[tl_nxt]
        if src_ids and dst_ids:
            edges_rows.append({
                "src": src_ids[0],
                "dst": dst_ids[0],
                "edge_type": "PRECEDES",
                "case_id": case_id,
                "attrs_json": "{}",
            })

    # 제약 (b): Y(t5) → (chain) → W(t0) CAUSE 체인 엣지 하드 삽입
    # W(t0) → EntropyConcept(t2) → FragilityEvent(t4) → Y(t5)
    # 단순화: W_t0 → Y_t5 까지 노드 시퀀스로 직결 CAUSE 삽입
    anchor_nodes = [w_node_id]
    # t2, t4 중간 앵커: 해당 층위의 첫 노드 사용
    for tl in ("t2", "t4"):
        nl = node_ids_by_layer.get(tl, [])
        if nl:
            anchor_nodes.append(nl[0])
    anchor_nodes.append(y_node_id)

    for j in range(len(anchor_nodes) - 1):
        edges_rows.append({
            "src": anchor_nodes[j],
            "dst": anchor_nodes[j + 1],
            "edge_type": "CAUSE",
            "case_id": case_id,
            "attrs_json": json.dumps({"chain_type": "backbone"}),
        })

    # SUPPORTED_BY: assertion → source 링크 (일부)
    for _, arow in assertions_df.iterrows():
        src_nodes = node_ids_by_layer.get(arow["time_layer"], [])
        if src_nodes:
            edges_rows.append({
                "src": arow["assertion_id"],
                "dst": src_nodes[0],
                "edge_type": "SUPPORTED_BY",
                "case_id": case_id,
                "attrs_json": "{}",
            })

    edges_df = pd.DataFrame(edges_rows, columns=EDGES_COLUMNS)

    # ── 5. 스키마 검증 ─────────────────────────────────────────────────────────
    validate_nodes(nodes_df)
    validate_assertions(assertions_df)

    logger.info(
        "케이스 '%s': nodes=%d, assertions=%d, edges=%d",
        case_id,
        len(nodes_df),
        len(assertions_df),
        len(edges_df),
    )
    return {"nodes": nodes_df, "assertions": assertions_df,
            "corpus_meta": corpus_meta_df, "edges": edges_df}


def main(cfg=None) -> None:
    """data/raw/case_*/ 전체 기록.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml에서 로드.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    cfg.paths.ensure_all()

    for case_id, meta in CASE_META.items():
        out_dir = cfg.paths.data_raw / case_id
        out_dir.mkdir(parents=True, exist_ok=True)

        dfs = generate_case(cfg, case_id)
        for name, df in dfs.items():
            path = out_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            logger.info("기록: %s (%d rows)", path, len(df))

    logger.info("더미 코퍼스 생성 완료.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config

    main(load_config(root / "params.yaml", root=root))
