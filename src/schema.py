"""schema.py — L0 Foundation.

변수 정의, 노드·엣지 스키마, CSV 컬럼 계약, 검증 함수, 이산화 규칙.
모든 상수는 이 모듈에서만 정의한다.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.config import Config

# ── 온톨로지 상수 ──────────────────────────────────────────────────────────────
NODE_TYPES: list[str] = [
    "Event",
    "Person",
    "Organization",
    "Place",
    "Concept",
    "Mention",
    "Source",
    "CausalAssertion",
]

EDGE_TYPES: list[str] = [
    "PRECEDES",
    "CAUSE",
    "EFFECT",
    "SUPPORTED_BY",
    "DERIVED_FROM",
]

TIME_LAYERS: list[str] = ["t0", "t1", "t2", "t3", "t4", "t5"]

M_STATES: list[str] = ["Low", "Mid", "High"]  # 이산화: M ∈ {0, 1, 2}

# ── CSV 컬럼 계약 ──────────────────────────────────────────────────────────────
NODES_COLUMNS: list[str] = [
    "node_id",
    "node_type",
    "label",
    "time_layer",
    "case_id",
    "attrs_json",
]

ASSERTIONS_COLUMNS: list[str] = [
    "assertion_id",
    "cause_concept",
    "effect_concept",
    "speaker",
    "confidence",
    "polarity",
    "source_id",
    "source_span",
    "time_layer",
    "case_id",
]

EDGES_COLUMNS: list[str] = [
    "src",
    "dst",
    "edge_type",
    "case_id",
    "attrs_json",
]

OBSERVED_COLUMNS: list[str] = ["W", "M", "Y"]  # U는 의도적 부재


# ── 검증 함수 ──────────────────────────────────────────────────────────────────
def validate_nodes(df: pd.DataFrame) -> None:
    """nodes DataFrame 스키마 검증.

    Args:
        df: 검증할 노드 DataFrame.

    Raises:
        ValueError: 컬럼 누락, 타입 오류, enum 위반 시. 위반 행 번호 포함.
    """
    _check_columns(df, NODES_COLUMNS, "nodes")

    # node_type enum 검증
    bad_rows = df.index[~df["node_type"].isin(NODE_TYPES)].tolist()
    if bad_rows:
        bad_vals = df.loc[bad_rows, "node_type"].unique().tolist()
        raise ValueError(
            f"nodes.node_type 위반 — 허용값: {NODE_TYPES}\n"
            f"위반 행: {bad_rows[:10]}, 위반값: {bad_vals}"
        )

    # time_layer enum 검증
    bad_rows = df.index[~df["time_layer"].isin(TIME_LAYERS)].tolist()
    if bad_rows:
        bad_vals = df.loc[bad_rows, "time_layer"].unique().tolist()
        raise ValueError(
            f"nodes.time_layer 위반 — 허용값: {TIME_LAYERS}\n"
            f"위반 행: {bad_rows[:10]}, 위반값: {bad_vals}"
        )

    # node_id 유일성
    dupes = df["node_id"][df["node_id"].duplicated()].tolist()
    if dupes:
        raise ValueError(f"nodes.node_id 중복: {dupes[:10]}")


def validate_assertions(df: pd.DataFrame) -> None:
    """assertions DataFrame 스키마 검증.

    Args:
        df: 검증할 assertions DataFrame.

    Raises:
        ValueError: 컬럼 누락, source_span 공백, enum 위반 시. 위반 행 번호 포함.
    """
    _check_columns(df, ASSERTIONS_COLUMNS, "assertions")

    # source_span 비어 있으면 실패
    blank_mask = df["source_span"].isna() | (df["source_span"].astype(str).str.strip() == "")
    bad_rows = df.index[blank_mask].tolist()
    if bad_rows:
        raise ValueError(
            f"assertions.source_span 공백/누락 — 위반 행: {bad_rows[:10]}\n"
            "모든 레코드에 source_span이 필수입니다."
        )

    # time_layer enum 검증
    bad_rows = df.index[~df["time_layer"].isin(TIME_LAYERS)].tolist()
    if bad_rows:
        raise ValueError(
            f"assertions.time_layer 위반 — 허용값: {TIME_LAYERS}\n"
            f"위반 행: {bad_rows[:10]}"
        )

    # polarity 검증
    valid_polarity = {"positive", "negative", "neutral"}
    bad_rows = df.index[~df["polarity"].isin(valid_polarity)].tolist()
    if bad_rows:
        raise ValueError(
            f"assertions.polarity 위반 — 허용값: {valid_polarity}\n"
            f"위반 행: {bad_rows[:10]}"
        )


def validate_edges(df: pd.DataFrame) -> None:
    """edges DataFrame 스키마 검증.

    Args:
        df: 검증할 엣지 DataFrame.

    Raises:
        ValueError: 컬럼 누락 또는 edge_type enum 위반 시.
    """
    _check_columns(df, EDGES_COLUMNS, "edges")

    bad_rows = df.index[~df["edge_type"].isin(EDGE_TYPES)].tolist()
    if bad_rows:
        bad_vals = df.loc[bad_rows, "edge_type"].unique().tolist()
        raise ValueError(
            f"edges.edge_type 위반 — 허용값: {EDGE_TYPES}\n"
            f"위반 행: {bad_rows[:10]}, 위반값: {bad_vals}"
        )


def discretize_m(pmi: float, cosd: float, cfg: "Config") -> int:
    """PMI·코사인 거리 합성 지표 → M ∈ {0, 1, 2}.

    PMI 하락(silo화) + cosd 증가(담론 분열)를 결합한 단일 entropy 척도.
    pmi를 [0,1]로 반전 정규화 후 cosd와 단순 평균 → 삼분위 이산화.

    Args:
        pmi: 해당 기간 PMI 값 (음수 가능).
        cosd: 해당 기간 1−cosθ 값 ∈ [0, 2].
        cfg: Config (현재는 임계값 고정, 향후 외부화 가능).

    Returns:
        M 이산값: 0(Low), 1(Mid), 2(High).
    """
    # PMI 정규화: 낮을수록 entropy 높음 → 1 - clip(pmi/max_pmi)
    max_pmi = 5.0  # 실제 데이터에서 PMI 상한 근사
    pmi_norm = float(np.clip(1.0 - pmi / max_pmi, 0.0, 1.0))
    cosd_norm = float(np.clip(cosd / 2.0, 0.0, 1.0))
    score = 0.5 * pmi_norm + 0.5 * cosd_norm

    if score < 1 / 3:
        return 0  # Low
    elif score < 2 / 3:
        return 1  # Mid
    else:
        return 2  # High


# ── 내부 헬퍼 ──────────────────────────────────────────────────────────────────
def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: 필수 컬럼 누락 — {missing}")
