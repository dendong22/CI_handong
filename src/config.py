"""config.py — L0 Foundation.

모든 수치 파라미터·경로의 유일한 정의처.
다른 모듈에서 수치 리터럴 사용 금지; 모든 함수는 cfg: Config를 첫 인자로 받는다.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass(frozen=True)
class Paths:
    """모든 입출력 절대 경로. root 기준으로 파생."""

    root: Path

    # ── data ──
    @property
    def data_raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def data_dgp(self) -> Path:
        return self.root / "data" / "dgp"

    @property
    def observed_csv(self) -> Path:
        return self.data_dgp / "observed.csv"

    @property
    def ground_truth_json(self) -> Path:
        return self.data_dgp / "ground_truth.json"

    # ── outputs ──
    @property
    def out_graph(self) -> Path:
        return self.root / "outputs" / "graph"

    @property
    def out_tables(self) -> Path:
        return self.root / "outputs" / "tables"

    @property
    def out_figures(self) -> Path:
        return self.root / "outputs" / "figures"

    # ── tables ──
    @property
    def proxy_series_csv(self) -> Path:
        return self.out_tables / "proxy_series.csv"

    @property
    def t2_estimates_csv(self) -> Path:
        return self.out_tables / "t2_estimates.csv"

    @property
    def do_c0_csv(self) -> Path:
        return self.out_tables / "do_c0.csv"

    @property
    def sensitivity_csv(self) -> Path:
        return self.out_tables / "sensitivity.csv"

    @property
    def motifs_csv(self) -> Path:
        return self.out_tables / "motifs.csv"

    @property
    def backtrace_json(self) -> Path:
        return self.out_tables / "backtrace.json"

    @property
    def transport_result_json(self) -> Path:
        return self.out_tables / "transport_result.json"

    # ── graph exports ──
    @property
    def nodes_csv(self) -> Path:
        return self.out_graph / "nodes.csv"

    @property
    def edges_csv(self) -> Path:
        return self.out_graph / "edges.csv"

    @property
    def graphml(self) -> Path:
        return self.out_graph / "ekg.graphml"

    # ── figures ──
    @property
    def f1_dag(self) -> Path:
        return self.out_figures / "F1_dag.png"

    @property
    def f2_unrolled(self) -> Path:
        return self.out_figures / "F2_unrolled.png"

    @property
    def f3_pmi(self) -> Path:
        return self.out_figures / "F3_pmi.png"

    @property
    def f4_cosine(self) -> Path:
        return self.out_figures / "F4_cosine.png"

    def ensure_all(self) -> None:
        """파이프라인 실행 전 출력 디렉토리를 일괄 생성."""
        for p in (self.data_dgp, self.out_graph, self.out_tables, self.out_figures):
            p.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    """파이프라인 전체 설정. 필드는 params.yaml에서 로드된다."""

    seed_dgp: int
    seed_proxy: int
    n_samples: int
    t_horizon: int
    n_periods: int

    # DGP 구조 파라미터
    p_u1: float
    p_w1_given_u: Dict[int, float]          # {0: 0.70, 1: 0.30}
    p_m_given_w: Dict[int, List[float]]     # {0: [...], 1: [...]}
    p_fragile: Dict[Tuple[int, int], float]  # {(m, u): p}
    p_c: float

    # 엔트로피 동역학
    alpha: float
    beta: float

    # 반사실 기본 시점
    t_k_default: int

    paths: Paths


def load_config(
    yaml_path: str = "params.yaml",
    root: Path | None = None,
) -> Config:
    """params.yaml을 읽어 Config 인스턴스를 반환한다.

    Args:
        yaml_path: params.yaml 경로 (절대 또는 CWD 상대).
        root: 프로젝트 루트. None이면 yaml_path의 부모 디렉토리.

    Returns:
        불변 Config 인스턴스.

    Raises:
        FileNotFoundError: yaml_path가 존재하지 않을 때.
        KeyError: 필수 파라미터 누락 시.
    """
    yaml_path = Path(yaml_path).resolve()
    if not yaml_path.exists():
        raise FileNotFoundError(f"params.yaml not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if root is None:
        root = yaml_path.parent

    # p_w1_given_u: 키를 int로 변환
    p_w1 = {int(k): float(v) for k, v in raw["p_w1_given_u"].items()}

    # p_m_given_w: 키를 int로 변환
    p_m = {int(k): [float(x) for x in v] for k, v in raw["p_m_given_w"].items()}

    # p_fragile: "m,u" → (int, int)
    p_frag = {
        tuple(int(x) for x in k.split(",")): float(v)
        for k, v in raw["p_fragile"].items()
    }

    return Config(
        seed_dgp=int(raw["seed_dgp"]),
        seed_proxy=int(raw["seed_proxy"]),
        n_samples=int(raw["n_samples"]),
        t_horizon=int(raw["t_horizon"]),
        n_periods=int(raw["n_periods"]),
        p_u1=float(raw["p_u1"]),
        p_w1_given_u=p_w1,
        p_m_given_w=p_m,
        p_fragile=p_frag,
        p_c=float(raw["p_c"]),
        alpha=float(raw["alpha"]),
        beta=float(raw["beta"]),
        t_k_default=int(raw["t_k_default"]),
        paths=Paths(root=Path(root)),
    )
