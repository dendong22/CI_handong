"""inference/interventions.py — L2 Analysis.

do(C=0) 누적 붕괴 확률 계산.
트리거 차단의 무력성(P2) 수치화:
    P(Y=1, t기간) = 1 − (1 − frag·p_c)^t  (반복 Bernoulli 충격)

불변량 (seed=42):
    t=5에서 누적 붕괴 확률 = 0.761 ± 0.005
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def do_c_zero(cfg, truth: dict) -> pd.DataFrame:
    """do(C=0) 개입 시 단기 억제 vs 누적 붕괴 확률.

    frag_do_w1 = Σ_{m,u} P(u)·P(m|do(W=1))·P_FRAGILE(m,u)

    누적 붕괴:
        P_cum(t) = 1 − (1 − frag_do_w1 · p_c)^t

    Args:
        cfg: Config 인스턴스.
        truth: ground_truth.json dict. frag_do_w1, params_used 사용.

    Returns:
        DataFrame[t, cum_collapse_prob]. outputs/tables/do_c0.csv로 기록.

    Invariant:
        t=5에서 cum_collapse_prob = 0.761 ± 0.005
    """
    frag = truth["frag_do_w1"]
    p_c = cfg.p_c

    rows = []
    for t in range(1, cfg.t_horizon + 1):
        cum_prob = 1.0 - (1.0 - frag * p_c) ** t
        rows.append({"t": t, "cum_collapse_prob": round(cum_prob, 6)})

    df = pd.DataFrame(rows)

    cfg.paths.ensure_all()
    df.to_csv(cfg.paths.do_c0_csv, index=False)

    logger.info(
        "do(C=0) 누적 붕괴 확률:\n%s",
        df.to_string(index=False),
    )
    return df


def main(cfg=None) -> None:
    """ground_truth.json을 읽어 do_c0.csv 기록.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    with cfg.paths.ground_truth_json.open() as f:
        truth = json.load(f)

    do_c_zero(cfg, truth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config

    main(load_config(root / "params.yaml", root=root))
