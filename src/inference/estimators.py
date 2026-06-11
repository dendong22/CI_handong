"""inference/estimators.py — L2 Analysis.

순진 추정·전단계(front-door) 추정·DGP 참값의 3중 대조.

전단계 추정식:
    P(Y|do(W=w)) = Σ_m P(m|w) Σ_{w'} P(Y|m,w') P(w')

불변량 (seed=42, N=5000):
    naive ATE  = +0.173 ± 0.002
    FD ATE     = +0.144 ± 0.002
    |FD−truth| ≤ 0.002
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def naive_estimate(df: pd.DataFrame) -> dict:
    """순진 추정: E[Y|W=w] (U에 의한 교란 미보정).

    Args:
        df: DataFrame[W, M, Y].

    Returns:
        {"0": P(Y=1|W=0), "1": P(Y=1|W=1), "ate": float}
    """
    p0 = df.loc[df["W"] == 0, "Y"].mean()
    p1 = df.loc[df["W"] == 1, "Y"].mean()
    return {"0": float(p0), "1": float(p1), "ate": float(p1 - p0)}


def front_door_estimate(df: pd.DataFrame) -> dict:
    """전단계 추정: Pearl (1995) 전단계 공식.

    P(Y|do(W=w)) = Σ_m P(m|w) Σ_{w'} P(Y|m,w') P(w')

    Args:
        df: DataFrame[W, M, Y]. U 열 불필요(관측 불가 설계).

    Returns:
        {"0": P(Y|do(W=0)), "1": P(Y|do(W=1)), "ate": float}
    """
    def estimate_do_w(w_do: int) -> float:
        total = 0.0
        for m in range(3):
            # P(M=m | W=w_do)
            w_mask = df["W"] == w_do
            p_m_w = ((df["M"] == m) & w_mask).sum() / max(w_mask.sum(), 1)

            inner = 0.0
            for wp in (0, 1):
                # P(Y=1 | M=m, W=w')
                mask = (df["W"] == wp) & (df["M"] == m)
                p_y = df.loc[mask, "Y"].mean() if mask.sum() > 0 else 0.0
                # P(W=w')
                p_wp = (df["W"] == wp).mean()
                inner += p_y * p_wp

            total += p_m_w * inner
        return total

    p0 = estimate_do_w(0)
    p1 = estimate_do_w(1)
    return {"0": float(p0), "1": float(p1), "ate": float(p1 - p0)}


def triple_contrast(cfg, df: pd.DataFrame, truth: dict) -> pd.DataFrame:
    """T2 테이블: naive·front-door·truth 3중 대조.

    Args:
        cfg: Config 인스턴스 (로깅·경로용).
        df: observed.csv DataFrame[W, M, Y].
        truth: ground_truth.json dict.

    Returns:
        DataFrame[w, naive, front_door, truth] + ATE 행.
        outputs/tables/t2_estimates.csv로 기록.
    """
    naive = naive_estimate(df)
    fd = front_door_estimate(df)
    tr = truth["p_y_do_w"]  # {"0": float, "1": float}
    ate_true = truth["ate_true"]

    rows = [
        {
            "w": "W=0",
            "naive": round(naive["0"], 4),
            "front_door": round(fd["0"], 4),
            "truth": round(float(tr["0"]), 4),
        },
        {
            "w": "W=1",
            "naive": round(naive["1"], 4),
            "front_door": round(fd["1"], 4),
            "truth": round(float(tr["1"]), 4),
        },
        {
            "w": "ATE",
            "naive": round(naive["ate"], 4),
            "front_door": round(fd["ate"], 4),
            "truth": round(float(ate_true), 4),
        },
    ]
    result_df = pd.DataFrame(rows)

    cfg.paths.ensure_all()
    result_df.to_csv(cfg.paths.t2_estimates_csv, index=False)

    logger.info(
        "T2 추정:\n  naive ATE=%.4f  FD ATE=%.4f  truth=%.4f  |FD-truth|=%.4f",
        naive["ate"],
        fd["ate"],
        ate_true,
        abs(fd["ate"] - ate_true),
    )
    return result_df


def main(cfg=None) -> None:
    """observed.csv + ground_truth.json을 읽어 t2_estimates.csv 기록.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    df = pd.read_csv(cfg.paths.observed_csv)
    with cfg.paths.ground_truth_json.open() as f:
        truth = json.load(f)

    triple_contrast(cfg, df, truth)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config

    main(load_config(root / "params.yaml", root=root))
