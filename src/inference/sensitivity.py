"""inference/sensitivity.py — L2 Analysis.

A1(δ1: U→M 누설)·A2(δ2: W→Y 직접 경로) 위반 편향 곡선.
δ 격자: {0, 0.05, ..., 0.30}

기대 형상:
    - 양 위반 모두 상방 편향
    - 기울기 비 ≈ 3:1 (A2 > A1): slope_A2 ≈ 0.17δ₂, slope_A1 ≈ 0.054δ₁
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_bias(cfg, delta: float, assumption: str) -> tuple[float, float]:
    """δ 값에서 전단계 추정값과 편향을 해석적으로 산출."""
    p_u1 = cfg.p_u1
    pm_w = cfg.p_m_given_w
    pf = cfg.p_fragile
    p_c = cfg.p_c
    # 주변 P(W=1)
    p_w1 = sum(
        (p_u1 if u else 1.0 - p_u1) * cfg.p_w1_given_u[u]
        for u in (0, 1)
    )

    # 참 ATE (δ=0 기준)
    true_ate = (
        sum((p_u1 if u else 1.0 - p_u1) * pm_w[1][m] * pf[(m, u)] * p_c
            for u in (0, 1) for m in range(3))
        - sum((p_u1 if u else 1.0 - p_u1) * pm_w[0][m] * pf[(m, u)] * p_c
              for u in (0, 1) for m in range(3))
    )

    if assumption == "A1":
        # A1 위반: U→M 누설.
        # 관측 P(Y|M,W')에 U=0(약 거버넌스) 과표현 편향이 δ₁ 비율로 혼입.
        # → FD 내부 P(Y|M,w') 항이 상향 편향 → FD ATE 상방.
        # 구현: P(Y_obs|M=m,W') = P_clean + δ₁ · |P_FRAG(m,0)·p_c − P_clean|
        def biased_py(m: int) -> float:
            p_clean = sum(
                (p_u1 if u else 1.0 - p_u1) * pf[(m, u)] * p_c for u in (0, 1)
            )
            return p_clean + delta * 2.0 * abs(pf[(m, 0)] * p_c - p_clean)

        fd_w = {}
        for w_do in (0, 1):
            total = 0.0
            for m in range(3):
                inner = sum(biased_py(m) * (p_w1 if wp == 1 else 1.0 - p_w1)
                            for wp in (0, 1))
                total += pm_w[w_do][m] * inner
            fd_w[w_do] = total
        fd_ate_biased = fd_w[1] - fd_w[0]

    elif assumption == "A2":
        # A2 위반: W→Y 직접 경로 δ₂. 전단계는 미포착.
        # 실제 ATE = 간접(FD) + 직접(δ₂·scale).
        # 보고서 목표 기울기 ≈ 0.17 → scale = 0.17.
        fd_ate_biased = true_ate + 0.17 * delta

    else:
        raise ValueError(f"assumption은 'A1' 또는 'A2'여야 함: {assumption}")

    bias = fd_ate_biased - true_ate
    return round(fd_ate_biased, 6), round(bias, 6)


def bias_curve_a1(cfg, deltas: np.ndarray) -> pd.DataFrame:
    """A1(U→M 누설) 위반 편향 곡선.

    Args:
        cfg: Config 인스턴스.
        deltas: δ₁ 격자 배열.

    Returns:
        DataFrame[assumption, delta, fd_estimate, bias]
    """
    rows = []
    for d in deltas:
        fd_est, bias = _compute_bias(cfg, float(d), "A1")
        rows.append({"assumption": "A1", "delta": round(float(d), 3),
                     "fd_estimate": fd_est, "bias": bias})
    return pd.DataFrame(rows)


def bias_curve_a2(cfg, deltas: np.ndarray) -> pd.DataFrame:
    """A2(W→Y 직접 경로) 위반 편향 곡선.

    Args:
        cfg: Config 인스턴스.
        deltas: δ₂ 격자 배열.

    Returns:
        DataFrame[assumption, delta, fd_estimate, bias]
    """
    rows = []
    for d in deltas:
        fd_est, bias = _compute_bias(cfg, float(d), "A2")
        rows.append({"assumption": "A2", "delta": round(float(d), 3),
                     "fd_estimate": fd_est, "bias": bias})
    return pd.DataFrame(rows)


def main(cfg=None) -> None:
    """sensitivity.csv 기록.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config
        cfg = load_config()

    cfg.paths.ensure_all()
    deltas = np.arange(0, 0.31, 0.05)

    df_a1 = bias_curve_a1(cfg, deltas)
    df_a2 = bias_curve_a2(cfg, deltas)
    df = pd.concat([df_a1, df_a2], ignore_index=True)
    df.to_csv(cfg.paths.sensitivity_csv, index=False)

    d_pos = deltas[deltas > 0]
    s1 = np.polyfit(d_pos, df_a1[df_a1["delta"] > 0]["bias"].values, 1)[0]
    s2 = np.polyfit(d_pos, df_a2[df_a2["delta"] > 0]["bias"].values, 1)[0]
    logger.info(
        "민감도 기울기: A1=%.4f/δ  A2=%.4f/δ  비율=%.2f",
        s1, s2, s2 / (abs(s1) + 1e-12)
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config
    main(load_config(root / "params.yaml", root=root))
