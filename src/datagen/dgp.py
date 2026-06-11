"""datagen/dgp.py — L1 Data.

시드 고정 DGP. 관측 표본(U 은닉)과 do-분포 해석적 참값을 분리 생성.
기존 sim_frontdoor.py의 로직을 함수화; print 의존 제거.

구조방정식 (이산):
    P(U=1) = p_u1
    P(W=1 | U=u) = p_w1_given_u[u]
    P(M=m | W=w) = p_m_given_w[w][m]          ← A1: U -/-> M
    P(Y=1 | M=m, U=u) = P_FRAGILE[(m,u)] * P_C  ← A2: W -/-> Y (직접 경로 없음)
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def generate(cfg) -> Tuple[pd.DataFrame, dict]:
    """DGP 실행: 관측 표본 + 해석적 참값 반환.

    Args:
        cfg: Config 인스턴스. seed_dgp, n_samples, p_* 파라미터 사용.

    Returns:
        (observed_df, truth_dict)
        - observed_df: DataFrame[W, M, Y] — U 열 의도적 부재
        - truth_dict: {
            "p_y_do_w": {"0": float, "1": float},
            "ate_true": float,
            "frag_do_w1": float,
            "params_used": {...}
          }

    불변량:
        seed=42, N=5000 → ate_true = +0.143 ± 0.001
    """
    rng = np.random.default_rng(cfg.seed_dgp)

    N = cfg.n_samples
    p_u1 = cfg.p_u1
    pw1 = cfg.p_w1_given_u      # {0: 0.70, 1: 0.30}
    pm_w = cfg.p_m_given_w      # {0: [...], 1: [...]}
    pf = cfg.p_fragile          # {(m, u): float}
    p_c = cfg.p_c

    # ── 구조방정식 샘플 생성 ──────────────────────────────────────────────────
    U = (rng.random(N) < p_u1).astype(int)
    W = (rng.random(N) < np.vectorize(pw1.__getitem__)(U)).astype(int)
    M = np.array([rng.choice(3, p=pm_w[w]) for w in W])

    def p_y_given_mu(m: int, u: int) -> float:
        """P(Y=1 | M=m, U=u): 취약성 * 충격 확률 (단일 기간)."""
        return pf[(m, u)] * p_c

    Y = (rng.random(N) < np.array([p_y_given_mu(int(m), int(u)) for m, u in zip(M, U)])).astype(int)

    observed_df = pd.DataFrame({"W": W, "M": M, "Y": Y})

    # ── 해석적 참값 P(Y=1 | do(W=w)) ─────────────────────────────────────────
    def truth(w_do: int) -> float:
        """전단계 식별: Σ_m P(m|w) Σ_u P(u) P(Y|m,u)."""
        val = 0.0
        for u in (0, 1):
            pu = p_u1 if u == 1 else 1.0 - p_u1
            for m in range(3):
                val += pu * pm_w[w_do][m] * p_y_given_mu(m, u)
        return val

    p_y_do_w = {0: truth(0), 1: truth(1)}
    ate_true = p_y_do_w[1] - p_y_do_w[0]

    # ── frag_do_w1: do(W=1) 하에서의 기대 취약성 (개입·민감도 모듈에서 사용) ──
    frag_do_w1 = sum(
        (p_u1 if u else 1.0 - p_u1) * pm_w[1][m] * pf[(m, u)]
        for u in (0, 1)
        for m in range(3)
    )

    truth_dict = {
        "p_y_do_w": {"0": round(p_y_do_w[0], 6), "1": round(p_y_do_w[1], 6)},
        "ate_true": round(ate_true, 6),
        "frag_do_w1": round(frag_do_w1, 6),
        "params_used": {
            "seed_dgp": cfg.seed_dgp,
            "n_samples": N,
            "p_u1": p_u1,
            "p_c": p_c,
        },
    }

    logger.info(
        "DGP 생성 완료: N=%d, ate_true=%.4f, frag_do_w1=%.4f",
        N,
        ate_true,
        frag_do_w1,
    )
    return observed_df, truth_dict


def main(cfg=None) -> None:
    """파일 기록: data/dgp/observed.csv, ground_truth.json.

    Args:
        cfg: Config 인스턴스. None이면 CWD의 params.yaml에서 로드.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    cfg.paths.ensure_all()

    observed_df, truth_dict = generate(cfg)

    # U 열 부재 재확인
    assert "U" not in observed_df.columns, "DGP 보안 위반: U 열이 관측 표본에 포함됨"

    observed_df.to_csv(cfg.paths.observed_csv, index=False)
    with cfg.paths.ground_truth_json.open("w", encoding="utf-8") as f:
        json.dump(truth_dict, f, indent=2, ensure_ascii=False)

    logger.info("기록 완료: %s, %s", cfg.paths.observed_csv, cfg.paths.ground_truth_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # CWD를 프로젝트 루트로 가정
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config

    main(load_config(root / "params.yaml", root=root))
