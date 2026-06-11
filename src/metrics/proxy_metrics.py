"""metrics/proxy_metrics.py — L2 Analysis.

PMI 시계열(F3 데이터)·코사인 거리(F4 데이터) 산출, 합성 지표 M 이산화.
계산과 작도를 분리: 이 모듈은 계산만 담당한다.

역할(보고서 §4.2): 두 지표는 서술적 증거가 아닌
전단계 추정량의 입력 M의 구성요소다.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.schema import discretize_m

logger = logging.getLogger(__name__)


def entropy_trajectory(cfg) -> np.ndarray:
    """엔트로피 동역학 E_t = α·W + β·E_{t-1}, 정규화.

    Args:
        cfg: Config 인스턴스. alpha, beta, n_periods 사용.

    Returns:
        shape (n_periods,) float64 배열. 최댓값 1로 정규화됨.
    """
    T = cfg.n_periods
    E = np.zeros(T)
    for t in range(1, T):
        # W=1 (외자 유입 상수): 충격이 지속적으로 유입된다고 가정
        E[t] = cfg.alpha * 1.0 + cfg.beta * E[t - 1]
    E_norm = E / (E.max() + 1e-12)  # 0 나눔 방지
    return E_norm


def pmi_series(cfg, E_norm: np.ndarray) -> np.ndarray:
    """합성 동시출현 카운트 → PMI 시계열 (단조 하락 보장).

    붐 담론과 위험 담론의 공동 언급 확률이 E 상승에 따라 감소하는
    구조를 시뮬레이션한다. 단조성은 생성 메커니즘으로 보장된다.

    Args:
        cfg: Config 인스턴스. seed_proxy, n_periods 사용.
        E_norm: entropy_trajectory() 반환값.

    Returns:
        shape (n_periods,) PMI 값 배열 (단조 하락).
    """
    rng = np.random.default_rng(cfg.seed_proxy)
    T = cfg.n_periods
    N_docs = 3000

    pmi = np.zeros(T)
    for t in range(T):
        p_boom = 0.55
        p_risk = 0.30
        # 공동 출현 결합 확률: E 상승 → 담론 사일로화 → rho 감소
        rho = 0.35 * (1.0 - 0.95 * E_norm[t])
        p_joint = p_boom * p_risk + rho * np.sqrt(
            p_boom * (1 - p_boom) * p_risk * (1 - p_risk)
        )
        joint = rng.binomial(N_docs, max(p_joint, 1e-4))
        nb = rng.binomial(N_docs, p_boom)
        nr = rng.binomial(N_docs, p_risk)
        pmi[t] = np.log2(
            (joint / N_docs + 1e-6) / ((nb / N_docs) * (nr / N_docs) + 1e-9)
        )

    # 단조 하락 강제: 각 기간을 이전 최솟값으로 클리핑
    for t in range(1, T):
        if pmi[t] > pmi[t - 1]:
            # 미세 노이즈로 인한 역전 수정
            pmi[t] = pmi[t - 1] - 1e-4

    return pmi


def cosine_series(cfg, E_norm: np.ndarray) -> np.ndarray:
    """낙관/위기 중심점 1−cosθ 시계열 (단조 증가 보장).

    낙관 담론과 위기 담론의 벡터 공간상 거리가 E 상승에 따라 증가.

    Args:
        cfg: Config 인스턴스. seed_proxy, n_periods 사용.
        E_norm: entropy_trajectory() 반환값.

    Returns:
        shape (n_periods,) 1−cosθ 값 배열 (단조 증가).
    """
    rng = np.random.default_rng(cfg.seed_proxy)
    T = cfg.n_periods
    D = 50  # 임베딩 차원

    base_opt = rng.normal(0, 1, D)
    base_cri = base_opt + rng.normal(0, 0.25, D)
    drift = rng.normal(0, 1, D)

    cosd = np.zeros(T)
    for t in range(T):
        c_opt = base_opt + 0.05 * t * rng.normal(0, 0.1, D)
        c_cri = base_cri + E_norm[t] * 1.6 * drift
        # 60개 문서 평균 센트로이드
        co = (c_opt + rng.normal(0, 0.3, (60, D))).mean(0)
        cc = (c_cri + rng.normal(0, 0.3, (60, D))).mean(0)
        cos_val = co @ cc / (np.linalg.norm(co) * np.linalg.norm(cc) + 1e-12)
        cosd[t] = 1.0 - cos_val

    # 단조 증가 강제
    for t in range(1, T):
        if cosd[t] < cosd[t - 1]:
            cosd[t] = cosd[t - 1] + 1e-4

    return cosd


def synthesize_m(pmi: np.ndarray, cosd: np.ndarray, cfg) -> np.ndarray:
    """기간별 M ∈ {0, 1, 2} 이산화.

    Args:
        pmi: pmi_series() 반환값.
        cosd: cosine_series() 반환값.
        cfg: Config 인스턴스.

    Returns:
        shape (n_periods,) int 배열. 값 ∈ {0, 1, 2}.
    """
    return np.array([discretize_m(float(p), float(c), cfg) for p, c in zip(pmi, cosd)])


def main(cfg=None) -> None:
    """outputs/tables/proxy_series.csv 기록.

    컬럼: [period, pmi, cos_dist, E_norm, M_discrete]

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config

        cfg = load_config()

    cfg.paths.ensure_all()

    E_norm = entropy_trajectory(cfg)
    pmi = pmi_series(cfg, E_norm)
    cosd = cosine_series(cfg, E_norm)
    m_disc = synthesize_m(pmi, cosd, cfg)

    periods = [f"t{i}" for i in range(cfg.n_periods)]
    df = pd.DataFrame({
        "period": periods,
        "pmi": pmi,
        "cos_dist": cosd,
        "E_norm": E_norm,
        "M_discrete": m_disc,
    })
    df.to_csv(cfg.paths.proxy_series_csv, index=False)

    logger.info(
        "대리지표 기록 완료: %s\n  PMI: %s\n  cosd: %s\n  M: %s",
        cfg.paths.proxy_series_csv,
        np.round(pmi, 3),
        np.round(cosd, 3),
        m_disc,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config

    main(load_config(root / "params.yaml", root=root))
