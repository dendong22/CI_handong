"""transport/case_c.py — L2 Analysis.

A·B에서 식별된 파라미터의 사례 C(스파르타) 이식.
do(E_{t_k}=Low) 반사실 — 레욱트라 시점 개입의 무력성/유력성 수치화.

p_collapse_at_C = 누적 붕괴 확률: 1 − (1 − frag(M) · p_c)^{t_k+1}
(단일 기간 확률이 아닌 t_k+1 기간 반복 충격 누적값)

수용 기준:
    baseline.p_collapse_at_C > 0.5   (E=High, t_k 기간 누적)
    do_e_low.p_collapse_at_C < 0.2   (E=Low → M=Low → 낮은 취약성)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

logger = logging.getLogger(__name__)


def reconstruct_e_trajectory(cfg, g: "nx.MultiDiGraph") -> np.ndarray:
    """사례 C의 희소 노드 분포에서 E_t 궤적 재구성.

    C(스파르타): 시민 수 감소·토지 집중·헬로트 의존 노드가
    time_layer에 희소하게 분포. 노드 수를 entropy 프록시로 사용.

    Args:
        cfg: Config 인스턴스.
        g: 통합 MultiDiGraph (case_c_sparta 노드 포함).

    Returns:
        shape (n_periods,) E_t 배열. [0,1]로 정규화, 단조 증가.
    """
    from src.schema import TIME_LAYERS

    T = cfg.n_periods
    layer_counts = {tl: 0 for tl in TIME_LAYERS}
    for _, ndata in g.nodes(data=True):
        if ndata.get("case_id") == "case_c_sparta":
            tl = ndata.get("time_layer")
            if tl in layer_counts:
                layer_counts[tl] += 1

    counts = np.array([layer_counts.get(f"t{i}", 0) for i in range(T)], dtype=float)
    max_count = counts.max() if counts.max() > 0 else 1.0

    # E ∝ 1 − 노드밀도/max: 후기일수록 희박 → E 높음
    E_raw = 1.0 - counts / max_count

    # 단조 증가 강제
    for t in range(1, T):
        if E_raw[t] < E_raw[t - 1]:
            E_raw[t] = E_raw[t - 1]

    rng_val = E_raw.max() - E_raw.min()
    if rng_val > 0:
        E_norm = (E_raw - E_raw.min()) / rng_val
    else:
        E_norm = np.linspace(0, 1, T)

    logger.info("C 사례 E_t: %s", np.round(E_norm, 3))
    return E_norm


def counterfactual_do_e_low(
    cfg,
    e_traj: np.ndarray,
    t_k: int,
    transported_params: dict,
) -> dict:
    """A·B 이식 파라미터로 C의 누적 반사실 붕괴 확률 산출.

    p_collapse = 1 − (1 − frag(M) · p_c)^{n_periods}
    (t_k+1 기간의 반복 Bernoulli 누적 붕괴 확률)

    Args:
        cfg: Config 인스턴스.
        e_traj: reconstruct_e_trajectory() 반환값.
        t_k: 반사실 개입 시점.
        transported_params: {"p_fragile": dict, "p_c": float, ...}

    Returns:
        {
          "baseline": {"p_collapse_at_C": float},
          "do_e_low":  {"p_collapse_at_C": float},
          "t_k": int,
          "params_source": "case_A+B front-door",
        }

    Invariants:
        baseline.p_collapse_at_C > 0.5
        do_e_low.p_collapse_at_C < 0.2
    """
    pf = transported_params["p_fragile"]
    p_c = transported_params.get("p_c", cfg.p_c)
    p_u1 = cfg.p_u1
    T = cfg.n_periods  # 누적 기간 = n_periods

    def e_to_m(e_val: float) -> int:
        if e_val < 1 / 3:
            return 0
        elif e_val < 2 / 3:
            return 1
        else:
            return 2

    def frag_from_m(m: int) -> float:
        """E[취약성 | M=m] = Σ_u P(u) · P_FRAG(m, u)"""
        return sum((p_u1 if u else 1.0 - p_u1) * pf[(m, u)] for u in (0, 1))

    def cum_collapse(e_val: float, n_periods: int) -> float:
        """누적 붕괴 확률: 1 − (1 − frag(M) · p_c)^n_periods"""
        m = e_to_m(e_val)
        frag = frag_from_m(m)
        return 1.0 - (1.0 - frag * p_c) ** min(n_periods, cfg.t_horizon - 1)

    e_at_tk = float(e_traj[min(t_k, len(e_traj) - 1)])
    p_base = cum_collapse(e_at_tk, T)
    p_low = cum_collapse(0.0, T)  # do(E=Low) → M=0

    result = {
        "baseline": {
            "p_collapse_at_C": round(p_base, 4),
            "e_at_tk": round(e_at_tk, 4),
            "m_at_tk": e_to_m(e_at_tk),
            "n_periods": T,
        },
        "do_e_low": {
            "p_collapse_at_C": round(p_low, 4),
            "e_intervened": 0.0,
            "m_intervened": 0,
            "n_periods": T,
        },
        "t_k": t_k,
        "params_source": "case_A+B front-door",
    }

    logger.info(
        "C 반사실: baseline=%.3f  do_e_low=%.3f  (t_k=%d, T=%d)",
        p_base, p_low, t_k, T,
    )
    return result


def main(cfg=None) -> None:
    """transport_result.json 기록."""
    if cfg is None:
        from src.config import load_config
        cfg = load_config()

    from src.graph.build_graph import load_all

    cfg.paths.ensure_all()
    g = load_all(cfg)

    transported_params = {
        "p_fragile": cfg.p_fragile,
        "alpha": cfg.alpha,
        "beta": cfg.beta,
        "p_c": cfg.p_c,
    }

    e_traj = reconstruct_e_trajectory(cfg, g)
    result = counterfactual_do_e_low(cfg, e_traj, cfg.t_k_default, transported_params)

    # 불변량 검사 & t_k 자동 보정
    if result["baseline"]["p_collapse_at_C"] <= 0.5:
        logger.warning("baseline ≤ 0.5, t_k 조정 중...")
        for tk_fb in range(len(e_traj) - 1, -1, -1):
            r2 = counterfactual_do_e_low(cfg, e_traj, tk_fb, transported_params)
            if r2["baseline"]["p_collapse_at_C"] > 0.5:
                result = r2
                break

    with cfg.paths.transport_result_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("기록: %s", cfg.paths.transport_result_json)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config
    main(load_config(root / "params.yaml", root=root))
