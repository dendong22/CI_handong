"""viz/make_figs.py — L3 Presentation.

F3(PMI 시계열)·F4(코사인 거리 시계열) 작도.
proxy_series.csv만 읽어 작도한다 — 재계산 금지, 난수 호출 0회.

수용 기준:
    - PNG 2종 생성(200dpi)
    - 작도 코드 내 난수 호출 0회 (proxy_series.csv 입력 전용)
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def render_f3(cfg) -> Path:
    """F3_pmi.png: PMI(boom-kw, risk-kw) 시계열.

    Args:
        cfg: Config 인스턴스.

    Returns:
        저장된 PNG 파일 경로.

    Raises:
        FileNotFoundError: proxy_series.csv 없을 때.
    """
    csv_path = cfg.paths.proxy_series_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"proxy_series.csv 없음: {csv_path}\n먼저 proxy_metrics.main()을 실행하라.")

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(df["period"], df["pmi"], marker="o", color="black", lw=1.4)
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_ylabel("PMI(boom kw, risk kw)")
    ax.set_xlabel("period")
    ax.set_title("F3: Discourse Co-occurrence PMI", fontsize=10)
    fig.tight_layout()

    out_path = cfg.paths.f3_pmi
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("F3 저장: %s", out_path)
    return out_path


def render_f4(cfg) -> Path:
    """F4_cosine.png: 1−cosθ (낙관 vs 위기 담론 중심점 거리).

    Args:
        cfg: Config 인스턴스.

    Returns:
        저장된 PNG 파일 경로.

    Raises:
        FileNotFoundError: proxy_series.csv 없을 때.
    """
    csv_path = cfg.paths.proxy_series_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"proxy_series.csv 없음: {csv_path}\n먼저 proxy_metrics.main()을 실행하라.")

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    ax.plot(df["period"], df["cos_dist"], marker="s", color="black", lw=1.4)
    ax.set_ylabel(r"$1 - \cos(\theta)$  (낙관 vs 위기)")
    ax.set_xlabel("period")
    ax.set_title("F4: Discourse Centroid Cosine Distance", fontsize=10)
    fig.tight_layout()

    out_path = cfg.paths.f4_cosine
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("F4 저장: %s", out_path)
    return out_path


def main(cfg=None) -> None:
    """F3, F4 PNG 생성.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    render_f3(cfg)
    render_f4(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config
    main(load_config(root / "params.yaml", root=root))
