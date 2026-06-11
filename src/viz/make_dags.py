"""viz/make_dags.py — L3 Presentation.

F1: 전단계 DAG (U 점선, A1·A2 금지 엣지 ✕ 표기, E 점선 격하).
F2: 시간 전개 DAG (W→E_t→S_t→Y, τ 지연 표기).

make_dags2.py 로직 이식; 출력 경로는 cfg.paths에서 결정.
난수 호출 없음.
"""
from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch

logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.family": "Malgun Gothic",
    "axes.unicode_minus": False,
    "font.size": 11
})

_NW, _NH = 0.11, 0.13  # 노드 반폭/반높이 (데이터 좌표)


# ── 공용 드로잉 헬퍼 ───────────────────────────────────────────────────────────
def _node(ax, xy: tuple, text: str, dashed: bool = False, scale: float = 1.0) -> None:
    """타원 노드 + 텍스트 레이블."""
    e = Ellipse(
        xy,
        2 * _NW * scale,
        2 * _NH * scale,
        fill=True,
        facecolor="white",
        edgecolor="black",
        lw=1.3,
        linestyle=(0, (4, 3)) if dashed else "-",
        zorder=3,
    )
    ax.add_patch(e)
    ax.text(*xy, text, ha="center", va="center", fontsize=12, style="italic", zorder=4)


def _arrow(
    ax,
    p: tuple,
    q: tuple,
    shrink: float = 26,
    rad: float = 0.0,
    ls: str = "solid",
    color: str = "black",
    lw: float = 1.4,
) -> None:
    """화살표 패치."""
    a = FancyArrowPatch(
        p, q,
        arrowstyle="-|>",
        mutation_scale=15,
        lw=lw,
        color=color,
        linestyle=ls,
        shrinkA=shrink,
        shrinkB=shrink,
        connectionstyle=f"arc3,rad={rad}",
        zorder=2,
    )
    ax.add_patch(a)


# ── F1: 전단계 식별 DAG ────────────────────────────────────────────────────────
def render_f1(cfg) -> Path:
    """F1_dag.png 생성.

    필수 요소:
        - U 점선 노드 (잠재 교란)
        - A1 금지 엣지 (U→M) ✕ 표기
        - A2 금지 엣지 (W→Y) ✕ 표기
        - E 점선 노드 (잠재 엔트로피, 하단 격하)

    Args:
        cfg: Config 인스턴스.

    Returns:
        저장된 PNG 파일 경로.
    """
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    P = {
        "U": (0.50, 0.88),
        "W": (0.07, 0.34),
        "M": (0.50, 0.34),
        "Y": (0.93, 0.34),
        "E": (0.20, 0.07),
        "C": (0.93, 0.86),
    }

    # ── 식별 엣지 [I] ─────────────────────────────────────────────────────────
    _arrow(ax, P["W"], P["M"], shrink=52)            # W→M (전단계 1단계)
    _arrow(ax, P["M"], P["Y"], shrink=52)            # M→Y (전단계 2단계)

    # ── 교란 엣지 [U] (점선) ──────────────────────────────────────────────────
    _arrow(ax, P["U"], P["W"], ls="dashed", shrink=42)
    _arrow(ax, P["U"], P["Y"], ls="dashed", shrink=42)

    # ── 주장 엣지 [A] ─────────────────────────────────────────────────────────
    _arrow(ax, P["C"], P["Y"], shrink=24)            # C→Y (트리거)

    # ── 잠재 엔트로피 → M (점선, 보조) ──────────────────────────────────────
    _arrow(ax, P["E"], P["M"], ls="dotted", lw=1.0, shrink=22)

    # ── A1 금지 엣지: U→M (점선 회색 + ✕ A1) ────────────────────────────────
    _arrow(ax, P["U"], P["M"], ls="dotted", color="gray", lw=1.0, shrink=24)
    ax.text(0.515, 0.615, "✕", fontsize=13, fontweight="bold", ha="center",
            color="crimson", zorder=5)
    ax.text(0.595, 0.615, "A1", fontsize=9, color="dimgray")

    # ── A2 금지 엣지: W→Y 직접 경로 (호형 점선 + ✕ A2) ─────────────────────
    _arrow(ax, P["W"], P["Y"], rad=0.50, ls="dotted", color="gray", lw=1.0, shrink=40)
    ax.text(0.55, -0.085, "✕", fontsize=13, fontweight="bold", ha="center",
            color="crimson", zorder=5)
    ax.text(0.615, -0.085, "A2", fontsize=9, color="dimgray")

    # ── 노드 ──────────────────────────────────────────────────────────────────
    _node(ax, P["U"], "U", dashed=True)
    _node(ax, P["C"], "C")
    _node(ax, P["W"], "W")
    _node(ax, P["M"], "M")
    _node(ax, P["Y"], "Y")
    _node(ax, P["E"], "E", dashed=True, scale=0.8)

    # ── 범례 텍스트 ───────────────────────────────────────────────────────────
    ax.text(
        -0.06, -0.225,
        "[A] C→Y : asserted in text     "
        "[I] W→M→Y : analyst-identified     "
        "dashed = unobserved     ✕ = assumed-absent",
        fontsize=8,
    )
    ax.text(0.28, 0.485, "[I]", fontsize=9)
    ax.text(0.715, 0.485, "[I]", fontsize=9)
    ax.text(0.955, 0.60, "[A]", fontsize=9)
    ax.text(0.245, 0.665, "[U]", fontsize=9)
    ax.text(0.74, 0.665, "[U]", fontsize=9)

    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(-0.27, 1.06)
    ax.axis("off")
    fig.tight_layout()

    out_path = cfg.paths.f1_dag
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("F1 저장: %s", out_path)
    return out_path


# ── F2: 시간 전개 DAG ─────────────────────────────────────────────────────────
def render_f2(cfg) -> Path:
    """F2_unrolled.png 생성.

    W→E_t (τ 지연 및 지속 효과), E_t→E_{t+1}, E_t→S_t, C·S→Y.

    Args:
        cfg: Config 인스턴스.

    Returns:
        저장된 PNG 파일 경로.
    """
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    T = cfg.n_periods  # 6
    xs = [0.05 + i * 0.18 for i in range(T)]
    yW, yE, yS = 0.95, 0.55, 0.15
    sc = 0.62

    # W→E_0 (직접 τ=1), W→E_1..E_5 (지연 효과, 연한 회색)
    _arrow(ax, (xs[0], yW), (xs[1], yE), shrink=18, lw=1.1)
    for i in range(2, T):
        _arrow(ax, (xs[0], yW), (xs[i], yE), shrink=18, lw=0.8,
               color="dimgray", rad=-0.18)

    # E_t→E_{t+1} (엔트로피 누적)
    for i in range(T - 1):
        _arrow(ax, (xs[i], yE), (xs[i + 1], yE), shrink=34, lw=1.2)

    # E_t→S_t (취약성 전이)
    for x in xs:
        _arrow(ax, (x, yE), (x, yS), shrink=16, lw=1.0)

    # S_{t5}→Y, C→Y
    _arrow(ax, (xs[-1], yS), (xs[-1], -0.28), shrink=16, lw=1.2)
    _arrow(ax, (xs[-2], -0.28), (xs[-1], -0.28), shrink=34, lw=1.2)

    # 노드
    _node(ax, (xs[0], yW), "W", scale=sc)
    for i, x in enumerate(xs):
        _node(ax, (x, yE), "E", dashed=True, scale=sc)
        ax.text(x, yE + 0.135, f"t{i}", fontsize=8, ha="center")
        _node(ax, (x, yS), "S", dashed=True, scale=sc)
    _node(ax, (xs[-1], -0.28), "Y", scale=sc)
    _node(ax, (xs[-2], -0.28), "C", scale=sc)

    # 범례
    ax.text(
        -0.04, -0.52,
        "τ : lagged onset of W→E;  "
        "E accumulates via E\u209c\u208b\u2081→E\u209c;  "
        "Y = 1[S\u209c < S_crit] · 1[C = 1]",
        fontsize=8.5,
    )

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.58, 1.14)
    ax.axis("off")
    fig.tight_layout()

    out_path = cfg.paths.f2_unrolled
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("F2 저장: %s", out_path)
    return out_path


def main(cfg=None) -> None:
    """F1, F2 PNG 생성.

    Args:
        cfg: Config 인스턴스. None이면 CWD params.yaml 로드.
    """
    if cfg is None:
        from src.config import load_config
        cfg = load_config()
    render_f1(cfg)
    render_f2(cfg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(__file__).resolve().parent.parent.parent
    from src.config import load_config
    main(load_config(root / "params.yaml", root=root))
