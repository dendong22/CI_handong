"""run_all.py — L3 Presentation.

전 파이프라인 오케스트레이션 단일 진입점.
사용법: python run_all.py [--params params.yaml]

파이프라인 순서:
    1. dgp.main            — observed.csv, ground_truth.json
    2. corpus_dummy.main   — data/raw/case_*/
    3. graph pipeline      — build → export → queries
    4. proxy_metrics.main  — proxy_series.csv
    5. estimators.main     — t2_estimates.csv
    6. interventions.main  — do_c0.csv
    7. sensitivity.main    — sensitivity.csv
    8. case_c.main         — transport_result.json
    9. make_dags.main      — F1, F2
   10. make_figs.main      — F3, F4

종료 시 verify() 전수 검사; 위반 시 비0 종료코드.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 헤드리스 렌더링

import numpy as np
import pandas as pd

# ── 프로젝트 루트를 sys.path에 등록 ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_all")


def _step(name: str, fn, *args, **kwargs):
    """단계 실행 래퍼: 경과 시간 로깅 + 예외 포획."""
    logger.info("▶ [%s] 시작", name)
    t0 = time.time()
    try:
        fn(*args, **kwargs)
    except Exception as exc:
        logger.error("✗ [%s] 실패: %s", name, exc)
        raise
    logger.info("✓ [%s] 완료 (%.1fs)", name, time.time() - t0)


def verify(cfg) -> None:
    """§4.3 전역 수치 불변량 전수 검사.

    위반 항목이 있으면 AssertionError를 누적하여 마지막에 일괄 보고.
    모든 위반이 0이어야 sys.exit(0), 아니면 sys.exit(1).
    """
    failures: list[str] = []

    def check(label: str, value: float, target: float, tol: float) -> None:
        if abs(value - target) > tol:
            failures.append(f"  {label}: {value:.4f}  (target={target} ±{tol})")

    # ── T2: DGP 불변량 ────────────────────────────────────────────────────────
    with cfg.paths.ground_truth_json.open() as f:
        truth = json.load(f)
    check("ATE 참값",   truth["ate_true"],          0.143, 0.001)

    # ── T7: 추정량 불변량 ─────────────────────────────────────────────────────
    t2_df = pd.read_csv(cfg.paths.t2_estimates_csv)
    ate_row = t2_df[t2_df["w"] == "ATE"].iloc[0]
    check("ATE 순진",       float(ate_row["naive"]),        0.173, 0.002)
    check("ATE 전단계",     float(ate_row["front_door"]),   0.144, 0.002)
    check("|FD−truth|",     abs(float(ate_row["front_door"]) - truth["ate_true"]), 0.0, 0.002)

    # ── T8: 개입 불변량 ───────────────────────────────────────────────────────
    do_df = pd.read_csv(cfg.paths.do_c0_csv)
    t5_val = float(do_df[do_df["t"] == 5]["cum_collapse_prob"].values[0])
    check("do(C=0) t=5",    t5_val,                         0.761, 0.005)

    # ── T8: 민감도 기울기 불변량 ──────────────────────────────────────────────
    sens_df = pd.read_csv(cfg.paths.sensitivity_csv)
    d_pos = sens_df[sens_df["delta"] > 0]
    a1_vals = d_pos[d_pos["assumption"] == "A1"]["bias"].values
    a2_vals = d_pos[d_pos["assumption"] == "A2"]["bias"].values
    d_grid = d_pos[d_pos["assumption"] == "A1"]["delta"].values
    if len(d_grid) > 1:
        s1 = np.polyfit(d_grid, a1_vals, 1)[0]
        s2 = np.polyfit(d_grid, a2_vals, 1)[0]
        check("slope A1",   s1,  0.054, 0.054 * 0.20 + 0.005)
        check("slope A2",   s2,  0.170, 0.170 * 0.20 + 0.005)
        ratio = s2 / (abs(s1) + 1e-12)
        if not (2 <= ratio <= 4):
            failures.append(f"  기울기 비 A2/A1: {ratio:.2f}  (허용 [2,4])")

    # ── T6: 대리지표 단조성 ───────────────────────────────────────────────────
    ps_df = pd.read_csv(cfg.paths.proxy_series_csv)
    pmi = ps_df["pmi"].values
    cosd = ps_df["cos_dist"].values
    if not all(pmi[i] >= pmi[i + 1] - 1e-6 for i in range(len(pmi) - 1)):
        failures.append("  PMI 단조 하락 위반")
    if not all(cosd[i] <= cosd[i + 1] + 1e-6 for i in range(len(cosd) - 1)):
        failures.append("  cosd 단조 증가 위반")

    # ── T5: 백트레이스 ────────────────────────────────────────────────────────
    with cfg.paths.backtrace_json.open() as f:
        bt = json.load(f)
    if not bt.get("reaches_w_at_t0", False):
        failures.append("  backtrace reaches_w_at_t0 = False")

    # ── 결과 보고 ─────────────────────────────────────────────────────────────
    if failures:
        logger.error("불변량 위반 %d건:\n%s", len(failures), "\n".join(failures))
        sys.exit(1)
    else:
        logger.info("✓ 전역 불변량 전수 검사 통과 (%d개 항목)", 8)


def main() -> None:
    parser = argparse.ArgumentParser(description="prosperity-decline 전체 파이프라인")
    parser.add_argument("--params", default="params.yaml", help="params.yaml 경로")
    args = parser.parse_args()

    # Config 로드
    from src.config import load_config
    cfg = load_config(args.params, root=ROOT)
    cfg.paths.ensure_all()
    logger.info("Config 로드: seed_dgp=%d, N=%d", cfg.seed_dgp, cfg.n_samples)

    # ── 파이프라인 실행 ────────────────────────────────────────────────────────
    from src.datagen import dgp, corpus_dummy
    from src.graph import build_graph, export as graph_export, queries as graph_queries
    from src.metrics import proxy_metrics
    from src.inference import estimators, interventions, sensitivity
    from src.transport import case_c
    from src.viz import make_dags, make_figs
    import json as _json

    _step("1. DGP",           dgp.main, cfg)
    _step("2. 더미 코퍼스",   corpus_dummy.main, cfg)

    # 3. 그래프 빌드·export·쿼리 (단일 모듈이 아닌 함수 조합)
    def graph_pipeline(cfg):
        g = build_graph.load_all(cfg)
        graph_export.to_csv(g, cfg.paths.out_graph)
        graph_export.to_graphml(g, cfg.paths.graphml)
        motifs = graph_queries.causal_motifs(g, top_k=5, data_raw_dir=cfg.paths.data_raw)
        motifs.to_csv(cfg.paths.motifs_csv, index=False)
        bt = graph_queries.backtrace(g)
        with cfg.paths.backtrace_json.open("w") as f:
            _json.dump(bt, f, indent=2, ensure_ascii=False)

    _step("3. 그래프",        graph_pipeline, cfg)
    _step("4. 대리지표",      proxy_metrics.main, cfg)
    _step("5. 3중 추정",      estimators.main, cfg)
    _step("6. 개입 분석",     interventions.main, cfg)
    _step("7. 민감도",        sensitivity.main, cfg)
    _step("8. 사례 C 이식",   case_c.main, cfg)
    _step("9. DAG 시각화",    make_dags.main, cfg)
    _step("10. 지표 시각화",  make_figs.main, cfg)

    logger.info("=" * 55)
    logger.info("파이프라인 완료 — 불변량 검증 시작")
    logger.info("=" * 55)
    verify(cfg)
    logger.info("모든 산출물 생성 및 검증 완료.")


if __name__ == "__main__":
    main()
