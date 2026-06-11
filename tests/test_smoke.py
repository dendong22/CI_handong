"""tests/test_smoke.py — 전역 수치 불변량 회귀 검증.

T2·T7·T8의 수치 불변량을 pytest로 재검.
클린 체크아웃 후 run_all.py로 산출물을 생성한 뒤 pytest를 실행한다.

실행:
    python run_all.py && pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def cfg():
    from src.config import load_config
    return load_config(ROOT / "params.yaml", root=ROOT)


@pytest.fixture(scope="session")
def truth(cfg):
    with cfg.paths.ground_truth_json.open() as f:
        return json.load(f)


@pytest.fixture(scope="session")
def t2_df(cfg):
    return pd.read_csv(cfg.paths.t2_estimates_csv)


@pytest.fixture(scope="session")
def do_c0_df(cfg):
    return pd.read_csv(cfg.paths.do_c0_csv)


@pytest.fixture(scope="session")
def sens_df(cfg):
    return pd.read_csv(cfg.paths.sensitivity_csv)


@pytest.fixture(scope="session")
def proxy_df(cfg):
    return pd.read_csv(cfg.paths.proxy_series_csv)


@pytest.fixture(scope="session")
def backtrace_result(cfg):
    with cfg.paths.backtrace_json.open() as f:
        return json.load(f)


# ── T2: DGP 불변량 ────────────────────────────────────────────────────────────
class TestDGP:
    def test_ate_true(self, truth):
        """ATE 참값 = +0.143 ± 0.001"""
        assert abs(truth["ate_true"] - 0.143) <= 0.001, \
            f"ate_true={truth['ate_true']:.4f}"

    def test_observed_csv_no_u_column(self, cfg):
        """observed.csv에 U 열 부재"""
        df = pd.read_csv(cfg.paths.observed_csv)
        assert "U" not in df.columns

    def test_observed_csv_columns(self, cfg):
        """observed.csv 컬럼 = [W, M, Y]"""
        df = pd.read_csv(cfg.paths.observed_csv)
        assert list(df.columns) == ["W", "M", "Y"]

    def test_deterministic(self, cfg):
        """동일 시드 2회 실행 결과 동일"""
        from src.datagen.dgp import generate
        obs1, truth1 = generate(cfg)
        obs2, truth2 = generate(cfg)
        pd.testing.assert_frame_equal(obs1, obs2)
        assert truth1["ate_true"] == truth2["ate_true"]


# ── T7: 추정량 불변량 ─────────────────────────────────────────────────────────
class TestEstimators:
    def _ate_row(self, t2_df):
        return t2_df[t2_df["w"] == "ATE"].iloc[0]

    def test_naive_ate(self, t2_df):
        """순진 ATE = +0.173 ± 0.002"""
        naive = float(self._ate_row(t2_df)["naive"])
        assert abs(naive - 0.173) <= 0.002, f"naive={naive:.4f}"

    def test_fd_ate(self, t2_df):
        """전단계 ATE = +0.144 ± 0.002"""
        fd = float(self._ate_row(t2_df)["front_door"])
        assert abs(fd - 0.144) <= 0.002, f"fd={fd:.4f}"

    def test_fd_truth_gap(self, t2_df, truth):
        """|FD − truth| ≤ 0.002"""
        fd = float(self._ate_row(t2_df)["front_door"])
        gap = abs(fd - truth["ate_true"])
        assert gap <= 0.002, f"|FD−truth|={gap:.4f}"


# ── T8: 개입·민감도 불변량 ────────────────────────────────────────────────────
class TestInterventions:
    def test_do_c0_t5(self, do_c0_df):
        """do(C=0) t=5 누적 붕괴 확률 = 0.761 ± 0.005"""
        val = float(do_c0_df[do_c0_df["t"] == 5]["cum_collapse_prob"].values[0])
        assert abs(val - 0.761) <= 0.005, f"t=5={val:.4f}"

    def test_sensitivity_upward_bias(self, sens_df):
        """양 위반(A1, A2) 모두 상방 편향"""
        for assumption in ("A1", "A2"):
            pos = sens_df[(sens_df["assumption"] == assumption) & (sens_df["delta"] > 0)]
            slopes = pos["bias"].values
            assert all(b >= 0 for b in slopes), \
                f"{assumption} 음의 편향 발견: {slopes}"

    def test_sensitivity_slope_a1(self, sens_df):
        """slope A1 ≈ 0.054/δ (±20% + 0.005)"""
        pos = sens_df[(sens_df["assumption"] == "A1") & (sens_df["delta"] > 0)]
        s = np.polyfit(pos["delta"].values, pos["bias"].values, 1)[0]
        assert abs(s - 0.054) <= 0.054 * 0.20 + 0.005, f"slope_A1={s:.4f}"

    def test_sensitivity_slope_a2(self, sens_df):
        """slope A2 ≈ 0.17/δ (±20% + 0.005)"""
        pos = sens_df[(sens_df["assumption"] == "A2") & (sens_df["delta"] > 0)]
        s = np.polyfit(pos["delta"].values, pos["bias"].values, 1)[0]
        assert abs(s - 0.17) <= 0.17 * 0.20 + 0.005, f"slope_A2={s:.4f}"

    def test_sensitivity_slope_ratio(self, sens_df):
        """기울기 비 A2:A1 ∈ [2, 4]"""
        def slope(assumption):
            pos = sens_df[(sens_df["assumption"] == assumption) & (sens_df["delta"] > 0)]
            return np.polyfit(pos["delta"].values, pos["bias"].values, 1)[0]
        s1, s2 = slope("A1"), slope("A2")
        ratio = s2 / (abs(s1) + 1e-12)
        assert 2 <= ratio <= 4, f"비율={ratio:.2f}"


# ── T6: 대리지표 단조성 ───────────────────────────────────────────────────────
class TestProxyMetrics:
    def test_pmi_monotone_decreasing(self, proxy_df):
        """PMI 단조 하락"""
        pmi = proxy_df["pmi"].values
        assert all(pmi[i] >= pmi[i + 1] - 1e-6 for i in range(len(pmi) - 1))

    def test_cosd_monotone_increasing(self, proxy_df):
        """cosd 단조 증가"""
        cosd = proxy_df["cos_dist"].values
        assert all(cosd[i] <= cosd[i + 1] + 1e-6 for i in range(len(cosd) - 1))

    def test_proxy_series_columns(self, proxy_df):
        """proxy_series.csv 5열 확인"""
        assert list(proxy_df.columns) == ["period", "pmi", "cos_dist", "E_norm", "M_discrete"]


# ── T5: 그래프 불변량 ─────────────────────────────────────────────────────────
class TestGraph:
    def test_backtrace_reaches_w_at_t0(self, backtrace_result):
        """백트레이스 W(t0) 도달 = True"""
        assert backtrace_result["reaches_w_at_t0"] is True
