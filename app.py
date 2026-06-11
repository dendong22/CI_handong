import streamlit as st
import pandas as pd
import json
from PIL import Image
from pathlib import Path

# 설정
st.set_page_config(page_title="풍요는 쇠퇴를 부른다 - 파이프라인 시각화", layout="wide")

# 폴더 경로
base_dir = Path(__file__).resolve().parent
outputs_dir = base_dir / "outputs"
tables_dir = outputs_dir / "tables"
figures_dir = outputs_dir / "figures"

# 제목
st.title("풍요는 쇠퇴를 부른다 — 인과추론 파이프라인 대시보드")
st.markdown("UC-SCM / Front-door 식별 파이프라인의 결과 산출물을 동적으로 확인하는 스트림릿 대시보드입니다.")

# 탭 구성
tab1, tab2, tab3 = st.tabs(["DAG 시각화 (인과 모형)", "프록시 지표 시계열", "인과 추정 결과"])

with tab1:
    st.header("DAG (Directed Acyclic Graph) 모델 구조")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("F1: 전단계(Front-door) 식별 DAG")
        try:
            img1 = Image.open(figures_dir / "F1_dag.png")
            st.image(img1, use_container_width=True)
        except Exception as e:
            st.error("F1 이미지를 불러오지 못했습니다.")
    
    with col2:
        st.subheader("F2: 시간 전개 DAG")
        try:
            img2 = Image.open(figures_dir / "F2_unrolled.png")
            st.image(img2, use_container_width=True)
        except Exception as e:
            st.error("F2 이미지를 불러오지 못했습니다.")
            
    st.markdown("""
    **범례**:
    - **W**: 풍요 유입 (진짜 처치 변수)
    - **Y**: 붕괴 (결과)
    - **M**: 매개 변수 (담론, 엔트로피 등)
    - **C**: 외부 충격 (트리거)
    - **U**: 관측 불가 교란 (거버넌스 품질 등)
    """)

with tab2:
    st.header("프록시 지표 시계열 분석")
    st.markdown("각 시점(period)에 따른 **PMI (낙관-위기 담론 공기 빈도)** 및 **코사인 거리**를 분석합니다.")
    
    try:
        proxy_df = pd.read_csv(tables_dir / "proxy_series.csv")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("지표 변화 트렌드 (동적 차트)")
            st.line_chart(proxy_df.set_index("period")[["pmi", "cos_dist"]])
            
        with col2:
            st.subheader("지표 데이터 테이블")
            st.dataframe(proxy_df, use_container_width=True)
            
        st.divider()
        st.markdown("#### Matplotlib 기반 정적 시각화 결과")
        col3, col4 = st.columns(2)
        with col3:
            try:
                img3 = Image.open(figures_dir / "F3_pmi.png")
                st.image(img3, use_container_width=True)
            except Exception as e:
                st.warning("F3 이미지가 없습니다.")
        with col4:
            try:
                img4 = Image.open(figures_dir / "F4_cosine.png")
                st.image(img4, use_container_width=True)
            except Exception as e:
                st.warning("F4 이미지가 없습니다.")

    except Exception as e:
        st.error("proxy_series.csv 파일을 찾을 수 없거나 불러오지 못했습니다.")

with tab3:
    st.header("인과 효과 추정 및 민감도 분석")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("T2 추정치 (T2 Estimates)")
        try:
            t2_df = pd.read_csv(tables_dir / "t2_estimates.csv")
            st.dataframe(t2_df, use_container_width=True)
            
            st.info("💡 **결과 해석**: Front-door ATE(전단계 효과)와 참값(Truth)이 오차 허용 범위 내에서 일치합니다. Naive ATE는 참값과 차이가 있습니다.")
        except Exception as e:
            st.error("t2_estimates.csv 파일을 불러오지 못했습니다.")
            
        st.subheader("개입(Intervention) 시뮬레이션: do(C=0)")
        try:
            do_df = pd.read_csv(tables_dir / "do_c0.csv")
            st.dataframe(do_df, use_container_width=True)
            st.line_chart(do_df.set_index("t"))
        except Exception as e:
            st.error("do_c0.csv 파일을 불러오지 못했습니다.")
            
    with col2:
        st.subheader("민감도 분석 (Sensitivity)")
        try:
            sens_df = pd.read_csv(tables_dir / "sensitivity.csv")
            st.dataframe(sens_df, use_container_width=True)
            st.info("💡 A1, A2 가정 위반 시 발생할 수 있는 편향(bias)을 분석합니다.")
        except Exception as e:
            st.error("sensitivity.csv 파일을 불러오지 못했습니다.")
            
        st.subheader("이식성 테스트 (Transportability)")
        try:
            with open(tables_dir / "transport_result.json", "r", encoding="utf-8") as f:
                trans_json = json.load(f)
            st.json(trans_json)
        except Exception as e:
            st.error("transport_result.json 파일을 불러오지 못했습니다.")
