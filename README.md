# 풍요는 쇠퇴를 부른다

**저자:** 최한동 | **과목:** 인과추론과 AI (Track B)

---

## 빠른 재현 (3줄)

```bash
git clone <repo_url> prosperity-decline && cd prosperity-decline
pip install -r requirements.txt
python run_all.py
```

`outputs/` 하위 전체 산출물이 생성되고 수치 불변량 검증이 자동 실행된다.

---

## 프로젝트 개요

외부 충격(C)은 붕괴(Y)의 트리거이지 근원이 아니다. 진짜 처치 변수는 풍요 유입(W)이며,
관측 불가 교란(U: 거버넌스 품질) 존재 하에서 Pearl(1995) **전단계 기준(front-door criterion)**으로
인과 효과를 식별한다.

```
W → M → Y      (전단계 식별 경로, [I])
U ⤳ W, U ⤳ Y  (잠재 교란, 점선)
C → Y           (트리거, [A])
A1: U -/→ M    (전단계 식별 조건 1)
A2: W -/→ Y    (전단계 식별 조건 2)
```

---

## 모듈 지도 (4계층)

```
L0 Foundation  : src/config.py, src/schema.py
L1 Data        : src/datagen/dgp.py, corpus_dummy.py, extraction/extract_stub.py
L2 Analysis    : src/graph/, src/metrics/, src/inference/, src/transport/
L3 Presentation: src/viz/, run_all.py, publish.sh
```

| 모듈 | 핵심 산출 |
|---|---|
| `datagen/dgp.py` | `data/dgp/observed.csv`, `ground_truth.json` |
| `datagen/corpus_dummy.py` | `data/raw/case_*/` (nodes, assertions, edges CSV) |
| `graph/build_graph.py` | in-memory `nx.MultiDiGraph` |
| `graph/export.py` | `outputs/graph/nodes.csv`, `edges.csv`, `ekg.graphml` |
| `graph/queries.py` | `outputs/tables/motifs.csv`, `backtrace.json` |
| `metrics/proxy_metrics.py` | `outputs/tables/proxy_series.csv` |
| `inference/estimators.py` | `outputs/tables/t2_estimates.csv` |
| `inference/interventions.py` | `outputs/tables/do_c0.csv` |
| `inference/sensitivity.py` | `outputs/tables/sensitivity.csv` |
| `transport/case_c.py` | `outputs/tables/transport_result.json` |
| `viz/make_dags.py` | `outputs/figures/F1_dag.png`, `F2_unrolled.png` |
| `viz/make_figs.py` | `outputs/figures/F3_pmi.png`, `F4_cosine.png` |

---

## 전역 수치 불변량

| 항목 | 기준값 | 허용 오차 |
|---|---|---|
| ATE 참값 | +0.143 | ±0.001 |
| ATE 순진 | +0.173 | ±0.002 |
| ATE 전단계 | +0.144 | ±0.002 |
| \|FD − truth\| | ≤ 0.002 | — |
| do(C=0) 누적, t=5 | 0.761 | ±0.005 |
| 민감도 기울기 비 (A2:A1) | ≈ 3 | ±1 |
| PMI 단조 하락 | True | — |
| 백트레이스 W(t0) 도달 | True | — |

```bash
pytest tests/test_smoke.py -v   # 불변량 회귀 검증
```

---

## Neo4j 이식

`outputs/graph/ekg.graphml` 또는 `nodes.csv`/`edges.csv`를 Neo4j에 직접 임포트할 수 있다.

```cypher
LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n:Node {node_id: row.node_id})
SET n += {node_type: row.node_type, label: row.label,
          time_layer: row.time_layer, case_id: row.case_id};

LOAD CSV WITH HEADERS FROM 'file:///edges.csv' AS row
MATCH (a:Node {node_id: row.src}), (b:Node {node_id: row.dst})
MERGE (a)-[r:EDGE {edge_type: row.edge_type, case_id: row.case_id}]->(b);
```

---

## 실데이터 대체 경로

현재 파이프라인은 더미 데이터(`corpus_dummy.py`)를 사용한다.
실제 텍스트 코퍼스로 교체하려면:

| 사례 | 데이터 소스 | 기간 |
|---|---|---|
| A (한국 외환위기) | [빅카인즈](https://www.bigkinds.or.kr) — 언론사 기사 | 1994–1998 |
| B (닷컴 버블) | [NYT Article Search API](https://developer.nytimes.com) | 1998–2002 |
| C (스파르타) | [Perseus Digital Library](http://www.perseus.tufts.edu) — Plutarch, Thucydides | BC 431–371 |

교체 절차:
1. 각 사례의 기사/문서를 `data/raw/case_*/corpus_meta.csv`에 경로 등록
2. `ANTHROPIC_API_KEY` 환경변수 설정
3. `src/extraction/extract_stub.py`의 `_call_llm()` 내 `NotImplementedError` 제거 후 활성화
4. `python run_all.py` 재실행 — 스키마 검증 및 불변량 게이트 자동 통과 확인

> **더미 데이터 고지:** 현재 커밋된 `data/raw/`는 구조 시연용 합성 데이터이며,
> 실증적 인과 주장의 근거로 사용할 수 없다.

---
