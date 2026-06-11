"""extraction/extract_stub.py — L1 Data.

LLM 구조화 추출 명세. 프롬프트 전문 + 스키마 검증 로직 보유.
오프라인 재현 보장을 위해 실제 API 호출은 미실행(NotImplementedError).

실데이터 적용 시:
    1. ANTHROPIC_API_KEY 환경변수 설정
    2. _call_llm() 내부의 NotImplementedError를 제거하고 실제 호출 활성화
    3. 출력은 반드시 schema.validate_assertions()를 통과해야 한다
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.schema import ASSERTIONS_COLUMNS, validate_assertions

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "causal_assertion_extraction.md"


def build_prompt(article_text: str, time_layer: str, case_id: str = "unknown") -> str:
    """프롬프트 템플릿에 인자를 주입하여 완성된 프롬프트를 반환한다.

    Args:
        article_text: 추출 대상 원문 텍스트.
        time_layer: 문서의 시간 위치 (TIME_LAYERS 중 하나).
        case_id: 케이스 식별자.

    Returns:
        완성된 프롬프트 문자열.

    Raises:
        FileNotFoundError: 프롬프트 템플릿 파일이 없을 때.

    Invariant:
        반환 문자열에 8종 NODE_TYPES 레이블이 전부 포함된다.
    """
    if not _PROMPT_PATH.exists():
        raise FileNotFoundError(f"프롬프트 템플릿 없음: {_PROMPT_PATH}")

    template = _PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.replace("{article_text}", article_text)
    prompt = prompt.replace("{time_layer}", time_layer)
    prompt = prompt.replace("{case_id}", case_id)
    return prompt


def _call_llm(prompt: str) -> str:
    """Anthropic API 호출 래퍼 — 오프라인 재현 보장을 위해 미실행.

    실데이터 적용 시 이 함수를 활성화하라:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        msg = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text

    Args:
        prompt: build_prompt()로 생성된 완성 프롬프트.

    Returns:
        모델 응답 문자열 (JSON array).

    Raises:
        NotImplementedError: 항상. 오프라인 재현 보장용.
    """
    raise NotImplementedError(
        "오프라인 재현 보장을 위해 미실행. "
        "실데이터 적용 시 ANTHROPIC_API_KEY 환경변수 설정 후 "
        "_call_llm()을 활성화하라. "
        "출력은 schema.validate_assertions를 통과해야 한다."
    )


def extract(article_text: str, time_layer: str, case_id: str = "unknown") -> list[dict[str, Any]]:
    """텍스트에서 인과 주장 구조를 추출한다.

    Args:
        article_text: 추출 대상 원문.
        time_layer: 문서 시간 위치.
        case_id: 케이스 식별자.

    Returns:
        검증 통과한 assertion dict 리스트.

    Raises:
        NotImplementedError: 오프라인 환경에서 항상. (설계 의도)
    """
    prompt = build_prompt(article_text, time_layer, case_id)
    raw_response = _call_llm(prompt)  # ← NotImplementedError 발생

    # 이하 코드는 _call_llm 활성화 후 실행되는 후처리 로직
    try:
        records = json.loads(raw_response)
    except json.JSONDecodeError as e:
        logger.error("LLM 응답 JSON 파싱 실패: %s", e)
        raise

    df = pd.DataFrame(records, columns=ASSERTIONS_COLUMNS)
    validate_assertions(df)  # 스키마 위반 시 ValueError

    return records
