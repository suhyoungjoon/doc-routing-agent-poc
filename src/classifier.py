"""Claude API 기반 문서 분류기 (3단계: 방어 없음, 의도적으로 취약).

- 카테고리/설명/필수필드는 config_loader.get_config() 에서만 가져온다.
- 응답은 structured outputs(JSON schema)로 받는다. 카테고리 enum 도 설정에서 생성.
- ⚠️ 이 단계의 프롬프트는 문서 본문을 지시문과 구분 없이 그대로 이어붙인다.
  (숨은 지시에 취약한 상태를 5단계에서 확인하고, 6단계 defense.py 에서 막는다.)

실행: python src/classifier.py <sample_docs 기준 상대경로> [...]
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any

from config_loader import AppConfig, get_config

DEFAULT_MODEL = "claude-opus-5"
MODEL_ENV = "DOC_AGENT_MODEL"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class ClassificationResult:
    doc_id: str
    category: str
    confidence: float
    reasoning: str
    extracted: dict[str, str]  # 선택된 카테고리의 required_fields 만
    missing_fields: list[str]
    model: str | None = None
    stop_reason: str | None = None
    error: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _all_fields(cfg: AppConfig) -> list[str]:
    seen: dict[str, None] = {}
    for c in cfg.categories.categories:
        for f in c.required_fields:
            seen.setdefault(f, None)
    return list(seen)


def build_output_schema(cfg: AppConfig) -> dict[str, Any]:
    fields = _all_fields(cfg)
    names = [*cfg.categories.names, cfg.categories.fallback.name]
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": names},
            "confidence": {"type": "number"},
            "reasoning": {"type": "string"},
            "extracted": {
                "type": "object",
                "properties": {f: {"type": "string"} for f in fields},
                "required": fields,
                "additionalProperties": False,
            },
        },
        "required": ["category", "confidence", "reasoning", "extracted"],
        "additionalProperties": False,
    }


def build_system_prompt(cfg: AppConfig) -> str:
    lines = [
        "당신은 사내 수신 문서를 분류하는 담당자입니다.",
        "문서를 읽고 아래 카테고리 중 하나로 분류하고, 해당 카테고리의 필수 필드 값을 문서에서 추출하세요.",
        "",
        "## 카테고리",
    ]
    for c in cfg.categories.categories:
        lines.append(f"- {c.name} ({c.label}): {c.description}")
        lines.append(f"  필수 필드: {', '.join(c.required_fields)}")
    fb = cfg.categories.fallback
    lines += [
        f"- {fb.name} ({fb.label}): 위 어느 카테고리에도 해당하지 않거나 판단이 불확실한 경우",
        "",
        "## 출력 규칙",
        "- confidence 는 0~1 사이 분류 확신도.",
        "- extracted 에는 모든 필드 키를 채우되, 선택한 카테고리와 무관하거나 문서에 없는 값은 빈 문자열로 둔다.",
        "- 금액은 숫자만(쉼표·통화 기호 제외), 날짜는 YYYY-MM-DD 형식.",
    ]
    return "\n".join(lines)


def build_user_message(text: str) -> str:
    # ⚠️ 의도적 취약점: 문서 본문을 데이터 경계 표시 없이 지시문 뒤에 그대로 붙인다.
    return "다음 문서를 분류하고 필드를 추출하세요.\n\n" + text


class Classifier:
    def __init__(self, client: Any = None, *, model: str | None = None, config: AppConfig | None = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self._client = client
        self.model = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
        self.cfg = config or get_config()
        self._schema = build_output_schema(self.cfg)
        self._system = build_system_prompt(self.cfg)

    def build_request(self, text: str) -> dict[str, Any]:
        return dict(
            model=self.model,
            max_tokens=MAX_TOKENS,
            betas=[FALLBACK_BETA],
            fallbacks="default",
            system=self._system,
            messages=[{"role": "user", "content": build_user_message(text)}],
            output_config={"format": {"type": "json_schema", "schema": self._schema}},
        )

    def _fallback(self, doc_id: str, error: str, **kw: Any) -> ClassificationResult:
        return ClassificationResult(
            doc_id=doc_id,
            category=self.cfg.categories.fallback.name,
            confidence=0.0,
            reasoning="",
            extracted={},
            missing_fields=[],
            error=error,
            **kw,
        )

    def classify(self, text: str, doc_id: str = "<inline>") -> ClassificationResult:
        response = self._client.beta.messages.create(**self.build_request(text))
        meta = dict(
            model=getattr(response, "model", None),
            stop_reason=getattr(response, "stop_reason", None),
            usage=_usage_dict(getattr(response, "usage", None)),
        )

        if meta["stop_reason"] == "refusal":
            return self._fallback(doc_id, "refusal", **meta)
        if meta["stop_reason"] == "max_tokens":
            return self._fallback(doc_id, "max_tokens", **meta)

        text_out = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text_out is None:
            return self._fallback(doc_id, "no_text_block", **meta)
        try:
            data = json.loads(text_out)
        except json.JSONDecodeError as e:
            return self._fallback(doc_id, f"invalid_json: {e}", **meta)

        return self._to_result(doc_id, data, meta)

    def _to_result(self, doc_id: str, data: dict[str, Any], meta: dict[str, Any]) -> ClassificationResult:
        cats = self.cfg.categories
        category = data.get("category")
        error = None
        if category not in (*cats.names, cats.fallback.name):
            error = f"unknown_category: {category!r}"
            category = cats.fallback.name

        try:
            confidence = min(1.0, max(0.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0

        raw_fields = data.get("extracted") or {}
        required = cats.get(category).required_fields
        extracted = {f: str(raw_fields.get(f, "") or "").strip() for f in required}
        missing = [f for f, v in extracted.items() if not v]

        return ClassificationResult(
            doc_id=doc_id,
            category=category,
            confidence=confidence,
            reasoning=str(data.get("reasoning", "")),
            extracted=extracted,
            missing_fields=missing,
            error=error,
            **meta,
        )


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    if isinstance(usage, dict):
        return usage
    return {k: v for k, v in vars(usage).items() if not k.startswith("_")}


if __name__ == "__main__":
    from connector import DocumentConnector
    from tracing import Tracer, traced_classifier, traced_connector

    if len(sys.argv) < 2:
        print("usage: python src/classifier.py <doc> [<doc> ...]", file=sys.stderr)
        sys.exit(2)
    tracer = Tracer()
    conn = traced_connector(DocumentConnector(), tracer)
    clf = traced_classifier(Classifier(), tracer)
    for rel in sys.argv[1:]:
        doc = conn.read(rel)
        result = clf.classify(doc.text, doc_id=doc.doc_id)
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    print(f"trace: {tracer.path}", file=sys.stderr)
