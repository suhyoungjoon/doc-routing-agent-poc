"""승인 게이트: approval_thresholds.yaml 의 규칙을 평가해 사람 승인 필요 여부를 결정.

- 규칙은 전부 설정에서 온다. 이 모듈은 op 의미(lt/gt/in/...)와 필드 경로 해석만 안다.
- 규칙 중 하나라도 걸리면 승인 필요 (any). 걸린 규칙은 설정 순서대로 모두 보고한다.
- 필드 경로는 점 표기(extracted.total_amount), 'a|b' 는 값이 있는 첫 경로를 사용.
- 숫자 비교에서 값이 비어 있으면 해당 규칙은 건너뛰고(누락은 missing 규칙이 담당),
  값은 있는데 숫자로 해석할 수 없으면(예: '1.2억', 'TBD') 안전하게 '걸림'으로 처리한다.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from config_loader import AppConfig, ApprovalRule, get_config

_NUMERIC_OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
                "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b}
# 허용: 선택적 통화기호/부호, 천단위 쉼표, 소수점, 선택적 통화 단위. 그 외(억/만/USD/?)는 해석 불가.
_NUMBER_RE = re.compile(r"^\s*[₩$]?\s*(-?\d{1,3}(?:,\d{3})+|-?\d+)(\.\d+)?\s*(원|KRW|₩)?\s*$", re.IGNORECASE)
_MISSING = object()


@dataclass
class GateDecision:
    requires_approval: bool
    triggered: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def context_from(result: Any) -> dict[str, Any]:
    """ClassificationResult → gate 평가 컨텍스트."""
    return {
        "category": result.category,
        "confidence": result.confidence,
        "extracted": dict(result.extracted),
        "missing_fields": list(result.missing_fields),
        "defense": dict(result.defense) if getattr(result, "defense", None) else None,
        "error": result.error,
    }


def parse_number(v: Any) -> float | int | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return v
    m = _NUMBER_RE.match(str(v))
    if not m:
        return None
    n = float(m.group(1).replace(",", "") + (m.group(2) or ""))
    return int(n) if n.is_integer() else n


def _lookup(ctx: dict[str, Any], path: str) -> Any:
    cur: Any = ctx
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return _MISSING
        cur = cur[part]
    return cur


def _is_empty(v: Any) -> bool:
    return v is _MISSING or v is None or (isinstance(v, (str, list, dict, tuple)) and len(v) == 0) \
        or (isinstance(v, str) and not v.strip())


def _resolve(ctx: dict[str, Any], rule: ApprovalRule) -> tuple[str, Any]:
    for path in rule.fields:
        v = _lookup(ctx, path)
        if not _is_empty(v):
            return path, v
    return rule.fields[0], _lookup(ctx, rule.fields[0])


class ApprovalGate:
    def __init__(self, config: AppConfig | None = None):
        self.cfg = config or get_config()

    def _check(self, rule: ApprovalRule, ctx: dict[str, Any]) -> dict[str, Any] | None:
        path, actual = _resolve(ctx, rule)
        note = None
        op = rule.op
        if op in _NUMERIC_OPS:
            if _is_empty(actual):
                return None
            num = parse_number(actual)
            if num is None:
                hit, note = True, "unparseable_number"
            else:
                actual = num
                hit = _NUMERIC_OPS[op](num, rule.value)
        elif op == "empty":
            hit = _is_empty(actual)
        elif op == "not_empty":
            hit = not _is_empty(actual)
        else:
            a = None if actual is _MISSING else actual
            if op == "eq":
                hit = a == rule.value
            elif op == "ne":
                hit = a != rule.value
            elif op == "in":
                hit = a in rule.value
            else:  # not_in
                hit = a not in rule.value
        if not hit:
            return None
        return {
            "rule_id": rule.id,
            "description": rule.description,
            "field": path,
            "actual": None if actual is _MISSING else actual,
            "op": op,
            "value": rule.value,
            "note": note,
        }

    def evaluate(self, ctx: dict[str, Any]) -> GateDecision:
        triggered = [t for r in self.cfg.approval.rules if (t := self._check(r, ctx))]
        if not self.cfg.approval.auto_approve and not triggered:
            triggered = [{"rule_id": "auto_approve_disabled", "description": "auto_approve=false",
                          "field": None, "actual": None, "op": None, "value": None, "note": None}]
        return GateDecision(requires_approval=bool(triggered), triggered=triggered)
