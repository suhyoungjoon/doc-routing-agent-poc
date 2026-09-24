"""config/*.yaml 로더.

카테고리명·임계값·방어 패턴·라우팅 경로 같은 비즈니스 값은 전부 YAML에만 있고,
다른 모듈은 반드시 이 모듈의 get_config()를 통해서만 참조한다.
이 파일에 있는 상수는 '스키마'(허용되는 op/action/severity 종류)뿐이다.

실행: python src/config_loader.py [config_dir]
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR_ENV = "DOC_AGENT_CONFIG_DIR"

CONFIG_FILES = {
    "categories": "categories.yaml",
    "defense": "injection_defense.yaml",
    "approval": "approval_thresholds.yaml",
    "eval": "eval_criteria.yaml",
}

# --- 스키마 (허용 값 종류) ---
SEVERITIES = ("low", "medium", "high")
PATTERN_TYPES = ("regex", "substring")
DEFENSE_ACTIONS = ("block", "sanitize", "flag")
GATE_OPS = ("lt", "le", "gt", "ge", "eq", "ne", "in", "not_in", "empty", "not_empty")


class ConfigError(ValueError):
    """설정 파일이 없거나 스키마에 맞지 않을 때."""


# ---------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class Category:
    name: str
    label: str
    description: str
    required_fields: tuple[str, ...]
    route_to: str


@dataclass(frozen=True)
class CategoriesConfig:
    categories: tuple[Category, ...]
    fallback: Category

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.categories)

    def get(self, name: str) -> Category:
        """카테고리 조회. fallback 이름도 허용. 없으면 KeyError."""
        for c in (*self.categories, self.fallback):
            if c.name == name:
                return c
        raise KeyError(name)


@dataclass(frozen=True)
class DefensePattern:
    id: str
    type: str
    pattern: str
    severity: str
    compiled: re.Pattern[str] = field(compare=False, repr=False)


@dataclass(frozen=True)
class DefenseConfig:
    normalize: dict[str, bool]
    patterns: tuple[DefensePattern, ...]
    action_by_severity: dict[str, str]
    replacement: str
    wrap_document_as_data: bool


@dataclass(frozen=True)
class ApprovalRule:
    id: str
    description: str
    fields: tuple[str, ...]  # "a|b" 표기는 여러 필드 중 존재하는 것 사용
    op: str
    value: Any


@dataclass(frozen=True)
class ApprovalConfig:
    rules: tuple[ApprovalRule, ...]
    auto_approve: bool


@dataclass(frozen=True)
class EvalCheck:
    id: str
    description: str
    weight: float


@dataclass(frozen=True)
class EvalConfig:
    checks: tuple[EvalCheck, ...]
    case_pass_score: float
    attack_hard_fail: bool
    min_pass_rate: float
    min_attack_block_rate: float
    field_normalize: dict[str, Any]
    failure_types: dict[str, str]  # id -> description


@dataclass(frozen=True)
class AppConfig:
    config_dir: Path
    categories: CategoriesConfig
    defense: DefenseConfig
    approval: ApprovalConfig
    eval: EvalConfig


# ---------------------------------------------------------------- helpers


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"설정 파일 없음: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML 파싱 실패: {path}: {e}") from e
    if not isinstance(data, dict):
        raise ConfigError(f"최상위가 매핑이 아님: {path}")
    return data


def _req(d: dict[str, Any], key: str, where: str, typ: type | tuple[type, ...] | None = None) -> Any:
    if key not in d or d[key] is None:
        raise ConfigError(f"{where}: 필수 키 '{key}' 누락")
    v = d[key]
    if typ is not None and not isinstance(v, typ):
        raise ConfigError(f"{where}.{key}: 타입 오류 (기대 {typ}, 실제 {type(v).__name__})")
    return v


def _num(d: dict[str, Any], key: str, where: str) -> float:
    v = _req(d, key, where)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{where}.{key}: 숫자가 아님 ({v!r})")
    return float(v)


def _unique(ids: list[str], where: str) -> None:
    dup = {x for x in ids if ids.count(x) > 1}
    if dup:
        raise ConfigError(f"{where}: 중복 id/name {sorted(dup)}")


# ---------------------------------------------------------------- parsers


def _parse_category(d: Any, where: str) -> Category:
    if not isinstance(d, dict):
        raise ConfigError(f"{where}: 매핑이 아님")
    rf = d.get("required_fields", [])
    if not isinstance(rf, list) or not all(isinstance(x, str) for x in rf):
        raise ConfigError(f"{where}.required_fields: 문자열 리스트여야 함")
    return Category(
        name=_req(d, "name", where, str),
        label=d.get("label", d["name"]),
        description=d.get("description", ""),
        required_fields=tuple(rf),
        route_to=_req(d, "route_to", where, str),
    )


def _parse_categories(data: dict[str, Any]) -> CategoriesConfig:
    raw = _req(data, "categories", "categories", list)
    if not raw:
        raise ConfigError("categories: 카테고리가 하나 이상 있어야 함")
    cats = tuple(_parse_category(c, f"categories[{i}]") for i, c in enumerate(raw))
    fallback = _parse_category(_req(data, "fallback", "categories", dict), "categories.fallback")
    _unique([c.name for c in cats] + [fallback.name], "categories")
    for c in (*cats, fallback):
        if Path(c.route_to).is_absolute() or ".." in Path(c.route_to).parts:
            raise ConfigError(f"categories.{c.name}.route_to: 상대경로(.. 없이)여야 함: {c.route_to}")
    return CategoriesConfig(categories=cats, fallback=fallback)


def _parse_defense(data: dict[str, Any]) -> DefenseConfig:
    normalize = data.get("normalize", {}) or {}
    if not isinstance(normalize, dict) or not all(isinstance(v, bool) for v in normalize.values()):
        raise ConfigError("defense.normalize: bool 값 매핑이어야 함")

    pats = []
    for i, p in enumerate(_req(data, "patterns", "defense", list)):
        where = f"defense.patterns[{i}]"
        pid = _req(p, "id", where, str)
        ptype = _req(p, "type", where, str)
        pattern = _req(p, "pattern", where, str)
        sev = _req(p, "severity", where, str)
        if ptype not in PATTERN_TYPES:
            raise ConfigError(f"{where}({pid}).type: {PATTERN_TYPES} 중 하나여야 함")
        if sev not in SEVERITIES:
            raise ConfigError(f"{where}({pid}).severity: {SEVERITIES} 중 하나여야 함")
        flags = re.IGNORECASE if normalize.get("lowercase") else 0
        src = pattern if ptype == "regex" else re.escape(pattern)
        try:
            compiled = re.compile(src, flags)
        except re.error as e:
            raise ConfigError(f"{where}({pid}): 정규식 컴파일 실패: {e}") from e
        pats.append(DefensePattern(pid, ptype, pattern, sev, compiled))
    _unique([p.id for p in pats], "defense.patterns")

    od = _req(data, "on_detect", "defense", dict)
    by_sev = _req(od, "by_severity", "defense.on_detect", dict)
    missing = set(SEVERITIES) - set(by_sev)
    if missing:
        raise ConfigError(f"defense.on_detect.by_severity: severity {sorted(missing)} 정책 누락")
    for sev, action in by_sev.items():
        if sev not in SEVERITIES or action not in DEFENSE_ACTIONS:
            raise ConfigError(f"defense.on_detect.by_severity: 잘못된 항목 {sev}: {action}")

    return DefenseConfig(
        normalize=dict(normalize),
        patterns=tuple(pats),
        action_by_severity=dict(by_sev),
        replacement=_req(od, "replacement", "defense.on_detect", str),
        wrap_document_as_data=bool(od.get("wrap_document_as_data", True)),
    )


def _resolve_ref(ref: str, raw: dict[str, dict[str, Any]], where: str) -> Any:
    """'categories.fallback.name' 같은 다른 설정 파일 참조를 값으로 해석."""
    parts = ref.split(".")
    cur: Any = raw
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise ConfigError(f"{where}.value_from: 참조 해석 실패 '{ref}'")
        cur = cur[p]
    return cur


def _parse_approval(data: dict[str, Any], raw: dict[str, dict[str, Any]]) -> ApprovalConfig:
    rules = []
    for i, r in enumerate(_req(data, "rules", "approval", list)):
        where = f"approval.rules[{i}]"
        rid = _req(r, "id", where, str)
        op = _req(r, "op", where, str)
        if op not in GATE_OPS:
            raise ConfigError(f"{where}({rid}).op: {GATE_OPS} 중 하나여야 함")
        if "value" in r and "value_from" in r:
            raise ConfigError(f"{where}({rid}): value와 value_from 동시 지정 불가")
        if "value_from" in r:
            value = _resolve_ref(r["value_from"], raw, where)
        elif op in ("empty", "not_empty"):
            value = None
        else:
            value = _req(r, "value", where)
        if op in ("lt", "le", "gt", "ge") and (isinstance(value, bool) or not isinstance(value, (int, float))):
            raise ConfigError(f"{where}({rid}): op '{op}'는 숫자 value 필요")
        if op in ("in", "not_in") and not isinstance(value, list):
            raise ConfigError(f"{where}({rid}): op '{op}'는 리스트 value 필요")
        fields = tuple(f.strip() for f in _req(r, "field", where, str).split("|"))
        rules.append(ApprovalRule(rid, r.get("description", ""), fields, op, value))
    _unique([r.id for r in rules], "approval.rules")
    return ApprovalConfig(rules=tuple(rules), auto_approve=bool(data.get("auto_approve", True)))


def _parse_eval(data: dict[str, Any]) -> EvalConfig:
    checks = []
    for i, c in enumerate(_req(data, "checks", "eval", list)):
        where = f"eval.checks[{i}]"
        checks.append(EvalCheck(_req(c, "id", where, str), c.get("description", ""), _num(c, "weight", where)))
    _unique([c.id for c in checks], "eval.checks")
    total = sum(c.weight for c in checks)
    if abs(total - 1.0) > 1e-6:
        raise ConfigError(f"eval.checks: weight 합이 1이 아님 ({total})")

    suite = _req(data, "suite", "eval", dict)
    ft = _req(data, "failure_types", "eval", list)
    failure_types = {_req(f, "id", f"eval.failure_types[{i}]", str): f.get("description", "") for i, f in enumerate(ft)}
    _unique([f["id"] for f in ft], "eval.failure_types")

    cfg = EvalConfig(
        checks=tuple(checks),
        case_pass_score=_num(data, "case_pass_score", "eval"),
        attack_hard_fail=bool(data.get("attack_hard_fail", True)),
        min_pass_rate=_num(suite, "min_pass_rate", "eval.suite"),
        min_attack_block_rate=_num(suite, "min_attack_block_rate", "eval.suite"),
        field_normalize=dict(data.get("field_normalize", {}) or {}),
        failure_types=failure_types,
    )
    for name in ("case_pass_score", "min_pass_rate", "min_attack_block_rate"):
        v = getattr(cfg, name)
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"eval.{name}: 0~1 범위여야 함 ({v})")
    return cfg


# ---------------------------------------------------------------- public API


def load_config(config_dir: str | Path | None = None) -> AppConfig:
    """config_dir의 4개 YAML을 읽어 검증된 AppConfig를 반환. 문제 시 ConfigError."""
    cdir = Path(config_dir or os.environ.get(CONFIG_DIR_ENV) or DEFAULT_CONFIG_DIR).resolve()
    raw = {key: _read_yaml(cdir / fname) for key, fname in CONFIG_FILES.items()}
    return AppConfig(
        config_dir=cdir,
        categories=_parse_categories(raw["categories"]),
        defense=_parse_defense(raw["defense"]),
        approval=_parse_approval(raw["approval"], raw),
        eval=_parse_eval(raw["eval"]),
    )


@lru_cache(maxsize=None)
def _cached(config_dir: str | None) -> AppConfig:
    return load_config(config_dir)


def get_config(config_dir: str | Path | None = None) -> AppConfig:
    """프로세스 내 캐시된 설정. 다른 모듈은 이 함수로만 설정을 참조한다."""
    return _cached(str(config_dir) if config_dir else None)


def reload_config() -> None:
    _cached.cache_clear()


def _summary(cfg: AppConfig) -> str:
    lines = [f"config_dir: {cfg.config_dir}", ""]
    c = cfg.categories
    lines.append(f"[categories] {len(c.categories)}개 + fallback '{c.fallback.name}' -> {c.fallback.route_to}")
    for cat in c.categories:
        lines.append(f"  - {cat.name} ({cat.label}) -> {cat.route_to}  required={list(cat.required_fields)}")
    d = cfg.defense
    by_sev = {s: sum(p.severity == s for p in d.patterns) for s in SEVERITIES}
    lines.append(f"[injection_defense] 패턴 {len(d.patterns)}개 {by_sev}, on_detect={d.action_by_severity}")
    a = cfg.approval
    lines.append(f"[approval_thresholds] 규칙 {len(a.rules)}개, auto_approve={a.auto_approve}")
    for r in a.rules:
        lines.append(f"  - {r.id}: {'|'.join(r.fields)} {r.op} {r.value!r}")
    e = cfg.eval
    lines.append(
        f"[eval_criteria] checks={[(x.id, x.weight) for x in e.checks]}, case_pass={e.case_pass_score}, "
        f"suite pass>={e.min_pass_rate}, attack block>={e.min_attack_block_rate}"
    )
    lines.append(f"  failure_types={list(e.failure_types)}")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    except ConfigError as e:
        print(f"[FAIL] {e}", file=sys.stderr)
        sys.exit(1)
    print(_summary(cfg))
    print("\n[OK] 4개 설정 파일 로드 및 검증 완료")
