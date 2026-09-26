"""gate 테스트. 임계값·카테고리는 설정에서 꺼내 쓰고 테스트에 숫자를 직접 쓰지 않는다."""
import shutil
from pathlib import Path

import pytest
import yaml

from cases import load_cases
from classifier import ClassificationResult
from config_loader import DEFAULT_CONFIG_DIR, get_config, load_config
from connector import DocumentConnector
from defense import GuardedClassifier, InjectionDefense
from gate import ApprovalGate, context_from


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def gate(cfg):
    return ApprovalGate(cfg)


def rule(cfg, rule_id):
    return next(r for r in cfg.approval.rules if r.id == rule_id)


def category_without_approval(cfg):
    """sensitive 규칙에 걸리지 않는 첫 카테고리."""
    sensitive = rule(cfg, "sensitive_category").value
    return next(c for c in cfg.categories.categories if c.name not in sensitive)


def category_with_field(cfg, field):
    return next(c for c in cfg.categories.categories if field in c.required_fields)


def ctx(cfg, category=None, confidence=None, extracted=None, missing=None, defense=None):
    cat = cfg.categories.get(category) if category else category_without_approval(cfg)
    conf_min = rule(cfg, "low_confidence").value
    return {
        "category": cat.name,
        "confidence": conf_min if confidence is None else confidence,
        "extracted": extracted if extracted is not None else {f: "1" for f in cat.required_fields},
        "missing_fields": missing or [],
        "defense": defense or {"flagged": False, "action": "allow"},
        "error": None,
    }


def ids(decision):
    return [t["rule_id"] for t in decision.triggered]


# ------------------------------------------------------------ 기본


def test_clean_context_auto_approved(gate, cfg):
    d = gate.evaluate(ctx(cfg))
    assert d.requires_approval is False and d.triggered == []


# ------------------------------------------------------------ 규칙별


def test_low_confidence_boundary(gate, cfg):
    v = rule(cfg, "low_confidence").value
    assert "low_confidence" not in ids(gate.evaluate(ctx(cfg, confidence=v)))
    assert "low_confidence" in ids(gate.evaluate(ctx(cfg, confidence=v - 0.01)))


@pytest.mark.parametrize("field_name", ["total_amount", "contract_amount"])
def test_high_amount_alternative_fields(gate, cfg, field_name):
    limit = int(rule(cfg, "high_amount").value)
    cat = category_with_field(cfg, field_name)
    base = {f: "1" for f in cat.required_fields}
    assert "high_amount" not in ids(gate.evaluate(ctx(cfg, cat.name, extracted={**base, field_name: str(limit)})))
    d = gate.evaluate(ctx(cfg, cat.name, extracted={**base, field_name: str(limit + 1)}))
    assert "high_amount" in ids(d)
    t = next(t for t in d.triggered if t["rule_id"] == "high_amount")
    assert t["field"] == f"extracted.{field_name}" and t["actual"] == limit + 1


def test_amount_with_separators_and_currency(gate, cfg):
    limit = int(rule(cfg, "high_amount").value)
    cat = category_with_field(cfg, "total_amount")
    formatted = f"{limit + 1000:,}원"
    d = gate.evaluate(ctx(cfg, cat.name, extracted={"total_amount": formatted}))
    assert "high_amount" in ids(d)


@pytest.mark.parametrize("value", ["1.2억", "약 5천만원", "TBD", "48,500,000 USD?"])
def test_unparseable_amount_fails_safe(gate, cfg, value):
    cat = category_with_field(cfg, "total_amount")
    d = gate.evaluate(ctx(cfg, cat.name, extracted={"total_amount": value}))
    t = next(t for t in d.triggered if t["rule_id"] == "high_amount")
    assert t["note"] == "unparseable_number"
    assert d.requires_approval


def test_missing_amount_does_not_trigger_amount_rule(gate, cfg):
    cat = category_with_field(cfg, "total_amount")
    d = gate.evaluate(ctx(cfg, cat.name, extracted={"total_amount": ""}, missing=["total_amount"]))
    assert "high_amount" not in ids(d)
    assert "missing_required_fields" in ids(d)


def test_sensitive_category(gate, cfg):
    for name in rule(cfg, "sensitive_category").value:
        assert "sensitive_category" in ids(gate.evaluate(ctx(cfg, name)))


def test_defense_flag(gate, cfg):
    d = gate.evaluate(ctx(cfg, defense={"flagged": True, "action": "flag"}))
    assert ids(d) == ["injection_flagged"]


def test_no_defense_info_does_not_crash(gate, cfg):
    c = ctx(cfg)
    c["defense"] = None
    assert gate.evaluate(c).requires_approval is False


def test_fallback_category(gate, cfg):
    fb = cfg.categories.fallback.name
    d = gate.evaluate(ctx(cfg, fb, extracted={}))
    assert "fallback_category" in ids(d)


def test_multiple_rules_all_reported_in_config_order(gate, cfg):
    sensitive = rule(cfg, "sensitive_category").value[0]
    d = gate.evaluate(ctx(cfg, sensitive, confidence=0.0, missing=["x"],
                          defense={"flagged": True}))
    assert ids(d) == [r for r in [r.id for r in cfg.approval.rules]
                      if r in {"low_confidence", "sensitive_category", "missing_required_fields",
                               "injection_flagged"}]


# ------------------------------------------------------------ 설정 주도


def cfg_with(tmp_path: Path, edit):
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    p = d / "approval_thresholds.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    edit(data)
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(d)


def test_threshold_change_in_config_changes_decision(tmp_path, cfg):
    cat = category_with_field(cfg, "total_amount")
    c = ctx(cfg, cat.name, extracted={"total_amount": "500"})
    assert "high_amount" not in ids(ApprovalGate(cfg).evaluate(c))

    def lower(d):
        next(r for r in d["rules"] if r["id"] == "high_amount")["value"] = 100
    assert "high_amount" in ids(ApprovalGate(cfg_with(tmp_path, lower)).evaluate(c))


def test_auto_approve_disabled_requires_approval_always(tmp_path, cfg):
    g = ApprovalGate(cfg_with(tmp_path, lambda d: d.update(auto_approve=False)))
    d = g.evaluate(ctx(cfg))
    assert d.requires_approval and ids(d) == ["auto_approve_disabled"]


# ------------------------------------------------------------ ClassificationResult 연동


def test_context_from_result(cfg):
    cat = category_without_approval(cfg)
    r = ClassificationResult("d", cat.name, 0.9, "", {"a": "1"}, ["b"], defense={"flagged": True, "action": "flag"})
    c = context_from(r)
    assert c["category"] == cat.name and c["confidence"] == 0.9
    assert c["extracted"] == {"a": "1"} and c["missing_fields"] == ["b"]
    assert c["defense"]["flagged"] is True


def test_blocked_attack_requires_approval(gate, cfg):
    cases = {c.id: c for c in load_cases()}
    guarded = GuardedClassifier.__new__(GuardedClassifier)  # 모델 없이 차단 경로만 사용
    guarded.defense, guarded.cfg, guarded.classifier = InjectionDefense(cfg), cfg, None
    for cid in ("atk001", "atk002"):
        r = guarded.classify(DocumentConnector().read_text(cases[cid].doc), doc_id=cid)
        d = gate.evaluate(context_from(r))
        assert d.requires_approval
        assert {"injection_flagged", "fallback_category"} <= set(ids(d))


def test_gate_agrees_with_expected_approval_in_test_cases(gate, cfg):
    """정상 케이스: 기대 카테고리/필드가 그대로 나왔다고 가정하면 gate 판정 == 기대 승인 여부."""
    for c in load_cases():
        if c.type != "normal":
            continue
        cat = cfg.categories.get(c.expected_category)
        extracted = {f: c.expected_fields.get(f, "값") for f in cat.required_fields}
        d = gate.evaluate(ctx(cfg, cat.name, confidence=1.0, extracted=extracted))
        assert d.requires_approval == c.expected_requires_approval, (c.id, ids(d))
