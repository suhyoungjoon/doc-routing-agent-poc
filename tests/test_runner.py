"""eval/runner.py 테스트 (가짜 모델). 가중치·기준·실패 유형은 eval_criteria.yaml 에서 온다."""
import json
import shutil
from types import SimpleNamespace

import pytest
import yaml

import runner
from cases import load_cases
from config_loader import DEFAULT_CONFIG_DIR, get_config, load_config
from connector import DocumentConnector
from pipeline import Pipeline
from tracing import Tracer

CASES = load_cases()
TEXTS = {c.id: DocumentConnector().read_text(c.doc) for c in CASES}


def model(answer):
    """answer(case) -> (category, confidence, fields) 를 돌려주는 가짜 모델. 문서 본문으로 케이스를 찾는다."""
    cfg = get_config()
    all_fields = {f for cat in cfg.categories.categories for f in cat.required_fields}
    calls = []

    def create(**kw):
        msg = kw["messages"][0]["content"]
        c = next(c for c in CASES if TEXTS[c.id] in msg)
        calls.append(c.id)
        category, confidence, fields = answer(c)
        extracted = {f: "" for f in all_fields}
        extracted.update(fields)
        payload = {"category": category, "confidence": confidence, "reasoning": "", "extracted": extracted}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))],
                               stop_reason="end_turn", model="fake", usage=None)

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create))), calls


def correct(c):
    cfg = get_config()
    fields = {f: "값" for f in cfg.categories.get(c.expected_category).required_fields}
    fields.update(c.expected_fields)
    return c.expected_category, 0.95, fields


def follow_injection(c):
    if c.type == "attack":
        return c.attack.injected_category, 0.99, dict(c.attack.injected_fields)
    return correct(c)


def evaluate(tmp_path, answer, defense=True, cfg=None, cases=None):
    client, calls = model(answer)
    p = Pipeline(client=client, config=cfg, tracer=Tracer(run_id="ev", trace_dir=tmp_path), defense=defense,
                 route_mode="simulate")
    return runner.run_eval(cases or CASES, p, cfg or get_config()), calls


def by_id(report):
    return {r["id"]: r for r in report["cases"]}


# ------------------------------------------------------------ 정상 동작


def test_all_correct_with_defense_passes_suite(tmp_path):
    rep, calls = evaluate(tmp_path, correct)
    assert rep["summary"]["suite_passed"] is True
    assert rep["summary"]["pass_rate"] == 1.0 and rep["summary"]["attack_block_rate"] == 1.0
    assert all(r["passed"] and r["score"] == pytest.approx(1.0) for r in rep["cases"])
    assert rep["failures"] == {}
    assert not any(c.startswith("atk") for c in calls)  # 공격 문서는 모델에 도달하지 않음


def test_blocked_attack_uses_acceptable_category_and_skips_fields(tmp_path):
    rep, _ = evaluate(tmp_path, correct)
    a = by_id(rep)["atk001"]
    assert a["predicted"]["category"] == get_config().categories.fallback.name
    assert a["checks"]["category_match"] == 1.0
    assert a["checks"]["fields_match"] == 1.0 and a["notes"] == ["fields_not_applicable:acceptable_category"]


# ------------------------------------------------------------ 실패 유형


def test_injection_without_defense_is_hard_fail(tmp_path):
    rep, _ = evaluate(tmp_path, follow_injection, defense=False)
    for cid in ("atk001", "atk002"):
        r = by_id(rep)[cid]
        assert r["passed"] is False and "injection_success" in r["failure_types"]
    assert set(rep["failures"]["injection_success"]) == {"atk001", "atk002"}
    assert rep["summary"]["attack_block_rate"] == 0.0 and rep["summary"]["suite_passed"] is False


def test_same_model_with_defense_is_safe(tmp_path):
    rep, _ = evaluate(tmp_path, follow_injection, defense=True)
    assert rep["summary"]["suite_passed"] is True


def test_misclassification_and_field_error(tmp_path):
    cfg = get_config()
    other = next(c.name for c in cfg.categories.categories if c.name != "invoice")

    def answer(c):
        if c.id == "nrm001":
            return other, 0.95, {}
        if c.id == "nrm003":
            cat, conf, fields = correct(c)
            return cat, conf, {**fields, "employee_name": "다른사람"}
        return correct(c)

    rep, _ = evaluate(tmp_path, answer)
    r1, r3 = by_id(rep)["nrm001"], by_id(rep)["nrm003"]
    assert "misclassification" in r1["failure_types"] and r1["checks"]["fields_match"] == 0.0
    assert r3["failure_types"] == ["field_extraction_error"]
    assert r3["checks"]["fields_match"] == pytest.approx(2 / 3)
    assert "nrm001" in rep["failures"]["misclassification"]


def test_gate_error_when_approval_differs(tmp_path):
    low = next(r for r in get_config().approval.rules if r.id == "low_confidence").value

    def answer(c):
        cat, _, fields = correct(c)
        return cat, (low / 2 if c.id == "nrm003" else 0.95), fields

    rep, _ = evaluate(tmp_path, answer)
    r = by_id(rep)["nrm003"]
    assert r["checks"]["approval_match"] == 0.0 and "gate_error" in r["failure_types"]


def test_runtime_error_recorded_and_fails_suite(tmp_path):
    def answer(c):
        raise RuntimeError("boom")

    rep, _ = evaluate(tmp_path, answer)
    normals = [r for r in rep["cases"] if r["type"] == "normal"]
    assert all(r["failure_types"] == ["runtime_error"] and r["score"] == 0.0 for r in normals)
    assert "RuntimeError" in normals[0]["error"]
    assert rep["summary"]["suite_passed"] is False
    assert rep["summary"]["attack_block_rate"] == 1.0  # 공격은 모델 전에 차단되어 오류와 무관


# ------------------------------------------------------------ 설정 주도


def cfg_with(tmp_path, edit):
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    p = d / "eval_criteria.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    edit(data)
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_config(d)


def test_weights_and_threshold_from_config(tmp_path):
    def edit(d):
        for c in d["checks"]:
            c["weight"] = 1.0 if c["id"] == "category_match" else 0.0
        d["case_pass_score"] = 1.0

    cfg = cfg_with(tmp_path, edit)

    def answer(c):
        cat, conf, fields = correct(c)
        return cat, conf, {k: "틀림" for k in fields}

    rep, _ = evaluate(tmp_path / "t", answer, cfg=cfg)
    r = by_id(rep)["nrm001"]
    assert r["score"] == 1.0 and r["passed"] is True  # 필드는 틀렸지만 가중치 0
    assert "field_extraction_error" in r["failure_types"]  # 유형 태깅은 그대로


def test_unknown_check_id_in_config_rejected(tmp_path):
    def edit(d):
        d["checks"][0]["id"] = "no_such_check"
    with pytest.raises(runner.EvalConfigError, match="no_such_check"):
        runner.validate_eval_config(cfg_with(tmp_path, edit))


def test_missing_failure_type_in_config_rejected(tmp_path):
    def edit(d):
        d["failure_types"] = [f for f in d["failure_types"] if f["id"] != "gate_error"]
    with pytest.raises(runner.EvalConfigError, match="gate_error"):
        runner.validate_eval_config(cfg_with(tmp_path, edit))


# ------------------------------------------------------------ CLI


def test_cli_writes_report_and_exit_code(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOC_AGENT_TRACE_DIR", str(tmp_path / "tr"))
    client, _ = model(correct)
    monkeypatch.setattr(runner, "make_client", lambda: client)
    out = tmp_path / "report.json"
    assert runner.main(["--json", str(out)]) == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["summary"]["suite_passed"] is True
    text = capsys.readouterr().out
    assert "SUITE PASS" in text and "atk001" in text


def test_cli_filter_and_failure_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DOC_AGENT_TRACE_DIR", str(tmp_path / "tr"))
    client, _ = model(follow_injection)
    monkeypatch.setattr(runner, "make_client", lambda: client)
    assert runner.main(["--no-defense", "atk002"]) == 1
    text = capsys.readouterr().out
    assert "injection_success" in text and "nrm001" not in text
