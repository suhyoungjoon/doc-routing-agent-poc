import json
from types import SimpleNamespace

import pytest

from cases import load_cases
from config_loader import get_config
from pipeline import Pipeline
from tracing import Tracer


def expected_answer_client(calls):
    """테스트 케이스의 정답을 돌려주는 가짜 모델 (문서 본문으로 케이스를 찾음)."""
    cfg = get_config()
    cases = load_cases()
    from connector import DocumentConnector
    texts = {c.id: DocumentConnector().read_text(c.doc) for c in cases}
    all_fields = {f for cat in cfg.categories.categories for f in cat.required_fields}

    def create(**kw):
        msg = kw["messages"][0]["content"]
        c = next(c for c in cases if texts[c.id] in msg)
        calls.append(c.id)
        extracted = {f: "" for f in all_fields}
        extracted.update({f: "값" for f in cfg.categories.get(c.expected_category).required_fields})
        extracted.update(c.expected_fields)
        payload = {"category": c.expected_category, "confidence": 0.95, "reasoning": "", "extracted": extracted}
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=json.dumps(payload, ensure_ascii=False))],
                               stop_reason="end_turn", model="fake", usage=None)

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=create)))


@pytest.fixture
def run(tmp_path):
    calls = []
    p = Pipeline(client=expected_answer_client(calls), tracer=Tracer(run_id="t", trace_dir=tmp_path / "tr"),
                 output_root=tmp_path / "out")
    return p, calls


def test_normal_docs_routed(run):
    p, calls = run
    cfg = get_config()
    for c in load_cases():
        if c.type != "normal":
            continue
        r = p.process(c.doc)
        assert r.classification["category"] == c.expected_category
        assert r.gate["requires_approval"] == c.expected_requires_approval
        assert r.route["destination"] == cfg.categories.get(c.expected_category).route_to
        assert r.route["status"] == ("pending_approval" if c.expected_requires_approval else "routed")
        assert r.route["performed"] is False  # simulate 기본
    assert len(calls) == 4


def test_attack_docs_held_without_model_call(run):
    p, calls = run
    cfg = get_config()
    for c in load_cases():
        if c.type == "attack":
            r = p.process(c.doc)
            assert r.classification["defense"]["action"] == "block"
            assert r.route["status"] == "pending_approval"
            assert r.route["destination"] == cfg.categories.fallback.route_to
    assert calls == []


def test_trace_covers_every_stage_in_order(run):
    p, _ = run
    p.process("invoice_001.txt")
    recs = [json.loads(l) for l in p.tracer.path.read_text(encoding="utf-8").splitlines()]
    assert [f"{r['component']}.{r['operation']}" for r in recs] == [
        "connector.read", "defense.scan", "classifier.classify", "gate.evaluate", "router.route",
        "pipeline.process"]
    assert {r["run_id"] for r in recs} == {"t"}


def test_copy_mode_end_to_end(tmp_path):
    calls = []
    p = Pipeline(client=expected_answer_client(calls), tracer=Tracer(run_id="t", trace_dir=tmp_path / "tr"),
                 output_root=tmp_path / "out", route_mode="copy")
    r = p.process("invoice_001.txt")
    assert r.route["performed"] is True
    assert (tmp_path / "out" / r.route["destination"] / "invoice_001.txt").exists()


def test_error_is_traced_and_raised(run):
    p, _ = run
    with pytest.raises(PermissionError):
        p.process("../config/categories.yaml")
    last = json.loads(p.tracer.path.read_text(encoding="utf-8").splitlines()[-1])
    assert last["component"] == "pipeline" and last["status"] == "error"
