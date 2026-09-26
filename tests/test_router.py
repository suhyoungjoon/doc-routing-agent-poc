"""router 테스트. 경로는 categories.yaml 의 route_to 에서만 온다."""
import shutil
from pathlib import Path

import pytest
import yaml

from config_loader import DEFAULT_CONFIG_DIR, get_config, load_config
from connector import DocumentConnector
from gate import GateDecision
from router import Router

APPROVED = GateDecision(requires_approval=False)
NEEDS_APPROVAL = GateDecision(requires_approval=True, triggered=[{"rule_id": "r"}])


@pytest.fixture
def cfg():
    return get_config()


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "sample_docs"
    root.mkdir()
    (root / "a.txt").write_text("문서 A", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("문서 B", encoding="utf-8")
    return DocumentConnector(root)


@pytest.fixture
def out(tmp_path):
    return tmp_path / "out"


def first(cfg):
    return cfg.categories.categories[0]


# ------------------------------------------------------------ 시뮬레이션 (기본)


def test_simulate_does_not_touch_filesystem(cfg, docs, out):
    r = Router(docs, cfg, output_root=out).route("a.txt", first(cfg).name, APPROVED)
    assert r.mode == "simulate" and r.status == "routed" and r.performed is False
    assert r.destination == first(cfg).route_to
    assert r.target_path == str(out / first(cfg).route_to / "a.txt")
    assert not out.exists()


def test_every_category_uses_its_route_to(cfg, docs, out):
    router = Router(docs, cfg, output_root=out)
    for c in (*cfg.categories.categories, cfg.categories.fallback):
        assert router.route("a.txt", c.name, APPROVED).destination == c.route_to


def test_pending_approval_not_routed(cfg, docs, out):
    r = Router(docs, cfg, output_root=out, mode="copy").route("a.txt", first(cfg).name, NEEDS_APPROVAL)
    assert r.status == "pending_approval" and r.performed is False
    assert r.destination == first(cfg).route_to  # 승인 후 갈 곳
    assert r.approval_rules == ["r"]
    assert not out.exists()


def test_human_approval_releases_pending(cfg, docs, out):
    r = Router(docs, cfg, output_root=out, mode="copy").route(
        "a.txt", first(cfg).name, NEEDS_APPROVAL, approved=True)
    assert r.status == "routed" and r.performed is True
    assert (out / first(cfg).route_to / "a.txt").read_text(encoding="utf-8") == "문서 A"


def test_unknown_category_goes_to_fallback(cfg, docs, out):
    r = Router(docs, cfg, output_root=out).route("a.txt", "no_such_category", APPROVED)
    assert r.destination == cfg.categories.fallback.route_to
    assert r.note == "unknown_category"


# ------------------------------------------------------------ 복사 모드


def test_copy_mode_copies_and_keeps_source(cfg, docs, out):
    r = Router(docs, cfg, output_root=out, mode="copy").route("sub/b.txt", first(cfg).name, APPROVED)
    target = out / first(cfg).route_to / "b.txt"
    assert r.performed is True and Path(r.target_path) == target
    assert target.read_text(encoding="utf-8") == "문서 B"
    assert (docs.root / "sub" / "b.txt").exists()  # 원본은 read-only, 그대로


def test_copy_same_content_is_idempotent(cfg, docs, out):
    router = Router(docs, cfg, output_root=out, mode="copy")
    router.route("a.txt", first(cfg).name, APPROVED)
    r = router.route("a.txt", first(cfg).name, APPROVED)
    assert r.performed is False and r.note == "already_present"
    assert len(list((out / first(cfg).route_to).iterdir())) == 1


def test_copy_name_collision_gets_suffix(cfg, docs, out):
    dest = out / first(cfg).route_to
    dest.mkdir(parents=True)
    (dest / "a.txt").write_text("다른 내용", encoding="utf-8")
    r = Router(docs, cfg, output_root=out, mode="copy").route("a.txt", first(cfg).name, APPROVED)
    assert Path(r.target_path).name == "a-1.txt"
    assert (dest / "a.txt").read_text(encoding="utf-8") == "다른 내용"  # 덮어쓰지 않음


def test_source_read_goes_through_connector_sandbox(cfg, docs, out):
    from connector import AccessDenied
    with pytest.raises(AccessDenied):
        Router(docs, cfg, output_root=out, mode="copy").route("../x.txt", first(cfg).name, APPROVED)
    assert not out.exists()


def test_symlinked_destination_escaping_output_root_rejected(cfg, docs, out, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    top = Path(first(cfg).route_to).parts[0]
    out.mkdir()
    (out / top).symlink_to(outside, target_is_directory=True)
    with pytest.raises(PermissionError):
        Router(docs, cfg, output_root=out, mode="copy").route("a.txt", first(cfg).name, APPROVED)
    assert list(outside.rglob("*.txt")) == []


def test_invalid_mode_rejected(cfg, docs, out):
    with pytest.raises(ValueError):
        Router(docs, cfg, output_root=out, mode="move")


def test_mode_and_root_from_env(cfg, docs, out, monkeypatch):
    monkeypatch.setenv("DOC_AGENT_ROUTE_MODE", "copy")
    monkeypatch.setenv("DOC_AGENT_ROUTE_ROOT", str(out))
    r = Router(docs, cfg).route("a.txt", first(cfg).name, APPROVED)
    assert r.mode == "copy" and r.performed


def test_follows_route_to_from_config(docs, out, tmp_path):
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    p = d / "categories.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    data["categories"][0]["route_to"] = "elsewhere/inbox"
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    cfg = load_config(d)
    r = Router(docs, cfg, output_root=out, mode="copy").route("a.txt", cfg.categories.categories[0].name, APPROVED)
    assert (out / "elsewhere" / "inbox" / "a.txt").exists() and r.destination == "elsewhere/inbox"
