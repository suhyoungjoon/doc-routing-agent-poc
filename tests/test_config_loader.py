import shutil
from pathlib import Path

import pytest
import yaml

from config_loader import DEFAULT_CONFIG_DIR, ConfigError, get_config, load_config


@pytest.fixture
def cfg_dir(tmp_path: Path) -> Path:
    d = tmp_path / "config"
    shutil.copytree(DEFAULT_CONFIG_DIR, d)
    return d


def _edit(d: Path, fname: str, fn) -> None:
    p = d / fname
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    fn(data)
    p.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")


def test_default_config_loads():
    cfg = load_config()
    assert cfg.categories.names
    assert cfg.categories.fallback.name not in cfg.categories.names
    for c in cfg.categories.categories:
        assert cfg.categories.get(c.name) is c
    assert cfg.defense.patterns
    assert cfg.approval.rules


def test_get_config_is_cached():
    assert get_config() is get_config()


def test_value_from_resolved_to_fallback_name():
    cfg = load_config()
    resolved = [r for r in cfg.approval.rules if r.value == cfg.categories.fallback.name]
    assert resolved


def test_env_var_overrides_dir(cfg_dir, monkeypatch):
    monkeypatch.setenv("DOC_AGENT_CONFIG_DIR", str(cfg_dir))
    assert load_config().config_dir == cfg_dir.resolve()


def test_missing_file(cfg_dir):
    (cfg_dir / "eval_criteria.yaml").unlink()
    with pytest.raises(ConfigError, match="설정 파일 없음"):
        load_config(cfg_dir)


def test_duplicate_category(cfg_dir):
    _edit(cfg_dir, "categories.yaml", lambda d: d["categories"].append(dict(d["categories"][0])))
    with pytest.raises(ConfigError, match="중복"):
        load_config(cfg_dir)


def test_route_to_escape_rejected(cfg_dir):
    _edit(cfg_dir, "categories.yaml", lambda d: d["categories"][0].update(route_to="../etc"))
    with pytest.raises(ConfigError, match="route_to"):
        load_config(cfg_dir)


def test_bad_regex(cfg_dir):
    _edit(cfg_dir, "injection_defense.yaml", lambda d: d["patterns"][0].update(pattern="(unclosed"))
    with pytest.raises(ConfigError, match="정규식"):
        load_config(cfg_dir)


def test_unknown_defense_action(cfg_dir):
    _edit(cfg_dir, "injection_defense.yaml", lambda d: d["on_detect"]["by_severity"].update(high="allow"))
    with pytest.raises(ConfigError, match="by_severity"):
        load_config(cfg_dir)


def test_numeric_op_needs_number(cfg_dir):
    def f(d):
        for r in d["rules"]:
            if r["op"] in ("lt", "gt"):
                r["value"] = "high"
                return
    _edit(cfg_dir, "approval_thresholds.yaml", f)
    with pytest.raises(ConfigError, match="숫자"):
        load_config(cfg_dir)


def test_bad_value_from(cfg_dir):
    _edit(cfg_dir, "approval_thresholds.yaml", lambda d: d["rules"].append(
        {"id": "x", "field": "category", "op": "eq", "value_from": "categories.nope"}))
    with pytest.raises(ConfigError, match="참조 해석 실패"):
        load_config(cfg_dir)


def test_eval_weights_must_sum_to_one(cfg_dir):
    _edit(cfg_dir, "eval_criteria.yaml", lambda d: d["checks"][0].update(weight=0.9))
    with pytest.raises(ConfigError, match="weight 합"):
        load_config(cfg_dir)
