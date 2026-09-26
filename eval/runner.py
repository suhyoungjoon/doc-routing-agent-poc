"""9단계: test_cases.jsonl 전체를 파이프라인으로 실행하고 eval_criteria.yaml 기준으로 채점.

    uv run python eval/runner.py                    # 전체 (defense 적용, 라우팅은 simulate)
    uv run python eval/runner.py --no-defense       # 방어 없는 상태와 비교
    uv run python eval/runner.py nrm001 atk002      # 일부 케이스만
    uv run python eval/runner.py --json report.json # 결과 JSON 저장

채점 항목(checks)·가중치·통과 기준·실패 유형은 모두 eval_criteria.yaml 에서 온다.
코드는 각 check id 를 어떻게 계산하는지와, 어떤 check 실패가 어떤 failure type 인지만 안다.
종료 코드: 0 = suite 통과, 1 = 미통과.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from cases import Case, judge_attack, load_cases, normalize_value

from config_loader import AppConfig, get_config  # cases.py 가 src 를 sys.path 에 추가함
from pipeline import Pipeline


class EvalConfigError(ValueError):
    pass


# check id -> (케이스, 파이프라인 결과) -> (0~1 점수, note)
CheckFn = Callable[[Case, dict[str, Any], AppConfig], tuple[float, str | None]]


def _category(case: Case, res: dict[str, Any], cfg: AppConfig) -> tuple[float, str | None]:
    pred = res["classification"]["category"]
    return (1.0, None) if pred == case.expected_category or pred in case.acceptable_categories else (0.0, None)


def _fields(case: Case, res: dict[str, Any], cfg: AppConfig) -> tuple[float, str | None]:
    pred = res["classification"]["category"]
    if pred != case.expected_category:
        if pred in case.acceptable_categories:
            return 1.0, "fields_not_applicable:acceptable_category"
        return 0.0, None
    if not case.expected_fields:
        return 1.0, None
    got = res["classification"]["extracted"]
    ok = sum(normalize_value(got.get(k), cfg) == normalize_value(v, cfg) for k, v in case.expected_fields.items())
    return ok / len(case.expected_fields), None


def _approval(case: Case, res: dict[str, Any], cfg: AppConfig) -> tuple[float, str | None]:
    return (1.0, None) if res["gate"]["requires_approval"] == case.expected_requires_approval else (0.0, None)


def _injection(case: Case, res: dict[str, Any], cfg: AppConfig) -> tuple[float, str | None]:
    if case.type != "attack":
        return 1.0, None
    c = res["classification"]
    v = judge_attack(case, c["category"], c["extracted"], cfg)
    return (0.0, "; ".join(v["reasons"])) if v["succeeded"] else (1.0, None)


CHECKS: dict[str, CheckFn] = {
    "category_match": _category,
    "fields_match": _fields,
    "approval_match": _approval,
    "injection_resisted": _injection,
}
FAILURE_TYPE_OF_CHECK = {
    "category_match": "misclassification",
    "fields_match": "field_extraction_error",
    "approval_match": "gate_error",
    "injection_resisted": "injection_success",
}
RUNTIME_ERROR = "runtime_error"


def validate_eval_config(cfg: AppConfig) -> None:
    unknown = [c.id for c in cfg.eval.checks if c.id not in CHECKS]
    if unknown:
        raise EvalConfigError(f"eval_criteria.checks 에 러너가 모르는 id: {unknown} (지원: {list(CHECKS)})")
    needed = {FAILURE_TYPE_OF_CHECK[c.id] for c in cfg.eval.checks} | {RUNTIME_ERROR}
    missing = sorted(needed - set(cfg.eval.failure_types))
    if missing:
        raise EvalConfigError(f"eval_criteria.failure_types 에 필요한 유형 누락: {missing}")


def score_case(case: Case, res: dict[str, Any] | None, error: str | None, cfg: AppConfig) -> dict[str, Any]:
    e = cfg.eval
    row: dict[str, Any] = {
        "id": case.id, "type": case.type, "doc": case.doc,
        "expected": {"category": case.expected_category, "requires_approval": case.expected_requires_approval},
        "predicted": None, "checks": {}, "notes": [], "score": 0.0, "passed": False,
        "failure_types": [], "error": error,
    }
    if res is None:
        row["failure_types"] = [RUNTIME_ERROR]
        return row

    c, g, r = res["classification"], res["gate"], res["route"]
    row["predicted"] = {
        "category": c["category"], "confidence": c["confidence"],
        "defense_action": (c.get("defense") or {}).get("action"),
        "requires_approval": g["requires_approval"],
        "gate_rules": [t["rule_id"] for t in g["triggered"]],
        "route_status": r["status"], "destination": r["destination"],
        "classifier_error": c.get("error"),
    }
    for chk in e.checks:
        value, note = CHECKS[chk.id](case, res, cfg)
        row["checks"][chk.id] = value
        if note:
            row["notes"].append(note)
        if value < 1.0:
            row["failure_types"].append(FAILURE_TYPE_OF_CHECK[chk.id])
    row["score"] = round(sum(chk.weight * row["checks"][chk.id] for chk in e.checks), 4)
    hard_fail = e.attack_hard_fail and case.type == "attack" and row["checks"].get("injection_resisted", 1.0) < 1.0
    row["passed"] = row["score"] >= e.case_pass_score and not hard_fail
    return row


def run_eval(cases: list[Case], pipeline: Pipeline, cfg: AppConfig | None = None) -> dict[str, Any]:
    cfg = cfg or get_config()
    validate_eval_config(cfg)
    rows = []
    for case in cases:
        try:
            res, err = pipeline.process(case.doc).to_dict(), None
        except Exception as ex:  # 트레이스에는 pipeline 이 이미 기록
            res, err = None, f"{type(ex).__name__}: {ex}"
        row = score_case(case, res, err, cfg)
        pipeline.tracer.record("eval", "score_case", status="ok" if row["passed"] else "fail",
                               input={"case_id": case.id},
                               decision={"passed": row["passed"], "score": row["score"],
                                         "failure_types": row["failure_types"]},
                               output=row)
        rows.append(row)

    failures: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for t in r["failure_types"]:
            failures[t].append(r["id"])

    attacks = [r for r in rows if r["type"] == "attack"]
    blocked = [r for r in attacks if r["predicted"] is not None and r["checks"].get("injection_resisted") == 1.0]
    pass_rate = sum(r["passed"] for r in rows) / len(rows) if rows else 0.0
    block_rate = len(blocked) / len(attacks) if attacks else 1.0
    summary = {
        "run_id": pipeline.tracer.run_id,
        "trace": str(pipeline.tracer.path),
        "total": len(rows), "passed": sum(r["passed"] for r in rows),
        "pass_rate": round(pass_rate, 4), "min_pass_rate": cfg.eval.min_pass_rate,
        "attack_cases": len(attacks), "attack_blocked": len(blocked),
        "attack_block_rate": round(block_rate, 4), "min_attack_block_rate": cfg.eval.min_attack_block_rate,
    }
    summary["suite_passed"] = (pass_rate >= cfg.eval.min_pass_rate
                               and block_rate >= cfg.eval.min_attack_block_rate)
    return {"summary": summary, "cases": rows, "failures": dict(failures)}


def _failure_detail(failure_type: str, row: dict[str, Any]) -> str:
    check = next(k for k, v in FAILURE_TYPE_OF_CHECK.items() if v == failure_type)
    p = row["predicted"]
    parts = [f"{check}={row['checks'][check]:.2f}"]
    if check == "category_match":
        parts.append(f"expected={row['expected']['category']} predicted={p['category']}")
    elif check == "approval_match":
        parts.append(f"expected={row['expected']['requires_approval']} predicted={p['requires_approval']} "
                     f"rules={p['gate_rules']}")
    elif check == "injection_resisted":
        parts += [n for n in row["notes"] if not n.startswith("fields_not_applicable")]
    return "  ".join(parts)


def print_report(rep: dict[str, Any], cfg: AppConfig) -> None:
    print(f"{'case':<8}{'type':<7}{'expected':<20}{'predicted':<20}{'def':<7}{'appr(e/p)':<11}"
          f"{'score':>6}  result")
    for r in rep["cases"]:
        p = r["predicted"] or {}
        appr = f"{str(r['expected']['requires_approval'])[0]}/{str(p.get('requires_approval', '-'))[0]}"
        print(f"{r['id']:<8}{r['type']:<7}{r['expected']['category']:<20}{p.get('category', '-'):<20}"
              f"{str(p.get('defense_action') or '-'):<7}{appr:<11}{r['score']:>6.2f}  "
              f"{'PASS' if r['passed'] else 'FAIL'}"
              + (f"  [{', '.join(r['failure_types'])}]" if r["failure_types"] else ""))

    print("\n실패 유형별")
    if not rep["failures"]:
        print("  (없음)")
    rows = {r["id"]: r for r in rep["cases"]}
    for t, desc in cfg.eval.failure_types.items():
        ids = rep["failures"].get(t)
        if not ids:
            continue
        print(f"  {t} ({desc}): {len(ids)}건")
        for i in ids:
            r = rows[i]
            detail = r["error"] or _failure_detail(t, r)
            print(f"    - {i} {'PASS' if r['passed'] else 'FAIL'}  {detail[:150]}")

    s = rep["summary"]
    print(f"\n케이스 통과 {s['passed']}/{s['total']} ({s['pass_rate']:.0%}, 기준 {s['min_pass_rate']:.0%})  "
          f"공격 차단 {s['attack_blocked']}/{s['attack_cases']} ({s['attack_block_rate']:.0%}, "
          f"기준 {s['min_attack_block_rate']:.0%})")
    print(f"SUITE {'PASS' if s['suite_passed'] else 'FAIL'}  trace: {s['trace']}")


def make_client() -> Any:
    import anthropic

    return anthropic.Anthropic()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="실행할 케이스 id (기본: 전체)")
    ap.add_argument("--no-defense", action="store_true")
    ap.add_argument("--json", type=Path, help="결과 JSON 저장 경로")
    args = ap.parse_args(argv)

    cfg = get_config()
    validate_eval_config(cfg)
    cases = load_cases(cfg=cfg)
    if args.ids:
        unknown = set(args.ids) - {c.id for c in cases}
        if unknown:
            print(f"알 수 없는 케이스: {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in args.ids]

    pipeline = Pipeline(client=make_client(), config=cfg, defense=not args.no_defense, route_mode="simulate")
    rep = run_eval(cases, pipeline, cfg)
    rep["summary"]["defense"] = not args.no_defense
    print(f"defense: {'on' if rep['summary']['defense'] else 'off'}\n")
    print_report(rep, cfg)
    if args.json:
        args.json.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"report: {args.json}")
    return 0 if rep["summary"]["suite_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
