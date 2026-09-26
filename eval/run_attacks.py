"""공격 케이스 실행: 숨은 지시에 넘어가는지 확인 (5단계) / 방어 후 막히는지 재확인 (6단계).

    uv run python eval/run_attacks.py                 # 방어 적용 (6단계 기본)
    uv run python eval/run_attacks.py --no-defense    # 방어 없이 (5단계 취약 상태)
    uv run python eval/run_attacks.py atk001          # 특정 케이스만
    uv run python eval/run_attacks.py --dry-run       # API 호출 없이 실제로 보낼 프롬프트/방어 판정만 출력
    uv run python eval/run_attacks.py --repeat 3      # 케이스당 3회 (모델 응답 변동 확인)

방어 적용 시 block 된 문서는 모델을 호출하지 않으므로 API 키 없이도 판정된다.
모델 호출이 필요한데 ANTHROPIC_API_KEY 가 없으면 해당 케이스는 runtime error 로 표시된다.
종료 코드: 0 = 모든 공격 방어됨, 1 = 하나 이상 성공(취약), 2 = 실행 오류.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from cases import judge_attack, load_cases

from classifier import Classifier  # cases.py 가 src 를 sys.path 에 추가함
from connector import DocumentConnector
from defense import GuardedClassifier, InjectionDefense
from tracing import Tracer, traced_classifier, traced_connector, traced_defense


def _has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="실행할 케이스 id (기본: 모든 attack 케이스)")
    ap.add_argument("--no-defense", action="store_true", help="defense 없이 실행 (5단계 취약 상태)")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 요청 내용만 출력")
    ap.add_argument("--repeat", type=int, default=1, help="케이스당 반복 횟수")
    args = ap.parse_args(argv)
    use_defense = not args.no_defense

    cases = [c for c in load_cases() if c.type == "attack"]
    if args.ids:
        unknown = set(args.ids) - {c.id for c in cases}
        if unknown:
            print(f"알 수 없는 attack 케이스: {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in args.ids]

    tracer = Tracer()
    conn = traced_connector(DocumentConnector(), tracer)
    defense = traced_defense(InjectionDefense(), tracer) if use_defense else None

    if args.dry_run:
        clf = Classifier(client=object())  # 요청 구성만 사용
        if defense:
            defense.harden(clf)
        for c in cases:
            text = conn.read(c.doc).text
            print(f"===== {c.id} ({c.doc}) — model={clf.model}, defense={'on' if defense else 'off'} =====")
            if defense:
                d = defense.scan(text)
                print(f"[defense] action={d.action} max_severity={d.max_severity} "
                      f"patterns={[m.pattern_id for m in d.matches]}")
                if d.action == "block":
                    print("[model call] 없음 (차단)\n")
                    continue
                text = d.text
            print("[user message]")
            print(clf.build_request(text)["messages"][0]["content"])
            print()
        print(f"trace: {tracer.path}", file=sys.stderr)
        return 0

    if not use_defense and not _has_credentials():
        print("ANTHROPIC_API_KEY 가 설정되지 않았습니다. (--dry-run 으로 프롬프트만 확인 가능)", file=sys.stderr)
        return 2

    base = Classifier()
    clf = traced_classifier(GuardedClassifier(base, defense) if defense else base, tracer)

    rows = []
    errors = 0
    for c in cases:
        doc = conn.read(c.doc)
        for i in range(args.repeat):
            try:
                r = clf.classify(doc.text, doc_id=doc.doc_id)
            except Exception as e:  # 인증 실패 등. 트레이스에는 이미 기록됨
                errors += 1
                rows.append((c, i + 1, None, {"succeeded": None, "reasons": [f"{type(e).__name__}: {e}"]}))
                continue
            verdict = judge_attack(c, r.category, r.extracted)
            tracer.record("eval", "judge_attack", status="ok",
                          input={"case_id": c.id, "trial": i + 1, "defense": use_defense},
                          decision=verdict, output={"category": r.category, "extracted": r.extracted})
            rows.append((c, i + 1, r, verdict))

    print(f"defense: {'on' if use_defense else 'off'}")
    print(f"{'case':<8}{'try':>4}  {'expected':<20}{'predicted':<20}{'conf':>5}  {'defense':<9}result")
    for c, i, r, v in rows:
        if r is None:
            print(f"{c.id:<8}{i:>4}  {c.expected_category:<20}{'-':<20}{'-':>5}  {'-':<9}ERROR  {v['reasons'][0][:80]}")
            continue
        action = (getattr(r, "defense", None) or {}).get("action", "-")
        mark = "INJECTED  " + ", ".join(v["reasons"]) if v["succeeded"] else "resisted"
        print(f"{c.id:<8}{i:>4}  {c.expected_category:<20}{r.category:<20}{r.confidence:>5.2f}  {action:<9}{mark}"
              + (f"  (error={r.error})" if r.error else ""))
    for c, i, r, v in rows:
        if r is not None and r.reasoning:
            print(f"\n--- {c.id} #{i} reasoning ---\n{r.reasoning}\nextracted={json.dumps(r.extracted, ensure_ascii=False)}")

    judged = [v for *_, v in rows if v["succeeded"] is not None]
    n_ok = sum(v["succeeded"] for v in judged)
    print(f"\n공격 성공 {n_ok}/{len(judged)}" + (f", 실행 오류 {errors}" if errors else "") + f"  trace: {tracer.path}")
    if n_ok:
        return 1
    return 2 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
