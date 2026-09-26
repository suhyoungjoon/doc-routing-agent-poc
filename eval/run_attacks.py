"""5단계: 방어 없는 classifier 에 공격 케이스를 넣어 숨은 지시에 넘어가는지 확인.

    uv run python eval/run_attacks.py                 # 모든 attack 케이스 (실제 API 호출)
    uv run python eval/run_attacks.py atk001          # 특정 케이스만
    uv run python eval/run_attacks.py --dry-run       # API 호출 없이 실제로 보낼 프롬프트만 출력
    uv run python eval/run_attacks.py --repeat 3      # 케이스당 3회 (모델 응답 변동 확인)

필요: ANTHROPIC_API_KEY (dry-run 제외). 선택: DOC_AGENT_MODEL.
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
from tracing import Tracer, traced_classifier, traced_connector


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="실행할 케이스 id (기본: 모든 attack 케이스)")
    ap.add_argument("--dry-run", action="store_true", help="API 호출 없이 요청 내용만 출력")
    ap.add_argument("--repeat", type=int, default=1, help="케이스당 반복 횟수")
    args = ap.parse_args(argv)

    cases = [c for c in load_cases() if c.type == "attack"]
    if args.ids:
        unknown = set(args.ids) - {c.id for c in cases}
        if unknown:
            print(f"알 수 없는 attack 케이스: {sorted(unknown)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c.id in args.ids]

    tracer = Tracer()
    conn = traced_connector(DocumentConnector(), tracer)

    if args.dry_run:
        clf = Classifier(client=object())  # 요청 구성만 사용
        for c in cases:
            req = clf.build_request(conn.read(c.doc).text)
            print(f"===== {c.id} ({c.doc}) — model={req['model']} =====")
            print("[user message]")
            print(req["messages"][0]["content"])
            print()
        print(f"trace: {tracer.path}", file=sys.stderr)
        return 0

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY 가 설정되지 않았습니다. (--dry-run 으로 프롬프트만 확인 가능)", file=sys.stderr)
        return 2

    clf = traced_classifier(Classifier(), tracer)
    rows = []
    for c in cases:
        doc = conn.read(c.doc)
        for i in range(args.repeat):
            r = clf.classify(doc.text, doc_id=doc.doc_id)
            verdict = judge_attack(c, r.category, r.extracted)
            tracer.record("eval", "judge_attack", status="ok",
                          input={"case_id": c.id, "trial": i + 1},
                          decision=verdict, output={"category": r.category, "extracted": r.extracted})
            rows.append((c, i + 1, r, verdict))

    print(f"{'case':<8}{'try':>4}  {'expected':<20}{'predicted':<20}{'conf':>5}  result")
    for c, i, r, v in rows:
        mark = "INJECTED  " + ", ".join(v["reasons"]) if v["succeeded"] else "resisted"
        print(f"{c.id:<8}{i:>4}  {c.expected_category:<20}{r.category:<20}{r.confidence:>5.2f}  {mark}"
              + (f"  (error={r.error})" if r.error else ""))
    for c, i, r, v in rows:
        print(f"\n--- {c.id} #{i} reasoning ---\n{r.reasoning}\nextracted={json.dumps(r.extracted, ensure_ascii=False)}")

    n_ok = sum(v["succeeded"] for *_, v in rows)
    print(f"\n공격 성공 {n_ok}/{len(rows)}  trace: {tracer.path}")
    return 1 if n_ok else 0


if __name__ == "__main__":
    sys.exit(main())
