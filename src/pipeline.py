"""전체 파이프라인: connector → defense → classifier → gate → router (모든 단계 트레이스).

실행: python src/pipeline.py [문서 ...] [--mode simulate|copy] [--no-defense]
      문서를 지정하지 않으면 sample_docs 전체.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from classifier import Classifier
from config_loader import AppConfig, get_config
from connector import DocumentConnector
from defense import GuardedClassifier, InjectionDefense
from gate import ApprovalGate, context_from
from router import Router
from tracing import Tracer, traced_classifier, traced_connector, traced_defense, traced_gate, traced_router


@dataclass
class PipelineResult:
    run_id: str
    doc_id: str
    classification: dict[str, Any]
    gate: dict[str, Any]
    route: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Pipeline:
    def __init__(self, *, client: Any = None, config: AppConfig | None = None, tracer: Tracer | None = None,
                 docs_root: str | Path | None = None, route_mode: str | None = None,
                 output_root: str | Path | None = None, defense: bool = True):
        self.cfg = config or get_config()
        self.tracer = tracer or Tracer()
        self.connector = traced_connector(DocumentConnector(docs_root), self.tracer)
        clf: Any = Classifier(client, config=self.cfg)
        if defense:
            clf = GuardedClassifier(clf, traced_defense(InjectionDefense(self.cfg), self.tracer))
        self.classifier = traced_classifier(clf, self.tracer)
        self.gate = traced_gate(ApprovalGate(self.cfg), self.tracer)
        self.router = traced_router(
            Router(self.connector, self.cfg, output_root=output_root, mode=route_mode), self.tracer)

    def process(self, doc_id: str) -> PipelineResult:
        start = time.perf_counter()
        try:
            doc = self.connector.read(doc_id)
            result = self.classifier.classify(doc.text, doc_id=doc.doc_id)
            decision = self.gate.evaluate(context_from(result))
            route = self.router.route(doc.doc_id, result.category, decision)
        except Exception as e:
            self.tracer.record("pipeline", "process", status="error", input={"doc_id": doc_id}, error=e,
                               duration_ms=(time.perf_counter() - start) * 1000)
            raise
        out = PipelineResult(self.tracer.run_id, doc.doc_id, result.to_dict(), decision.to_dict(), route.to_dict())
        self.tracer.record("pipeline", "process", status="ok", input={"doc_id": doc_id},
                           decision={"category": result.category,
                                     "defense_action": (result.defense or {}).get("action"),
                                     "requires_approval": decision.requires_approval,
                                     "route_status": route.status, "destination": route.destination},
                           duration_ms=(time.perf_counter() - start) * 1000)
        return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("docs", nargs="*")
    ap.add_argument("--mode", choices=["simulate", "copy"], default=None)
    ap.add_argument("--no-defense", action="store_true")
    ap.add_argument("--json", action="store_true", help="결과 전체를 JSON 으로 출력")
    args = ap.parse_args(argv)

    p = Pipeline(route_mode=args.mode, defense=not args.no_defense)
    docs = args.docs or p.connector.list_documents()
    failed = 0
    for d in docs:
        try:
            r = p.process(d)
        except Exception as e:
            failed += 1
            print(f"{d:<40} ERROR {type(e).__name__}: {str(e)[:100]}")
            continue
        if args.json:
            print(json.dumps(r.to_dict(), ensure_ascii=False, indent=2))
        else:
            c, g, rt = r.classification, r.gate, r.route
            print(f"{d:<40} {c['category']:<20} defense={(c.get('defense') or {}).get('action', '-'):<6} "
                  f"approval={str(g['requires_approval']):<5} {rt['status']:<17} -> {rt['destination']}")
    print(f"\ntrace: {p.tracer.path}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
