"""실행 트레이스 공통 유틸.

모든 컴포넌트 호출의 입력·판단(decision)·출력을 logs/traces/<run_id>.jsonl 에 한 줄씩 기록한다.

    tracer = Tracer()
    conn = traced_connector(DocumentConnector(), tracer)
    clf = traced_classifier(Classifier(), tracer)

래핑된 객체는 원래 객체와 동일하게 동작하며(반환값·예외 그대로), 공개 메서드 호출마다 레코드를 남긴다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import itertools
import json
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRACE_DIR = PROJECT_ROOT / "logs" / "traces"
TRACE_DIR_ENV = "DOC_AGENT_TRACE_DIR"
DEFAULT_MAX_TEXT = 4000

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
_SECRET_KEY_RE = re.compile(r"(api[_-]?key|authorization|token|secret|password)", re.IGNORECASE)

Decide = Callable[[Any], dict[str, Any]]


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


class Tracer:
    def __init__(self, run_id: str | None = None, trace_dir: str | Path | None = None,
                 *, max_text: int = DEFAULT_MAX_TEXT):
        self.run_id = run_id if run_id is not None else new_run_id()
        if not _RUN_ID_RE.match(self.run_id):
            raise ValueError(f"잘못된 run_id: {self.run_id!r}")
        self.trace_dir = Path(trace_dir or os.environ.get(TRACE_DIR_ENV) or DEFAULT_TRACE_DIR)
        self.max_text = max_text
        self._seq = itertools.count(1)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self.trace_dir / f"{self.run_id}.jsonl"

    # ------------------------------------------------------------ 기록

    def record(self, component: str, operation: str, *, status: str, input: Any = None,
               decision: Any = None, output: Any = None, error: BaseException | dict | None = None,
               duration_ms: float | None = None) -> dict[str, Any]:
        if isinstance(error, BaseException):
            error = {"type": type(error).__name__, "message": str(error)}
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "seq": None,
            "component": component,
            "operation": operation,
            "status": status,
            "duration_ms": None if duration_ms is None else round(duration_ms, 2),
            "input": self._jsonable(input),
            "decision": self._jsonable(decision),
            "output": self._jsonable(output),
            "error": error,
        }
        with self._lock:
            rec["seq"] = next(self._seq)
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    def call(self, component: str, operation: str, fn: Callable[..., Any], *args: Any,
             decide: Decide | None = None, **kwargs: Any) -> Any:
        """fn(*args, **kwargs) 실행 + 기록. 반환값/예외는 그대로 전달."""
        inputs = _bind_args(fn, args, kwargs)
        start = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            denied = isinstance(e, PermissionError)
            self.record(component, operation, status="denied" if denied else "error", input=inputs,
                        decision={"allowed": False} if denied else None, error=e,
                        duration_ms=(time.perf_counter() - start) * 1000)
            raise
        decision = decide(result) if decide else None
        self.record(component, operation, status="ok", input=inputs, decision=decision, output=result,
                    duration_ms=(time.perf_counter() - start) * 1000)
        return result

    def wrap(self, target: Any, component: str, decisions: dict[str, Decide] | None = None) -> "TracedProxy":
        return TracedProxy(target, self, component, decisions or {})

    # ------------------------------------------------------------ 직렬화

    def _jsonable(self, v: Any, key: str | None = None) -> Any:
        if key is not None and _SECRET_KEY_RE.search(key):
            return "[REDACTED]"
        if v is None or isinstance(v, (bool, int, float)):
            return v
        if isinstance(v, str):
            if len(v) <= self.max_text:
                return v
            return {"truncated": True, "len": len(v), "sha256": _sha(v.encode("utf-8")),
                    "head": v[: self.max_text]}
        if isinstance(v, (bytes, bytearray)):
            return {"bytes": len(v), "sha256": _sha(bytes(v))}
        if isinstance(v, Path):
            return str(v)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {f.name: self._jsonable(getattr(v, f.name), f.name) for f in dataclasses.fields(v)}
        if isinstance(v, dict):
            return {str(k): self._jsonable(x, str(k)) for k, x in v.items()}
        if isinstance(v, (list, tuple, set, frozenset)):
            return [self._jsonable(x) for x in v]
        return repr(v)


class TracedProxy:
    """대상 객체의 공개 메서드 호출을 Tracer 로 기록하는 프록시. 속성 접근은 그대로 통과."""

    def __init__(self, target: Any, tracer: Tracer, component: str, decisions: dict[str, Decide]):
        object.__setattr__(self, "_target", target)
        object.__setattr__(self, "_tracer", tracer)
        object.__setattr__(self, "_component", component)
        object.__setattr__(self, "_decisions", decisions)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def traced(*args: Any, **kwargs: Any) -> Any:
            return self._tracer.call(self._component, name, attr, *args,
                                     decide=self._decisions.get(name), **kwargs)

        return traced

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._target, name, value)

    @property
    def unwrapped(self) -> Any:
        return self._target


def _bind_args(fn: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)
    except (TypeError, ValueError):
        return {"args": list(args), "kwargs": kwargs}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# ------------------------------------------------------------ 컴포넌트별 판단 추출


def _doc_decision(doc: Any) -> dict[str, Any]:
    return {"allowed": True, "doc_id": doc.doc_id, "sha256": doc.sha256, "size": doc.size}


CONNECTOR_DECISIONS: dict[str, Decide] = {
    "read": _doc_decision,
    "read_text": lambda text: {"allowed": True, "chars": len(text)},
    "read_bytes": lambda data: {"allowed": True, "size": len(data)},
    "list_documents": lambda docs: {"count": len(docs)},
}

CLASSIFIER_DECISIONS: dict[str, Decide] = {
    "classify": lambda r: {
        "category": r.category,
        "confidence": r.confidence,
        "missing_fields": list(r.missing_fields),
        "error": r.error,
    },
}


def traced_connector(connector: Any, tracer: Tracer) -> TracedProxy:
    return tracer.wrap(connector, "connector", CONNECTOR_DECISIONS)


def traced_classifier(classifier: Any, tracer: Tracer) -> TracedProxy:
    return tracer.wrap(classifier, "classifier", CLASSIFIER_DECISIONS)
