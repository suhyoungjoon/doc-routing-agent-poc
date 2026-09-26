"""라우터: categories.yaml 의 route_to 로 문서를 보낸다.

- mode="simulate" (기본): 파일시스템을 건드리지 않고 어디로 갈지만 결정·기록.
- mode="copy": output_root/<route_to>/ 로 복사. 원본(sample_docs)은 connector 가 read-only 로만 읽으므로
  '이동'이 아니라 복사다. 같은 내용이 이미 있으면 건너뛰고, 이름만 겹치면 -1, -2 접미사로 저장(덮어쓰기 없음).
- gate 가 승인 필요로 판정하면 보내지 않고 pending_approval 로 남긴다. approved=True(사람 승인) 이면 진행.
- 목적지 경로가 (심볼릭 링크 등으로) output_root 밖으로 해석되면 PermissionError.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from config_loader import AppConfig, get_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ROUTE_ROOT_ENV = "DOC_AGENT_ROUTE_ROOT"
ROUTE_MODE_ENV = "DOC_AGENT_ROUTE_MODE"
MODES = ("simulate", "copy")


@dataclass
class RouteDecision:
    doc_id: str
    category: str
    destination: str  # route_to (설정값 그대로)
    status: str  # routed | pending_approval
    mode: str
    target_path: str
    performed: bool
    approval_rules: list[str] = field(default_factory=list)
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Router:
    def __init__(self, connector: Any, config: AppConfig | None = None, *,
                 output_root: str | Path | None = None, mode: str | None = None):
        self.connector = connector
        self.cfg = config or get_config()
        self.mode = mode or os.environ.get(ROUTE_MODE_ENV) or "simulate"
        if self.mode not in MODES:
            raise ValueError(f"mode 는 {MODES} 중 하나: {self.mode!r}")
        self.output_root = Path(output_root or os.environ.get(ROUTE_ROOT_ENV) or PROJECT_ROOT)

    def route(self, doc_id: str, category: str, gate: Any, *, approved: bool = False) -> RouteDecision:
        cats = self.cfg.categories
        note = None
        try:
            cat = cats.get(category)
        except KeyError:
            cat, note = cats.fallback, "unknown_category"

        target_dir = self.output_root / cat.route_to
        target = target_dir / Path(doc_id).name
        decision = RouteDecision(
            doc_id=doc_id, category=cat.name, destination=cat.route_to, status="routed", mode=self.mode,
            target_path=str(target), performed=False,
            approval_rules=[t["rule_id"] for t in getattr(gate, "triggered", [])], note=note,
        )

        if gate.requires_approval and not approved:
            decision.status = "pending_approval"
            return decision
        if self.mode == "simulate":
            return decision

        data = self.connector.read_bytes(doc_id)  # 샌드박스 검사 먼저 (위반 시 아무것도 쓰지 않음)
        self._ensure_inside(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_inside(target_dir)

        final = self._free_name(target, data)
        if final is None:
            decision.note = "already_present"
            return decision
        with open(final, "xb") as f:  # 덮어쓰기 금지
            f.write(data)
        decision.target_path = str(final)
        decision.performed = True
        return decision

    def _ensure_inside(self, p: Path) -> None:
        root = Path(os.path.realpath(self.output_root))
        if not Path(os.path.realpath(p)).is_relative_to(root):
            raise PermissionError(f"라우팅 목적지가 출력 루트 밖: {p}")

    @staticmethod
    def _free_name(target: Path, data: bytes) -> Path | None:
        """같은 내용이 이미 있으면 None, 아니면 사용할 수 있는 경로."""
        digest = hashlib.sha256(data).hexdigest()
        candidate, n = target, 0
        while candidate.exists():
            if hashlib.sha256(candidate.read_bytes()).hexdigest() == digest:
                return None
            n += 1
            candidate = target.with_name(f"{target.stem}-{n}{target.suffix}")
        return candidate
