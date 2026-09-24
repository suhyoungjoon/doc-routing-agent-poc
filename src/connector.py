"""sample_docs 전용 read-only 문서 커넥터.

- 입력 경로는 루트(기본 ./sample_docs) 기준 상대경로만 허용한다.
- 심볼릭 링크를 끝까지 해석한 실제 경로가 루트 밖이면 AccessDenied.
- 쓰기/삭제/이동 API는 의도적으로 제공하지 않는다.
"""
from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS_DIR = PROJECT_ROOT / "sample_docs"
DOCS_DIR_ENV = "DOC_AGENT_DOCS_DIR"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024


class AccessDenied(PermissionError):
    """화이트리스트(루트 디렉토리) 밖 접근 또는 허용되지 않은 대상."""


@dataclass(frozen=True)
class Document:
    doc_id: str  # 루트 기준 상대경로 (posix)
    path: Path  # 해석된 실제 경로
    text: str
    size: int
    sha256: str


class DocumentConnector:
    def __init__(self, root: str | Path | None = None, *, max_bytes: int = DEFAULT_MAX_BYTES):
        raw = Path(root or os.environ.get(DOCS_DIR_ENV) or DEFAULT_DOCS_DIR)
        if not raw.exists():
            raise FileNotFoundError(f"문서 루트 없음: {raw}")
        resolved = raw.resolve(strict=True)
        if not resolved.is_dir():
            raise NotADirectoryError(f"문서 루트가 디렉토리가 아님: {raw}")
        self._root = resolved
        self._max_bytes = max_bytes

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------ 경로 검증

    def _resolve(self, rel_path: str) -> Path:
        """상대경로를 검증해 루트 안의 실제 경로로 변환. 위반 시 AccessDenied."""
        if not isinstance(rel_path, str):
            raise AccessDenied(f"경로는 문자열이어야 함: {rel_path!r}")
        if "\x00" in rel_path:
            raise AccessDenied("경로에 NUL 문자 포함")
        if Path(rel_path).is_absolute():
            raise AccessDenied(f"절대경로 불가 (루트 기준 상대경로만 허용): {rel_path}")

        # realpath는 존재하지 않는 경로도 처리하므로, 존재 여부보다 봉쇄 검사를 먼저 한다.
        real = Path(os.path.realpath(self._root / rel_path))
        if not real.is_relative_to(self._root):
            raise AccessDenied(f"허용 디렉토리 밖 접근: {rel_path}")
        if real == self._root:
            raise AccessDenied(f"파일이 아님: {rel_path!r}")
        return real

    def _is_inside(self, p: Path) -> bool:
        try:
            return Path(os.path.realpath(p)).is_relative_to(self._root)
        except OSError:
            return False

    # ------------------------------------------------------------ 읽기 API

    def read_bytes(self, rel_path: str) -> bytes:
        real = self._resolve(rel_path)
        try:
            # real 은 심볼릭 링크가 모두 해석된 경로. 검사 후 교체되는 경우를 막기 위해 NOFOLLOW.
            fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        except FileNotFoundError:
            raise FileNotFoundError(f"문서 없음: {rel_path}") from None
        except IsADirectoryError:
            raise AccessDenied(f"파일이 아님: {rel_path}") from None
        except OSError as e:
            raise AccessDenied(f"열 수 없음: {rel_path}: {e}") from None
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            os.close(fd)
            raise AccessDenied(f"파일이 아님: {rel_path}")
        with os.fdopen(fd, "rb") as f:
            if st.st_size > self._max_bytes:
                raise AccessDenied(f"파일 크기 초과 ({st.st_size} > {self._max_bytes}): {rel_path}")
            data = f.read(self._max_bytes + 1)
        if len(data) > self._max_bytes:
            raise AccessDenied(f"파일 크기 초과: {rel_path}")
        return data

    def read_text(self, rel_path: str, encoding: str = "utf-8") -> str:
        return self.read_bytes(rel_path).decode(encoding)

    def read(self, rel_path: str, encoding: str = "utf-8") -> Document:
        data = self.read_bytes(rel_path)
        real = self._resolve(rel_path)
        doc_id = Path(os.path.normpath(rel_path)).as_posix()
        return Document(
            doc_id=doc_id,
            path=real,
            text=data.decode(encoding),
            size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def list_documents(self, pattern: str = "*") -> list[str]:
        """루트 아래 파일 목록(상대경로, 정렬). 숨김 파일과 루트 밖을 가리키는 링크는 제외."""
        out = []
        for p in self._root.rglob(pattern):
            rel = p.relative_to(self._root)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if not self._is_inside(p) or not p.is_file():
                continue
            out.append(rel.as_posix())
        return sorted(out)
