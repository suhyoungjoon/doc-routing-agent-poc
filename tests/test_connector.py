"""connector 화이트리스트 테스트.

위반 케이스(샌드박스 밖 접근)를 먼저 정의하고, 정상 케이스는 그 뒤에 둔다.
"""
import os
from pathlib import Path

import pytest

from connector import AccessDenied, DocumentConnector


@pytest.fixture
def layout(tmp_path: Path):
    root = tmp_path / "sample_docs"
    root.mkdir()
    (root / "a.txt").write_text("문서 A", encoding="utf-8")
    (root / "sub").mkdir()
    (root / "sub" / "b.txt").write_text("문서 B", encoding="utf-8")

    outside = tmp_path / "secret.txt"
    outside.write_text("TOP SECRET", encoding="utf-8")
    # 접두사만 같은 형제 디렉토리 (startswith 비교 버그 탐지용)
    evil = tmp_path / "sample_docs_evil"
    evil.mkdir()
    (evil / "x.txt").write_text("EVIL", encoding="utf-8")
    return root, outside, evil


@pytest.fixture
def conn(layout):
    return DocumentConnector(layout[0])


# ------------------------------------------------------------ 위반 케이스


@pytest.mark.parametrize("path", [
    "../secret.txt",
    "../../etc/passwd",
    "sub/../../secret.txt",
    "./../secret.txt",
    "sub/../../sample_docs_evil/x.txt",
    "..",
    "../sample_docs_evil/x.txt",
])
def test_parent_traversal_denied(conn, path):
    with pytest.raises(AccessDenied):
        conn.read_text(path)


def test_absolute_path_outside_denied(conn, layout):
    _, outside, _ = layout
    with pytest.raises(AccessDenied):
        conn.read_text(str(outside))
    with pytest.raises(AccessDenied):
        conn.read_text("/etc/passwd")


def test_absolute_path_even_inside_root_denied(conn, layout):
    # 입력은 항상 sample_docs 기준 상대경로만 허용
    root, _, _ = layout
    with pytest.raises(AccessDenied):
        conn.read_text(str(root / "a.txt"))


def test_symlink_file_escaping_root_denied(conn, layout):
    root, outside, _ = layout
    (root / "link.txt").symlink_to(outside)
    with pytest.raises(AccessDenied):
        conn.read_text("link.txt")


def test_symlink_dir_escaping_root_denied(conn, layout):
    root, outside, _ = layout
    (root / "linkdir").symlink_to(outside.parent)
    with pytest.raises(AccessDenied):
        conn.read_text("linkdir/secret.txt")


def test_nul_byte_denied(conn):
    with pytest.raises(AccessDenied):
        conn.read_text("a.txt\x00../../secret.txt")


@pytest.mark.parametrize("path", ["", ".", "sub", "sub/"])
def test_non_file_denied(conn, path):
    with pytest.raises(AccessDenied):
        conn.read_text(path)


def test_nonexistent_inside_root_is_not_found(conn):
    with pytest.raises(FileNotFoundError):
        conn.read_text("missing.txt")


def test_encoded_traversal_is_literal_not_decoded(conn):
    # URL 인코딩을 디코딩하지 않음 -> 루트 안의 없는 파일로 취급
    with pytest.raises(FileNotFoundError):
        conn.read_text("%2e%2e/secret.txt")


def test_list_skips_escaping_symlinks(conn, layout):
    root, outside, _ = layout
    (root / "link.txt").symlink_to(outside)
    (root / "linkdir").symlink_to(outside.parent)
    docs = conn.list_documents()
    assert "link.txt" not in docs
    assert not any(d.startswith("linkdir") for d in docs)


def test_too_large_file_denied(layout):
    root, _, _ = layout
    (root / "big.txt").write_bytes(b"x" * 100)
    with pytest.raises(AccessDenied):
        DocumentConnector(root, max_bytes=10).read_text("big.txt")


def test_read_only_api(conn):
    for name in ("write", "write_text", "write_bytes", "delete", "remove", "move", "rename", "unlink"):
        assert not hasattr(conn, name)


def test_root_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        DocumentConnector(tmp_path / "nope")


def test_root_must_be_dir(tmp_path):
    f = tmp_path / "f.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        DocumentConnector(f)


def test_file_not_modified_after_read(conn, layout):
    root, _, _ = layout
    before = os.stat(root / "a.txt")
    conn.read("a.txt")
    after = os.stat(root / "a.txt")
    assert (before.st_mtime_ns, before.st_size) == (after.st_mtime_ns, after.st_size)


# ------------------------------------------------------------ 정상 케이스


def test_read_text_ok(conn):
    assert conn.read_text("a.txt") == "문서 A"
    assert conn.read_text("sub/b.txt") == "문서 B"
    assert conn.read_text("./sub/../a.txt") == "문서 A"  # 루트 안에서 끝나는 .. 은 허용


def test_symlink_inside_root_ok(conn, layout):
    root, _, _ = layout
    (root / "alias.txt").symlink_to(root / "sub" / "b.txt")
    assert conn.read_text("alias.txt") == "문서 B"


def test_read_returns_document(conn):
    doc = conn.read("sub/b.txt")
    assert doc.doc_id == "sub/b.txt"
    assert doc.text == "문서 B"
    assert doc.size == len("문서 B".encode())
    assert len(doc.sha256) == 64


def test_list_documents(conn):
    assert conn.list_documents() == ["a.txt", "sub/b.txt"]
    assert conn.list_documents("*.txt") == ["a.txt", "sub/b.txt"]


def test_default_root_env(layout, monkeypatch):
    root, _, _ = layout
    monkeypatch.setenv("DOC_AGENT_DOCS_DIR", str(root))
    assert DocumentConnector().root == root.resolve()
