"""Тесты безопасной работы с путями (ZIP traversal, абсолютные пути, '..')."""
import io
import zipfile

import pytest
from fastapi import HTTPException

from src.filetools.safe_paths import safe_name, resolve_within, safe_extract_zip


def test_safe_name_accepts_plain_component():
    assert safe_name("a1b2c") == "a1b2c"
    assert safe_name("  study5 ") == "study5"


@pytest.mark.parametrize("bad", ["..", ".", "", "a/b", "a\\b", "/etc/passwd", "C:\\win", "x\x00y"])
def test_safe_name_rejects_unsafe(bad):
    with pytest.raises(HTTPException):
        safe_name(bad)


def test_resolve_within_keeps_inside(tmp_path):
    p = resolve_within(tmp_path, "sub", "file.txt")
    assert p.is_relative_to(tmp_path.resolve())


def test_resolve_within_blocks_escape(tmp_path):
    with pytest.raises(HTTPException):
        resolve_within(tmp_path, "..", "..", "etc")


def _zip_with(names):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for n in names:
            zf.writestr(n, "data")
    buf.seek(0)
    return buf


def test_zip_traversal_rejected(tmp_path):
    with zipfile.ZipFile(_zip_with(["../evil.txt"])) as zf:
        with pytest.raises(HTTPException):
            safe_extract_zip(zf, tmp_path / "out")


def test_zip_absolute_rejected(tmp_path):
    with zipfile.ZipFile(_zip_with(["/abs.txt"])) as zf:
        with pytest.raises(HTTPException):
            safe_extract_zip(zf, tmp_path / "out")


def test_zip_safe_archive_extracts(tmp_path):
    target = tmp_path / "out"
    with zipfile.ZipFile(_zip_with(["a/b.txt", "c.txt"])) as zf:
        safe_extract_zip(zf, target)
    assert (target / "a" / "b.txt").exists()
    assert (target / "c.txt").exists()
