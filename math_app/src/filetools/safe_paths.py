"""Утилиты безопасной работы с путями в fileStorage.

Защищают от:
- абсолютных путей (`/etc/passwd`, `C:\\...`),
- обхода каталогов через `..`,
- разделителей пути в "имени" исследования/папки,
- ZIP traversal (Zip Slip) и символьных ссылок в архивах.
"""
from pathlib import Path

from fastapi import HTTPException, status


def safe_name(name: str) -> str:
    """Проверяет, что `name` — безопасный одиночный компонент пути.

    Возвращает очищенное имя или бросает 400, если оно небезопасно.
    """
    candidate = (name or "").strip()
    if not candidate or candidate in (".", ".."):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid name")
    if "/" in candidate or "\\" in candidate or "\x00" in candidate:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid name")
    # Абсолютные пути и диски Windows (например, "C:")
    if Path(candidate).is_absolute() or (len(candidate) >= 2 and candidate[1] == ":"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid name")
    return candidate


def resolve_within(base: Path, *parts: str) -> Path:
    """Возвращает путь внутри `base`, гарантируя отсутствие выхода за её пределы."""
    base_resolved = base.resolve()
    target = base_resolved.joinpath(*parts).resolve()
    if target != base_resolved and not target.is_relative_to(base_resolved):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path escapes base directory",
        )
    return target


def safe_extract_zip(zip_file, target_dir: Path) -> None:
    """Безопасно распаковывает ZIP в `target_dir`, отклоняя Zip Slip и симлинки."""
    target_resolved = target_dir.resolve()
    for member in zip_file.infolist():
        member_name = member.filename
        # Абсолютные пути и обход через ..
        if member_name.startswith(("/", "\\")) or ".." in Path(member_name).parts:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsafe path in ZIP")
        member_path = (target_resolved / member_name).resolve()
        if member_path != target_resolved and not member_path.is_relative_to(target_resolved):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsafe path in ZIP")
        # Символьные ссылки в архиве (внешний бит режима 0xA000) запрещены
        mode = member.external_attr >> 16
        if mode and (mode & 0o170000) == 0o120000:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Symlink in ZIP is not allowed")
    zip_file.extractall(target_resolved)
