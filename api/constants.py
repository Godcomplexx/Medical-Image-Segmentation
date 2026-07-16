"""Канонические имена статусов исследования и их жизненный цикл.

Жизненный цикл: UPLOADED → QUEUED → PROCESSING → READY | FAILED
"""

STATUS_UPLOADED = "Загружен"
STATUS_QUEUED = "В очереди"
STATUS_PROCESSING = "В обработке"
STATUS_READY = "Готов"
STATUS_FAILED = "Ошибка"

# Полный набор статусов для инициализации БД.
ALL_STATUSES = [
    STATUS_UPLOADED,
    STATUS_QUEUED,
    STATUS_PROCESSING,
    STATUS_READY,
    STATUS_FAILED,
]

# Статусы, в которых исследование считается "в работе" — повторный запуск запрещён.
IN_FLIGHT_STATUSES = {STATUS_QUEUED, STATUS_PROCESSING}


class InvalidStatusTransition(ValueError):
    """Raised when a research lifecycle transition is not allowed."""


ALLOWED_STATUS_TRANSITIONS = {
    STATUS_UPLOADED: {STATUS_QUEUED},
    STATUS_QUEUED: {STATUS_PROCESSING, STATUS_FAILED},
    STATUS_PROCESSING: {STATUS_READY, STATUS_FAILED},
    STATUS_READY: set(),
    STATUS_FAILED: set(),
}


def validate_status_transition(current: str, target: str) -> None:
    """Validate the canonical lifecycle while allowing idempotent repeats."""
    if current == target:
        return
    if target not in ALLOWED_STATUS_TRANSITIONS.get(current, set()):
        raise InvalidStatusTransition(
            f"Invalid research status transition: {current!r} -> {target!r}"
        )
