import pytest

from constants import (
    STATUS_FAILED,
    STATUS_PROCESSING,
    STATUS_QUEUED,
    STATUS_READY,
    STATUS_UPLOADED,
    InvalidStatusTransition,
    validate_status_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (STATUS_UPLOADED, STATUS_QUEUED),
        (STATUS_QUEUED, STATUS_PROCESSING),
        (STATUS_QUEUED, STATUS_FAILED),
        (STATUS_PROCESSING, STATUS_READY),
        (STATUS_PROCESSING, STATUS_FAILED),
        (STATUS_READY, STATUS_READY),
        (STATUS_FAILED, STATUS_FAILED),
    ],
)
def test_allowed_status_transitions(current, target):
    validate_status_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (STATUS_UPLOADED, STATUS_READY),
        (STATUS_QUEUED, STATUS_READY),
        (STATUS_PROCESSING, STATUS_QUEUED),
        (STATUS_READY, STATUS_PROCESSING),
        (STATUS_FAILED, STATUS_QUEUED),
    ],
)
def test_rejects_invalid_status_transitions(current, target):
    with pytest.raises(InvalidStatusTransition):
        validate_status_transition(current, target)
