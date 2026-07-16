"""Training-tool logging objects configured through the shared implementation."""
from shared.logging_config import configure_logging

from .settings import settings


globals().update(configure_logging(settings))
