"""Application logging configuration with request correlation support."""

from __future__ import annotations

import json
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any, cast

from loguru import logger

from packages.core.config import get_settings

request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def configure_logging() -> None:
    """Configure loguru sinks and correlation patching."""
    settings = get_settings()
    logger.remove()
    logger.configure(patcher=cast(Any, _patch_record))

    level = settings.log.level.upper()
    log_file = Path(settings.app.log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    if settings.app.log_json:
        logger.add(
            sys.stdout,
            level=level,
            serialize=True,
            enqueue=True,
            backtrace=False,
            diagnose=False,
        )
    else:
        logger.add(
            sys.stdout,
            level=level,
            enqueue=True,
            backtrace=False,
            diagnose=False,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | req={extra[request_id]} "
                "| {name}:{function}:{line} | {message}"
            ),
        )

    logger.add(
        str(log_file),
        level=level,
        rotation="10 MB",
        retention="14 days",
        compression="zip",
        enqueue=True,
        backtrace=False,
        diagnose=False,
        serialize=settings.app.log_json,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | req={extra[request_id]} "
            "| {name}:{function}:{line} | {message}"
        ),
    )


def _patch_record(record: dict[str, Any]) -> None:
    record["extra"]["request_id"] = request_id_var.get()
    if isinstance(record["message"], dict):
        record["message"] = json.dumps(record["message"])
