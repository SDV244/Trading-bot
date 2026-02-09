"""Runtime startup customizations for local development.

This module is auto-imported by Python at startup when present on ``sys.path``.
"""

from __future__ import annotations

import os
import platform
import sys


def _disable_windows_wmi_platform_query() -> None:
    """Avoid hangs in platform.win32_ver() WMI query on some Windows setups.

    Python 3.13 may block inside ``platform._wmi_query`` when WMI is unhealthy.
    SQLAlchemy imports ``platform.machine()`` during module import, which can
    trigger this path and stall app startup indefinitely.
    """

    if not sys.platform.startswith("win"):
        return
    if os.getenv("APP_DISABLE_WMI_PLATFORM_QUERY", "true").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    # Force-disable WMI lookup path even if platform has not lazily created
    # the attribute yet.
    platform._wmi = None  # type: ignore[attr-defined]


_disable_windows_wmi_platform_query()
