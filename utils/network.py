"""
Fast network-quality check used to AUTO-detect poor connectivity and switch
the pipeline into offline-safe mode on its own — even if the caller didn't
explicitly ask for it. Never overrides an explicit offline=True; only
upgrades offline=False -> True when the network genuinely looks bad, so
farmers on 2G/patchy connections don't get stuck waiting on doomed API calls.

Deliberately NOT an HTTP request (that itself can hang) — a raw TCP connect
to a small, fixed set of highly-reliable anycast hosts, with a strict
wall-clock timeout, run in a background thread we never block on.
"""
from __future__ import annotations

import logging
import socket
from typing import Optional

from utils.timeout import call_with_timeout

logger = logging.getLogger(__name__)

# Cloudflare + Google public DNS — extremely high uptime, answer TCP connects
# in single-digit milliseconds on any working internet connection. Checking
# two guards against one being firewalled/blocked in a specific network.
_PROBE_HOSTS = [("1.1.1.1", 443), ("8.8.8.8", 443)]

# A connection attempt slower than this counts as "poor", not just "present" —
# a farmer on a barely-alive 2G link should be treated as effectively offline
# rather than let every downstream LLM call queue up behind a crawling link.
DEFAULT_TIMEOUT_SECONDS = 1.5


def _tcp_connect_ok(host: str, port: int, timeout: float) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def is_network_healthy(timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Return True only if at least one reliable host answers quickly.

    Bounded to ~``timeout`` seconds total regardless of how badly the network
    is behaving (uses :func:`utils.timeout.call_with_timeout` so a hung
    socket can never block the caller past the budget).
    """
    for host, port in _PROBE_HOSTS:
        try:
            ok = call_with_timeout(_tcp_connect_ok, host, port, timeout, timeout=timeout)
        except TimeoutError:
            ok = False
        if ok:
            return True
    return False


def should_force_offline(requested_offline: bool, health_check: Optional[callable] = None) -> bool:
    """Decide the effective offline flag.

    An explicit offline=True request is always honored as-is (no network
    check needed). Otherwise, run a fast health check and auto-upgrade to
    offline when the network looks poor — never the reverse.
    """
    if requested_offline:
        return True
    check = health_check or is_network_healthy
    healthy = check()
    if not healthy:
        logger.warning("Network health check failed — auto-switching to offline mode")
    return not healthy
