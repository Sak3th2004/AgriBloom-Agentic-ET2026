"""Unit tests for automatic network-quality detection (utils/network.py,
utils/timeout.py) — the "if the network is very poor, auto-switch to
offline" requirement.
"""
from __future__ import annotations

import time

import pytest

from utils.network import is_network_healthy, should_force_offline
from utils.timeout import call_with_timeout


# ── call_with_timeout ────────────────────────────────────────────────────────
def test_call_with_timeout_returns_fast_result():
    assert call_with_timeout(lambda: 42, timeout=1.0) == 42


def test_call_with_timeout_raises_and_does_not_block_past_budget():
    def slow():
        time.sleep(5)
        return "too late"

    t0 = time.time()
    with pytest.raises(TimeoutError):
        call_with_timeout(slow, timeout=0.2)
    elapsed = time.time() - t0
    # Must return control near the budget, NOT wait for the 5s sleep — this
    # is the exact bug class found in vision_agent's old
    # `with ThreadPoolExecutor() as pool:` pattern (blocks on __exit__).
    assert elapsed < 1.0


def test_call_with_timeout_propagates_exceptions():
    def boom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        call_with_timeout(boom, timeout=1.0)


# ── is_network_healthy ──────────────────────────────────────────────────────
def test_is_network_healthy_true_when_any_probe_connects(monkeypatch):
    import utils.network as net

    calls = []

    def fake_connect(host, port, timeout):
        calls.append(host)
        return host == "8.8.8.8"  # first probe fails, second succeeds

    monkeypatch.setattr(net, "_tcp_connect_ok", fake_connect)
    assert is_network_healthy(timeout=0.1) is True
    assert calls == ["1.1.1.1", "8.8.8.8"]


def test_is_network_healthy_false_when_all_probes_fail(monkeypatch):
    import utils.network as net

    monkeypatch.setattr(net, "_tcp_connect_ok", lambda host, port, timeout: False)
    assert is_network_healthy(timeout=0.1) is False


def test_is_network_healthy_bounded_even_if_socket_hangs(monkeypatch):
    import utils.network as net

    def hangs(host, port, timeout):
        time.sleep(5)
        return True

    monkeypatch.setattr(net, "_tcp_connect_ok", hangs)
    t0 = time.time()
    result = is_network_healthy(timeout=0.2)
    elapsed = time.time() - t0
    assert result is False
    assert elapsed < 2.0  # two probes x ~0.2s budget each, not 2x5s


# ── should_force_offline: the auto-upgrade policy ───────────────────────────
def test_explicit_offline_always_honored_without_checking_network():
    calls = {"n": 0}

    def check():
        calls["n"] += 1
        return True  # network is actually fine

    assert should_force_offline(True, health_check=check) is True
    assert calls["n"] == 0  # never even checked — explicit request short-circuits


def test_healthy_network_does_not_force_offline():
    assert should_force_offline(False, health_check=lambda: True) is False


def test_poor_network_auto_forces_offline():
    assert should_force_offline(False, health_check=lambda: False) is True


def test_never_downgrades_a_fine_network_to_offline():
    # Sanity: requesting online with a healthy network stays online, always.
    for _ in range(5):
        assert should_force_offline(False, health_check=lambda: True) is False
