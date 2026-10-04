"""Clocks: the only difference between a simulated run and a real one, as far as the engine cares."""

from __future__ import annotations

import time


class VirtualClock:
    """Simulated time. Nothing sleeps: waiting jumps the clock, and a generation advances it by the
    backend's reported latency."""

    simulated = True

    def __init__(self, start_ms: float = 0.0):
        self._now = start_ms

    def now(self) -> float:
        return self._now

    def advance_to(self, t_ms: float) -> None:
        self._now = max(self._now, t_ms)

    def after_service(self, latency_ms: float) -> None:
        self._now += latency_ms

    def charge(self, ms: float) -> None:
        self._now += ms


class WallClock:
    """Real time in ms since construction. Waiting sleeps; generation time passes by itself."""

    simulated = False

    def __init__(self):
        self._t0 = time.perf_counter()

    def now(self) -> float:
        return (time.perf_counter() - self._t0) * 1000.0

    def advance_to(self, t_ms: float) -> None:
        delay = t_ms - self.now()
        if delay > 0:
            time.sleep(delay / 1000.0)

    def after_service(self, latency_ms: float) -> None:
        pass  # the blocking generate() call already took that long

    def charge(self, ms: float) -> None:
        pass  # real predictor time is already on the clock
