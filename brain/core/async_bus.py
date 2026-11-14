"""Asynchronous event-driven Thalamus Global Workspace Bus for multi-rate brain regions."""

from __future__ import annotations

from collections import deque
import threading
import time
from typing import Callable

from brain.core.types import SensoryPacket, WorkspaceState
from brain.core.workspace import ThalamusWorkspace

WorkspaceListener = Callable[[WorkspaceState, dict[str, SensoryPacket]], None]


class RateTracker:
    """Thread-safe sliding-window frequency meter (Hz) per neural stream."""

    def __init__(self, window_size: int = 32, stale_after_s: float = 1.5) -> None:
        self.window_size = window_size
        self.stale_after_s = stale_after_s
        self._stamps: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def record(self, stream_name: str, now: float | None = None) -> None:
        ts = time.perf_counter() if now is None else now
        with self._lock:
            dq = self._stamps.get(stream_name)
            if dq is None:
                dq = deque(maxlen=self.window_size)
                self._stamps[stream_name] = dq
            dq.append(ts)

    def rates_hz(self) -> dict[str, float]:
        now = time.perf_counter()
        with self._lock:
            out: dict[str, float] = {}
            for name, dq in self._stamps.items():
                if len(dq) < 2 or (now - dq[-1]) > self.stale_after_s:
                    out[name] = 0.0
                    continue
                span = dq[-1] - dq[0]
                out[name] = (len(dq) - 1) / span if span > 1e-6 else 0.0
            return out

    def reset(self) -> None:
        with self._lock:
            self._stamps.clear()


class AsyncThalamusBus:
    """Runs ThalamusWorkspace arbitration asynchronously on incoming multi-rate region packets."""

    def __init__(
        self,
        workspace: ThalamusWorkspace | None = None,
        coincidence_window_s: float = 0.35,
        max_queue: int = 256,
    ) -> None:
        self.workspace = workspace or ThalamusWorkspace()
        self.coincidence_window_s = coincidence_window_s
        self.max_queue = max(8, int(max_queue))
        self.rates = RateTracker()

        self._cond = threading.Condition()
        self._queue: deque[SensoryPacket] = deque(maxlen=self.max_queue)
        self._latest_packets: dict[str, SensoryPacket] = {}
        self._latest_semantic: dict[str, SensoryPacket] = {}
        self._latest_state: WorkspaceState | None = None
        self._listeners: list[WorkspaceListener] = []

        self._thread: threading.Thread | None = None
        self._stop_requested = False
        self._busy = False
        self._last_error: BaseException | None = None

        self.published_count = 0
        self.processed_count = 0
        self.dropped_count = 0

    @staticmethod
    def _stream_key(packet: SensoryPacket) -> str:
        if packet.modality_event == "semantic":
            return f"{packet.region_name}.ventral"
        if packet.modality_event in ("reflex", "saccade"):
            return f"{packet.region_name}.reflex"
        return packet.region_name

    def start(self) -> AsyncThalamusBus:
        with self._cond:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_requested = False
            self._last_error = None
            self._thread = threading.Thread(
                target=self._arbiter_loop,
                name="thalamus-arbiter",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self, timeout: float = 2.0) -> None:
        with self._cond:
            self._stop_requested = True
            self._cond.notify_all()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._cond:
            self._thread = None

    def __enter__(self) -> AsyncThalamusBus:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def subscribe(self, listener: WorkspaceListener) -> None:
        with self._cond:
            self._listeners.append(listener)

    def publish(self, packet: SensoryPacket) -> None:
        """Non-blocking packet publish from any asynchronous brain region/substream."""
        with self._cond:
            if len(self._queue) >= self.max_queue:
                self._queue.popleft()
                self.dropped_count += 1
            self._queue.append(packet)
            if packet.modality_event == "semantic":
                self._latest_semantic[packet.region_name] = packet
                self._latest_packets.setdefault(packet.region_name, packet)
            else:
                # Never overwrite live reflex gaze with an older delayed semantic packet
                self._latest_packets[packet.region_name] = packet
            self.published_count += 1
            self.rates.record(self._stream_key(packet))
            self._cond.notify_all()

    def flush(self, timeout: float = 3.0) -> None:
        """Wait until all currently queued packets have been arbitrated by Thalamus."""
        deadline = time.perf_counter() + timeout
        with self._cond:
            while (self._queue or self._busy) and not self._stop_requested:
                if self._last_error is not None:
                    raise self._last_error
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("AsyncThalamusBus.flush timed out waiting for arbiter")
                self._cond.wait(timeout=remaining)
            if self._last_error is not None:
                raise self._last_error

    def get_snapshot(
        self,
    ) -> tuple[WorkspaceState | None, dict[str, SensoryPacket], dict[str, float]]:
        """Non-blocking thread-safe snapshot for observers or downstream feedback consumers."""
        with self._cond:
            if self._last_error is not None:
                raise self._last_error
            return self._latest_state, dict(self._latest_packets), self.rates.rates_hz()

    def reset(self) -> None:
        with self._cond:
            self._queue.clear()
            self._latest_packets.clear()
            self._latest_semantic.clear()
            self._latest_state = None
            self.published_count = 0
            self.processed_count = 0
            self.dropped_count = 0
            self._last_error = None
            self.rates.reset()
            self.workspace.reset()
            self._cond.notify_all()

    def _arbiter_loop(self) -> None:
        while True:
            with self._cond:
                while not self._queue and not self._stop_requested:
                    self._cond.wait()
                if self._stop_requested and not self._queue:
                    self._busy = False
                    self._cond.notify_all()
                    return
                packet = self._queue.popleft()
                self._busy = True

            try:
                self.rates.record("thalamus")
                rates_now = self.rates.rates_hz()
                state = self.workspace.process_packet(
                    packet,
                    coincidence_window_s=self.coincidence_window_s,
                    region_rates_hz=rates_now,
                )
                with self._cond:
                    self._latest_state = state
                    packets_copy = dict(self._latest_packets)
                    listeners = list(self._listeners)
                    self.processed_count += 1
                    self._busy = False
                    self._cond.notify_all()

                for fn in listeners:
                    fn(state, packets_copy)
            except BaseException as exc:  # pragma: no cover - surfaced via flush/get_snapshot
                with self._cond:
                    self._last_error = exc
                    self._busy = False
                    self._cond.notify_all()
                return
