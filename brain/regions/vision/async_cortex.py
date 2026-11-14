"""Asynchronous multi-rate Visual Cortex decoupling fast gaze reflexes from deep semantics."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from typing import Any, Callable

import torch

from brain.core.types import GazeState, SensoryPacket, WorkspaceState
from brain.regions.vision.cortex import VisualCortex
from brain.regions.vision.ventral import VentralSemanticStream


@dataclass(slots=True)
class _VentralJob:
    fovea: torch.Tensor
    cuda_event: torch.cuda.Event | None
    surprise_peak: float
    motion_energy: float
    peak_xy: tuple[float, float]
    gaze: GazeState
    timestamp: float
    diagnostics: dict[str, Any]


class AsyncVisualCortex(VisualCortex):
    """Dual-rate Visual Cortex: fast Dorsal+Colliculus reflex loop + async event-gated Ventral stream."""

    def __init__(
        self,
        periphery_size: int = 96,
        fovea_size: int = 140,
        fovea_span: float = 0.44,
        ventral_backend: str = "auto",
        model_dir: Path | str | None = None,
        device: str | torch.device | None = None,
        bgr_input: bool = True,
        on_semantic_packet: Callable[[SensoryPacket], None] | None = None,
        ventral_surprise_thresh: float = 0.020,
        ventral_calm_interval_s: float = 0.25,
        auto_start: bool = True,
    ) -> None:
        super().__init__(
            periphery_size=periphery_size,
            fovea_size=fovea_size,
            fovea_span=fovea_span,
            ventral_backend=ventral_backend,
            model_dir=model_dir,
            device=device,
            bgr_input=bgr_input,
        )
        self.on_semantic_packet = on_semantic_packet
        self.ventral_surprise_thresh = ventral_surprise_thresh
        self.ventral_calm_interval_s = ventral_calm_interval_s

        if self.device.type == "cuda":
            self._fast_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self.device)
            self._ventral_stream: torch.cuda.Stream | None = torch.cuda.Stream(device=self.device)
        else:
            self._fast_stream = None
            self._ventral_stream = None

        self._mailbox_cond = threading.Condition()
        self._pending_job: _VentralJob | None = None
        self._ventral_busy = False
        self._stop_requested = False
        self._worker_error: BaseException | None = None
        self._worker_thread: threading.Thread | None = None

        self._state_lock = threading.Lock()
        self._latest_semantic_cpu: torch.Tensor | None = None
        self._last_ventral_submit_ts = 0.0
        self.last_ventral_ms = 0.0
        self.reflex_steps = 0
        self.ventral_updates = 0
        self.ventral_skipped_calm = 0

        if auto_start:
            self.start()

    def start(self) -> AsyncVisualCortex:
        with self._mailbox_cond:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return self
            self._stop_requested = False
            self._worker_error = None
            self._worker_thread = threading.Thread(
                target=self._ventral_worker_loop,
                name="vision-ventral-worker",
                daemon=True,
            )
            self._worker_thread.start()
        return self

    def stop(self, timeout: float = 3.0) -> None:
        with self._mailbox_cond:
            self._stop_requested = True
            self._pending_job = None
            self._mailbox_cond.notify_all()
            thread = self._worker_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        with self._mailbox_cond:
            self._worker_thread = None

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> AsyncVisualCortex:
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _should_wake_ventral(self, gaze: GazeState, surprise_peak: float, now: float) -> bool:
        with self._state_lock:
            if self._latest_semantic_cpu is None:
                return True
            dt = now - self._last_ventral_submit_ts
        return (
            gaze.mode == "saccade"
            or surprise_peak >= self.ventral_surprise_thresh
            or dt >= self.ventral_calm_interval_s
        )

    def _run_ventral_encode(self, job: _VentralJob) -> tuple[torch.Tensor, float]:
        t0 = time.perf_counter()
        stream_ctx = (
            torch.cuda.stream(self._ventral_stream)
            if self._ventral_stream is not None
            else nullcontext()
        )
        with stream_ctx:
            if self._ventral_stream is not None and job.cuda_event is not None:
                self._ventral_stream.wait_event(job.cuda_event)
            if self._ventral_stream is not None and job.fovea.is_cuda:
                job.fovea.record_stream(self._ventral_stream)
            latent_cpu = self.ventral.encode(
                fovea=job.fovea,
                surprise_energy=job.surprise_peak,
                gaze_vx=job.gaze.vx,
                gaze_vy=job.gaze.vy,
            )
            if self._ventral_stream is not None:
                self._ventral_stream.synchronize()
        ventral_ms = (time.perf_counter() - t0) * 1000.0
        with self._state_lock:
            self._latest_semantic_cpu = latent_cpu
            self.last_ventral_ms = ventral_ms
            self.ventral_updates += 1
        return latent_cpu, ventral_ms

    def subconscious_step(
        self, raw_input: Any, feedback: WorkspaceState | None = None
    ) -> SensoryPacket:
        """Execute fast foveated gaze reflex immediately while overlapping deep Ventral perception."""
        with self._mailbox_cond:
            if self._worker_error is not None:
                raise self._worker_error

        t0 = time.perf_counter()
        fast_ctx = (
            torch.cuda.stream(self._fast_stream)
            if self._fast_stream is not None
            else nullcontext()
        )
        with fast_ctx:
            periphery, fovea, span_xy = self.retina.sample(
                raw_input, self.colliculus.gaze, bgr_input=self.bgr_input
            )
            surprise_map, surprise_peak, motion_energy, flow_xy = self.dorsal.step(periphery)
            topdown_boredom = feedback.familiarity_boredom if feedback is not None else 0.0
            gaze, peak_xy, effective_salience = self.colliculus.step(
                surprise_map, flow_xy=flow_xy, topdown_boredom=topdown_boredom
            )

            if gaze.mode == "saccade":
                _, fovea, span_xy = self.retina.sample(raw_input, gaze, bgr_input=self.bgr_input)

            diag_cpu = {
                "periphery": periphery.detach().to("cpu"),
                "fovea": fovea.detach().to("cpu"),
                "surprise_map": surprise_map.detach().to("cpu"),
                "effective_salience": effective_salience.detach().to("cpu"),
                "ior_map": self.colliculus._ior_map.detach().to("cpu"),
            }
            cuda_ev = None
            if self._fast_stream is not None:
                cuda_ev = torch.cuda.Event()
                cuda_ev.record(self._fast_stream)

        now = time.time()
        mono_now = time.perf_counter()
        with self._state_lock:
            self.reflex_steps += 1
            reflex_steps = self.reflex_steps
            cached_semantic = self._latest_semantic_cpu

        gaze_snapshot = GazeState(
            x=gaze.x,
            y=gaze.y,
            vx=gaze.vx,
            vy=gaze.vy,
            mode=gaze.mode,
        )
        base_diag: dict[str, Any] = {
            **diag_cpu,
            "span_xy": span_xy,
            "backend": self.ventral.backend_name,
            "device": str(self.device),
            "async_mode": True,
        }

        job = _VentralJob(
            fovea=fovea,
            cuda_event=cuda_ev,
            surprise_peak=surprise_peak,
            motion_energy=motion_energy,
            peak_xy=peak_xy,
            gaze=gaze_snapshot,
            timestamp=now,
            diagnostics=base_diag,
        )

        # Warm-start Ventral synchronously only once on boot so step 0 has real semantics
        if cached_semantic is None:
            with self._state_lock:
                self._last_ventral_submit_ts = mono_now
            latent_state, ventral_ms = self._run_ventral_encode(job)
            is_fresh = True
            modality_event = "semantic"
        else:
            if self._should_wake_ventral(gaze_snapshot, surprise_peak, mono_now):
                with self._state_lock:
                    self._last_ventral_submit_ts = mono_now
                with self._mailbox_cond:
                    self._pending_job = job
                    self._mailbox_cond.notify_all()
            else:
                with self._state_lock:
                    self.ventral_skipped_calm += 1

            latent_state = VentralSemanticStream.attach_kinematics(
                cached_semantic, surprise_peak, gaze.vx, gaze.vy
            )
            is_fresh = False
            modality_event = "saccade" if gaze.mode == "saccade" else "reflex"
            with self._state_lock:
                ventral_ms = self.last_ventral_ms

        latency_ms = (time.perf_counter() - t0) * 1000.0
        vram_mb = (
            torch.cuda.memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )
        with self._state_lock:
            ventral_updates = self.ventral_updates
            ventral_skipped = self.ventral_skipped_calm

        return SensoryPacket(
            region_name=self.name,
            timestamp=now,
            latent_state=latent_state,
            surprise_energy=surprise_peak,
            motion_energy=motion_energy,
            salience_peak_xy=peak_xy,
            gaze=gaze_snapshot,
            modality_event=modality_event,
            is_fresh_semantic=is_fresh,
            diagnostics={
                **base_diag,
                "latency_ms": latency_ms,
                "ventral_latency_ms": ventral_ms,
                "vram_mb": vram_mb,
                "reflex_steps": reflex_steps,
                "ventral_updates": ventral_updates,
                "ventral_skipped_calm": ventral_skipped,
            },
        )

    def flush(self, timeout: float = 3.0) -> None:
        """Wait until any currently queued/running asynchronous Ventral pass finishes."""
        deadline = time.perf_counter() + timeout
        with self._mailbox_cond:
            while (self._pending_job is not None or self._ventral_busy) and not self._stop_requested:
                if self._worker_error is not None:
                    raise self._worker_error
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    raise TimeoutError("AsyncVisualCortex.flush timed out waiting for Ventral worker")
                self._mailbox_cond.wait(timeout=remaining)
            if self._worker_error is not None:
                raise self._worker_error

    def reset(self) -> None:
        self.flush()
        with self._mailbox_cond:
            self._pending_job = None
        with self._state_lock:
            self._latest_semantic_cpu = None
            self._last_ventral_submit_ts = 0.0
            self.last_ventral_ms = 0.0
            self.reflex_steps = 0
            self.ventral_updates = 0
            self.ventral_skipped_calm = 0
        super().reset()

    def _ventral_worker_loop(self) -> None:
        while True:
            with self._mailbox_cond:
                while self._pending_job is None and not self._stop_requested:
                    self._mailbox_cond.wait()
                if self._stop_requested and self._pending_job is None:
                    self._ventral_busy = False
                    self._mailbox_cond.notify_all()
                    return
                job = self._pending_job
                self._pending_job = None
                self._ventral_busy = True

            try:
                assert job is not None
                latent_cpu, ventral_ms = self._run_ventral_encode(job)
                if self.on_semantic_packet is not None:
                    vram_mb = (
                        torch.cuda.memory_allocated(self.device) / (1024 * 1024)
                        if self.device.type == "cuda"
                        else 0.0
                    )
                    with self._state_lock:
                        reflex_steps = self.reflex_steps
                        ventral_updates = self.ventral_updates
                        ventral_skipped = self.ventral_skipped_calm
                    pkt = SensoryPacket(
                        region_name=self.name,
                        timestamp=time.time(),
                        latent_state=latent_cpu,
                        surprise_energy=job.surprise_peak,
                        motion_energy=job.motion_energy,
                        salience_peak_xy=job.peak_xy,
                        gaze=job.gaze,
                        modality_event="semantic",
                        is_fresh_semantic=True,
                        diagnostics={
                            **job.diagnostics,
                            "latency_ms": ventral_ms,
                            "ventral_latency_ms": ventral_ms,
                            "vram_mb": vram_mb,
                            "reflex_steps": reflex_steps,
                            "ventral_updates": ventral_updates,
                            "ventral_skipped_calm": ventral_skipped,
                        },
                    )
                    self.on_semantic_packet(pkt)
            except BaseException as exc:  # pragma: no cover - surfaced on flush/step
                with self._mailbox_cond:
                    self._worker_error = exc
                    self._ventral_busy = False
                    self._mailbox_cond.notify_all()
                return
            finally:
                with self._mailbox_cond:
                    self._ventral_busy = False
                    self._mailbox_cond.notify_all()
