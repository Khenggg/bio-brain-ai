"""Main runner for the Bio-Inspired Visual Brain (no bounding boxes or discrete object IDs)."""

from __future__ import annotations

import argparse
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from brain.core import AsyncThalamusBus, EpisodicAttractorMemory, ThalamusWorkspace, WorkspaceState
from brain.observer.visualizer import BrainObserver
from brain.regions.vision import AsyncVisualCortex, VisualCortex
from sim.cat_motion_env import CatMotionSimulator


@dataclass
class _RunStats:
    """Tracks per-frame behavioral and memory telemetry for end-of-run comparison."""

    steps: int = 0
    ignitions: int = 0
    saccades: int = 0
    boredom_sum: float = 0.0
    resonance_sum: float = 0.0
    gaze_jump_sum: float = 0.0
    _prev_gaze: tuple[float, float] | None = None

    def record(self, gaze_x: float, gaze_y: float, gaze_mode: str, ws: WorkspaceState | None) -> None:
        self.steps += 1
        if gaze_mode == "saccade":
            self.saccades += 1
        cur_xy = (gaze_x, gaze_y)
        if self._prev_gaze is not None:
            self.gaze_jump_sum += math.hypot(cur_xy[0] - self._prev_gaze[0], cur_xy[1] - self._prev_gaze[1])
        self._prev_gaze = cur_xy
        if ws is not None:
            if ws.conscious_ignition:
                self.ignitions += 1
            self.boredom_sum += ws.familiarity_boredom
            self.resonance_sum += ws.memory_resonance

    def reset(self) -> None:
        self.steps = 0
        self.ignitions = 0
        self.saccades = 0
        self.boredom_sum = 0.0
        self.resonance_sum = 0.0
        self.gaze_jump_sum = 0.0
        self._prev_gaze = None

    def print_summary(self, memory: EpisodicAttractorMemory, mode_label: str) -> None:
        n = max(1, self.steps)
        avg_boredom = self.boredom_sum / n
        avg_res = self.resonance_sum / n
        mean_jump = self.gaze_jump_sum / max(1, self.steps - 1)
        print(
            f"[Summary:{mode_label}] steps={self.steps} ignitions={self.ignitions} "
            f"kept={memory.count} ltm={memory.long_term_count} forgotten={memory.forgotten_count} "
            f"avg_boredom={avg_boredom:.3f} avg_res={avg_res:.3f} "
            f"saccades={self.saccades} mean_gaze_shift={mean_jump:.4f}"
        )


def _log_headless_step(
    step: int,
    packet: Any,
    ws: WorkspaceState | None,
    rates: dict[str, float] | None = None,
) -> None:
    ign = ws.conscious_ignition if ws is not None else False
    eps = ws.memory_episode_count if ws is not None else 0
    ltm = ws.long_term_episode_count if ws is not None else 0
    forgot = ws.forgotten_count if ws is not None else 0
    bored = ws.familiarity_boredom if ws is not None else 0.0
    res = ws.memory_resonance if ws is not None else 0.0
    rate_part = (
        f" reflex={rates.get('vision.reflex', 0.0):.1f}Hz ventral={rates.get('vision.ventral', 0.0):.1f}Hz"
        if rates is not None
        else ""
    )
    print(
        f"step={step} mode={packet.gaze.mode} gaze=({packet.gaze.x:+.2f},{packet.gaze.y:+.2f}) "
        f"surprise={packet.surprise_energy:.4f} ignition={ign} kept={eps} ltm={ltm} forgot={forgot} "
        f"boredom={bored:.2f} res={res:.2f}{rate_part} "
        f"lat={packet.diagnostics['latency_ms']:.1f}ms"
    )


def _load_session_memory(memory: EpisodicAttractorMemory, args: argparse.Namespace) -> None:
    if args.no_memory_io:
        return
    if args.fresh_memory:
        print("[Memory] --fresh-memory set: starting with blank episodic memory.")
        return
    if memory.load(args.memory_path):
        print(
            f"[Memory] Loaded {memory.count} attractors (LTM={memory.long_term_count}) "
            f"from {args.memory_path} (total_forgotten={memory.forgotten_count})."
        )
    else:
        print(f"[Memory] Starting fresh (no saved memory at {args.memory_path}).")


def _save_session_memory(memory: EpisodicAttractorMemory, args: argparse.Namespace) -> None:
    if args.no_memory_io:
        return
    memory.save(args.memory_path)
    print(
        f"[Memory] Saved {memory.count} attractors (LTM={memory.long_term_count}, "
        f"forgotten={memory.forgotten_count}) -> {args.memory_path}"
    )


def build_brain(backend: str = "auto", device: str | None = None) -> tuple[ThalamusWorkspace, VisualCortex]:
    memory = EpisodicAttractorMemory(latent_dim=384, max_episodes=50_000)
    thalamus = ThalamusWorkspace(memory=memory)
    vision = VisualCortex(ventral_backend=backend, device=device)
    thalamus.register_region(vision)
    return thalamus, vision


def _next_frame(
    sim: CatMotionSimulator | None,
    cap: cv2.VideoCapture | None,
    loop_video: bool,
) -> tuple[np.ndarray | None, tuple[float, float] | None]:
    if sim is not None:
        frame, meta = sim.step()
        return frame, meta.cat_norm_xy
    assert cap is not None
    ok, frame = cap.read()
    if not ok and loop_video:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    return (frame, None) if ok else (None, None)


def _run_async(
    args: argparse.Namespace,
    sim: CatMotionSimulator | None,
    cap: cv2.VideoCapture | None,
    observer: BrainObserver | None,
) -> int:
    memory = EpisodicAttractorMemory(latent_dim=384, max_episodes=50_000)
    _load_session_memory(memory, args)
    thalamus = ThalamusWorkspace(memory=memory)
    bus = AsyncThalamusBus(workspace=thalamus)
    vision = AsyncVisualCortex(
        ventral_backend=args.backend,
        device=args.device,
        on_semantic_packet=bus.publish,
    )
    thalamus.register_region(vision)

    stop_event = threading.Event()
    paused_flag = threading.Event()
    reset_flag = threading.Event()
    frame_lock = threading.Lock()
    latest_visual: tuple[int, np.ndarray, Any, tuple[float, float] | None] | None = None
    loop_video = args.source != "webcam" and args.max_steps == 0 and not args.headless
    stats = _RunStats()

    pace_hz = 0.0
    if observer is not None and args.source != "webcam":
        if cap is not None:
            src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            pace_hz = src_fps if 5.0 <= src_fps <= 60.0 else 25.0
        else:
            pace_hz = 30.0
    pace_dt = (1.0 / pace_hz) if pace_hz > 0.0 else 0.0

    def _sensory_worker() -> None:
        nonlocal latest_visual
        step = 0
        while not stop_event.is_set():
            tick_t0 = time.perf_counter()
            if reset_flag.is_set():
                reset_flag.clear()
                vision.reset()
                bus.reset()
                stats.reset()
                with frame_lock:
                    latest_visual = None
                step = 0

            if paused_flag.is_set():
                stop_event.wait(0.02)
                continue

            frame, gt_xy = _next_frame(sim, cap, loop_video=loop_video)
            if frame is None:
                stop_event.set()
                break

            ws_fb, _, _ = bus.get_snapshot()
            packet = vision.subconscious_step(frame, feedback=ws_fb)
            bus.publish(packet)
            step += 1
            stats.record(packet.gaze.x, packet.gaze.y, packet.gaze.mode, ws_fb)

            with frame_lock:
                # Bind raw frame and its exact reflex packet atomically so GUI gaze never lags/mismatches
                latest_visual = (step, frame, packet, gt_xy)

            if args.headless and step % 30 == 0:
                ws, _, rates = bus.get_snapshot()
                _log_headless_step(step, packet, ws, rates=rates)

            if args.max_steps > 0 and step >= args.max_steps:
                stop_event.set()
                break

            if pace_dt > 0.0:
                rem = pace_dt - (time.perf_counter() - tick_t0)
                if rem > 0.001:
                    stop_event.wait(rem)

    bus.start()
    vision.start()
    sensory_thread = threading.Thread(target=_sensory_worker, name="sensory-vision", daemon=True)
    sensory_thread.start()

    show_probe = False
    last_drawn_step = -1
    try:
        if observer is None:
            while sensory_thread.is_alive():
                sensory_thread.join(timeout=0.1)
        else:
            while not stop_event.is_set():
                with frame_lock:
                    vis_item = latest_visual
                ws, _, _ = bus.get_snapshot()
                if vis_item is not None and ws is not None and vis_item[0] != last_drawn_step:
                    vis_step, frame_view, pkt, gt_view = vis_item
                    dash = observer.render_dashboard(
                        frame_view,
                        pkt,
                        ws,
                        sim_gt_xy=gt_view if sim is not None else None,
                        show_gt_probe=show_probe,
                    )
                    cv2.imshow("Bio-Brain AI :: Async Multi-Rate Cortex", dash)
                    last_drawn_step = vis_step

                key = cv2.waitKey(8) & 0xFF
                if key == ord("q"):
                    stop_event.set()
                    break
                if key == ord("r"):
                    reset_flag.set()
                if key == ord("m"):
                    show_probe = not show_probe
                if key == ord(" "):
                    if paused_flag.is_set():
                        paused_flag.clear()
                    else:
                        paused_flag.set()
    finally:
        stop_event.set()
        sensory_thread.join(timeout=3.0)
        vision.flush(timeout=3.0)
        bus.flush(timeout=3.0)
        vision.stop()
        bus.stop()
        stats.print_summary(memory, "async")
        _save_session_memory(memory, args)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Bio-Inspired Visual Brain v1")
    parser.add_argument(
        "--source",
        default="sim",
        help="'sim' for procedural leaping cat, 'webcam' for camera 0, or path to a video file",
    )
    parser.add_argument("--backend", default="auto", choices=["auto", "dinov2", "light"])
    parser.add_argument("--device", default=None, help="Force 'cuda' or 'cpu'")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window")
    parser.add_argument("--max-steps", type=int, default=0, help="Stop after N steps (0=unlimited)")
    parser.add_argument(
        "--async-brain",
        action="store_true",
        help="Run dual-rate AsyncVisualCortex + AsyncThalamusBus instead of lockstep loop",
    )
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=Path("data/episodic_memory.npz"),
        help="Path to load/save persistent episodic attractors (.npz)",
    )
    parser.add_argument(
        "--fresh-memory",
        action="store_true",
        help="Start with blank memory (ignore saved file on startup, still saves on exit)",
    )
    parser.add_argument(
        "--no-memory-io",
        action="store_true",
        help="Disable disk load/save of episodic memory",
    )
    args = parser.parse_args()

    observer = None if args.headless else BrainObserver()

    sim = None
    cap = None
    if args.source == "sim":
        sim = CatMotionSimulator()
    elif args.source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam 0", file=sys.stderr)
            return 1
    else:
        vid_path = Path(args.source)
        if not vid_path.exists():
            print(f"Error: Video file not found: {vid_path}", file=sys.stderr)
            return 1
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video: {vid_path}", file=sys.stderr)
            return 1

    if args.async_brain:
        try:
            return _run_async(args, sim, cap, observer)
        finally:
            if cap is not None:
                cap.release()
            if observer is not None:
                cv2.destroyAllWindows()

    thalamus, _ = build_brain(backend=args.backend, device=args.device)
    _load_session_memory(thalamus.memory, args)
    stats = _RunStats()
    loop_video = args.source != "webcam" and args.max_steps == 0 and not args.headless
    paused = False
    show_probe = False
    step = 0
    last_frame = None
    gt_xy = None
    packet = None
    workspace = None

    try:
        while True:
            if not paused or last_frame is None:
                frame, gt_xy = _next_frame(sim, cap, loop_video=loop_video)
                if frame is None:
                    break
                last_frame = frame
                workspace, packets = thalamus.step({"vision": frame})
                packet = packets["vision"]
                step += 1
                stats.record(packet.gaze.x, packet.gaze.y, packet.gaze.mode, workspace)

                if args.headless and step % 30 == 0:
                    _log_headless_step(step, packet, workspace)

            if observer is not None and last_frame is not None:
                dash = observer.render_dashboard(
                    last_frame,
                    packet,
                    workspace,
                    sim_gt_xy=gt_xy if sim is not None else None,
                    show_gt_probe=show_probe,
                )
                cv2.imshow("Bio-Brain AI :: Visual Cortex v1 (No BBox / No Object ID)", dash)
                key = cv2.waitKey(16) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("r"):
                    thalamus.reset()
                    stats.reset()
                if key == ord("m"):
                    show_probe = not show_probe
                if key == ord(" "):
                    paused = not paused

            if args.max_steps > 0 and step >= args.max_steps:
                break
    finally:
        stats.print_summary(thalamus.memory, "sync")
        _save_session_memory(thalamus.memory, args)
        if cap is not None:
            cap.release()
        if observer is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
