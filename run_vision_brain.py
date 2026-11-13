"""Main runner for the Bio-Inspired Visual Brain (no bounding boxes or discrete object IDs)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from brain.core import EpisodicAttractorMemory, ThalamusWorkspace
from brain.observer.visualizer import BrainObserver
from brain.regions.vision import VisualCortex
from sim.cat_motion_env import CatMotionSimulator


def build_brain(backend: str = "auto", device: str | None = None) -> tuple[ThalamusWorkspace, VisualCortex]:
    memory = EpisodicAttractorMemory(latent_dim=384, max_episodes=50_000)
    thalamus = ThalamusWorkspace(memory=memory)
    vision = VisualCortex(ventral_backend=backend, device=device)
    thalamus.register_region(vision)
    return thalamus, vision


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
    args = parser.parse_args()

    thalamus, _ = build_brain(backend=args.backend, device=args.device)
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
                gt_xy = None
                if sim is not None:
                    frame, meta = sim.step()
                    gt_xy = meta.cat_norm_xy
                else:
                    assert cap is not None
                    ok, frame = cap.read()
                    if not ok:
                        if args.source != "webcam":
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok, frame = cap.read()
                        if not ok:
                            break
                last_frame = frame
                workspace, packets = thalamus.step({"vision": frame})
                packet = packets["vision"]
                step += 1

                if args.headless and step % 30 == 0:
                    print(
                        f"step={step} mode={packet.gaze.mode} gaze=({packet.gaze.x:+.2f},{packet.gaze.y:+.2f}) "
                        f"surprise={packet.surprise_energy:.4f} ignition={workspace.conscious_ignition} "
                        f"episodes={workspace.memory_episode_count} lat={packet.diagnostics['latency_ms']:.1f}ms"
                    )

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
                if key == ord("m"):
                    show_probe = not show_probe
                if key == ord(" "):
                    paused = not paused

            if args.max_steps > 0 and step >= args.max_steps:
                break
    finally:
        if cap is not None:
            cap.release()
        if observer is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
