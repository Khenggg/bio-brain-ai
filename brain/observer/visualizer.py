"""Diagnostic observer displaying biological gaze, surprise energy, and conscious ignition."""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from brain.core.types import SensoryPacket, WorkspaceState


class BrainObserver:
    """Visualizes what the silent visual brain experiences WITHOUT bounding boxes or object IDs."""

    def __init__(self, panel_w: int = 440, panel_h: int = 330) -> None:
        self.panel_w = panel_w
        self.panel_h = panel_h
        self._gaze_trail: deque[tuple[int, int]] = deque(maxlen=28)
        self._ignition_history: deque[tuple[float, float, bool]] = deque(maxlen=120)

    def render_dashboard(
        self,
        raw_bgr: np.ndarray,
        packet: SensoryPacket,
        workspace: WorkspaceState,
        sim_gt_xy: tuple[float, float] | None = None,
        show_gt_probe: bool = False,
    ) -> np.ndarray:
        h, w = raw_bgr.shape[:2]

        # Panel 1: Foveated Vision Spotlight (dimmed periphery + sharp fovea circle + unconscious gaze trail)
        gx_px = int((packet.gaze.x + 1.0) * 0.5 * w)
        gy_px = int((packet.gaze.y + 1.0) * 0.5 * h)
        span_x, span_y = packet.diagnostics["span_xy"]
        rad_x = max(12, int(span_x * 0.5 * w))
        rad_y = max(12, int(span_y * 0.5 * h))

        blurred = cv2.GaussianBlur(raw_bgr, (21, 21), 0)
        dimmed = (blurred.astype(np.float32) * 0.42).astype(np.uint8)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (gx_px, gy_px), (rad_x, rad_y), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (25, 25), 0)[..., None]
        p1 = (raw_bgr.astype(np.float32) * mask + dimmed.astype(np.float32) * (1.0 - mask)).astype(np.uint8)

        self._gaze_trail.append((gx_px, gy_px))
        pts = list(self._gaze_trail)
        for i in range(1, len(pts)):
            alpha = i / len(pts)
            col = (0, int(180 * alpha + 60), int(255 * alpha))
            cv2.line(p1, pts[i - 1], pts[i], col, 2, cv2.LINE_AA)

        gaze_col = (0, 255, 255) if packet.gaze.mode == "saccade" else (
            (80, 255, 120) if packet.gaze.mode == "pursuit" else (200, 200, 200)
        )
        cv2.ellipse(p1, (gx_px, gy_px), (rad_x, rad_y), 0, 0, 360, gaze_col, 2, cv2.LINE_AA)
        cv2.drawMarker(p1, (gx_px, gy_px), gaze_col, cv2.MARKER_CROSS, 14, 2)

        if show_gt_probe and sim_gt_xy is not None:
            tx = int((sim_gt_xy[0] + 1.0) * 0.5 * w)
            ty = int((sim_gt_xy[1] + 1.0) * 0.5 * h)
            cv2.circle(p1, (tx, ty), 5, (255, 120, 255), -1)

        p1 = cv2.resize(p1, (self.panel_w, self.panel_h))
        cv2.putText(
            p1,
            f"1. Unconscious Gaze [{packet.gaze.mode.upper()}]",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            2,
        )

        # Panel 2: Dorsal Surprise Energy Field (background suppressed, novel motion glows)
        s_map = packet.diagnostics["surprise_map"].numpy()
        s_norm = np.clip(s_map / max(0.06, float(s_map.max()) + 1e-6) * 255.0, 0, 255).astype(np.uint8)
        p2 = cv2.applyColorMap(s_norm, cv2.COLORMAP_INFERNO)
        p2 = cv2.resize(p2, (self.panel_w, self.panel_h), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            p2,
            f"2. Dorsal Surprise (peak={packet.surprise_energy:.3f})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
        )

        # Panel 3: High-res Fovea Input entering Ventral ("What") Cortex
        fov = packet.diagnostics["fovea"][0].permute(1, 2, 0).numpy()
        fov_bgr = np.clip(fov[:, :, ::-1] * 255.0, 0, 255).astype(np.uint8)
        p3 = cv2.resize(fov_bgr, (self.panel_w, self.panel_h), interpolation=cv2.INTER_LINEAR)
        cv2.putText(
            p3,
            f"3. Fovea -> Ventral ({packet.diagnostics['backend']})",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (240, 240, 240),
            2,
        )

        # Panel 4: Thalamus Workspace Consciousness & Memory Resonance Telemetry
        p4 = np.full((self.panel_h, self.panel_w, 3), 20, dtype=np.uint8)
        self._ignition_history.append(
            (workspace.ignition_strength, workspace.adaptive_threshold, workspace.conscious_ignition)
        )
        if workspace.expectation_violation:
            status_txt = "ANOMALY / SURPRISE!"
            status_col = (50, 210, 255)
        elif workspace.conscious_ignition:
            status_txt = "CONSCIOUS IGNITION"
            status_col = (80, 240, 140)
        else:
            status_txt = "SILENT SUBCONSCIOUS"
            status_col = (150, 160, 175)
        cv2.putText(p4, f"4. Thalamus: {status_txt}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_col, 2)

        rates = workspace.region_rates_hz
        boredom = workspace.familiarity_boredom
        mood_tag = "BORED/EXPLORE" if boredom >= 0.45 else ("FAMILIAR" if boredom >= 0.18 else "CURIOUS")
        rate_line = (
            f"Rates: R={rates.get('vision.reflex', 0.0):.0f}Hz | "
            f"V={rates.get('vision.ventral', 0.0):.0f}Hz | "
            f"Th={rates.get('thalamus', 0.0):.0f}Hz"
            if rates
            else "Controls: [Q] Quit  [R] Reset Brain  [M] Probe  [Space] Pause"
        )
        stats = [
            f"Step: {workspace.step_id} | Latency: {packet.diagnostics['latency_ms']:.1f} ms",
            f"Device: {packet.diagnostics['device']} | VRAM: {packet.diagnostics['vram_mb']:.1f} MB",
            f"Ignition: {workspace.ignition_strength:.3f} / Thresh: {workspace.adaptive_threshold:.3f}",
            (
                f"Res: {workspace.memory_resonance:.3f} | "
                f"Kept: {workspace.memory_episode_count} (LTM:{workspace.long_term_episode_count}) | "
                f"Forgot: {workspace.forgotten_count}"
            ),
            (
                f"Boredom: {boredom:.2f} [{mood_tag}] | "
                f"Links:{workspace.transition_count} Pred:{workspace.predicted_next_idx}"
            ),
            rate_line,
        ]
        for idx, line in enumerate(stats):
            cv2.putText(
                p4,
                line,
                (12, 56 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (215, 220, 230),
                1,
            )

        # Plot ignition waveform in bottom half of Panel 4
        gx0, gy0, gw, gh = 16, 195, self.panel_w - 32, self.panel_h - 212
        cv2.rectangle(p4, (gx0, gy0), (gx0 + gw, gy0 + gh), (48, 54, 66), 1)
        hist = list(self._ignition_history)
        if len(hist) > 1:
            scale = max(0.25, max(max(s, t) for s, t, _ in hist))
            for i in range(1, len(hist)):
                x1 = gx0 + int((i - 1) / 119.0 * gw)
                x2 = gx0 + int(i / 119.0 * gw)
                y1s = gy0 + gh - int(min(1.0, hist[i - 1][0] / scale) * gh)
                y2s = gy0 + gh - int(min(1.0, hist[i][0] / scale) * gh)
                y1t = gy0 + gh - int(min(1.0, hist[i - 1][1] / scale) * gh)
                y2t = gy0 + gh - int(min(1.0, hist[i][1] / scale) * gh)
                cv2.line(p4, (x1, y1t), (x2, y2t), (90, 140, 220), 1)
                col = (80, 245, 140) if hist[i][2] else (190, 190, 200)
                cv2.line(p4, (x1, y1s), (x2, y2s), col, 2)

        top = np.hstack([p1, p2])
        bottom = np.hstack([p3, p4])
        return np.vstack([top, bottom])
