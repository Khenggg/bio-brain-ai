"""Integration and unit tests for the Visual Cortex (no bounding boxes or object IDs)."""

import math

import numpy as np
import torch

from brain.core import EpisodicAttractorMemory, GazeState, ThalamusWorkspace
from brain.regions.vision import DualResolutionRetina, DorsalMotionStream, SuperiorColliculusGaze, VisualCortex
from sim.cat_motion_env import CatMotionSimulator


def test_retina_compression_and_bounds() -> None:
    retina = DualResolutionRetina(periphery_size=96, fovea_size=140, fovea_span=0.28, device="cpu")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    periphery, fovea, span_xy = retina.sample(frame, GazeState(x=0.9, y=-0.9))
    assert periphery.shape == (1, 3, 96, 96)
    assert fovea.shape == (1, 3, 140, 140)
    assert span_xy[0] < span_xy[1]  # square in physical pixels for 640x480


def test_dorsal_suppresses_static_and_detects_motion() -> None:
    dorsal = DorsalMotionStream(periphery_size=96, device="cpu")
    static = torch.full((1, 3, 96, 96), 0.5)

    for _ in range(12):
        s_map, peak, motion_static, _ = dorsal.step(static)
    assert peak < 0.01

    # Introduce sudden moving object patch (2.4% of peripheral field)
    moving = static.clone()
    moving[:, :, 20:34, 60:76] = 0.95
    s_map, peak_move, motion_e, _ = dorsal.step(moving)
    assert peak_move > 0.03
    assert motion_e > max(0.002, motion_static * 10.0)
    # Peak surprise must localize around y~27, x~68
    flat_idx = int(torch.argmax(s_map).item())
    py, px = divmod(flat_idx, 96)
    assert 15 <= py <= 40
    assert 55 <= px <= 82


def test_colliculus_unconscious_tracking_and_ior() -> None:
    sc = SuperiorColliculusGaze(periphery_size=96, device="cpu")
    coords = torch.linspace(-1.0, 1.0, 96)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")

    # Place continuous surprise energy around (0.6, -0.4) — no bbox passed!
    target_x, target_y = 0.6, -0.4
    surprise = 0.25 * torch.exp(-((xx - target_x) ** 2 + (yy - target_y) ** 2) / (2.0 * 0.08**2))
    for _ in range(15):
        gaze, _, _ = sc.step(surprise)

    err = math.hypot(gaze.x - target_x, gaze.y - target_y)
    assert err < 0.15

    # If scene goes completely quiet, IOR builds at current gaze
    quiet = torch.zeros(96, 96)
    for _ in range(10):
        sc.step(quiet)
    assert float(sc._ior_map.max().item()) > 0.2


def test_visual_brain_unconsciously_tracks_leaping_cat_sim() -> None:
    mem = EpisodicAttractorMemory(latent_dim=384)
    thalamus = ThalamusWorkspace(memory=mem)
    vision = VisualCortex(ventral_backend="light", device="cpu")
    thalamus.register_region(vision)

    sim = CatMotionSimulator()
    min_err = 999.0
    late_errors = []
    ignitions = 0

    for step_idx in range(140):
        frame, meta = sim.step()
        ws, packets = thalamus.step({"vision": frame})
        pkt = packets["vision"]
        if ws.conscious_ignition:
            ignitions += 1
        err = math.hypot(pkt.gaze.x - meta.cat_norm_xy[0], pkt.gaze.y - meta.cat_norm_xy[1])
        min_err = min(min_err, err)
        if step_idx >= 40 and meta.behavior in ("leap", "walk"):
            late_errors.append(err)

    assert ignitions >= 1
    assert min_err < 0.30
    assert sum(late_errors) / max(1, len(late_errors)) < 0.45


def test_cuda_dinov2_visual_brain_tracks_leaping_cat() -> None:
    if not torch.cuda.is_available():
        return
    mem = EpisodicAttractorMemory(latent_dim=384)
    thalamus = ThalamusWorkspace(memory=mem)
    vision = VisualCortex(ventral_backend="dinov2", device="cuda")
    assert vision.ventral.backend_name == "dinov2-s-reg"
    thalamus.register_region(vision)

    sim = CatMotionSimulator(seed=7)
    min_err = 999.0
    max_vram_mb = 0.0
    ignitions = 0

    for _ in range(120):
        frame, meta = sim.step()
        ws, packets = thalamus.step({"vision": frame})
        pkt = packets["vision"]
        assert torch.isfinite(pkt.latent_state).all()
        if ws.conscious_ignition:
            ignitions += 1
        err = math.hypot(pkt.gaze.x - meta.cat_norm_xy[0], pkt.gaze.y - meta.cat_norm_xy[1])
        min_err = min(min_err, err)
        max_vram_mb = max(max_vram_mb, pkt.diagnostics["vram_mb"])

    assert ignitions >= 1
    assert min_err < 0.30
    assert max_vram_mb < 1500.0


def test_dorsal_cancels_camera_pan_and_isolates_independent_motion() -> None:
    dorsal = DorsalMotionStream(periphery_size=96, device="cpu")
    # Textured background stripe pattern
    xs = torch.linspace(0.0, 12.0, 96).view(1, 1, 1, 96)
    ys = torch.linspace(0.0, 8.0, 96).view(1, 1, 96, 1)
    bg0 = 0.5 + 0.2 * torch.sin(xs) * torch.cos(ys)
    frame0 = bg0.repeat(1, 3, 1, 1)
    dorsal.step(frame0)

    # Pan camera right by 4 pixels (scene shifts left by 4 px) + independent white patch near (65, 65)
    bg1 = 0.5 + 0.2 * torch.sin(xs + 4.0 * 12.0 / 95.0) * torch.cos(ys)
    frame1 = bg1.repeat(1, 3, 1, 1)
    frame1[:, :, 58:72, 58:72] = 0.98
    s_map, peak, _, _ = dorsal.step(frame1)
    assert peak > 0.02
    flat_idx = int(torch.argmax(s_map).item())
    py, px = divmod(flat_idx, 96)
    assert 50 <= py <= 80
    assert 50 <= px <= 80


