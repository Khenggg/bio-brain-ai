"""Unit tests for Thalamus Global Workspace and Episodic Attractor Memory."""

import torch
import torch.nn.functional as F

from brain.core import BrainRegion, EpisodicAttractorMemory, GazeState, SensoryPacket, ThalamusWorkspace, WorkspaceState


class _DummyRegion(BrainRegion):
    name = "dummy"

    def __init__(self) -> None:
        self.surprise = 0.0
        self.motion = 0.0
        self.latent = F.normalize(torch.ones(16), dim=0)

    def subconscious_step(self, raw_input, feedback: WorkspaceState | None = None) -> SensoryPacket:
        return SensoryPacket(
            region_name=self.name,
            timestamp=0.0,
            latent_state=self.latent,
            surprise_energy=self.surprise,
            motion_energy=self.motion,
            salience_peak_xy=(0.0, 0.0),
            gaze=GazeState(),
        )


def test_episodic_memory_hebbian_consolidation() -> None:
    mem = EpisodicAttractorMemory(latent_dim=16, consolidation_threshold=0.92)
    base = F.normalize(torch.arange(1, 17, dtype=torch.float32), dim=0)
    _, created1 = mem.consolidate(base)
    assert created1 is True
    assert mem.count == 1

    # Very similar experience reinforces existing basin without creating duplicate ID
    near = F.normalize(base + 0.02 * torch.randn(16), dim=0)
    res, created2 = mem.consolidate(near)
    assert res >= 0.92
    assert created2 is False
    assert mem.count == 1

    # Orthogonal novel experience creates a distinct attractor basin
    novel = F.normalize(-base, dim=0)
    _, created3 = mem.consolidate(novel)
    assert created3 is True
    assert mem.count == 2


def test_thalamus_subconscious_vs_conscious_gating() -> None:
    mem = EpisodicAttractorMemory(latent_dim=16)
    thalamus = ThalamusWorkspace(memory=mem, base_ignition_floor=0.10)
    region = _DummyRegion()
    thalamus.register_region(region)

    # First low-surprise tick below floor stays silent (note novelty bonus=0.08 + 0 = 0.08 < 0.10)
    region.surprise = 0.005
    ws_quiet, _ = thalamus.step({"dummy": None})
    assert ws_quiet.conscious_ignition is False
    assert ws_quiet.active_latent is None
    assert mem.count == 0

    # Sudden jump surprise ignites Global Workspace and consolidates into memory
    region.surprise = 0.25
    region.motion = 0.15
    ws_alert, _ = thalamus.step({"dummy": None})
    assert ws_alert.conscious_ignition is True
    assert ws_alert.active_latent is not None
    assert mem.count == 1
