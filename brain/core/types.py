"""Core continuous neural signaling structures (no bounding boxes or discrete object IDs)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class GazeState:
    """Continuous normalized foveal gaze coordinates and angular velocity in [-1, 1]."""

    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    mode: str = "fixate"  # fixate | pursuit | saccade


@dataclass(slots=True)
class SensoryPacket:
    """Output emitted silently by a subconscious brain region toward the Thalamus."""

    region_name: str
    timestamp: float
    latent_state: torch.Tensor
    surprise_energy: float
    motion_energy: float
    salience_peak_xy: tuple[float, float]
    gaze: GazeState
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkspaceState:
    """Global Workspace broadcast state after Thalamus subconscious-to-conscious gating."""

    step_id: int
    conscious_ignition: bool
    ignition_strength: float
    active_latent: torch.Tensor | None
    memory_resonance: float
    memory_episode_count: int
    adaptive_threshold: float
    subconscious_regions: dict[str, float] = field(default_factory=dict)
