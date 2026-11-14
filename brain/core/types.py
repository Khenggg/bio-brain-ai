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
    modality_event: str = "tick"  # tick | reflex | saccade | semantic
    is_fresh_semantic: bool = True


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
    source_region: str = ""
    region_rates_hz: dict[str, float] = field(default_factory=dict)
    familiarity_boredom: float = 0.0
    forgotten_count: int = 0
    long_term_episode_count: int = 0
    active_episode_idx: int = -1
    active_episode_strength: float = 0.0
    predicted_next_idx: int = -1
    prediction_confidence: float = 0.0
    expectation_violation: bool = False
    transition_count: int = 0
