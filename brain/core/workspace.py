"""Thalamus & Global Workspace: routes silent subconscious regions into conscious ignition."""

from __future__ import annotations

from abc import ABC, abstractmethod
import threading
from typing import Any

import torch

from brain.core.memory import EpisodicAttractorMemory
from brain.core.types import SensoryPacket, WorkspaceState


class BrainRegion(ABC):
    """Base contract for any autonomous sensory or cognitive lobe."""

    name: str

    @abstractmethod
    def subconscious_step(
        self, raw_input: Any, feedback: WorkspaceState | None = None
    ) -> SensoryPacket:
        """Process sensory/internal streams silently without forced symbolic labels."""

    def reset(self) -> None:
        """Reset transient recurrent states."""


class ThalamusWorkspace:
    """Coordinates multiple silent brain regions and gates Global Workspace ignition."""

    def __init__(
        self,
        memory: EpisodicAttractorMemory | None = None,
        base_ignition_floor: float = 0.08,
        habituation_rate: float = 0.05,
        surprise_gain: float = 1.65,
        silent_fixation_steps: int = 15,
        silent_fixation_cooldown: int = 24,
        silent_init_strength: float = 0.48,
    ) -> None:
        self.regions: dict[str, BrainRegion] = {}
        self.memory = memory or EpisodicAttractorMemory()
        self.base_ignition_floor = base_ignition_floor
        self.habituation_rate = habituation_rate
        self.surprise_gain = surprise_gain
        self.silent_fixation_steps = max(2, int(silent_fixation_steps))
        self.silent_fixation_cooldown = max(1, int(silent_fixation_cooldown))
        self.silent_init_strength = float(silent_init_strength)

        self._step_id = 0
        self._decay_step = 0
        self._surprise_ema = 0.04
        self._surprise_var = 0.001
        self._familiar_surprise_ema = 0.035
        self._boredom = 0.0
        self._recent_episode_idx = -1
        self._same_episode_streak = 0
        self._last_matched_idx = -1
        self._expected_next_idx = -1
        self._expected_next_conf = 0.0
        self._silent_dwell_latent: torch.Tensor | None = None
        self._silent_dwell_steps = 0
        self._silent_fresh_seen = False
        self._last_silent_encode_step = -(10**9)
        self._last_state: WorkspaceState | None = None
        self._active_packets: dict[str, SensoryPacket] = {}
        self._lock = threading.RLock()

    def register_region(self, region: BrainRegion) -> None:
        with self._lock:
            self.regions[region.name] = region

    @property
    def latest_state(self) -> WorkspaceState | None:
        with self._lock:
            return self._last_state

    def step(self, inputs: dict[str, Any]) -> tuple[WorkspaceState, dict[str, SensoryPacket]]:
        """Run all active regions silently, then decide conscious Workspace ignition."""
        with self._lock:
            regions_snapshot = list(self.regions.items())
            feedback = self._last_state

        packets: dict[str, SensoryPacket] = {}
        for name, region in regions_snapshot:
            if name not in inputs:
                continue
            packet = region.subconscious_step(inputs[name], feedback=feedback)
            packets[name] = packet

        with self._lock:
            self._step_id += 1
            self._active_packets.update(packets)
            state = self._evaluate_packets(
                packets,
                source_region=",".join(packets.keys()),
                update_habituation=True,
            )
            return state, packets

    def process_packet(
        self,
        packet: SensoryPacket,
        coincidence_window_s: float = 0.35,
        region_rates_hz: dict[str, float] | None = None,
    ) -> WorkspaceState:
        """Asynchronously integrate one arriving SensoryPacket into Global Workspace."""
        with self._lock:
            self._step_id += 1
            self._active_packets[packet.region_name] = packet

            newest_ts = max(p.timestamp for p in self._active_packets.values())
            stale = [
                name
                for name, pkt in self._active_packets.items()
                if (newest_ts - pkt.timestamp) > coincidence_window_s
            ]
            for name in stale:
                del self._active_packets[name]

            update_habituation = packet.modality_event != "semantic"
            return self._evaluate_packets(
                dict(self._active_packets),
                source_region=packet.region_name,
                update_habituation=update_habituation,
                region_rates_hz=region_rates_hz,
            )

    def _reset_silent_dwell(self) -> None:
        self._silent_dwell_latent = None
        self._silent_dwell_steps = 0
        self._silent_fresh_seen = False

    def _consolidate_with_sync(
        self, latent: torch.Tensor, init_strength: float | None = None
    ) -> tuple[float, int, bool]:
        prev_mem_count = self.memory.count
        resonance, created_new = self.memory.consolidate(
            latent, step_id=self._decay_step, init_strength=init_strength
        )
        _, best_idx = self.memory.resonate(latent)
        if self.memory.count < prev_mem_count:
            self._last_matched_idx = best_idx
            self._recent_episode_idx = best_idx
            self._expected_next_idx = -1
            self._expected_next_conf = 0.0
        return resonance, best_idx, created_new

    def _evaluate_packets(
        self,
        packets: dict[str, SensoryPacket],
        source_region: str = "",
        update_habituation: bool = True,
        region_rates_hz: dict[str, float] | None = None,
    ) -> WorkspaceState:
        subconscious_activity = {name: pkt.surprise_energy for name, pkt in packets.items()}
        rates = dict(region_rates_hz) if region_rates_hz is not None else {}

        if not packets:
            state = WorkspaceState(
                step_id=self._step_id,
                conscious_ignition=False,
                ignition_strength=0.0,
                active_latent=None,
                memory_resonance=0.0,
                memory_episode_count=self.memory.count,
                adaptive_threshold=self.base_ignition_floor,
                subconscious_regions=subconscious_activity,
                source_region=source_region,
                region_rates_hz=rates,
                familiarity_boredom=float(self._boredom),
                forgotten_count=self.memory.forgotten_count,
                long_term_episode_count=self.memory.long_term_count,
                active_episode_idx=-1,
                active_episode_strength=0.0,
                predicted_next_idx=-1,
                prediction_confidence=0.0,
                expectation_violation=False,
                transition_count=self.memory.transition_count,
            )
            self._last_state = state
            return state

        if update_habituation:
            self._decay_step += 1

        # Competition among sensory regions: highest surprise + novelty wins Workspace access
        dominant = max(packets.values(), key=lambda p: p.surprise_energy)
        resonance, best_idx = self.memory.resonate(dominant.latent_state)
        matched_before = best_idx >= 0 and resonance >= 0.85

        expectation_violation = False
        if matched_before:
            surprise_spike = dominant.surprise_energy > max(0.070, self._familiar_surprise_ema * 2.4)
            seq_break = (
                self._expected_next_idx >= 0
                and self._expected_next_conf >= 0.55
                and best_idx != self._last_matched_idx
                and best_idx != self._expected_next_idx
            )
            expectation_violation = surprise_spike or seq_break

        novelty_bonus = (1.0 - resonance) * 0.06 if self.memory.count > 0 else 0.08
        violation_boost = 0.035 if expectation_violation else 0.0
        salience_score = (
            dominant.surprise_energy
            + 0.35 * dominant.motion_energy
            + novelty_bonus
            + violation_boost
        )

        adaptive_threshold = max(
            self.base_ignition_floor,
            self._surprise_ema + self.surprise_gain * (self._surprise_var**0.5),
        ) + (0.03 * self._boredom * (0.0 if expectation_violation else 1.0))
        conscious_ignition = salience_score >= adaptive_threshold

        # Update habituation statistics slowly so steady clutter raises tolerance
        if update_habituation:
            delta = dominant.surprise_energy - self._surprise_ema
            self._surprise_ema += self.habituation_rate * delta
            self._surprise_var = (1.0 - self.habituation_rate) * self._surprise_var + (
                self.habituation_rate * (delta * delta)
            )

        active_latent: torch.Tensor | None = None
        created_new = False
        if conscious_ignition:
            # Reflex ticks can ignite immediately while memory consolidates on fresh semantic updates
            if dominant.is_fresh_semantic or self.memory.count == 0:
                resonance, best_idx, created_new = self._consolidate_with_sync(dominant.latent_state)
            active_latent = dominant.latent_state
            self._reset_silent_dwell()
            self._last_silent_encode_step = self._decay_step
        else:
            # Silent Fixation Encoding: sustained calm gaze gently imprints novel scenes without shock
            latent_cpu = dominant.latent_state.detach().to("cpu", dtype=torch.float32).flatten()
            latent_norm = float(torch.linalg.vector_norm(latent_cpu).item())
            if latent_norm > 1e-8 and dominant.gaze.mode != "saccade":
                unit_cpu = latent_cpu / latent_norm
                same_focus = (
                    self._silent_dwell_latent is not None
                    and float(torch.dot(unit_cpu, self._silent_dwell_latent).item()) >= 0.90
                )
                if not same_focus:
                    self._silent_dwell_latent = unit_cpu
                    self._silent_dwell_steps = 1 if update_habituation else 0
                    self._silent_fresh_seen = bool(dominant.is_fresh_semantic)
                else:
                    if dominant.is_fresh_semantic:
                        self._silent_fresh_seen = True
                    if update_habituation:
                        self._silent_dwell_steps += 1

                if (
                    self._silent_fresh_seen
                    and self._silent_dwell_steps >= self.silent_fixation_steps
                    and (self._decay_step - self._last_silent_encode_step) >= self.silent_fixation_cooldown
                    and (best_idx < 0 or resonance < self.memory.consolidation_threshold)
                ):
                    resonance, best_idx, created_new = self._consolidate_with_sync(
                        dominant.latent_state, init_strength=self.silent_init_strength
                    )
                    self._last_silent_encode_step = self._decay_step
                    self._silent_dwell_steps = 0
            else:
                self._reset_silent_dwell()

        if created_new or expectation_violation:
            self._boredom *= 0.45
            if created_new:
                self._same_episode_streak = 0
                self._recent_episode_idx = best_idx

        if update_habituation:
            exclude_idx = best_idx if best_idx >= 0 and resonance >= 0.80 else -1
            if self.memory.step_decay(step_id=self._decay_step, exclude_idx=exclude_idx) > 0:
                resonance, best_idx = self.memory.resonate(dominant.latent_state)
                self._last_matched_idx = -1
                self._expected_next_idx = -1
                self._expected_next_conf = 0.0
                if self._recent_episode_idx >= self.memory.count:
                    self._recent_episode_idx = best_idx

            if not created_new and not expectation_violation:
                if best_idx >= 0 and resonance >= 0.85:
                    self._familiar_surprise_ema += 0.08 * (
                        dominant.surprise_energy - self._familiar_surprise_ema
                    )
                    is_new_visit = best_idx != self._recent_episode_idx
                    self.memory.note_recognition(
                        best_idx, step_id=self._decay_step, bump_hit=is_new_visit
                    )
                    if is_new_visit:
                        self._recent_episode_idx = best_idx
                        self._same_episode_streak = 1
                    else:
                        self._same_episode_streak += 1

                    _, ep_hits = self.memory.exposure_of(best_idx)
                    familiarity_drive = min(
                        1.0,
                        0.12 * max(0, ep_hits - 1) + 0.012 * max(0, self._same_episode_streak - 45),
                    )
                    self._boredom += 0.06 * (familiarity_drive - self._boredom)
                elif best_idx < 0 or resonance < 0.72:
                    # Gentle inertia keeps familiarity mood across brief saccades over clutter
                    self._boredom *= 0.90
                    self._same_episode_streak = max(0, self._same_episode_streak - 2)
                    self._recent_episode_idx = -1
                else:
                    self._boredom *= 0.97
                    self._same_episode_streak = max(0, self._same_episode_streak - 1)

        matched_idx = best_idx if best_idx >= 0 and resonance >= 0.80 else -1
        if update_habituation and matched_idx >= 0:
            if self._last_matched_idx >= 0 and self._last_matched_idx != matched_idx:
                self.memory.note_transition(self._last_matched_idx, matched_idx)
            self._last_matched_idx = matched_idx
            pred_next, pred_conf = self.memory.predict_next(matched_idx)
            self._expected_next_idx = pred_next
            self._expected_next_conf = pred_conf
        else:
            pred_next, pred_conf = (
                self.memory.predict_next(matched_idx) if matched_idx >= 0 else (-1, 0.0)
            )

        ep_strength = self.memory.exposure_of(matched_idx)[0] if matched_idx >= 0 else 0.0

        state = WorkspaceState(
            step_id=self._step_id,
            conscious_ignition=conscious_ignition,
            ignition_strength=float(salience_score),
            active_latent=active_latent,
            memory_resonance=float(resonance),
            memory_episode_count=self.memory.count,
            adaptive_threshold=float(adaptive_threshold),
            subconscious_regions=subconscious_activity,
            source_region=source_region,
            region_rates_hz=rates,
            familiarity_boredom=float(self._boredom),
            forgotten_count=self.memory.forgotten_count,
            long_term_episode_count=self.memory.long_term_count,
            active_episode_idx=matched_idx,
            active_episode_strength=float(ep_strength),
            predicted_next_idx=pred_next,
            prediction_confidence=float(pred_conf),
            expectation_violation=expectation_violation,
            transition_count=self.memory.transition_count,
        )
        self._last_state = state
        return state

    def reset(self) -> None:
        with self._lock:
            self._step_id = 0
            self._decay_step = 0
            self._surprise_ema = 0.04
            self._surprise_var = 0.001
            self._familiar_surprise_ema = 0.035
            self._boredom = 0.0
            self._recent_episode_idx = -1
            self._same_episode_streak = 0
            self._last_matched_idx = -1
            self._expected_next_idx = -1
            self._expected_next_conf = 0.0
            self._reset_silent_dwell()
            self._last_silent_encode_step = -(10**9)
            self._last_state = None
            self._active_packets.clear()
            self.memory.reset()
            for region in self.regions.values():
                region.reset()
