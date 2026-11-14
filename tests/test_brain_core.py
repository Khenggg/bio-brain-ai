"""Unit tests for Thalamus Global Workspace, Async Bus, and Episodic Attractor Memory."""

import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from brain.core import (
    AsyncThalamusBus,
    BrainRegion,
    EpisodicAttractorMemory,
    GazeState,
    SensoryPacket,
    ThalamusWorkspace,
    WorkspaceState,
)


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


def test_async_thalamus_bus_multi_rate_competition() -> None:
    mem = EpisodicAttractorMemory(latent_dim=16)
    thalamus = ThalamusWorkspace(memory=mem, base_ignition_floor=0.10)
    ignited_sources: list[str] = []

    with AsyncThalamusBus(workspace=thalamus, coincidence_window_s=0.5) as bus:
        bus.subscribe(lambda ws, _: ignited_sources.append(ws.source_region) if ws.conscious_ignition else None)

        vis_vec = F.normalize(torch.ones(16), dim=0)
        aud_vec = F.normalize(-torch.ones(16), dim=0)

        def _publish_vision() -> None:
            for _ in range(12):
                bus.publish(
                    SensoryPacket(
                        region_name="vision",
                        timestamp=time.time(),
                        latent_state=vis_vec,
                        surprise_energy=0.01,
                        motion_energy=0.01,
                        salience_peak_xy=(0.0, 0.0),
                        gaze=GazeState(),
                        modality_event="reflex",
                        is_fresh_semantic=False,
                    )
                )
                time.sleep(0.002)

        t_vis = threading.Thread(target=_publish_vision)
        t_vis.start()
        time.sleep(0.006)
        bus.publish(
            SensoryPacket(
                region_name="audio",
                timestamp=time.time(),
                latent_state=aud_vec,
                surprise_energy=0.35,
                motion_energy=0.0,
                salience_peak_xy=(0.0, 0.0),
                gaze=GazeState(),
                modality_event="event",
                is_fresh_semantic=True,
            )
        )
        t_vis.join()
        bus.flush(timeout=2.0)
        ws, packets, rates = bus.get_snapshot()

        assert bus.published_count == 13
        assert bus.processed_count == 13
        assert ws is not None
        assert "audio" in ignited_sources
        assert mem.count >= 1
        assert "vision" in packets and "audio" in packets
        assert "thalamus" in rates and "vision.reflex" in rates


def test_async_thalamus_bus_preserves_live_reflex_gaze() -> None:
    thalamus = ThalamusWorkspace(memory=EpisodicAttractorMemory(latent_dim=16))
    with AsyncThalamusBus(workspace=thalamus) as bus:
        vec = F.normalize(torch.ones(16), dim=0)
        bus.publish(
            SensoryPacket(
                region_name="vision",
                timestamp=1.0,
                latent_state=vec,
                surprise_energy=0.04,
                motion_energy=0.02,
                salience_peak_xy=(0.5, -0.2),
                gaze=GazeState(x=0.5, y=-0.2, mode="pursuit"),
                modality_event="reflex",
                is_fresh_semantic=False,
            )
        )
        bus.publish(
            SensoryPacket(
                region_name="vision",
                timestamp=1.02,
                latent_state=-vec,
                surprise_energy=0.04,
                motion_energy=0.02,
                salience_peak_xy=(-0.8, 0.8),
                gaze=GazeState(x=-0.8, y=0.8, mode="saccade"),
                modality_event="semantic",
                is_fresh_semantic=True,
            )
        )
        bus.flush(timeout=2.0)
        _, packets, _ = bus.get_snapshot()
        assert packets["vision"].modality_event == "reflex"
        assert packets["vision"].gaze.x == 0.5


def test_memory_forgetting_persistence_and_consolidation(tmp_path: Path) -> None:
    mem = EpisodicAttractorMemory(
        latent_dim=16,
        consolidation_threshold=0.85,
        init_strength=0.50,
        decay_rate=0.05,
        forget_threshold=0.20,
        grace_steps=2,
    )
    v0 = F.normalize(torch.tensor([1.0] + [0.0] * 15), dim=0)
    v1 = F.normalize(torch.tensor([0.0, 1.0] + [0.0] * 14), dim=0)

    mem.consolidate(v0, step_id=1)
    mem.consolidate(v1, step_id=1)
    assert mem.count == 2

    # Repeatedly revisit v0 while leaving v1 unattended -> v1 fades and is pruned
    for step in range(2, 14):
        if step in (2, 4, 6, 8):
            mem.note_recognition(0, step_id=step, bump_hit=True)
        mem.step_decay(step_id=step, exclude_idx=0)

    assert mem.forgotten_count == 1
    assert mem.count == 1
    assert mem.long_term_count == 1
    assert mem.is_long_term(0) is True
    str0, hits0 = mem.exposure_of(0)
    assert str0 > 0.54
    assert hits0 == 5
    assert mem.resonate(v0)[0] > 0.99

    save_file = tmp_path / "episodic.npz"
    mem.save(save_file)

    reloaded = EpisodicAttractorMemory(latent_dim=16)
    assert reloaded.load(save_file, inter_session_decay=0.90) is True
    assert reloaded.count == 1
    assert reloaded.long_term_count == 1
    assert reloaded.forgotten_count == 1
    assert reloaded.resonate(v0)[0] > 0.99
    assert reloaded.exposure_of(0)[1] == hits0


def test_long_term_memory_anchors_across_thousands_of_steps_and_reloads(tmp_path: Path) -> None:
    mem = EpisodicAttractorMemory(
        latent_dim=16,
        consolidation_threshold=0.85,
        init_strength=0.65,
        decay_rate=0.01,
        forget_threshold=0.18,
        grace_steps=5,
        long_term_hits=4,
        long_term_strength=0.95,
        ltm_floor=0.35,
    )
    v_ltm = F.normalize(torch.eye(16)[0], dim=0)
    v_stm = F.normalize(torch.eye(16)[1], dim=0)

    mem.consolidate(v_ltm, step_id=1)
    for step in (2, 4, 6):
        mem.note_recognition(0, step_id=step, bump_hit=True)
    assert mem.is_long_term(0) is True

    # Short-term item formed near end of session stays unpromoted
    mem.consolidate(v_stm, step_id=20)
    assert mem.is_long_term(1) is False

    # Advance 2,000 unvisited steps: STM is pruned quickly, LTM stays anchored >= ltm_floor
    for s in range(21, 2021):
        mem.step_decay(step_id=s)

    assert mem.forgotten_count == 1
    assert mem.count == 1
    assert mem.long_term_count == 1
    assert mem.exposure_of(0)[0] >= mem.ltm_floor - 1e-6

    # Persist & reload 10 sessions in a row without re-seeing v_ltm; still survives at >= ltm_floor
    mem_path = tmp_path / "ltm_persist.npz"
    cur = mem
    for _ in range(10):
        cur.save(mem_path)
        nxt = EpisodicAttractorMemory(
            latent_dim=16,
            decay_rate=0.01,
            forget_threshold=0.18,
            grace_steps=5,
            long_term_hits=4,
            ltm_floor=0.35,
        )
        assert nxt.load(mem_path, inter_session_decay=0.90) is True
        cur = nxt

    assert cur.count == 1
    assert cur.long_term_count == 1
    assert cur.exposure_of(0)[0] >= cur.ltm_floor - 1e-6

    # Verify cross-session relative age: STM nearly forgotten at end of session 1 continues aging at step 1
    stm_age_mem = EpisodicAttractorMemory(
        latent_dim=16,
        init_strength=0.30,
        decay_rate=0.08,
        forget_threshold=0.18,
        grace_steps=5,
    )
    stm_age_mem.consolidate(v_stm, step_id=1)
    for s in range(2, 100):
        stm_age_mem.step_decay(step_id=s, exclude_idx=0)  # advance cursor to 99 while keeping idx 0 unvisited near 1
    # Make strength 0.25 and last_step 100 steps behind cursor 200
    stm_age_mem.consolidate(v_ltm, step_id=200)
    stm_age_mem.note_recognition(0, step_id=100, bump_hit=False)
    stm_age_mem.save(mem_path)

    reloaded_stm = EpisodicAttractorMemory(
        latent_dim=16,
        init_strength=0.30,
        decay_rate=0.08,
        forget_threshold=0.18,
        grace_steps=5,
    )
    reloaded_stm.load(mem_path, inter_session_decay=0.75)
    # Age at step 1 of new session is 1 - (100 - 200) = 101 > grace_steps(5), so decays immediately
    pruned = reloaded_stm.step_decay(step_id=1)
    assert pruned == 1


def test_workspace_familiarity_boredom_rises_and_resets_on_novelty() -> None:
    thalamus = ThalamusWorkspace(memory=EpisodicAttractorMemory(latent_dim=16, consolidation_threshold=0.82))
    v_a = F.normalize(torch.eye(16)[0], dim=0)
    v_b = F.normalize(torch.eye(16)[1], dim=0)
    v_c = F.normalize(torch.eye(16)[2], dim=0)

    def _pkt(vec: torch.Tensor, surprise: float, ts: float) -> SensoryPacket:
        return SensoryPacket(
            region_name="vision",
            timestamp=ts,
            latent_state=vec,
            surprise_energy=surprise,
            motion_energy=0.05,
            salience_peak_xy=(0.0, 0.0),
            gaze=GazeState(),
            modality_event="reflex",
            is_fresh_semantic=True,
        )

    thalamus.process_packet(_pkt(v_a, 0.25, 1.0))
    thalamus.process_packet(_pkt(v_b, 0.25, 1.1))

    # Repeated visits across A and B build cumulative familiarity boredom
    state = thalamus.process_packet(_pkt(v_a, 0.06, 1.2))
    for i in range(4):
        thalamus.process_packet(_pkt(v_b, 0.06, 1.3 + i * 0.2))
        state = thalamus.process_packet(_pkt(v_a, 0.06, 1.4 + i * 0.2))
    for j in range(25):
        state = thalamus.process_packet(_pkt(v_a, 0.04, 3.0 + j * 0.05))

    boredom_after_repeat = state.familiarity_boredom
    assert boredom_after_repeat > 0.25

    # Brand new high-surprise pattern C sharply lowers boredom
    novel_state = thalamus.process_packet(_pkt(v_c, 0.35, 10.0))
    assert novel_state.familiarity_boredom < boredom_after_repeat * 0.60
