"""Continuous Episodic Attractor Memory hosted in system RAM (no discrete string labels)."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import torch

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    faiss = None


class EpisodicAttractorMemory:
    """Stores continuous latent attractors with biological consolidation, forgetting, and disk persistence."""

    def __init__(
        self,
        latent_dim: int = 384,
        max_episodes: int = 50_000,
        consolidation_threshold: float = 0.90,
        hebbian_rate: float = 0.15,
        init_strength: float = 0.65,
        decay_rate: float = 0.0025,
        forget_threshold: float = 0.18,
        grace_steps: int = 30,
        long_term_hits: int = 4,
        long_term_strength: float = 0.95,
        ltm_floor: float = 0.35,
    ) -> None:
        self.latent_dim = latent_dim
        self.max_episodes = max_episodes
        self.consolidation_threshold = consolidation_threshold
        self.hebbian_rate = hebbian_rate
        self.init_strength = init_strength
        self.decay_rate = decay_rate
        self.forget_threshold = forget_threshold
        self.grace_steps = max(0, int(grace_steps))
        self.long_term_hits = max(2, int(long_term_hits))
        self.long_term_strength = float(long_term_strength)
        self.ltm_floor = min(0.95, max(float(ltm_floor), float(forget_threshold) + 0.05))

        self._vectors = np.zeros((max_episodes, latent_dim), dtype=np.float32)
        self._strength = np.zeros(max_episodes, dtype=np.float32)
        self._hits = np.zeros(max_episodes, dtype=np.int32)
        self._last_step = np.zeros(max_episodes, dtype=np.int64)
        self._long_term = np.zeros(max_episodes, dtype=bool)
        self._count = 0
        self._step_cursor = 0
        self.forgotten_count = 0
        self._faiss_index = faiss.IndexFlatIP(latent_dim) if faiss is not None else None
        self._lock = threading.RLock()

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    @property
    def long_term_count(self) -> int:
        with self._lock:
            if self._count == 0:
                return 0
            return int(np.count_nonzero(self._long_term[: self._count]))

    def is_long_term(self, idx: int) -> bool:
        with self._lock:
            if 0 <= idx < self._count:
                return bool(self._long_term[idx])
            return False

    def _promote_if_ready(self, idx: int) -> None:
        if (
            self._hits[idx] >= self.long_term_hits
            or float(self._strength[idx]) >= self.long_term_strength
        ):
            self._long_term[idx] = True

    @staticmethod
    def _to_unit_np(latent: torch.Tensor) -> np.ndarray:
        vec = latent.detach().to("cpu", dtype=torch.float32).flatten().numpy()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32, copy=False)

    def resonate(self, latent: torch.Tensor) -> tuple[float, int]:
        """Return (max_cosine_similarity, best_index) against stored latent attractors."""
        query = self._to_unit_np(latent)
        with self._lock:
            if self._count == 0:
                return 0.0, -1

            if self._faiss_index is not None and self._faiss_index.ntotal == self._count:
                sims, idxs = self._faiss_index.search(query[None, :], 1)
                return float(np.clip(sims[0, 0], 0.0, 1.0)), int(idxs[0, 0])

            sims = self._vectors[: self._count] @ query
            best_idx = int(np.argmax(sims))
            return float(np.clip(sims[best_idx], 0.0, 1.0)), best_idx

    def exposure_of(self, idx: int) -> tuple[float, int]:
        """Return (strength, hit_count) for an active attractor index."""
        with self._lock:
            if 0 <= idx < self._count:
                return float(self._strength[idx]), int(self._hits[idx])
            return 0.0, 0

    def note_recognition(self, idx: int, step_id: int | None = None, bump_hit: bool = False) -> None:
        """Refresh recency and lightly reinforce a recognized attractor without FAISS rebuild."""
        with self._lock:
            if not (0 <= idx < self._count):
                return
            step = self._step_cursor if step_id is None else int(step_id)
            self._last_step[idx] = max(int(self._last_step[idx]), step)
            s = float(self._strength[idx])
            self._strength[idx] = min(1.0, s + 0.03 * (1.0 - s))
            if bump_hit:
                self._hits[idx] += 1
            self._promote_if_ready(idx)

    def consolidate(self, latent: torch.Tensor, step_id: int | None = None) -> tuple[float, bool]:
        """Resonate with existing attractors; deepen/blend if familiar, append or evict weakest if novel."""
        unit = self._to_unit_np(latent)
        if float(np.linalg.norm(unit)) <= 1e-8:
            return 0.0, False

        with self._lock:
            step = self._step_cursor if step_id is None else int(step_id)
            self._step_cursor = max(self._step_cursor, step)
            resonance, best_idx = self.resonate(latent)
            if best_idx >= 0 and resonance >= self.consolidation_threshold:
                # Hebbian prototype refinement + synaptic strengthening
                blended = (1.0 - self.hebbian_rate) * self._vectors[best_idx] + self.hebbian_rate * unit
                norm = float(np.linalg.norm(blended))
                if norm > 1e-8:
                    self._vectors[best_idx] = blended / norm
                s = float(self._strength[best_idx])
                self._strength[best_idx] = min(1.0, s + 0.22 * (1.0 - s))
                self._hits[best_idx] += 1
                self._last_step[best_idx] = step
                self._promote_if_ready(best_idx)
                if self._faiss_index is not None:
                    self._rebuild_faiss()
                return resonance, False

            if self._count < self.max_episodes:
                slot = self._count
                self._count += 1
                self._vectors[slot] = unit
                self._strength[slot] = self.init_strength
                self._hits[slot] = 1
                self._last_step[slot] = step
                self._long_term[slot] = False
                if self._faiss_index is not None:
                    self._faiss_index.add(unit[None, :])
                return resonance, True

            # Evict the weakest short-term (then weakest LTM) slot only when capacity is full
            evict_scores = self._strength[: self._count] + np.where(self._long_term[: self._count], 1.0, 0.0)
            evict_idx = int(np.argmin(evict_scores))
            self._vectors[evict_idx] = unit
            self._strength[evict_idx] = self.init_strength
            self._hits[evict_idx] = 1
            self._last_step[evict_idx] = step
            self._long_term[evict_idx] = False
            self.forgotten_count += 1
            if self._faiss_index is not None:
                self._rebuild_faiss()
            return resonance, True

    def step_decay(self, step_id: int | None = None, exclude_idx: int = -1) -> int:
        """Apply Ebbinghaus-like forgetting decay while protecting consolidated long-term attractors."""
        with self._lock:
            if step_id is None:
                self._step_cursor += 1
                step = self._step_cursor
            else:
                step = int(step_id)
                self._step_cursor = max(self._step_cursor, step)

            n = self._count
            if n == 0 or self.decay_rate <= 0.0:
                return 0

            ages = step - self._last_step[:n]
            decay_mask = ages > self.grace_steps
            if 0 <= exclude_idx < n:
                decay_mask[exclude_idx] = False
            if not np.any(decay_mask):
                return 0

            hits = self._hits[:n][decay_mask].astype(np.float32)
            is_ltm = self._long_term[:n][decay_mask]
            # Short-term fades fast unless revisited; consolidated long-term decays ultra-slowly to ltm_floor
            eff_decay_stm = self.decay_rate / (1.0 + 1.2 * np.power(np.maximum(0.0, hits - 1.0), 1.5))
            eff_decay_ltm = (self.decay_rate * 0.03) / (1.0 + 0.5 * hits)
            eff_decay = np.where(is_ltm, eff_decay_ltm, eff_decay_stm)

            decayed = self._strength[:n][decay_mask] - eff_decay
            self._strength[:n][decay_mask] = np.where(
                is_ltm, np.maximum(decayed, self.ltm_floor), decayed
            )

            keep_mask = (self._strength[:n] >= self.forget_threshold) | self._long_term[:n]
            if 0 <= exclude_idx < n:
                keep_mask[exclude_idx] = True

            kept = int(np.count_nonzero(keep_mask))
            pruned = n - kept
            if pruned > 0:
                self._vectors[:kept] = self._vectors[:n][keep_mask]
                self._strength[:kept] = self._strength[:n][keep_mask]
                self._hits[:kept] = self._hits[:n][keep_mask]
                self._last_step[:kept] = self._last_step[:n][keep_mask]
                self._long_term[:kept] = self._long_term[:n][keep_mask]
                self._count = kept
                self.forgotten_count += pruned
                if self._faiss_index is not None:
                    self._rebuild_faiss()
            return pruned

    def save(self, path: Path | str) -> None:
        """Persist current episodic attractors and LTM metadata to disk (.npz)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            n = self._count
            np.savez_compressed(
                target,
                latent_dim=np.array([self.latent_dim], dtype=np.int32),
                vectors=self._vectors[:n],
                strength=self._strength[:n],
                hits=self._hits[:n],
                last_step=self._last_step[:n],
                long_term=self._long_term[:n],
                forgotten_count=np.array([self.forgotten_count], dtype=np.int64),
                step_cursor=np.array([self._step_cursor], dtype=np.int64),
            )

    def load(self, path: Path | str, inter_session_decay: float = 0.90) -> bool:
        """Load saved attractors, normalize recency to step 0, and anchor consolidated LTM."""
        target = Path(path)
        if not target.exists():
            return False
        with np.load(target) as data:
            saved_dim = int(data["latent_dim"][0])
            if saved_dim != self.latent_dim:
                raise ValueError(
                    f"Saved memory latent_dim={saved_dim} does not match expected {self.latent_dim}"
                )
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            strength = np.asarray(data["strength"], dtype=np.float32) * float(inter_session_decay)
            hits = np.asarray(data["hits"], dtype=np.int32)
            last_step = np.asarray(data["last_step"], dtype=np.int64)
            long_term = (
                np.asarray(data["long_term"], dtype=bool)
                if "long_term" in data
                else (hits >= self.long_term_hits)
            )
            forgotten = int(data["forgotten_count"][0]) if "forgotten_count" in data else 0
            cursor = int(data["step_cursor"][0]) if "step_cursor" in data else 0

        strength = np.where(long_term, np.maximum(strength, self.ltm_floor), strength)
        rel_last = np.minimum(last_step, cursor) - cursor

        keep_mask = (strength >= self.forget_threshold) | long_term
        kept_vectors = vectors[keep_mask]
        kept_strength = strength[keep_mask]
        kept_hits = hits[keep_mask]
        kept_last = rel_last[keep_mask]
        kept_ltm = long_term[keep_mask]
        session_pruned = int(len(vectors) - len(kept_vectors))

        with self._lock:
            n = min(len(kept_vectors), self.max_episodes)
            self._count = n
            self._strength.fill(0.0)
            self._hits.fill(0)
            self._last_step.fill(0)
            self._long_term.fill(False)
            if n > 0:
                self._vectors[:n] = kept_vectors[:n]
                self._strength[:n] = kept_strength[:n]
                self._hits[:n] = kept_hits[:n]
                self._last_step[:n] = kept_last[:n]
                self._long_term[:n] = kept_ltm[:n]
            self._step_cursor = 0
            self.forgotten_count = forgotten + session_pruned
            if self._faiss_index is not None:
                self._rebuild_faiss()
        return True

    def _rebuild_faiss(self) -> None:
        if self._faiss_index is None:
            return
        self._faiss_index.reset()
        if self._count > 0:
            self._faiss_index.add(self._vectors[: self._count])

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._step_cursor = 0
            self.forgotten_count = 0
            self._strength.fill(0.0)
            self._hits.fill(0)
            self._last_step.fill(0)
            self._long_term.fill(False)
            if self._faiss_index is not None:
                self._faiss_index.reset()
