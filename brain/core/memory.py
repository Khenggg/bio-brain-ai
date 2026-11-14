"""Continuous Episodic Attractor Memory hosted in system RAM (no discrete string labels)."""

from __future__ import annotations

import os
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
        decay_rate: float = 0.00006,
        forget_threshold: float = 0.18,
        grace_steps: int = 450,
        long_term_hits: int = 4,
        long_term_strength: float = 0.95,
        ltm_floor: float = 0.35,
        transition_decay: float = 0.0006,
        transition_prune_threshold: float = 0.15,
        merge_threshold: float = 0.96,
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
        self.transition_decay = min(0.25, max(0.0, float(transition_decay)))
        self.transition_prune_threshold = max(0.01, float(transition_prune_threshold))
        self.merge_threshold = float(merge_threshold)

        self._vectors = np.zeros((max_episodes, latent_dim), dtype=np.float32)
        self._strength = np.zeros(max_episodes, dtype=np.float32)
        self._hits = np.zeros(max_episodes, dtype=np.int32)
        self._last_step = np.zeros(max_episodes, dtype=np.int64)
        self._long_term = np.zeros(max_episodes, dtype=bool)
        self._count = 0
        self._step_cursor = 0
        self.forgotten_count = 0
        self.merged_count = 0
        self.faiss_min_count = 512
        self._faiss_index = faiss.IndexFlatIP(latent_dim) if faiss is not None else None
        self._faiss_dirty = False
        self._transitions: dict[int, dict[int, float]] = {}
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

    @property
    def transition_count(self) -> int:
        with self._lock:
            return sum(len(edges) for edges in self._transitions.values())

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

    def _remap_transitions(self, index_map: dict[int, int]) -> None:
        remapped: dict[int, dict[int, float]] = {}
        for src, edges in self._transitions.items():
            new_src = index_map.get(src)
            if new_src is None:
                continue
            dest_edges = remapped.setdefault(new_src, {})
            for dst, w in edges.items():
                new_dst = index_map.get(dst)
                if new_dst is not None and new_dst != new_src:
                    dest_edges[new_dst] = min(100.0, dest_edges.get(new_dst, 0.0) + float(w))
            while len(dest_edges) > 16:
                weakest = min(dest_edges, key=dest_edges.__getitem__)
                del dest_edges[weakest]
            if not dest_edges:
                del remapped[new_src]
        self._transitions = remapped

    def _merge_if_redundant(self, idx: int) -> int:
        """Collapse two attractors into one if Hebbian drift pushed them closer than merge_threshold."""
        n = self._count
        if n <= 1 or self.merge_threshold >= 1.0 or not (0 <= idx < n):
            return idx
        sims = self._vectors[:n] @ self._vectors[idx]
        sims[idx] = -2.0
        other = int(np.argmax(sims))
        if float(sims[other]) < self.merge_threshold:
            return idx

        def _prio(i: int) -> tuple[bool, float, int, int]:
            return (bool(self._long_term[i]), float(self._strength[i]), int(self._hits[i]), -i)

        keep, drop = (idx, other) if _prio(idx) >= _prio(other) else (other, idx)
        w_keep = max(1e-3, float(self._strength[keep]))
        w_drop = max(1e-3, float(self._strength[drop]))
        blended = w_keep * self._vectors[keep] + w_drop * self._vectors[drop]
        norm = float(np.linalg.norm(blended))
        if norm > 1e-8:
            self._vectors[keep] = blended / norm
        self._hits[keep] = int(self._hits[keep]) + int(self._hits[drop])
        self._strength[keep] = min(1.0, max(float(self._strength[keep]), float(self._strength[drop])) + 0.05)
        self._last_step[keep] = max(int(self._last_step[keep]), int(self._last_step[drop]))
        self._long_term[keep] = bool(self._long_term[keep] or self._long_term[drop])
        self._promote_if_ready(keep)

        survivor = keep - (1 if drop < keep else 0)
        index_map = {
            i: (survivor if i == drop else (i if i < drop else i - 1))
            for i in range(n)
        }
        self._remap_transitions(index_map)

        if drop < n - 1:
            self._vectors[drop : n - 1] = self._vectors[drop + 1 : n]
            self._strength[drop : n - 1] = self._strength[drop + 1 : n]
            self._hits[drop : n - 1] = self._hits[drop + 1 : n]
            self._last_step[drop : n - 1] = self._last_step[drop + 1 : n]
            self._long_term[drop : n - 1] = self._long_term[drop + 1 : n]
        self._count = n - 1
        self.merged_count += 1
        if self._count >= self.faiss_min_count:
            self._faiss_dirty = True
        return survivor

    def note_transition(self, prev_idx: int, curr_idx: int, weight: float = 1.0) -> None:
        """Record directed temporal link prev_idx -> curr_idx in episodic sequence memory."""
        with self._lock:
            if (
                prev_idx == curr_idx
                or not (0 <= prev_idx < self._count)
                or not (0 <= curr_idx < self._count)
            ):
                return
            edges = self._transitions.setdefault(prev_idx, {})
            edges[curr_idx] = min(100.0, edges.get(curr_idx, 0.0) + float(weight))
            if len(edges) > 16:
                weakest = min(edges, key=edges.__getitem__)
                del edges[weakest]

    def predict_next(self, idx: int) -> tuple[int, float]:
        """Predict most likely next attractor from idx with Laplace-smoothed confidence in [0, 1)."""
        with self._lock:
            edges = self._transitions.get(idx)
            if not edges:
                return -1, 0.0
            total = sum(edges.values())
            best_next = max(edges, key=edges.__getitem__)
            # +1.0 prior prevents 1-shot coincidental transitions from claiming 100% certainty
            confidence = float(edges[best_next] / (total + 1.0))
            return best_next, confidence

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

            if self._faiss_index is not None and self._count >= self.faiss_min_count:
                if self._faiss_dirty or self._faiss_index.ntotal != self._count:
                    self._rebuild_faiss()
                    self._faiss_dirty = False
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

    def consolidate(
        self,
        latent: torch.Tensor,
        step_id: int | None = None,
        init_strength: float | None = None,
    ) -> tuple[float, bool]:
        """Resonate with existing attractors; deepen/blend if familiar, append or evict weakest if novel."""
        unit = self._to_unit_np(latent)
        if float(np.linalg.norm(unit)) <= 1e-8:
            return 0.0, False

        start_strength = (
            self.init_strength
            if init_strength is None
            else float(np.clip(init_strength, min(1.0, self.forget_threshold + 0.02), 1.0))
        )

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
                self._merge_if_redundant(best_idx)
                if self._count >= self.faiss_min_count:
                    self._faiss_dirty = True
                return resonance, False

            if self._count < self.max_episodes:
                slot = self._count
                self._count += 1
                self._vectors[slot] = unit
                self._strength[slot] = start_strength
                self._hits[slot] = 1
                self._last_step[slot] = step
                self._long_term[slot] = False
                if self._faiss_index is not None and self._count >= self.faiss_min_count:
                    if not self._faiss_dirty and self._faiss_index.ntotal == slot:
                        self._faiss_index.add(unit[None, :])
                    else:
                        self._faiss_dirty = True
                return resonance, True

            # Evict the weakest short-term (then weakest LTM) slot only when capacity is full
            evict_scores = self._strength[: self._count] + np.where(self._long_term[: self._count], 1.0, 0.0)
            evict_idx = int(np.argmin(evict_scores))
            self._vectors[evict_idx] = unit
            self._strength[evict_idx] = start_strength
            self._hits[evict_idx] = 1
            self._last_step[evict_idx] = step
            self._long_term[evict_idx] = False
            self._remap_transitions({i: i for i in range(self._count) if i != evict_idx})
            self.forgotten_count += 1
            if self._count >= self.faiss_min_count:
                self._faiss_dirty = True
            return resonance, True

    def _decay_transitions(self) -> None:
        """Gradually fade unvisited sequence edges and prune accidental one-shot links."""
        if not self._transitions or self.transition_decay <= 0.0:
            return
        retain = 1.0 - self.transition_decay
        thresh = self.transition_prune_threshold
        for src in list(self._transitions):
            edges = self._transitions[src]
            for dst in list(edges):
                w = edges[dst] * retain
                if w < thresh:
                    del edges[dst]
                else:
                    edges[dst] = w
            if not edges:
                del self._transitions[src]

    def step_decay(self, step_id: int | None = None, exclude_idx: int = -1) -> int:
        """Apply Ebbinghaus-like forgetting decay while protecting consolidated long-term attractors."""
        with self._lock:
            if step_id is None:
                self._step_cursor += 1
                step = self._step_cursor
            else:
                step = int(step_id)
                self._step_cursor = max(self._step_cursor, step)

            self._decay_transitions()

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
                kept_old_indices = np.flatnonzero(keep_mask)
                self._remap_transitions(
                    {int(old_i): new_i for new_i, old_i in enumerate(kept_old_indices)}
                )
                self._vectors[:kept] = self._vectors[:n][keep_mask]
                self._strength[:kept] = self._strength[:n][keep_mask]
                self._hits[:kept] = self._hits[:n][keep_mask]
                self._last_step[:kept] = self._last_step[:n][keep_mask]
                self._long_term[:kept] = self._long_term[:n][keep_mask]
                self._count = kept
                self.forgotten_count += pruned
                if self._count >= self.faiss_min_count:
                    self._faiss_dirty = True
            return pruned

    def save(self, path: Path | str) -> None:
        """Atomically persist episodic attractors, sequence transitions, and LTM metadata (.npz)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            n = self._count
            triples = [
                (src, dst, w)
                for src, edges in self._transitions.items()
                for dst, w in edges.items()
                if src < n and dst < n
            ]
            trans_src = np.array([t[0] for t in triples], dtype=np.int32)
            trans_dst = np.array([t[1] for t in triples], dtype=np.int32)
            trans_w = np.array([t[2] for t in triples], dtype=np.float32)
            tmp_path = target.with_name(f".{target.stem}.{os.getpid()}.tmp.npz")
            try:
                with open(tmp_path, "wb") as f:
                    np.savez_compressed(
                        f,
                        latent_dim=np.array([self.latent_dim], dtype=np.int32),
                        vectors=self._vectors[:n].copy(),
                        strength=self._strength[:n].copy(),
                        hits=self._hits[:n].copy(),
                        last_step=self._last_step[:n].copy(),
                        long_term=self._long_term[:n].copy(),
                        trans_src=trans_src,
                        trans_dst=trans_dst,
                        trans_w=trans_w,
                        forgotten_count=np.array([self.forgotten_count], dtype=np.int64),
                        merged_count=np.array([self.merged_count], dtype=np.int64),
                        step_cursor=np.array([self._step_cursor], dtype=np.int64),
                    )
                os.replace(tmp_path, target)
            except BaseException:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                raise

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
            trans_src = np.asarray(data["trans_src"], dtype=np.int32) if "trans_src" in data else np.empty(0, dtype=np.int32)
            trans_dst = np.asarray(data["trans_dst"], dtype=np.int32) if "trans_dst" in data else np.empty(0, dtype=np.int32)
            trans_w = np.asarray(data["trans_w"], dtype=np.float32) if "trans_w" in data else np.empty(0, dtype=np.float32)
            forgotten = int(data["forgotten_count"][0]) if "forgotten_count" in data else 0
            merged = int(data["merged_count"][0]) if "merged_count" in data else 0
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
        kept_old_indices = np.flatnonzero(keep_mask)
        index_map = {
            int(old_i): new_i
            for new_i, old_i in enumerate(kept_old_indices[: self.max_episodes])
        }

        with self._lock:
            n = min(len(kept_vectors), self.max_episodes)
            self._count = n
            self._strength.fill(0.0)
            self._hits.fill(0)
            self._last_step.fill(0)
            self._long_term.fill(False)
            self._transitions.clear()
            if n > 0:
                self._vectors[:n] = kept_vectors[:n]
                self._strength[:n] = kept_strength[:n]
                self._hits[:n] = kept_hits[:n]
                self._last_step[:n] = kept_last[:n]
                self._long_term[:n] = kept_ltm[:n]
                for s_old, d_old, w in zip(trans_src, trans_dst, trans_w):
                    ns = index_map.get(int(s_old))
                    nd = index_map.get(int(d_old))
                    if ns is not None and nd is not None and ns != nd:
                        self._transitions.setdefault(ns, {})[nd] = float(w) * float(inter_session_decay)
            self._step_cursor = 0
            self.forgotten_count = forgotten + session_pruned
            self.merged_count = merged
            self._faiss_dirty = n >= self.faiss_min_count
        return True

    def _rebuild_faiss(self) -> None:
        if self._faiss_index is None:
            return
        self._faiss_index.reset()
        if self._count > 0:
            self._faiss_index.add(self._vectors[: self._count])
        self._faiss_dirty = False

    def reset(self) -> None:
        with self._lock:
            self._count = 0
            self._step_cursor = 0
            self.forgotten_count = 0
            self.merged_count = 0
            self._faiss_dirty = False
            self._transitions.clear()
            self._strength.fill(0.0)
            self._hits.fill(0)
            self._last_step.fill(0)
            self._long_term.fill(False)
            if self._faiss_index is not None:
                self._faiss_index.reset()
