"""Continuous Episodic Attractor Memory hosted in system RAM (no discrete string labels)."""

from __future__ import annotations

import numpy as np
import torch

try:
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional acceleration
    faiss = None


class EpisodicAttractorMemory:
    """Stores continuous latent perceptual attractors in RAM via cosine resonance."""

    def __init__(
        self,
        latent_dim: int = 384,
        max_episodes: int = 50_000,
        consolidation_threshold: float = 0.90,
        hebbian_rate: float = 0.15,
    ) -> None:
        self.latent_dim = latent_dim
        self.max_episodes = max_episodes
        self.consolidation_threshold = consolidation_threshold
        self.hebbian_rate = hebbian_rate

        self._vectors = np.zeros((max_episodes, latent_dim), dtype=np.float32)
        self._count = 0
        self._faiss_index = faiss.IndexFlatIP(latent_dim) if faiss is not None else None

    @property
    def count(self) -> int:
        return self._count

    @staticmethod
    def _to_unit_np(latent: torch.Tensor) -> np.ndarray:
        vec = latent.detach().to("cpu", dtype=torch.float32).flatten().numpy()
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm
        return vec.astype(np.float32, copy=False)

    def resonate(self, latent: torch.Tensor) -> tuple[float, int]:
        """Return (max_cosine_similarity, best_index) against stored latent attractors."""
        if self._count == 0:
            return 0.0, -1

        query = self._to_unit_np(latent)
        if self._faiss_index is not None and self._faiss_index.ntotal == self._count:
            sims, idxs = self._faiss_index.search(query[None, :], 1)
            return float(np.clip(sims[0, 0], 0.0, 1.0)), int(idxs[0, 0])

        sims = self._vectors[: self._count] @ query
        best_idx = int(np.argmax(sims))
        return float(np.clip(sims[best_idx], 0.0, 1.0)), best_idx

    def consolidate(self, latent: torch.Tensor) -> tuple[float, bool]:
        """Resonate with existing attractors; blend if familiar, append if novel."""
        unit = self._to_unit_np(latent)
        if float(np.linalg.norm(unit)) <= 1e-8:
            return 0.0, False

        resonance, best_idx = self.resonate(latent)
        if best_idx >= 0 and resonance >= self.consolidation_threshold:
            # Hebbian prototype refinement: deepen familiar attractor instead of duplicating
            blended = (1.0 - self.hebbian_rate) * self._vectors[best_idx] + self.hebbian_rate * unit
            norm = float(np.linalg.norm(blended))
            if norm > 1e-8:
                self._vectors[best_idx] = blended / norm
            if self._faiss_index is not None:
                self._rebuild_faiss()
            return resonance, False

        if self._count < self.max_episodes:
            self._vectors[self._count] = unit
            self._count += 1
            if self._faiss_index is not None:
                self._faiss_index.add(unit[None, :])
            return resonance, True

        # Replace weakest/oldest slot cleanly once capacity is full
        evict_idx = self._count - 1
        self._vectors[evict_idx] = unit
        if self._faiss_index is not None:
            self._rebuild_faiss()
        return resonance, True

    def _rebuild_faiss(self) -> None:
        if self._faiss_index is None:
            return
        self._faiss_index.reset()
        if self._count > 0:
            self._faiss_index.add(self._vectors[: self._count])

    def reset(self) -> None:
        self._count = 0
        if self._faiss_index is not None:
            self._faiss_index.reset()
