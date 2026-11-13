"""Ventral ("What") stream: self-supervised latent representation + temporal binding without labels."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LightVentralEncoder(nn.Module):
    """Fallback compact multi-scale convolutional encoder producing 384-d latent vectors."""

    def __init__(self, out_dim: int = 384) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, fovea: torch.Tensor) -> torch.Tensor:
        return self.net(fovea).flatten(1)


class VentralSemanticStream:
    """Encodes sharp foveal view into continuous self-supervised latent attractors."""

    def __init__(
        self,
        latent_dim: int = 384,
        device: torch.device | str = "cpu",
        backend: str = "auto",
        model_dir: Path | str | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.backend_name = "light"
        self._dinov2 = None
        self._light_encoder: _LightVentralEncoder | None = None

        default_model_dir = Path(__file__).resolve().parents[3] / "models" / "dinov2-with-registers-small"
        resolved_dir = Path(model_dir) if model_dir is not None else default_model_dir

        if backend in ("auto", "dinov2") and (resolved_dir / "model.safetensors").exists():
            try:
                from transformers import AutoModel  # type: ignore

                self._dinov2 = AutoModel.from_pretrained(
                    str(resolved_dir),
                    local_files_only=True,
                ).to(self.device)
                self._dinov2.eval()
                self.backend_name = "dinov2-s-reg"
            except Exception:
                self._dinov2 = None

        if self._dinov2 is None:
            self._light_encoder = _LightVentralEncoder(out_dim=latent_dim).to(self.device)
            self._light_encoder.eval()
            self.backend_name = "light-conv"

        # ImageNet normalization constants for DINOv2
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Temporal binding attractor (replaces discrete object IDs with continuous presence)
        self._bound_state: torch.Tensor | None = None

    @torch.no_grad()
    def encode(
        self,
        fovea: torch.Tensor,
        surprise_energy: float,
        gaze_vx: float,
        gaze_vy: float,
    ) -> torch.Tensor:
        """Encode sharp fovea patch and integrate into continuous temporal latent trajectory."""
        if self._dinov2 is not None:
            normed = (fovea - self._mean) / self._std
            use_amp = self.device.type == "cuda"
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self._dinov2(pixel_values=normed)
                # Blend CLS token + mean patch representation for rich local + global shape
                cls_tok = out.last_hidden_state[:, 0]
                patch_mean = out.last_hidden_state[:, 1:].mean(dim=1)
                raw_vec = (0.6 * cls_tok + 0.4 * patch_mean).float()
        else:
            assert self._light_encoder is not None
            raw_vec = self._light_encoder(fovea).float()

        # Fold subtle motion-kinematic resonance into latent tail without destroying semantic similarity
        raw_vec = F.normalize(raw_vec.squeeze(0), dim=-1)
        motion_sig = torch.zeros_like(raw_vec)
        motion_sig[0] = float(gaze_vx) * 2.0
        motion_sig[1] = float(gaze_vy) * 2.0
        motion_sig[2] = float(surprise_energy)
        combined = F.normalize(raw_vec + 0.08 * motion_sig, dim=-1)

        if self._bound_state is None:
            self._bound_state = combined
        else:
            # High surprise updates bound perception faster; calm pauses preserve continuity
            alpha = min(0.55, max(0.12, 0.12 + 2.5 * surprise_energy))
            self._bound_state = F.normalize(
                (1.0 - alpha) * self._bound_state + alpha * combined, dim=-1
            )
        return self._bound_state.detach().to("cpu")

    def reset(self) -> None:
        self._bound_state = None
