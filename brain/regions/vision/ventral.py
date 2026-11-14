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
        self._foveal_weight_cache: dict[tuple[int, str], torch.Tensor] = {}

    def _get_foveal_weights(self, side: int, device: torch.device) -> torch.Tensor:
        key = (side, str(device))
        cached = self._foveal_weight_cache.get(key)
        if cached is not None:
            return cached
        coords = torch.linspace(-1.0, 1.0, side, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        w = torch.exp(-(xx**2 + yy**2) / (2.0 * 0.48 * 0.48)).flatten()
        w = w / w.sum()
        self._foveal_weight_cache[key] = w
        return w

    @torch.no_grad()
    def encode(
        self,
        fovea: torch.Tensor,
        surprise_energy: float,
        gaze_vx: float = 0.0,
        gaze_vy: float = 0.0,
    ) -> torch.Tensor:
        """Encode sharp fovea content into pure semantic trajectory (kept free of raw gaze speed)."""
        if self._dinov2 is not None:
            normed = (fovea - self._mean) / self._std
            use_amp = getattr(self, "_use_amp", None)
            if use_amp is None:
                if self.device.type == "cuda":
                    major, minor = torch.cuda.get_device_capability(self.device)
                    dev_name = torch.cuda.get_device_name(self.device).upper()
                    # GTX GPUs (incl. GTX 16xx sm_75) lack Tensor Cores and run ~1.7x faster in pure FP32
                    use_amp = (major, minor) >= (7, 5) and "GTX" not in dev_name
                else:
                    use_amp = False
                self._use_amp = use_amp
            with torch.amp.autocast(device_type=self.device.type, dtype=torch.float16, enabled=use_amp):
                out = self._dinov2(pixel_values=normed)
                cls_tok = out.last_hidden_state[:, 0]
                num_reg = int(getattr(self._dinov2.config, "num_register_tokens", 0))
                patch_tokens = out.last_hidden_state[:, 1 + num_reg :]
                n_p = patch_tokens.shape[1]
                side = int(round(n_p**0.5))
                if side * side == n_p:
                    # Weight center foveal patches higher than peripheral corner clutter
                    fw = self._get_foveal_weights(side, patch_tokens.device).to(patch_tokens.dtype)
                    patch_vec = (patch_tokens * fw.view(1, n_p, 1)).sum(dim=1)
                else:
                    patch_vec = patch_tokens.mean(dim=1)
                raw_vec = (0.55 * cls_tok + 0.45 * patch_vec).float()
        else:
            assert self._light_encoder is not None
            raw_vec = self._light_encoder(fovea).float()

        semantic = F.normalize(raw_vec.squeeze(0), dim=-1)

        if self._bound_state is None:
            self._bound_state = semantic
        else:
            # High surprise updates bound perception faster; calm pauses preserve continuity
            alpha = min(0.55, max(0.12, 0.12 + 2.5 * surprise_energy))
            self._bound_state = F.normalize(
                (1.0 - alpha) * self._bound_state + alpha * semantic, dim=-1
            )
        return self._bound_state.detach().to("cpu")

    def reset(self) -> None:
        self._bound_state = None
