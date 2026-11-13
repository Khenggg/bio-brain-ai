"""Dual-resolution foveated retina compressing raw frames into periphery + movable fovea."""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from brain.core.types import GazeState


class DualResolutionRetina:
    """Mimics biological periphery (wide, coarse, motion-sensitive) + fovea (narrow, sharp)."""

    def __init__(
        self,
        periphery_size: int = 96,
        fovea_size: int = 140,
        fovea_span: float = 0.44,
        device: torch.device | str = "cpu",
    ) -> None:
        self.periphery_size = periphery_size
        self.fovea_size = fovea_size
        self.fovea_span = fovea_span
        self.device = torch.device(device)

        # Precompute base normalized grid in [-1, 1] for fast tensor foveal sampling
        lin = torch.linspace(-1.0, 1.0, fovea_size, device=self.device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1, F, F, 2]

    def to_tensor(self, frame: np.ndarray | torch.Tensor, bgr_input: bool = True) -> torch.Tensor:
        """Convert uint8 HxWx3 or float tensor into [1, 3, H, W] RGB tensor in [0, 1]."""
        if isinstance(frame, torch.Tensor):
            t = frame.to(self.device, dtype=torch.float32)
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.max() > 1.0:
                t = t / 255.0
            return t.clamp(0.0, 1.0)

        arr = frame
        if bgr_input and arr.ndim == 3 and arr.shape[2] == 3:
            arr = arr[:, :, ::-1].copy()
        t = torch.from_numpy(arr).to(self.device, dtype=torch.float32) / 255.0
        t = t.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
        return t.clamp(0.0, 1.0)

    def _compute_spans(self, h: int, w: int) -> tuple[float, float]:
        if w >= h:
            return self.fovea_span * (h / float(w)), self.fovea_span
        return self.fovea_span, self.fovea_span * (w / float(h))

    def _np_to_gpu_tensor(self, img_np: np.ndarray, bgr_input: bool) -> torch.Tensor:
        if bgr_input:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device, dtype=torch.float32)
        return (t / 255.0).clamp(0.0, 1.0)

    def sample(
        self, frame: np.ndarray | torch.Tensor, gaze: GazeState, bgr_input: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float]]:
        """Return (periphery [1,3,P,P], fovea [1,3,F,F], (span_x, span_y))."""
        if isinstance(frame, np.ndarray) and frame.ndim == 3 and frame.dtype == np.uint8:
            h, w = frame.shape[:2]
            span_x, span_y = self._compute_spans(h, w)
            cx = float(np.clip(gaze.x, -1.0 + span_x * 0.5, 1.0 - span_x * 0.5))
            cy = float(np.clip(gaze.y, -1.0 + span_y * 0.5, 1.0 - span_y * 0.5))

            # Fast CPU resize/crop avoids uploading full 1080p float32 frame over PCIe every tick
            periph_np = cv2.resize(
                frame, (self.periphery_size, self.periphery_size), interpolation=cv2.INTER_AREA
            )
            half_wx = max(8, int(round(span_x * 0.5 * w)))
            half_wy = max(8, int(round(span_y * 0.5 * h)))
            px_c = int(round((cx + 1.0) * 0.5 * (w - 1)))
            py_c = int(round((cy + 1.0) * 0.5 * (h - 1)))

            x0, x1 = px_c - half_wx, px_c + half_wx
            y0, y1 = py_c - half_wy, py_c + half_wy
            pad_l = max(0, -x0)
            pad_t = max(0, -y0)
            pad_r = max(0, x1 - w)
            pad_b = max(0, y1 - h)
            crop = frame[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
            if pad_l or pad_t or pad_r or pad_b:
                crop = cv2.copyMakeBorder(crop, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
            fovea_np = cv2.resize(
                crop, (self.fovea_size, self.fovea_size), interpolation=cv2.INTER_LINEAR
            )
            return (
                self._np_to_gpu_tensor(periph_np, bgr_input),
                self._np_to_gpu_tensor(fovea_np, bgr_input),
                (span_x, span_y),
            )

        img = self.to_tensor(frame, bgr_input=bgr_input)
        _, _, h, w = img.shape

        periphery = F.interpolate(
            img,
            size=(self.periphery_size, self.periphery_size),
            mode="bilinear",
            align_corners=False,
        )

        span_x, span_y = self._compute_spans(h, w)
        cx = float(np.clip(gaze.x, -1.0 + span_x * 0.5, 1.0 - span_x * 0.5))
        cy = float(np.clip(gaze.y, -1.0 + span_y * 0.5, 1.0 - span_y * 0.5))

        grid = self.register_grid.clone()
        grid[..., 0] = cx + grid[..., 0] * span_x
        grid[..., 1] = cy + grid[..., 1] * span_y

        fovea = F.grid_sample(
            img,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return periphery, fovea, (span_x, span_y)
