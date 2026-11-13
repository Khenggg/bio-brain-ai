"""Dorsal ("Where / Motion") stream using online Predictive Coding & Predictive Suppression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PredictiveConvGRU(nn.Module):
    """Lightweight spatio-temporal predictor predicting frame t features from t-1."""

    def __init__(self, in_ch: int = 4, hidden_ch: int = 16) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch
        self.conv_gates = nn.Conv2d(
            in_ch + hidden_ch, 2 * hidden_ch, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.conv_cand = nn.Conv2d(
            in_ch + hidden_ch, hidden_ch, kernel_size=3, padding=1, padding_mode="replicate"
        )
        self.pred_head = nn.Sequential(
            nn.Conv2d(hidden_ch, hidden_ch, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.GELU(),
            nn.Conv2d(hidden_ch, in_ch, kernel_size=1),
        )
        # Start near persistence prior (x_t ~= x_{t-1}) to avoid random initial surprise
        nn.init.zeros_(self.pred_head[-1].weight)
        nn.init.zeros_(self.pred_head[-1].bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if h is None:
            h = torch.zeros(
                x.shape[0], self.hidden_ch, x.shape[2], x.shape[3], device=x.device, dtype=x.dtype
            )
        combined = torch.cat([x, h], dim=1)
        r, z = torch.sigmoid(self.conv_gates(combined)).chunk(2, dim=1)
        cand_in = torch.cat([x, r * h], dim=1)
        h_tilde = torch.tanh(self.conv_cand(cand_in))
        h_next = (1.0 - z) * h + z * h_tilde
        pred_delta = self.pred_head(h_next)
        pred_next = x + pred_delta
        return pred_next, h_next


class DorsalMotionStream:
    """Computes continuous surprise energy via online self-supervised predictive suppression."""

    def __init__(
        self,
        periphery_size: int = 96,
        device: torch.device | str = "cpu",
        online_lr: float = 2e-3,
    ) -> None:
        self.periphery_size = periphery_size
        self.device = torch.device(device)

        self.predictor = _PredictiveConvGRU(in_ch=4, hidden_ch=16).to(self.device)
        self.optimizer = torch.optim.AdamW(self.predictor.parameters(), lr=online_lr, weight_decay=1e-4)

        # Gaussian kernel to fuse fragmented motion edges into coherent surprise basins
        self._gauss = self._make_gaussian_kernel(kernel_size=9, sigma=2.0).to(self.device)
        # 2D Hann window for clean FFT phase-correlation global camera-pan estimation (MSTd)
        win_1d = torch.hann_window(periphery_size, periodic=False, device=self.device)
        self._hann2d = torch.outer(win_1d, win_1d).view(1, 1, periphery_size, periphery_size)
        lin = torch.linspace(-1.0, 1.0, periphery_size, device=self.device)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self._base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)

        self._prev_features: torch.Tensor | None = None
        self._hidden: torch.Tensor | None = None
        self._prev_surprise: torch.Tensor | None = None
        self._habituation_floor: torch.Tensor | None = None

    @staticmethod
    def _make_gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    @staticmethod
    def _extract_magnocellular_features(periphery: torch.Tensor) -> torch.Tensor:
        """Convert RGB periphery [1,3,P,P] into 4 biological opponent + edge energy channels."""
        r, g, b = periphery[:, 0:1], periphery[:, 1:2], periphery[:, 2:3]
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        rg = (r - g) * 0.5 + 0.5
        by = (b - 0.5 * (r + g)) * 0.5 + 0.5
        dx = F.pad(lum[:, :, :, 1:] - lum[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(lum[:, :, 1:, :] - lum[:, :, :-1, :], (0, 0, 0, 1))
        edge = torch.sqrt(dx * dx + dy * dy + 1e-6).clamp(0.0, 1.0)
        return torch.cat([lum, rg, by, edge], dim=1)

    def _estimate_global_pan(
        self, prev_lum: torch.Tensor, curr_lum: torch.Tensor
    ) -> tuple[float, float]:
        """Estimate dominant camera pan (dx, dy) in pixels via FFT phase correlation."""
        p = self.periphery_size
        a = (prev_lum - prev_lum.mean()) * self._hann2d
        b = (curr_lum - curr_lum.mean()) * self._hann2d
        fa = torch.fft.rfft2(a)
        fb = torch.fft.rfft2(b)
        cross = fb * torch.conj(fa)
        cross = cross / (torch.abs(cross) + 1e-6)
        corr = torch.fft.irfft2(cross, s=(p, p)).squeeze(0).squeeze(0)
        peak_val = float(corr.max().item())
        if peak_val < 0.08:
            return 0.0, 0.0
        idx = int(torch.argmax(corr).item())
        dy, dx = divmod(idx, p)
        if dy > p // 2:
            dy -= p
        if dx > p // 2:
            dx -= p
        max_shift = p // 4
        if abs(dx) > max_shift or abs(dy) > max_shift:
            return 0.0, 0.0
        return float(dx), float(dy)

    def _warp_shift(self, tensor: torch.Tensor, dx_px: float, dy_px: float) -> torch.Tensor:
        """Shift a [1,C,P,P] or [P,P] map by (dx_px, dy_px) to cancel global camera pan."""
        if abs(dx_px) < 0.5 and abs(dy_px) < 0.5:
            return tensor
        is_2d = tensor.ndim == 2
        t = tensor.unsqueeze(0).unsqueeze(0) if is_2d else tensor
        p = self.periphery_size
        norm_dx = 2.0 * dx_px / max(1, p - 1)
        norm_dy = 2.0 * dy_px / max(1, p - 1)
        grid = self._base_grid.clone()
        # Sample previous frame at (x - dx, y - dy) to align it with current frame at (x, y)
        grid[..., 0] = grid[..., 0] - norm_dx
        grid[..., 1] = grid[..., 1] - norm_dy
        warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=False)
        return warped.squeeze(0).squeeze(0) if is_2d else warped

    def _pan_valid_mask(self, dx_px: float, dy_px: float) -> torch.Tensor:
        """Suppress newly entered edge strips caused purely by camera panning."""
        p = self.periphery_size
        mask = torch.ones((1, 1, p, p), device=self.device)
        pad_x = int(abs(dx_px)) + 2
        pad_y = int(abs(dy_px)) + 2
        if dx_px > 0.5:
            mask[:, :, :, : min(p, pad_x)] = 0.0
        elif dx_px < -0.5:
            mask[:, :, :, max(0, p - pad_x) :] = 0.0
        if dy_px > 0.5:
            mask[:, :, : min(p, pad_y), :] = 0.0
        elif dy_px < -0.5:
            mask[:, :, max(0, p - pad_y) :, :] = 0.0
        return mask

    def step(self, periphery: torch.Tensor) -> tuple[torch.Tensor, float, float, tuple[float, float]]:
        """Process one peripheral frame; return (surprise_map [P,P], surprise_peak, motion_energy, flow_xy)."""
        x_t = self._extract_magnocellular_features(periphery)

        if self._prev_features is None:
            self._prev_features = x_t.detach()
            zeros = torch.zeros(self.periphery_size, self.periphery_size, device=self.device)
            self._habituation_floor = torch.zeros_like(x_t[:, :1])
            self._prev_surprise = zeros
            return zeros, 0.0, 0.0, (0.0, 0.0)

        # 0. Vestibular / MSTd Global Ego-Motion Cancellation (compensates panning cameras)
        dx_pan, dy_pan = self._estimate_global_pan(self._prev_features[:, :1], x_t[:, :1])
        aligned_prev = self._warp_shift(self._prev_features, dx_pan, dy_pan)
        aligned_hidden = (
            self._warp_shift(self._hidden, dx_pan, dy_pan) if self._hidden is not None else None
        )
        pan_mask = self._pan_valid_mask(dx_pan, dy_pan)

        # 1. Instantaneous motion residual (fast magnocellular transient after pan alignment)
        inst_diff = (x_t - aligned_prev).abs().mean(dim=1, keepdim=True) * pan_mask

        # 2. Online predictive coding step: predict x_t from aligned (x_{t-1}, h_{t-1})
        self.predictor.train()
        pred_t, h_next = self.predictor(aligned_prev, aligned_hidden)
        loss = F.smooth_l1_loss(pred_t * pan_mask, x_t * pan_mask)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.predictor.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            pred_err = (x_t - pred_t.detach()).abs().mean(dim=1, keepdim=True) * pan_mask
            # Blend learned predictive error with transient motion energy
            raw_energy = 0.65 * pred_err + 0.35 * inst_diff

            # Subtract median frame-wide background residual
            global_floor = torch.quantile(raw_energy.flatten(), 0.55)
            raw_energy = (raw_energy - global_floor).clamp(min=0.0)

            # Adaptive spatial habituation: repetitive local flicker/sway raises local floor
            if self._habituation_floor is None:
                self._habituation_floor = torch.zeros_like(raw_energy)
            aligned_hab = self._warp_shift(self._habituation_floor, dx_pan, dy_pan)
            active_mask = (raw_energy > 0.015).float()
            # Accumulate slowly where motion repeats endlessly, decay where quiet
            hab_rate = 0.025 * active_mask + 0.012 * (1.0 - active_mask)
            self._habituation_floor = (1.0 - hab_rate) * aligned_hab + hab_rate * raw_energy
            suppressed = (raw_energy - 0.85 * self._habituation_floor).clamp(min=0.0)

            # Smooth spatially so coherent moving bodies form continuous attractors
            pad = self._gauss.shape[-1] // 2
            smoothed = F.conv2d(suppressed, self._gauss, padding=pad).squeeze(0).squeeze(0)

            # Estimate coarse velocity cue around peak surprise for Smooth Pursuit
            if self._prev_surprise is not None:
                self._prev_surprise = self._warp_shift(self._prev_surprise, dx_pan, dy_pan)
            flow_x, flow_y = self._estimate_peak_flow(smoothed)

            self._prev_features = x_t.detach()
            self._hidden = h_next.detach()
            self._prev_surprise = smoothed.detach()

            surprise_peak = float(smoothed.max().item())
            motion_energy = float(inst_diff.mean().item())
            return smoothed, surprise_peak, motion_energy, (flow_x, flow_y)

    def _estimate_peak_flow(self, curr_surprise: torch.Tensor) -> tuple[float, float]:
        if self._prev_surprise is None or float(curr_surprise.max()) < 0.01:
            return 0.0, 0.0
        p = self.periphery_size
        coords = torch.linspace(-1.0, 1.0, p, device=self.device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")

        def _centroid(s: torch.Tensor) -> tuple[float, float]:
            w = torch.pow(s.clamp(min=0.0), 2.0)
            total = w.sum()
            if float(total) < 1e-6:
                return 0.0, 0.0
            cx = float((w * xx).sum().item() / total.item())
            cy = float((w * yy).sum().item() / total.item())
            return cx, cy

        px, py = _centroid(self._prev_surprise)
        cx, cy = _centroid(curr_surprise)
        return float(max(-0.2, min(0.2, cx - px))), float(max(-0.2, min(0.2, cy - py)))

    def reset(self) -> None:
        self._prev_features = None
        self._hidden = None
        self._prev_surprise = None
        self._habituation_floor = None
