"""Fourier Neural Operator (FNO) baseline, operating on a physical grid
(not the B-spline coefficient space -- see learning.projected_fno for the
Pi_h projection that brings its output into V_h so it can be certified).

Conditioned on (tau, mu, payoff) by concatenating tau_norm and mu_tilde as
constant-valued extra channels alongside the payoff sampled on the grid --
a standard way to condition an FNO on non-spatial parameters.

The raw output of this network is NOT an element of V_h: it is a set of
pointwise values on a fixed physical grid, with no guarantee of matching
any B-spline coefficient representation, and no exact-initial-condition
structure of its own. Both properties are added by ProjectedFNO
(learning.projected_fno), not here.
"""

from __future__ import annotations

import torch
from torch import nn


class SpectralConv1d(nn.Module):
    """1D spectral convolution layer (truncated Fourier-mode mixing).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        n_modes: Number of Fourier modes retained.
    """

    def __init__(self, in_channels: int, out_channels: int, n_modes: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        scale = 1.0 / (in_channels * out_channels)
        # Stored as two REAL parameters, not one complex64 parameter.
        # nn.Module.to(dtype)/.float()/.double() casts a complex parameter
        # to the target dtype via a complex->real cast when the target is
        # real, which DISCARDS the imaginary part silently (PyTorch even
        # warns "Casting complex values to real discards the imaginary
        # part") -- calling .double() anywhere on a module containing this
        # layer would permanently zero out weight_imag. Real parameters
        # follow ordinary (lossless) dtype-casting semantics, so combining
        # them into a complex tensor in forward() (via torch.complex, which
        # picks up whatever real dtype .to()/.float()/.double() left them
        # in) is fully robust to those calls.
        self.weight_real = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_channels, out_channels, n_modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply truncated spectral convolution along the last (grid) dimension.

        Args:
            x: (batch, in_channels, n_grid)
        Returns:
            (batch, out_channels, n_grid)
        """
        batch, _, n_grid = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        n_modes = min(self.n_modes, x_ft.shape[-1])
        weight = torch.complex(self.weight_real, self.weight_imag).to(x_ft.dtype)

        out_ft = torch.zeros(batch, self.out_channels, x_ft.shape[-1], dtype=x_ft.dtype, device=x.device)
        out_ft[:, :, :n_modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :n_modes], weight[:, :, :n_modes]
        )
        return torch.fft.irfft(out_ft, n=n_grid, dim=-1)


class FNO1d(nn.Module):
    """1D Fourier Neural Operator mapping gridded input functions to gridded outputs.

    Args:
        in_channels: Number of input channels (payoff-on-grid + conditioning
            channels: 1 (payoff) + 1 (tau_norm) + mu_dim, by convention --
            see learning.projected_fno).
        out_channels: Number of output channels (1: the raw solution-grid
            correction).
        width: Channel width of the lifted representation.
        n_modes: Number of Fourier modes retained per spectral layer.
        n_layers: Number of spectral-convolution layers.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 32,
        n_modes: int = 16,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.n_modes = n_modes
        self.n_layers = n_layers

        self.lift = nn.Linear(in_channels, width)
        self.spectral_layers = nn.ModuleList(
            [SpectralConv1d(width, width, n_modes) for _ in range(n_layers)]
        )
        self.pointwise_layers = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(n_layers)]
        )
        self.project = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of gridded input functions to gridded output functions.

        Args:
            x: (batch, n_grid, in_channels)
        Returns:
            (batch, n_grid, out_channels)
        """
        h = self.lift(x)  # (batch, n_grid, width)
        h = h.permute(0, 2, 1)  # (batch, width, n_grid)
        for spectral, pointwise in zip(self.spectral_layers, self.pointwise_layers):
            h = torch.nn.functional.gelu(spectral(h) + pointwise(h))
        h = h.permute(0, 2, 1)  # (batch, n_grid, width)
        return self.project(h)
