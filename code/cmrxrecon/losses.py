"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        """
        Args:
            win_size: Window size for SSIM calculation.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
        """
        super().__init__()
        self.win_size = win_size
        self.k1, self.k2 = k1, k2
        self.register_buffer("w", torch.ones(1, 1, win_size, win_size) / win_size**2)
        NP = win_size**2
        self.cov_norm = NP / (NP - 1)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ):
        assert isinstance(self.w, torch.Tensor)

        S = []
        for chan in range(X.shape[1]):
            x_chan = X[:, [chan], :, :]
            y_chan = Y[:, [chan], :, :]
            cur_range = torch.max(y_chan.amax((1, 2, 3)) - y_chan.amin((1, 2, 3)), x_chan.amax((1, 2, 3)) - x_chan.amin((1, 2, 3)))
            C1 = ((self.k1 * cur_range) ** 2)[:, None, None, None]
            C2 = ((self.k2 * cur_range) ** 2)[:, None, None, None]
            ux = F.conv2d(x_chan, self.w)  # typing: ignore
            uy = F.conv2d(y_chan, self.w)  #
            uxx = F.conv2d(x_chan * x_chan, self.w)
            uyy = F.conv2d(y_chan * y_chan, self.w)
            uxy = F.conv2d(x_chan * y_chan, self.w)
            vx = self.cov_norm * (uxx - ux * ux)
            vy = self.cov_norm * (uyy - uy * uy)
            vxy = self.cov_norm * (uxy - ux * uy)
            A1, A2, B1, B2 = (
                2 * ux * uy + C1,
                2 * vxy + C2,
                ux**2 + uy**2 + C1,
                vx + vy + C2,
            )
            D = B1 * B2
            s_chan = A1 * A2 / D
            S.append((A1 * A2) / D)

        return 1 - torch.cat(S).mean()

class NormL1L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, X, Y):
        return (X - Y).norm(2)/Y.norm(2) + (X - Y).norm(1)/Y.norm(1)

class L1L2Loss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.l2_loss = torch.nn.MSELoss()
        self.l1_loss = torch.nn.L1Loss()
        pass

    def forward(self, X, Y):
        return self.l2_loss(X, Y) + self.l1_loss(X, Y)

class SSIM_L1(nn.Module):
    def __init__(self, gamma) -> None:
        super().__init__()
        self.ssim = SSIMLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.gamma = gamma
        pass

    def forward(self, X, Y):
        return self.ssim(X, Y) * self.gamma + self.l1_loss(X, Y)



