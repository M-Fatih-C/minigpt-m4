import torch
import torch.nn as nn

from config_v3 import RMS_NORM_EPS


class RMSNorm(nn.Module):

    def __init__(
        self,
        dimension: int,
        eps: float = RMS_NORM_EPS,
    ):
        super().__init__()

        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(dimension)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        input_dtype = x.dtype

        x_float = x.float()

        variance = (
            x_float
            .pow(2)
            .mean(
                dim=-1,
                keepdim=True,
            )
        )

        normalized = (
            x_float
            * torch.rsqrt(
                variance
                + self.eps
            )
        )

        return (
            normalized
            .to(input_dtype)
            * self.weight
        )
