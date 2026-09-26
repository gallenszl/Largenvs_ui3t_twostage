from torch import Tensor, nn


class DtypePreservingLayerNorm(nn.LayerNorm):
    """LayerNorm that returns the same dtype as its input."""

    def forward(self, x: Tensor) -> Tensor:
        out = super().forward(x)
        return out.to(dtype=x.dtype)
