"""DINOv3 target encoder for REPA-style feature alignment.

Loads a frozen DINOv3-ViT-L/16 from a local torch hub clone and exposes
patch tokens of the input image. Used as the alignment target for VGGT's
aggregator intermediate features. The encoder is never trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3TargetEncoder(nn.Module):
    def __init__(self, weights_path: str, hub_dir: str, resolution: int = 512):
        super().__init__()
        # Load architecture without auto-downloading weights, then load explicit state dict.
        self.model = torch.hub.load(
            hub_dir, "dinov3_vitl16", source="local", pretrained=False
        )
        state_dict = torch.load(weights_path, map_location="cpu")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.resolution = resolution
        self.embed_dim = 1024
        self.grid = resolution // 16  # 512 -> 32 (exact match with aggregator's 448/14=32 grid)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def train(self, mode: bool = True):
        # Frozen teacher must stay in eval mode no matter what the outer Trainer does.
        # nn.Module.train() recurses into all submodules by default, which would flip
        # the teacher back into train mode.
        super().train(mode)
        self.model.eval()
        return self

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 3, H, W] in [0, 1]
        if x.shape[-1] != self.resolution or x.shape[-2] != self.resolution:
            # antialias=True applies low-pass filter when downsampling (no-op for upsample).
            # Matches torchvision's standard Resize default; keeps the teacher target clean
            # under ablations that switch target_resolution to a downsample (e.g. 448 -> 224).
            x = F.interpolate(
                x,
                size=self.resolution,
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        x = (x - self.mean) / self.std
        out = self.model.forward_features(x)
        return out["x_norm_patchtokens"]  # [N, grid*grid, 1024]
