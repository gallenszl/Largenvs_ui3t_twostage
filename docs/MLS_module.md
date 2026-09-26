# MLS (MultiLayer-SimpleAdd) Module

A self-contained reference for the **multi-layer intermediate feature aggregation** technique used by representation autoencoders to extract richer latent features from frozen Vision Transformer encoders.

This document is portable — it does not depend on any specific repository. Copy it alongside the MLS code into any project that needs it.

---

## 1. Overview

**Problem.** A frozen ViT encoder (DINOv2, DINOv3, SigLIP2, …) is typically used by reading only the **last layer's** patch tokens. But the last layer is semantically dominant — fine-grained texture and structural cues from earlier layers get discarded. A downstream decoder must then re-learn or hallucinate that information.

**MLS solution.** Aggregate patch tokens from *k* selected intermediate layers (each layer-normalized) by **mean**, then add a broadcast of the deepest layer's global mean. The result has the same per-token dimensionality as a single layer but encodes a mixture of low/mid/high-level features plus an explicit global semantic prior.

**One-line formula.**

```
z = mean( stack( [LN_i(block_i(x)) for i in layer_indices] ) )    # cross-layer mean
    + broadcast_over_tokens( mean_over_tokens( block_{i_last}(x) ) )  # global semantic prior
```

**Key invariant.** Aggregation uses `mean` (not `concat`), so `latent_dim == embed_dim` regardless of *k*. Switching from k=1 to k=23 does **not** widen the latent.

---

## 2. Encoder String DSL

The factory function expects a string of the form:

```
"<encoder_type>-<architecture>-<model_config>[<flags>]"
```

| Segment | Example | Purpose |
|---|---|---|
| `encoder_type` | `dinov3mls` | Maps to an encoder class via a registry. `*mls` suffix selects the multi-layer variant. |
| `architecture` | `vit` | Architecture family (currently a placeholder; ViT is the only supported family). |
| `model_config` | `l16` | Backbone size code: `s16`, `b16`, `l16`, `h16plus`, `7b16` for DINOv3; `s/b/l/g` for DINOv2; `b/l/so400m/g` for SigLIP2. |
| `flags` (optional, in `[...]`) | `[layers=11.13.15.17.19.21.23]` | Comma-separated key=value flags. Currently supported: `layers=<dot-separated indices>`, `norm` (keep LN affine). |

**Parsing** (regex):

```python
import re
match = re.match(r'^(.+?)\[([^\]]+)\]$', model_config)
if match:
    base = match.group(1)                                     # 'l16'
    flags = [f.strip() for f in match.group(2).split(',')]    # ['layers=11.13.15.17.19.21.23']
else:
    base, flags = model_config, []
```

**Layer-indices extraction**:

```python
layers_flag = [f for f in flags if f.startswith('layers=')]
if layers_flag:
    layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
else:
    layer_indices = DEFAULT_LAYERS[base]   # fallback to k=4 quartile sampling
```

**Examples**:

| String | Backbone | `layer_indices` | `k` |
|---|---|---|---|
| `dinov3-vit-l16` | DINOv3-L (24 blocks) | (not MLS — single last layer) | 1 |
| `dinov3mls-vit-l16` | DINOv3-L | `[5, 11, 17, 23]` (default) | 4 |
| `dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]` | DINOv3-L | `[11, 13, 15, 17, 19, 21, 23]` | 7 |
| `dinov3mls-vit-l16[layers=1.2.3.4.5.6.7.8.9.10.11.12.13.14.15.16.17.18.19.20.21.22.23]` | DINOv3-L | all 23 transformer blocks | 23 |

Note that k=7 here is **not** "the last 7 layers" — it's "every other layer from block 11 to 23", reflecting an empirical preference for mid-to-deep features.

---

## 3. Layer Index Configuration

Default `layer_indices` per backbone (used when no `layers=` flag is provided):

| Backbone family | `s16` / `s` / `b` | `b16` / `b` | `l16` / `l` | `h16plus` | `g` | `so400m` |
|---|---|---|---|---|---|---|
| DINOv2 | `[2,5,8,11]` | `[2,5,8,11]` | `[5,11,17,23]` | — | `[10,20,30,39]` | — |
| DINOv3 | `[2,5,8,11]` | `[2,5,8,11]` | `[5,11,17,23]` | `[8,16,24,31]` | — | — |
| SigLIP2 | — | `[2,5,8,11]` | `[5,11,17,23]` | — | `[10,20,30,39]` | `[5,12,19,26]` |

The pattern is **quartile sampling**: take 4 layers evenly spaced through the depth (e.g., for a 24-layer ViT-L, that's blocks 5, 11, 17, 23).

For larger *k*, the project's training configs override these defaults via the `layers=` flag.

---

## 4. `get_intermediate_layers()` — DINOv3-Native API

⚠ **Important**: this is **not a forward hook**. DINOv3 / DINOv2 backbones expose a built-in method that takes layer indices and returns post-block features. The signature:

```python
outputs = model.get_intermediate_layers(
    x,                            # [B, 3, H, W] preprocessed input
    n=layer_indices,              # list[int], e.g. [11, 13, 15, 17, 19, 21, 23]
    reshape=False,                # keep [B, N, D]; if True → [B, D, h, w]
    return_class_token=False,     # drop CLS + register tokens, keep only patch tokens
    norm=True,                    # apply per-layer LayerNorm before returning
)
# returns: list[Tensor], len = len(n), each shape [B, num_patch_tokens, D]
```

### Execution flow

1. **Patch embedding**: `[B, 3, H, W]` → `[B, num_patches + num_special_tokens, D]`
   - For DINOv3-L at 256×256: `num_patches = 256` (16×16 grid, patch_size=16), `num_special_tokens = 5` (1 CLS + 4 registers), `D = 1024`.
2. **All transformer blocks are executed** (necessary, since deeper layers depend on earlier ones — there's no skipping).
3. **At each requested layer index `n[i]`**, a snapshot is taken:
   - Apply the layer-specific LayerNorm (`norm=True`).
   - Drop CLS + register tokens (`return_class_token=False`).
   - Save `[B, num_patches, D]`.
4. **Return list** of `len(n) == k` tensors, each `[B, num_patches, D]`.

### Why `norm=True` matters

Early ViT blocks have activation RMS values **5–10× larger** than later blocks. Averaging raw activations would let early layers dominate the mean. Per-layer LayerNorm equalizes scale, so each layer contributes its *information content*, not its *amplitude*.

This single flag is the difference between MLS working and producing garbage. Do not omit it.

### For non-DINO backbones (SigLIP2)

HuggingFace `SiglipVisionModel` doesn't have `get_intermediate_layers`. The MLS variant calls `output_hidden_states=True` and manually applies the model's post-attention LayerNorm to non-final layers:

```python
hs = model(x, output_hidden_states=True).hidden_states
post_ln = model.vision_model.post_layernorm
N = model.config.num_hidden_layers
outputs = []
for li in layer_indices:
    if li == N - 1:
        outputs.append(hs[N])             # final layer already LN-normalized
    else:
        outputs.append(post_ln(hs[li + 1]))  # apply post-LN manually
```

---

## 5. Aggregation Formula

The core of MLS, 4 lines:

```python
def forward_features(self, x):
    outputs = self.model.get_intermediate_layers(
        x, n=self.layer_indices,
        reshape=False, return_class_token=False, norm=True,
    )
    # outputs: list of k tensors, each [B, N, D]

    patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
    # stack → [k, B, N, D]
    # mean over dim=0 → [B, N, D]  ← per-token, equal-weight cross-layer average

    final_mean = outputs[-1].mean(dim=1, keepdim=True)
    # outputs[-1]: deepest selected layer [B, N, D]
    # mean over token dim with keepdim → [B, 1, D]
    # interpretation: the image's "semantic fingerprint" — a CLS-like global summary
    # derived from the deepest layer's patch tokens

    patch_tokens = patch_tokens + final_mean
    # broadcast addition: [B, N, D] + [B, 1, D] → [B, N, D]
    # every patch token receives the same global semantic offset

    return {
        'x_norm_clstoken':  final_mean.squeeze(1),   # [B, D] — optional global feature
        'x_norm_patchtokens': patch_tokens,          # [B, N, D] — the actual MLS latent
    }
```

### Plain-English translation

> Average the LN-normalized patch tokens from *k* selected mid-to-deep transformer blocks — this is a "mixed-layer semantic map". Then take the deepest selected layer's patch tokens, average them over the token dimension to get a per-image 1-D semantic fingerprint. Broadcast that fingerprint back onto every spatial position. The result: a per-token feature map where every position knows both its local content and the global scene.

### Shape pipeline at a glance (DINOv3-L, 256×256 input, k=7)

```
input          [B, 3, 256, 256]
↓ patch_embed + 24 transformer blocks
hidden states  (internal, never materialized as list)
↓ get_intermediate_layers(n=[11,13,15,17,19,21,23], norm=True, return_class_token=False)
outputs        list of 7 × [B, 256, 1024]
↓ stack + mean(dim=0)
patch_tokens   [B, 256, 1024]
↓ + final_mean (broadcast)
z              [B, 256, 1024]     ← MLS output
```

---

## 6. Reference Implementation

Complete, self-contained Python code. Drop into any project that has PyTorch + a DINOv2/DINOv3 backbone.

### 6.1 Base class

```python
import re
from typing import Dict, Optional
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Base class. Subclasses override load_model() and preprocess()."""

    def __init__(
        self,
        encoder_type: str,
        architecture: str,
        model_config: str,
        device: torch.device,
        resolution: int = 256,
        accelerator=None,
    ):
        super().__init__()
        self.encoder_type = encoder_type
        self.architecture = architecture
        self.model_config = model_config
        self.device = device
        self.resolution = resolution
        self.accelerator = accelerator
        self._embed_dim: Optional[int] = None
        self.model: Optional[nn.Module] = None
        self.patch_size: Optional[int] = None

    def load_model(self):
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Map input image (B,3,H,W) in [0,255] or [0,1] to encoder-ready tensor."""
        raise NotImplementedError

    def forward_features(self, x: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        out = self.model.forward_features(x)
        if isinstance(out, dict):
            return out
        return {'x_norm_clstoken': None, 'x_norm_patchtokens': out}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return patch tokens [B, N, D]."""
        x = self.preprocess(x)
        features = self.forward_features(x)
        return features['x_norm_patchtokens']

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    @property
    def hidden_size(self) -> int:
        return self._embed_dim

    def eval(self):
        if self.model is not None:
            self.model.eval()
        return self

    def to(self, device):
        if self.model is not None:
            self.model = self.model.to(device)
        self.device = device
        return self
```

### 6.2 DINOv3 single-layer (k=1) encoder

```python
class DINOv3Encoder(VisionEncoder):
    """Single-layer DINOv3 encoder. Returns last layer patch tokens."""

    _KNOWN_FLAGS = {'norm'}
    _KNOWN_BASES = {'s16', 's16plus', 'b16', 'l16', 'h16plus', '7b16'}

    def _parse_config(self):
        """Parse 'l16[norm]' or 'l16norm' style config strings."""
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            return match.group(1), set(f.strip() for f in match.group(2).split(','))

        # Concatenated suffix fallback: e.g. 'l16norm'
        cfg = self.model_config
        for known_base in sorted(self._KNOWN_BASES, key=len, reverse=True):
            if cfg.startswith(known_base):
                suffix = cfg[len(known_base):]
                if not suffix:
                    return known_base, set()
                flags = set()
                remaining = suffix
                while remaining:
                    matched = False
                    for flag in self._KNOWN_FLAGS:
                        if remaining.startswith(flag):
                            flags.add(flag)
                            remaining = remaining[len(flag):]
                            matched = True
                            break
                    if not matched:
                        return self.model_config, set()
                return known_base, flags
        return self.model_config, set()

    def load_model(self):
        from .models.dinov3_loader import load_dinov3  # PROJECT-SPECIFIC: replace with your loader
        base_config, flags = self._parse_config()
        use_norm_affine = 'norm' in flags
        self.model = load_dinov3(f"dinov3_vit{base_config}").to(self.device).eval()
        self._embed_dim = self.model.embed_dim
        self.patch_size = 16
        if not use_norm_affine:
            # Strip the encoder's final LayerNorm affine, keep only the normalization op.
            self.model.norm = nn.LayerNorm(self._embed_dim, elementwise_affine=False)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        from .models.dinov3_loader import make_dinov3_transform  # PROJECT-SPECIFIC
        return make_dinov3_transform(resize_size=self.resolution)(x)

    def forward_features(self, x):
        out = self.model.forward_features(x)
        return {
            'x_norm_clstoken':  out.get('x_norm_clstoken'),
            'x_norm_patchtokens': out.get('x_norm_patchtokens'),
        }
```

### 6.3 DINOv3 MLS variant (the core of this doc)

```python
class DINOv3MultiLayerSimpleAddEncoder(DINOv3Encoder):
    """Multi-layer DINOv3 with mean aggregation + global mean broadcast."""

    DEFAULT_LAYERS = {
        's16':     [2, 5, 8, 11],
        'b16':     [2, 5, 8, 11],
        'l16':     [5, 11, 17, 23],
        'h16plus': [8, 16, 24, 31],
    }

    def _parse_config(self):
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            return match.group(1), [f.strip() for f in match.group(2).split(',')]
        return self.model_config, []

    def load_model(self):
        super().load_model()
        base_config, flags = self._parse_config()
        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def forward_features(self, x):
        outputs = self.model.get_intermediate_layers(
            x,
            n=self.layer_indices,
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        # outputs: list[Tensor], len=k, each [B, N, D]
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            'x_norm_clstoken':  final_mean.squeeze(1),
            'x_norm_patchtokens': patch_tokens,
        }
```

### 6.4 DINOv2 MLS variant — **no** global broadcast

```python
class DINOv2MultiLayerSimpleAddEncoder(DINOv2Encoder):  # assume DINOv2Encoder analogous to DINOv3Encoder
    """DINOv2 multi-layer: pure cross-layer mean, no broadcast."""

    DEFAULT_LAYERS = {
        's': [2, 5, 8, 11],
        'b': [2, 5, 8, 11],
        'l': [5, 11, 17, 23],
        'g': [10, 20, 30, 39],
    }

    def load_model(self):
        super().load_model()
        base_config, flags = self._parse_config()
        for f in flags:
            if f.startswith('layers='):
                self.layer_indices = [int(i) for i in f.split('=')[1].split('.')]
                return
        self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def forward_features(self, x):
        outputs = self.model.get_intermediate_layers(
            x, n=self.layer_indices, reshape=False,
            return_class_token=False, norm=True,
        )
        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        return {
            'x_norm_clstoken':  patch_tokens.mean(dim=1),
            'x_norm_patchtokens': patch_tokens,
        }
```

### 6.5 SigLIP2 MLS variant — manual post-LN for non-final layers

```python
class SigLIP2MultiLayerSimpleAddEncoder(SigLIP2Encoder):
    """SigLIP2 multi-layer with manual post-LN, mean aggregation + broadcast."""

    DEFAULT_LAYERS = {
        'b':       [2, 5, 8, 11],
        'l':       [5, 11, 17, 23],
        'so400m':  [5, 12, 19, 26],
        'g':       [10, 20, 30, 39],
    }

    def _parse_config(self):
        match = re.match(r'^(.+?)\[([^\]]+)\]$', self.model_config)
        if match:
            return match.group(1), [f.strip() for f in match.group(2).split(',')]
        return self.model_config, []

    def load_model(self):
        from transformers import SiglipVisionModel
        base_config, flags = self._parse_config()
        model_map = {
            'b':       'google/siglip2-base-patch16-256',
            'l':       'google/siglip2-large-patch16-256',
            'so400m':  'google/siglip2-so400m-patch16-256',
            'g':       'google/siglip2-giant-opt-patch16-256',
        }
        if base_config not in model_map:
            raise ValueError(f"Unknown SigLIP2 model config: {base_config}")

        self.model = SiglipVisionModel.from_pretrained(model_map[base_config])
        self.model.to(self.device).eval()
        self.patch_size = 16
        self._embed_dim = self.model.config.hidden_size
        self._num_hidden_layers = self.model.config.num_hidden_layers

        layers_flag = [f for f in flags if f.startswith('layers=')]
        if layers_flag:
            self.layer_indices = [int(i) for i in layers_flag[0].split('=')[1].split('.')]
        else:
            self.layer_indices = self.DEFAULT_LAYERS.get(base_config, [2, 5, 8, 11])

    def forward_features(self, x):
        hs = self.model(x, output_hidden_states=True).hidden_states
        post_ln = self.model.vision_model.post_layernorm
        N = self._num_hidden_layers

        outputs = []
        for li in self.layer_indices:
            if li == N - 1:
                outputs.append(hs[N])
            else:
                outputs.append(post_ln(hs[li + 1]))

        patch_tokens = torch.stack(outputs, dim=0).mean(dim=0)
        final_mean = outputs[-1].mean(dim=1, keepdim=True)
        patch_tokens = patch_tokens + final_mean
        return {
            'x_norm_clstoken':  None,
            'x_norm_patchtokens': patch_tokens,
        }
```

### 6.6 Factory function

```python
ENCODER_REGISTRY = {
    'dinov3':    DINOv3Encoder,
    'dinov3mls': DINOv3MultiLayerSimpleAddEncoder,
    'dinov2':    DINOv2Encoder,
    'dinov2mls': DINOv2MultiLayerSimpleAddEncoder,
    'siglip2':   SigLIP2Encoder,
    'siglip2mls': SigLIP2MultiLayerSimpleAddEncoder,
    # Add more as needed: 'mae', 'webssl', 'clip', 'mocov3', 'jepa', ...
}


def create_encoder(
    encoder_string: str,
    device: torch.device,
    resolution: int = 256,
    accelerator=None,
) -> VisionEncoder:
    """Factory. encoder_string format: '<type>-<arch>-<model_config>'."""
    parts = encoder_string.split('-')
    if len(parts) != 3:
        raise ValueError(
            f"Invalid encoder string: {encoder_string!r}. "
            f"Expected '<type>-<arch>-<model_config>'."
        )
    encoder_type, architecture, model_config = parts

    if encoder_type not in ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder type: {encoder_type!r}. "
            f"Available: {sorted(ENCODER_REGISTRY)}"
        )

    encoder_class = ENCODER_REGISTRY[encoder_type]
    encoder = encoder_class(
        encoder_type, architecture, model_config,
        device, resolution, accelerator,
    )
    encoder.load_model()
    return encoder
```

---

## 7. Cross-Encoder Variants

Not all MLS encoders use the same formula. Aggregation choices reflect each backbone family's characteristics:

| Class | Aggregation formula | Notes |
|---|---|---|
| `DINOv2MLS` | `mean(stack(outputs))` | No broadcast. DINOv2 has no register tokens, single-layer semantics is already strong. |
| `DINOv3MLS` | `mean(stack) + broadcast(outputs[-1].mean(token_dim))` | Primary recipe. Adds global semantic prior. |
| `SigLIP2MLS` | `mean(stack) + broadcast(outputs[-1].mean(token_dim))` | Same formula as DINOv3, but manually applies `post_layernorm` to non-final layers (HF SiglipVisionModel only normalizes the final output). |
| `EUPEMLS` | `sum(stack)` | No division by `k`, no broadcast. EUPE outputs are pre-normalized. |

Treat "MLS" / "SimpleAdd" as a **family name**, not a single recipe. When porting, verify the exact aggregation for the specific backbone you target.

---

## 8. Why It Works

The technique gives downstream models substantially faster convergence (the original paper reports ~10× over single-layer baselines on representation autoencoders). Mechanisms:

1. **Multi-scale information packed into one tensor.**  
   Shallow ViT layers carry low-level features (edges, color, texture); deep layers carry semantics (object identity, scene category). MLS pre-mixes them — the decoder no longer needs to learn which layer to query for which type of information.

2. **`norm=True` removes magnitude bias.**  
   Without LN, early layers' larger activations (5–10× the final layer) would dominate the mean. Per-layer LayerNorm gives every layer equal influence on information content rather than amplitude.

3. **`+ final_mean` is zero-cost global modulation.**  
   The broadcast term is essentially a per-image style code (AdaLN-style) implemented as pure addition. Every spatial token "knows" the global content, improving consistency (e.g., a cat's two eyes should be the same color).

4. **Better backbone → better starting point.**  
   When using stronger frozen encoders (DINOv3-L vs DINOv2-B), the representations are richer to begin with, multiplying the benefit of multi-layer aggregation.

These four factors compound: stronger encoder × multi-scale features × scale-equalized aggregation × global prior = much easier optimization for the downstream decoder/diffusion model.

---

## 9. Integration Checklist

When porting MLS to a new project, walk through this list:

### 9.1 Code dependencies

- [ ] PyTorch ≥ 2.0
- [ ] Access to the backbone's `get_intermediate_layers()` method.  
      → DINOv2/DINOv3: available on the hub model. Load via `torch.hub.load('facebookresearch/dinov3', 'dinov3_vitl16')` or from a local `DINOV3_REPO_DIR`.  
      → For HF backbones (SigLIP2 et al.): you must call `model(x, output_hidden_states=True)` and manually apply post-LN to non-final layers.
- [ ] A `preprocess()` transform matching the backbone's expected input distribution (DINOv3 uses ImageNet mean/std; CLIP uses its own values).

### 9.2 Choosing layer indices

- **Quick start**: use the `DEFAULT_LAYERS` quartile sampling (k=4). Already a strong baseline.
- **More layers**: increase to k=7 (every other layer in the back half) or k=23 (all blocks). Cost is **forward-pass only** — the backbone is frozen, so no extra training memory.
- **Empirical rule**: focus on the back half of the network for reconstruction-style downstream tasks. Front-half layers add noise more than information for that goal.

### 9.3 Verify before training

- [ ] Confirm `latent_dim` matches the backbone's embed dim (not multiplied by *k* — aggregation is mean, not concat).
- [ ] Confirm `norm=True` is set in the `get_intermediate_layers()` call. Without it, the cross-layer mean is dominated by early-layer amplitudes.
- [ ] Pick the right MLS variant: with-broadcast (DINOv3/SigLIP2 style) vs. without-broadcast (DINOv2 style). When in doubt, start with broadcast — it's been validated on more setups.
- [ ] If downstream code consumes a `[B, C, H, W]` latent (e.g., a 2D diffusion model), reshape the `[B, N, D]` MLS output: `z.transpose(1, 2).view(B, D, int(N**0.5), int(N**0.5))`.

### 9.4 Latent statistics for normalization

If your downstream model expects z-score-normalized latents:

1. Run the MLS encoder on a representative dataset (e.g., ImageNet) in eval mode.
2. Accumulate `mean` and `var` per `(C, H, W)` position using Welford's online algorithm (numerically stable).
3. Save as `{'mean': tensor[C, H, W], 'var': tensor[C, H, W]}`.
4. At inference: `z_normed = (z - mean) / sqrt(var + eps)`. Decoder operates on normalized latents.

**Critical**: re-compute stats whenever you change the backbone, *k*, or the layer indices. The latent distribution shifts substantially across configurations — using mismatched stats will silently corrupt reconstructions.

### 9.5 Common pitfalls

- ❌ **Concatenating instead of averaging.** Some implementations of "multi-layer features" use concat, which multiplies the channel dim by *k*. MLS uses `mean` to keep the dim fixed. Don't conflate them.
- ❌ **Omitting `norm=True`.** Easy to miss; produces silently bad latents.
- ❌ **Using the wrong layer indexing convention.** DINOv3's `get_intermediate_layers(n=...)` takes 0-indexed block positions. Make sure `max(layer_indices) < num_blocks`.
- ❌ **Forgetting to drop CLS / register tokens.** `return_class_token=False` handles both for DINOv3, but for HF models you may need to manually slice `hidden_states[:, num_special_tokens:]`.
- ❌ **Mismatched preprocessing.** Each backbone family expects specific image normalization. Use the backbone's official `preprocess()`/transform, never assume defaults.

### 9.6 Minimal usage example

```python
import torch

encoder = create_encoder(
    "dinov3mls-vit-l16[layers=11.13.15.17.19.21.23]",
    device=torch.device("cuda"),
    resolution=256,
)
encoder.eval()
for p in encoder.parameters():
    p.requires_grad_(False)

x = torch.randn(2, 3, 256, 256, device="cuda")  # B=2 dummy images
with torch.no_grad():
    z = encoder(x)  # [2, 256, 1024]   (B, num_patches, embed_dim)

# Reshape to 2D feature map for a convolutional / diffusion downstream:
B, N, D = z.shape
H = W = int(N ** 0.5)
z_2d = z.transpose(1, 2).reshape(B, D, H, W)  # [2, 1024, 16, 16]
```

---

## Appendix: Glossary

- **MLS**: MultiLayer-SimpleAdd — the technique described in this document.
- **Layer index**: 0-indexed transformer block position within the encoder. `layer_indices=[5, 11, 17, 23]` means "the outputs of blocks 5, 11, 17, and 23".
- **k**: the number of layers selected for aggregation, equal to `len(layer_indices)`.
- **Patch tokens**: ViT token outputs corresponding to image patches (excluding CLS and register tokens). For a 256×256 input with patch_size=16, there are 256 patch tokens.
- **Register tokens**: Extra learned tokens added to some ViTs (e.g., DINOv3 has 4) that absorb high-norm artifacts and improve attention quality. Always dropped before MLS aggregation.
- **Embed dim (D)**: The hidden width of the backbone (768 for ViT-B, 1024 for ViT-L, 1280 for ViT-H, etc.). This is also the MLS output's channel count — invariant to k.
