"""
MambaX-Net: A Dual-scan architecture combining Mamba and cross-attention mechanisms.

This file contains classes adapted from multiple sources:

1. AttentionBlock and CrossAttention classes adapted from:
   https://github.com/ipc-lab/NDIC-CAM/blob/main/models/distributed_models/attention_block.py
   Original work: "Neural Distributed Image Compression with Cross-Attention Feature Alignment"
   Original authors: Nitish Mital, Ezgi Ozyilkan, Ali Garjani, Deniz Gunduz
   Original license: MIT License

2. MambaBlock adapted from:
   https://github.com/state-spaces/mamba/tree/main/mamba_ssm
   Original work: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
   Original license: Apache License 2.0

Modifications by: Yovin Yahathugoda (yovin.yahathugoda@kcl.ac.uk)
Modifications: Adapted and extended for MambaX-Net architecture with 3D support,
               Mamba integration, and additional shape capture capabilities.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Note: Portions of this code (AttentionBlock and CrossAttention classes) were
originally licensed under MIT License from NDIC-CAM project. The MIT License
is compatible with Apache License 2.0.
"""

import gc
import pydoc
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from einops import rearrange
from einops.layers.torch import Rearrange
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint as grad_ckpt


class AttentionBlock(nn.Module):
    """Multi-Head Attention Block

    Adapted from NDIC-CAM: https://github.com/ipc-lab/NDIC-CAM
    Modified by Yovin Yahathugoda (yovin.yahathugoda@kcl.ac.uk) for MambaX-Net
    """

    def __init__(
        self, dim: int, heads: int = 1, dim_head: int = None, dropout: float = 0.0
    ) -> None:
        """Initialize the Multi-Head Attention Block.

        Args:
            dim (int): The input dimension.
            heads (int, optional): The number of attention heads. Defaults to 1.
            dim_head (int, optional): The dimension of each attention head. Defaults to None.
            dropout (float, optional): The dropout rate. Defaults to 0.0.
        """
        super().__init__()
        if dim_head is None:
            dim_head = dim
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q_x = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_y = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_y = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the Multi-Head Attention Block.

        Args:
            x (torch.Tensor): The input tensor for the first sequence.
            y (torch.Tensor): The input tensor for the second sequence.

        Returns:
            torch.Tensor: The output tensor after applying multi-head attention.
        """
        q_x = rearrange(self.to_q_x(x), "b n (h d) -> b h n d", h=self.heads)
        k_y = rearrange(self.to_k_y(y), "b n (h d) -> b h n d", h=self.heads)
        v_y = rearrange(self.to_v_y(y), "b n (h d) -> b h n d", h=self.heads)

        out = F.scaled_dot_product_attention(q_x, k_y, v_y, scale=self.scale)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


def bidirectional_scan(mamba: nn.Module, seq: torch.Tensor) -> torch.Tensor:
    """Run a causal Mamba scan in both directions and average the two passes.

    Args:
        mamba (nn.Module): The Mamba layer to apply.
        seq (torch.Tensor): Token sequence of shape (b, n, d).

    Returns:
        torch.Tensor: Bidirectionally scanned sequence, same shape as input.
    """
    return 0.5 * (mamba(seq) + mamba(seq.flip(1)).flip(1))


class CrossAttention(nn.Module):
    """CrossAttention module for efficient attention.

    Adapted from NDIC-CAM: https://github.com/ipc-lab/NDIC-CAM
    Modified by Yovin Yahathugoda (yovin.yahathugoda@kcl.ac.uk) for MambaX-Net with:
    - 3D tensor support (added depth dimension)
    - Mamba layer integration
    - Shape feature incorporation
    - Additional channel convolution
    """

    def __init__(
        self,
        input_size: Tuple[int, int, int],
        num_filters: int = 192,
        heads: int = 1,
        ch_patch_size: int = 1,
        num_patches: int = 4,
        dim: int = None,
        dim_head: int = None,
        dropout: float = 0.2,
        in_channels: int = 48,
    ) -> None:
        """CrossAttention module for efficient attention.

        Args:
            input_size (Tuple[int, int, int]): _description_
            num_filters (int, optional): Number of filters in the convolutional layers. Defaults to 192.
            heads (int, optional): Number of attention heads. Defaults to 1.
            ch_patch_size (int, optional): Patch size for the channel dimension. Defaults to 1.
            num_patches (int, optional): Number of patches. Defaults to 4.
            dim (int, optional): Dimension of the output embeddings. Defaults to None.
            dim_head (int, optional): Dimension of the attention heads. Defaults to None.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
            in_channels (int, optional): Number of input channels. Defaults to 48.
        """
        super().__init__()

        assert (
            num_filters % ch_patch_size == 0
        ), "num_filters must be divisible by the patch size."
        self.patch_size = [None] * 4
        self.patch_size[0] = ch_patch_size
        self.patch_size[1] = input_size[0] // num_patches
        self.patch_size[2] = input_size[1] // num_patches
        self.patch_size[3] = input_size[2] // num_patches
        patch_dim = (
            self.patch_size[1] * self.patch_size[2] * self.patch_size[3] * ch_patch_size
        )
        if dim is None:
            dim = patch_dim

        self.to_patch_embedding_x = nn.Sequential(
            Rearrange(
                "b (c p0) (d p1) (h p2) (w p3) -> b (c d h w) (p0 p1 p2 p3)",
                p0=ch_patch_size,
                p1=self.patch_size[1],
                p2=self.patch_size[2],
                p3=self.patch_size[3],
            ),
            nn.Linear(patch_dim, dim),
        )
        self.to_patch_embedding_y = nn.Sequential(
            Rearrange(
                "b (c p0) (d p1) (h p2) (w p3) -> b (c d h w) (p0 p1 p2 p3)",
                p0=ch_patch_size,
                p1=self.patch_size[1],
                p2=self.patch_size[2],
                p3=self.patch_size[3],
            ),
            nn.Linear(patch_dim, dim),
        )
        self.unpack_embedding_y = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange(
                "b (c d h w) (p0 p1 p2 p3) -> b (c p0) (d p1) (h p2) (w p3)",
                d=num_patches,
                h=num_patches,
                w=num_patches,
                p0=ch_patch_size,
                p1=self.patch_size[1],
                p2=self.patch_size[2],
                p3=self.patch_size[3],
            ),
        )

        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)
        self.norm_shape = nn.LayerNorm(dim)
        self.shape_proj = nn.Conv3d(in_channels, num_filters, kernel_size=1)
        self.to_shape_embedding = nn.Sequential(
            Rearrange(
                "b (c p0) (d p1) (h p2) (w p3) -> b (c d h w) (p0 p1 p2 p3)",
                p0=ch_patch_size,
                p1=self.patch_size[1],
                p2=self.patch_size[2],
                p3=self.patch_size[3],
            ),
            nn.Linear(patch_dim, dim),
        )
        self.mamba_layer_x = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)
        self.mamba_layer_y = Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

        self.attn = AttentionBlock(dim, heads, dim_head, dropout)

        self.to_patch_embedding_y[1].load_state_dict(
            self.to_patch_embedding_x[1].state_dict()
        )
        self.mamba_layer_y.load_state_dict(self.mamba_layer_x.state_dict())
        self.attn.to_k_y.load_state_dict(self.attn.to_q_x.state_dict())

        nn.init.constant_(self.norm_shape.weight, 0.01)
        nn.init.zeros_(self.norm_shape.bias)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        shape_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for the CrossAttention module.

        Args:
            x (torch.Tensor): Input tensor for the x branch.
            y (torch.Tensor): Input tensor for the y branch.
            shape_feats (Optional[torch.Tensor], optional): Shape features tensor. Defaults to None.

        Returns:
            torch.Tensor: Output tensor with cross-attended features added via residual.
        """
        x_emb = self.to_patch_embedding_x(x)
        y_emb = self.to_patch_embedding_y(y)

        if shape_feats is not None:
            shape_feats = F.interpolate(
                shape_feats, size=y.shape[2:], mode="trilinear", align_corners=False
            )
            shape_feats = self.shape_proj(shape_feats)
            shape_emb = self.norm_shape(self.to_shape_embedding(shape_feats))
            y_emb = y_emb + shape_emb

        x_norm = self.norm_x(x_emb)
        y_norm = self.norm_y(y_emb)

        x_norm = F.layer_norm(
            bidirectional_scan(self.mamba_layer_x, x_norm), (x_norm.shape[-1],)
        )
        y_norm = F.layer_norm(
            bidirectional_scan(self.mamba_layer_y, y_norm), (y_norm.shape[-1],)
        )

        att_out = self.attn(x_norm, y_norm)
        att_out = self.unpack_embedding_y(att_out)

        return x + att_out


class ConvBlock3d(nn.Module):
    """3×3×3 conv with InstanceNorm and LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        """Initialize the conv, instance norm and activation layers.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            stride (int, optional): Stride of the 3x3x3 convolution. Defaults to 1.
        """
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)
        self.norm = nn.InstanceNorm3d(out_ch, affine=True)
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution, instance norm and activation in sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (B, in_ch, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, out_ch, D', H', W'), where
                the spatial dims depend on `stride`.
        """
        return self.act(self.norm(self.conv(x)))


class ShapeExtractorModule(nn.Module):
    """3D Conv shape extractor for prostate zone masks.

    Input: binary zone mask (B, C, D, H, W).
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the three-layer 3D conv stack.

        Args:
            in_channels (int): Number of channels in the input zone mask.
            out_channels (int): Number of channels in the final extracted
                shape feature map.
        """
        super(ShapeExtractorModule, self).__init__()
        self.block1 = ConvBlock3d(in_channels, 16, stride=1)
        self.block2 = ConvBlock3d(16, 32, stride=1)
        self.block3 = ConvBlock3d(32, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract shape features from a binary zone mask.

        Args:
            x (torch.Tensor): Binary zone mask of shape (B, in_channels, D, H, W).

        Returns:
            torch.Tensor: Extracted shape features of shape
                (B, out_channels, D, H, W).
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class MambaXNet(PlainConvUNet):
    """nnU-Net backbone extended with shape-conditioned Mamba/cross-attention fusion.

    Extends `PlainConvUNet`, optionally initialised from a pretrained nnU-Net
    fold checkpoint. At each of the first `num_attention_layers` encoder skip
    levels, a `CrossAttention` block fuses the current-image features with the
    previous-image features, conditioned on shape features extracted from a
    prostate zone mask via `ShapeExtractorModule`, before the fused skips are
    passed to the decoder.
    """

    def __init__(
        self,
        config: dict,
        configuration_manager: callable,
        model_path: Optional[str],
        pretrained_fold: int = 0,
        num_attention_layers: int = 2,
        mask_channels: int = 3,
        shape_out_channels: int = 32,
        crossattention_dim: int = 256,
        crossattention_heads: int = 4,
        crossattention_num_patches: int = 4,
        gradient_checkpointing: bool = False,
    ) -> None:
        """MambaXNet model for 3D medical image segmentation.

        Args:
            config (dict): Configuration dictionary for the model.
            configuration_manager (callable): Manager for configuration settings.
            model_path (Optional[str]): Path to the pre-trained model, or None to train
                the encoder/decoder from scratch.
            pretrained_fold (int, optional): Which nnU-Net cross-validation fold to
                initialise from. Folds are trained from independent random inits, so
                their weights sit in different permutation basins and cannot be
                averaged — a single fold is used. Defaults to 0.
            num_attention_layers (int, optional): Number of attention layers in the model. Defaults to 2.
            mask_channels (int, optional): Number of channels in the input mask. Defaults to 3.
            shape_out_channels (int, optional): Number of output channels from the
                `ShapeExtractorModule` used to condition the cross-attention on the
                mask shape features. Defaults to 32.
            crossattention_dim (int, optional): Dimension of the cross-attention layer. Defaults to 256.
            crossattention_heads (int, optional): Number of heads in the cross-attention layer. Defaults to 4.
            crossattention_num_patches (int, optional): Number of patches in the cross-attention layer. Defaults to 4.
            gradient_checkpointing (bool): Recompute encoder activations during backward to save memory. Defaults to False.
        """
        self.num_attention_layers = num_attention_layers
        self.configuration_manager = configuration_manager
        self.config = config

        # Convert string class references to actual classes as nnU-Net does
        arch_kwargs = dict(configuration_manager.network_arch_init_kwargs)
        for key in getattr(
            configuration_manager, "network_arch_init_kwargs_req_import", []
        ):
            if (
                key in arch_kwargs
                and isinstance(arch_kwargs[key], str)
                and arch_kwargs[key] is not None
            ):
                arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

        super().__init__(
            input_channels=config["in_channels"],
            num_classes=config["out_channels"],
            deep_supervision=False,
            **arch_kwargs,
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_checkpointing = gradient_checkpointing

        if model_path is not None:
            fold_path = self._resolve_fold_checkpoint(model_path, pretrained_fold)
            print(f"Loading pretrained fold {pretrained_fold} from {fold_path}")
            state_dict = torch.load(fold_path, map_location=str(self.device))
            self.load_state_dict(state_dict)

        self.shape_cnn = ShapeExtractorModule(
            in_channels=mask_channels, out_channels=shape_out_channels
        )

        # Initialize CrossAttention layers
        self.cross_attention_layers = nn.ModuleList()

        feat_map_list, filters_list = self.get_feature_list()

        for i in range(self.num_attention_layers):
            self.cross_attention_layers.append(
                CrossAttention(
                    input_size=feat_map_list[i],
                    heads=crossattention_heads,
                    num_filters=filters_list[i],
                    dim=crossattention_dim,
                    num_patches=crossattention_num_patches,
                    in_channels=shape_out_channels,
                )
            )

    @staticmethod
    def _resolve_fold_checkpoint(model_path: str, fold: int) -> Path:
        """Locate the pretrained checkpoint for a given nnU-Net fold.

        Args:
            model_path (str): Either a checkpoint file, or a directory searched
                recursively for a file matching the requested fold.
            fold (int): Which cross-validation fold to load.

        Returns:
            Path: The resolved checkpoint path.

        Raises:
            FileNotFoundError: If no checkpoint matches the fold.
            ValueError: If more than one checkpoint matches the fold.
        """
        path = Path(model_path)
        if path.is_file():
            return path

        pattern = f"*fold_{fold}*.pt"
        matches = sorted(path.glob(f"**/{pattern}"))
        if not matches:
            raise FileNotFoundError(f"No checkpoint matching '{pattern}' under {path}")
        if len(matches) > 1:
            raise ValueError(
                f"Multiple checkpoints match '{pattern}' under {path}, pass the "
                f"file directly as model_path: {[str(m) for m in matches]}"
            )
        return matches[0]

    def _inspect_input_size(self, spatial_shape: tuple) -> None:
        """Inspect the input size for the model.

        Args:
            spatial_shape (tuple): The spatial shape of the input tensor.

        Raises:
            ValueError: If the input shape is not 3D or if any dimension is smaller than the minimum size.
        """
        # Expecting 3D input with minimum size in each dimension
        min_size = 16
        if len(spatial_shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {spatial_shape}")
        if any(s < min_size for s in spatial_shape):
            raise ValueError(
                f"Each spatial dimension must be >= {min_size}, got {spatial_shape}"
            )

    def get_feature_list(self) -> tuple[list[tuple], list[int]]:
        """Get the feature map and filter sizes from the encoder.

        Returns:
            Tuple[list[tuple], list[int]]: A tuple containing a list of feature map sizes and a list of filter sizes.
        """
        # Dynamically get feature map and filter sizes
        input_shape = tuple(self.configuration_manager.patch_size)
        dummy_shape = (
            1,
            self.config["in_channels"],
            *input_shape,
        )
        x = torch.randn(*dummy_shape).to(next(self.encoder.parameters()).device)

        with torch.no_grad():
            skips = self.encoder(x)
        feat_map_list = [
            tuple(skip.shape[2:]) for skip in skips[: self.num_attention_layers]
        ]
        filters_list = [skip.shape[1] for skip in skips[: self.num_attention_layers]]

        del dummy_shape, x, skips
        gc.collect()
        torch.cuda.empty_cache()
        return feat_map_list, filters_list

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for the model.

        x_1: Input feature for the current image.
        x_2: Input feature for the previous image.
        mask: Previous segmentation mask for the patient

        Args:
            x (torch.Tensor): Input tensor.
            mask (torch.Tensor): Mask tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x_1 = x[:, 0, :, :, :].unsqueeze(1)
        x_2 = x[:, 1, :, :, :].unsqueeze(1)

        # Custom input size inspection
        if not torch.jit.is_scripting() and not torch.jit.is_tracing():
            self._inspect_input_size(x_1.shape[2:])
            self._inspect_input_size(x_2.shape[2:])

        shape_feats = self.shape_cnn(mask)

        if self.gradient_checkpointing and self.training:
            curr_skips = grad_ckpt(self.encoder, x_1, use_reentrant=False)
            prev_skips = grad_ckpt(self.encoder, x_2, use_reentrant=False)
        else:
            curr_skips = self.encoder(x_1)
            prev_skips = self.encoder(x_2)

        for map_idx in range(self.num_attention_layers):
            net = self.cross_attention_layers[map_idx]
            curr_skips[map_idx] = net(
                curr_skips[map_idx], prev_skips[map_idx], shape_feats
            )
            prev_skips[map_idx] = None

        return self.decoder(curr_skips)
