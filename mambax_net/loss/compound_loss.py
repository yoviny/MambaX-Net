from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from monai.losses import MaskedDiceLoss, TverskyLoss
from segmentation_models_pytorch.losses import TverskyLoss as SMP_TverskyLoss

from .monai_dice_loss import ModifiedDiceCELoss as DiceCELoss


class FocalTverskyLoss(TverskyLoss):
    """
    Focal Tversky Loss implementation.

    If alpha is greater than beta, more emphasis is put on minimizing false positives.
    If beta is greater than alpha, more emphasis is put on minimizing false negatives.
    If alpha equals beta, it becomes the Dice coefficient, which treats false positives and false negatives equally.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = True,
        reduction: str = "mean",
    ) -> None:
        """Focal Tversky Loss implementation.

        Args:
            alpha (float, optional): Weight for false positives. Defaults to 0.5.
            beta (float, optional): Weight for false negatives. Defaults to 0.5.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            include_background (bool, optional): Whether to include background class in the loss. Defaults to True.
            to_onehot_y (bool, optional): Whether to convert target to one-hot encoding. Defaults to False.
            sigmoid (bool, optional): Whether to apply sigmoid activation to the input. Defaults to True.
            reduction (str, optional): Reduction method to apply to the loss. Defaults to "mean".
        """
        super(FocalTverskyLoss, self).__init__(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            reduction=reduction,
            alpha=alpha,
            beta=beta,
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Focal Tversky Loss computation.

        Args:
            input (torch.Tensor): Input tensor.
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Focal Tversky Loss value.
        """
        # MONAI's TverskyLoss already returns (1 - T), so the focal term is that
        # quantity raised to 1/gamma (Abraham & Khan, 2019).
        tversky_loss = super().forward(input, target)
        return torch.pow(tversky_loss, 1.0 / self.gamma)


class MultiClassFocalTverskyLoss(SMP_TverskyLoss):
    """Multi-class Focal Tversky Loss implementation."""

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 2.0,
        mode: str = "multiclass",
        ignore_index: Optional[int] = None,
    ) -> None:
        """Multi-class Focal Tversky Loss implementation.

        Args:
            alpha (float, optional): Weight for false positives. Defaults to 0.5.
            beta (float, optional): Weight for false negatives. Defaults to 0.5.
            gamma (float, optional): Focusing parameter. Defaults to 2.0.
            mode (str, optional): Mode for loss computation. Defaults to "multiclass".
            ignore_index (int, optional): Index to ignore during loss computation. Defaults to None.
        """
        # smp's TverskyLoss applies the focal exponent itself in aggregate_loss,
        # returning (1 - T)**gamma, so the 1/gamma exponent is handed to it directly.
        super(MultiClassFocalTverskyLoss, self).__init__(
            alpha=alpha, beta=beta, mode=mode, ignore_index=ignore_index, gamma=1.0 / gamma
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Multi-class Focal Tversky Loss computation.

        Args:
            y_pred (torch.Tensor): Predicted logits.
            y_true (torch.Tensor): Ground truth labels.

        Returns:
            torch.Tensor: Multi-class Focal Tversky Loss value.
        """
        return super().forward(y_pred.contiguous(), y_true.contiguous().long())


class TverskyCE(nn.Module):
    """Tversky Cross-Entropy Loss with balanced weight mapping."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        to_onehot_y: bool = False,
        background: bool = True,
        sigmoid: bool = True,
        softmax: bool = False,
        reduction: str = "mean",
        nnunet: bool = False,
    ) -> None:
        """Tversky Cross-Entropy Loss with balanced weight mapping.

        Args:
            alpha (float, optional): Weight for false positives in the Tversky term. Defaults to 0.7.
            beta (float, optional): Weight for false negatives in the Tversky term. Defaults to 0.3.
            to_onehot_y (bool, optional): Whether to convert target to one-hot encoding. Defaults to False.
            background (bool, optional): Whether to include the background class in the Tversky loss. Defaults to True.
            sigmoid (bool, optional): Whether to apply sigmoid activation to the input in the Tversky loss. Defaults to True.
            softmax (bool, optional): Unused; kept for interface parity with the other losses in this module. Defaults to False.
            reduction (str, optional): Reduction method to apply to the loss. Defaults to "mean".
            nnunet (bool, optional): Whether targets are in nnU-Net's (B, C, D, H, W) layout, which is rearranged to (B, C, H, W, D) when computing the balanced weight map. Defaults to False.
        """
        super(TverskyCE, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.nnunet = nnunet

        self.tv = TverskyLoss(
            alpha=self.alpha,
            beta=self.beta,
            include_background=background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            reduction=reduction,
        )

    def create_balanced_weight_map(self, y_true: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Create a balanced weight map where foreground and background have balanced contribution.

        Args:
            y_true (torch.Tensor): Binary mask of shape (B, C, H, W) or (B, C, H, W, D)
            eps (float): Small epsilon value for numerical stability

        Returns:
            torch.Tensor: Weight map with balanced foreground/background contribution
        """
        # Calculate foreground and background counts for each batch and channel
        if y_true.dim() == 5:
            if self.nnunet:
                # if shape is (B, C, D, H, W) change to (B, C, H, W, D)
                y_true = rearrange(y_true, "b c d h w -> b c h w d")
            # 3D case
            spatial_dims = (2, 3, 4)
        elif y_true.dim() == 4:
            # 2D case
            spatial_dims = (2, 3)
        else:
            raise ValueError("y_true must be 4D or 5D tensor")

        fg_counts = y_true.sum(dim=spatial_dims, keepdim=True).float()

        # Total pixels per (B,C)
        total_px = torch.prod(torch.tensor([y_true.shape[d] for d in spatial_dims])).to(
            device=y_true.device, dtype=fg_counts.dtype
        )
        bg_counts = total_px - fg_counts

        fg_weights = bg_counts.float() / (fg_counts.float() + eps)
        raw_map = torch.where(y_true == 1, fg_weights, 1.0)

        # Normalize weight map
        mean_map = raw_map.mean(dim=spatial_dims, keepdim=True)
        weight_map = raw_map / (mean_map + eps)

        return weight_map

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Tversky Cross-Entropy Loss.

        Args:
            y_pred (torch.Tensor): Predicted logits
            y_true (torch.Tensor): Ground truth labels
            mask (torch.Tensor): Mask tensor

        Returns:
            torch.Tensor: Combined Tversky and Cross-Entropy loss
        """
        if y_true.shape[1] > 1:
            weight_map = self.create_balanced_weight_map(y_true[:, 1:, ...])

            bg_true = y_true[:, 0:, :, :].float()
            bg_pred = y_pred[:, 0:, :, :]

            fg_true = y_true[:, 1:, :, :].float()
            fg_pred = y_pred[:, 1:, :, :]

            ce_fg = torch.nn.functional.binary_cross_entropy_with_logits(
                fg_pred, fg_true, weight_map, reduction=self.reduction
            )
            ce_bg = torch.nn.functional.binary_cross_entropy_with_logits(
                bg_pred, bg_true.float(), None, reduction=self.reduction
            )
            ce = ce_fg + ce_bg
        else:
            weight_map = self.create_balanced_weight_map(y_true)
            ce = torch.nn.functional.binary_cross_entropy_with_logits(y_pred, y_true, weight_map, reduction=self.reduction)

        tv = self.tv(y_pred, y_true)

        return tv + ce


class FocalTverskyDiceCE(nn.Module):
    """Focal Tversky Dice Cross-Entropy Loss for segmentation tasks."""

    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.4,
        gamma: float = 0.8,
        focal_gamma: float = 2.0,
        to_onehot_y: bool = True,
        background: bool = True,
        sigmoid: bool = False,
        softmax: bool = True,
        reduction: str = "mean",
        mode: str = "binary",
        masked: bool = False,
    ) -> None:
        """Focal Tversky Dice Cross-Entropy Loss for segmentation tasks.

        Args:
            alpha (float, optional): Weight for false positives in the focal Tversky term. Defaults to 0.7.
            beta (float, optional): Weight for false negatives in the focal Tversky term. Defaults to 0.4.
            gamma (float, optional): Weighting factor balancing the focal Tversky and Dice CE terms in the combined loss. Defaults to 0.8.
            focal_gamma (float, optional): Focusing parameter for the focal Tversky term. Defaults to 2.0.
            to_onehot_y (bool, optional): Whether to convert target to one-hot encoding. Defaults to True.
            background (bool, optional): Whether to include the background class in the loss. Defaults to True.
            sigmoid (bool, optional): Whether to apply sigmoid activation to the input. Defaults to False.
            softmax (bool, optional): Whether to apply softmax activation to the input. If True while mode is "binary", mode is switched to "multiclass". Defaults to True.
            reduction (str, optional): Reduction method to apply to the loss. Defaults to "mean".
            mode (str, optional): Loss mode, "binary" or "multiclass". Defaults to "binary".
            masked (bool, optional): Whether to use MaskedDiceLoss instead of DiceCELoss for the Dice CE term. Defaults to False.
        """
        super(FocalTverskyDiceCE, self).__init__()
        self.alpha = alpha  # FP reduction
        self.beta = beta  # FN reduction
        self.gamma = gamma  # weighting factor for the two losses
        self.focal_gamma = focal_gamma
        self.reduction = reduction
        self.masked = masked
        self.mode = mode
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        self.sigmoid = sigmoid
        self.background = background

        # For 2-class softmax, use multiclass mode
        if softmax and mode == "binary":
            self.mode = "multiclass"

        if self.mode == "multiclass":
            self.ftv = MultiClassFocalTverskyLoss(
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.focal_gamma,
                mode="multiclass",
            )
        else:
            self.ftv = FocalTverskyLoss(
                alpha=self.alpha,
                beta=self.beta,
                include_background=background,
                to_onehot_y=to_onehot_y,
                sigmoid=sigmoid,
                reduction=reduction,
                gamma=self.focal_gamma,
            )

        if masked:
            self.dice_ce = MaskedDiceLoss(
                include_background=background,
                to_onehot_y=to_onehot_y,
                softmax=softmax,
                sigmoid=sigmoid,
                squared_pred=True,
                reduction=reduction,
            )
        else:
            self.dice_ce = DiceCELoss(
                include_background=background,
                to_onehot_y=to_onehot_y,
                softmax=softmax,
                sigmoid=sigmoid,
                squared_pred=True,
                reduction=reduction,
            )

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Focal Tversky Dice Cross-Entropy Loss.

        Args:
            y_pred (torch.Tensor): Predicted logits of shape [B, C, H, W]
            y_true (torch.Tensor): Ground truth labels of shape [B, H, W] (class indices) or [B, C, H, W] (one-hot)
            mask (Optional[torch.Tensor]): Optional mask tensor

        Returns:
            torch.Tensor: Combined focal tversky and dice cross-entropy loss
        """
        # y_pred: [B, C, H, W] logits. Both loss terms apply their own activation
        # internally, so raw logits are passed straight through — sigmoid/softmax
        # must not be applied here as well.
        # y_true: [B, H, W] (class indices) or [B, C, H, W] (one-hot)
        # For multiclass, MultiClassFocalTverskyLoss expects class indices, not one-hot
        if self.mode == "multiclass":
            # Convert one-hot to class indices if needed
            if y_true.dim() == 4 and y_true.shape[1] > 1:
                y_true_indices = torch.argmax(y_true, dim=1)
            else:
                y_true_indices = y_true
            ftv = self.ftv(y_pred, y_true_indices)
            dice_ce = self.dice_ce(y_pred, y_true)
        else:
            # For binary, can use one-hot if needed
            if self.to_onehot_y and y_true.dim() == 3:
                y_true_oh = F.one_hot(y_true.long(), num_classes=y_pred.shape[1]).permute(0, 3, 1, 2).float()
            else:
                y_true_oh = y_true
            ftv = self.ftv(y_pred, y_true_oh)
            if self.masked:
                dice_ce = self.dice_ce(y_pred, y_true, mask)
            else:
                dice_ce = self.dice_ce(y_pred, y_true)

        combined_loss = (1 - self.gamma) * ftv + self.gamma * dice_ce

        # Handle MetaTensor conversion for mixed precision training
        # MetaTensors can occur with autocast and need to be converted to regular tensors
        loss_type_str = str(type(combined_loss))

        if "MetaTensor" in loss_type_str:
            # Convert MetaTensor to regular tensor while preserving gradients and computational graph
            combined_loss = combined_loss.as_tensor().to(torch.float32)

        return combined_loss
