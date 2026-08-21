# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by: Yovin Yahathugoda (yovin.yahathugoda@kcl.ac.uk)
# Modifications:
#   - Route the cross-entropy term to BCE for multi-label (sigmoid) targets


from __future__ import annotations

import torch
from monai.losses import DiceCELoss


class ModifiedDiceCELoss(DiceCELoss):
    """
    Dice Cross-Entropy loss for multi-label targets.

    MONAI's DiceCELoss only selects the binary cross-entropy term when the input
    has a single channel; anything wider falls through to softmax cross-entropy.
    The prostate zones here are multi-label — a voxel inside the peripheral zone
    is also inside the whole gland — so the channels are not mutually exclusive
    and softmax is the wrong family. This subclass dispatches on the activation
    instead of the channel count, so a sigmoid model gets per-channel BCE.

    Args:
        *args: Additional positional arguments passed to parent DiceCELoss
        **kwargs: Additional keyword arguments passed to parent DiceCELoss
    """

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice + cross-entropy over logits.

        Args:
            input (torch.Tensor): Input logits tensor
            target (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Weighted sum of the Dice and cross-entropy terms
        """
        if len(input.shape) != len(target.shape):
            raise ValueError(
                "the number of dimensions for input and target should be the same, "
                f"got shape {input.shape} and {target.shape}."
            )

        use_bce = self.dice.sigmoid or input.shape[1] == 1
        ce_loss = self.bce(input, target) if use_bce else self.ce(input, target)
        return self.lambda_dice * self.dice(input, target) + self.lambda_ce * ce_loss

    def ce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute CrossEntropy loss for the input logits and target.

        Will remove the channel dim according to PyTorch CrossEntropyLoss:
        https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

        Args:
            input (torch.Tensor): Input logits tensor
            target (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Computed cross-entropy loss
        """
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        if n_pred_ch != n_target_ch and n_target_ch == 1:
            target = torch.squeeze(target, dim=1)
            target = target.long()
        elif not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.cross_entropy(input, target)

    def bce(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary CrossEntropy loss for the input logits and target.

        Args:
            input (torch.Tensor): Input logits tensor
            target (torch.Tensor): Target tensor

        Returns:
            torch.Tensor: Computed binary cross-entropy loss
        """
        if not torch.is_floating_point(target):
            target = target.to(dtype=input.dtype)

        return self.binary_cross_entropy(input, target)
