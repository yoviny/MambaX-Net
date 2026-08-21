from typing import List, Optional, Tuple

import torch


def _write_patch(
    volume: torch.Tensor,
    patch: torch.Tensor,
    coord: Tuple[List[int], List[int], List[int], List[int]],
) -> None:
    """Copy a patch into volume at the location given by coord, in place.

    Clips the patch to volume's bounds along each spatial axis and does nothing
    if the resulting overlap is empty (non-positive extent) in any dimension.

    Args:
        volume (torch.Tensor): Destination tensor of shape [C, H, W, D]; modified in place.
        patch (torch.Tensor): Source patch tensor of shape [C, h, w, d].
        coord (Tuple[List[int], List[int], List[int], List[int]]): Patch coordinates as
            (c_coord, h_coord, w_coord, d_coord), where h_coord/w_coord/d_coord are each
            a two-element [start, end] range in volume space.
    """
    _, h_coord, w_coord, d_coord = coord
    h0, h1 = int(h_coord[0]), int(h_coord[1])
    w0, w1 = int(w_coord[0]), int(w_coord[1])
    d0, d1 = int(d_coord[0]), int(d_coord[1])

    h1 = min(h1, volume.shape[1])
    w1 = min(w1, volume.shape[2])
    d1 = min(d1, volume.shape[3])

    patch_h = h1 - h0
    patch_w = w1 - w0
    patch_d = d1 - d0
    if patch_h <= 0 or patch_w <= 0 or patch_d <= 0:
        return

    volume[:, h0:h1, w0:w1, d0:d1] = patch[:, :patch_h, :patch_w, :patch_d]


def recreate_image(
    batch_sz: int,
    logits: torch.Tensor,
    probs: torch.Tensor,
    img_patches: torch.Tensor,
    mask_patches: torch.Tensor,
    img_size: List[Tuple[int, ...]],
    mask_size: List[Tuple[int, ...]],
    coords: List[Tuple[List[int], List[int], List[int], List[int]]],
    patches_per_image: Optional[List[int]] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Recreate full-size images from patches by assembling them according to their coordinates.

    This function takes patches of images, masks, logits, and probabilities along with their
    coordinate information and reconstructs the original full-size tensors. It handles
    overlapping patches and boundary conditions for medical image segmentation tasks.

    Args:
        batch_sz (int): Original batch size before patch extraction
        logits (torch.Tensor): Predicted logits from model of shape [N, C, H, W, D]
        probs (torch.Tensor): Predicted probabilities from model of shape [N, C, H, W, D]
        img_patches (torch.Tensor): Tensor containing all image patches
        mask_patches (torch.Tensor): Tensor containing all mask patches
        img_size (List[Tuple[int, ...]]): Original image sizes for each item in batch
        mask_size (List[Tuple[int, ...]]): Original mask sizes for each item in batch
        coords (List[Tuple[List[int], List[int], List[int], List[int]]]):
            Patch coordinates as (c_coord, h_coord, w_coord, d_coord) for each patch
        patches_per_image (Optional[List[int]]): Number of patches for each image in batch.
            If None, assumes uniform distribution (legacy behavior).

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            A tuple containing:
            - new_logits: List of reconstructed logit tensors
            - new_probabilities: List of reconstructed probability tensors
            - new_images: List of reconstructed image tensors
            - new_masks: List of reconstructed mask tensors

    Note:
        This function handles variable numbers of patches per image when patches_per_image
        is provided, enabling batch_size > 1 with different patch counts per image.
    """
    true_batch_sz: int = logits.shape[0]

    # Calculate patch indices for each image
    if patches_per_image is None:
        # Legacy behavior: assume uniform distribution
        multiplier: int = true_batch_sz // batch_sz
        patch_indices = [(i * multiplier, (i + 1) * multiplier) for i in range(batch_sz)]
    else:
        # Variable patches per image
        patch_indices = []
        start_idx = 0
        for count in patches_per_image:
            patch_indices.append((start_idx, start_idx + count))
            start_idx += count

    new_images, new_masks, new_logits, new_probabilities = [], [], [], []
    for i, (im_sz, m_sz) in enumerate(zip(img_size, mask_size)):
        # create empty image and mask
        new_img: torch.Tensor = torch.zeros(im_sz, dtype=img_patches.dtype)
        new_mask: torch.Tensor = torch.zeros(m_sz, dtype=mask_patches.dtype)
        new_logit: torch.Tensor = torch.zeros(m_sz, dtype=logits.dtype)
        new_probs: torch.Tensor = torch.zeros(m_sz, dtype=probs.dtype)

        # Get patches for this image using calculated indices
        start_idx, end_idx = patch_indices[i]

        # Skip images with no patches
        if start_idx == end_idx:
            print(f"Warning: Image {i} has no patches, skipping reconstruction")
            new_images.append(new_img)
            new_masks.append(new_mask)
            new_logits.append(new_logit)
            new_probabilities.append(new_probs)
            continue

        img = img_patches[start_idx:end_idx]
        mask = mask_patches[start_idx:end_idx]
        logits_ = logits[start_idx:end_idx]
        probs_ = probs[start_idx:end_idx]
        coords_ = coords[start_idx:end_idx]

        for logit, prob, img_patch, mask_patch, coord in zip(logits_, probs_, img, mask, coords_):
            _write_patch(new_img, img_patch, coord)
            _write_patch(new_mask, mask_patch, coord)
            _write_patch(new_logit, logit, coord)
            _write_patch(new_probs, prob, coord)

        new_images.append(new_img)
        new_masks.append(new_mask)
        new_logits.append(new_logit)
        new_probabilities.append(new_probs)
    return new_logits, new_probabilities, new_images, new_masks
