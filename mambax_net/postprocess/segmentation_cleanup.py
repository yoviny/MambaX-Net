"""
Segmentation postprocessing utilities for medical image analysis.

This module provides functions for cleaning up segmentation masks by removing
small blobs and applying distance-based filtering.
"""

from typing import Union, Tuple
import cupy as cp
import numpy as np
import torch
from cupyx.scipy.ndimage import label


def distance_based_cleanup(
    segmentation: Union[torch.Tensor, np.ndarray], min_size: int = 75, distance_threshold: float = 100
) -> torch.Tensor:
    """
    Remove small blobs from the segmentation mask based on distance from image center.

    Args:
        segmentation: Input segmentation mask as torch.Tensor or numpy array.
                     Expected shape: (H, W, D, C) for multi-channel.
        min_size: Minimum size (in voxels) for blobs to be retained.
        distance_threshold: Maximum distance from image center for small blobs.
                          Blobs farther than this distance will be removed.

    Returns:
        torch.Tensor: Cleaned segmentation mask with small distant blobs removed.
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if not isinstance(segmentation, torch.Tensor):
        segmentation = torch.tensor(segmentation, dtype=torch.float64)

    if segmentation.ndim > 3:
        for i in range(segmentation.shape[-1]):
            segmentation[..., i] = min_voxels(segmentation[..., i], min_size=min_size)

    image_h = segmentation.shape[0]
    image_w = segmentation.shape[1]

    # Calculate the center of the image
    image_center = torch.tensor([image_w // 2, image_h // 2], dtype=torch.float64).to(device)

    for i in range(segmentation.shape[-1]):
        labeled_seg, num_seg_blobs = label(
            cp.asarray(segmentation[..., i]),
            structure=cp.ones((3, 3, 3), dtype=cp.uint8),
        )
        labeled_seg = torch.from_numpy(cp.asnumpy(labeled_seg)).to(device)
        num_seg_blobs = torch.from_numpy(cp.asnumpy(num_seg_blobs)).to(device)

        for blob_id in range(1, num_seg_blobs + 1):
            blob_mask = (labeled_seg == blob_id).to(device)
            if not blob_mask.any():
                continue

            coords = torch.nonzero(blob_mask, as_tuple=False).to(device)
            center = coords.float().mean(dim=0)[0:2]  # Exclude the z-coordinate

            # Calculate the distance from the center of the image
            distances = torch.norm(center.float() - image_center.float())

            # If the distance is greater than the threshold, remove the blob
            if distances > distance_threshold:
                labeled_seg[blob_mask] = 0

        segmentation[..., i] = labeled_seg.to(torch.float64)
    return segmentation


def min_voxels(segmentation: Union[torch.Tensor, np.ndarray], min_size: int = 100) -> torch.Tensor:
    """
    Remove connected components smaller than the specified minimum size.

    Args:
        segmentation: Input segmentation mask as torch.Tensor or numpy array.
                     Expected shape: (H, W, D) for 3D data.
        min_size: Minimum size (in voxels) for connected components to be retained.

    Returns:
        torch.Tensor: Segmentation mask with small components removed.
    """
    device = torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
    if not isinstance(segmentation, torch.Tensor):
        segmentation = torch.tensor(segmentation, dtype=torch.float64)

    labeled_seg, num_seg_blobs = label(
        cp.asarray(segmentation),
        structure=cp.ones((3, 3, 3), dtype=cp.uint8),
    )
    labeled_seg = torch.from_numpy(cp.asnumpy(labeled_seg)).to(device)
    num_seg_blobs = torch.from_numpy(cp.asnumpy(num_seg_blobs)).to(device)

    if num_seg_blobs == 0:
        return segmentation

    sizes = torch.bincount(labeled_seg.ravel())
    keep_blobs = sizes >= min_size
    keep_blobs[0] = False  # Ignore background

    return (labeled_seg * keep_blobs[labeled_seg]).to(torch.float64)


def postprocess_segmentation(
    segmentation: Union[torch.Tensor, np.ndarray], min_size: int = 100, distance_threshold: float = 100
) -> torch.Tensor:
    """
    Post-process segmentation mask by removing small blobs and applying distance-based filtering.

    This function handles both binary and multilabel segmentation masks. For multilabel masks,
    it processes each label separately before combining the results.

    Args:
        segmentation: Input segmentation mask as torch.Tensor or numpy array.
                     Expected shape: (H, W, D) for 3D data.
        min_size: Minimum size (in voxels) for connected components to be retained.
        distance_threshold: Maximum distance from image center for small blobs.

    Returns:
        torch.Tensor: Post-processed segmentation mask.
    """
    # Make a copy to avoid modifying the original
    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.clone()
    elif isinstance(segmentation, np.ndarray):
        segmentation = segmentation.copy()

    if isinstance(segmentation, np.ndarray):
        segmentation = torch.tensor(segmentation, dtype=torch.float64)

    # Check for multilabel segmentation
    if len(torch.unique(segmentation)) > 2:
        # For multilabel segmentation, split into separate channels for each label (assuming labels 1 and 2)
        seg_split_labels = torch.stack(
            [(segmentation == 1).to(segmentation.dtype), (segmentation == 2).to(segmentation.dtype)], dim=-1
        )

        # Process each label separately
        for i in range(seg_split_labels.shape[-1]):
            seg_split_labels[..., i] = distance_based_cleanup(
                seg_split_labels[..., i], min_size=min_size, distance_threshold=distance_threshold
            )

        # Combine the processed labels back into a single segmentation
        pz = seg_split_labels[..., 0]
        tz = seg_split_labels[..., 1]

        processed_segmentation = torch.zeros_like(segmentation)
        processed_segmentation[pz > 0] = 1
        processed_segmentation[tz > 0] = 2
        processed_segmentation = torch.clip(processed_segmentation, 0, 2)
    else:
        processed_segmentation = distance_based_cleanup(segmentation, min_size=min_size, distance_threshold=distance_threshold)

    return processed_segmentation
