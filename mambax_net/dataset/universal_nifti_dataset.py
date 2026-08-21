"""Universal NIfTI Dataset for inference on arbitrary NIfTI image directories.

This dataset class is designed to work with any NIfTI image directory structure,
making it suitable for inference on new datasets without requiring specific
DataFrame structures or mask paths.
"""

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import monai
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from monai.transforms import apply_transform

import mambax_net.utilities.nifti_utilities as nutil

TensorTuple = Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...], Tuple[int, ...]]
TensorTupleWithNifti = Tuple[
    torch.Tensor, torch.Tensor, Tuple[int, ...], Tuple[int, ...], Any, Any
]
TensorTupleWithOriginalInfo = Tuple[
    torch.Tensor,
    torch.Tensor,
    Tuple[int, ...],
    Tuple[int, ...],
    Any,
    Any,
    Tuple[int, ...],
    Tuple[float, ...],
    np.ndarray,
]
PatchTuple = Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]
PatchList = List[PatchTuple]
DataDict = Dict[str, torch.Tensor]
GetItemReturnType = Tuple[
    str,
    PatchList,
    Tuple[int, ...],
    Tuple[int, ...],
    Any,
    Any,
    Tuple[int, ...],
    Tuple[float, ...],
    np.ndarray,
]


class UniversalNiftiDataset(monai.data.Dataset):
    """Universal dataset for NIfTI image inference.

    This dataset class accepts a simple list of filenames or a DataFrame with a
    'filename' column, making it compatible with any NIfTI dataset structure.
    """

    def __init__(
        self,
        data: Union[pd.DataFrame, List[str]],
        patch_iter: Optional[Callable],
        img_folder: str,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        patch_transform: Optional[Callable] = None,
        with_coordinates: bool = False,
        extension: str = ".nii.gz",
        filename_column: str = "filename",
    ) -> None:
        """Initialize the Universal NIfTI dataset.

        Args:
            data: Either a pandas DataFrame with filenames or a list of filenames.
                  If DataFrame, should have a column specified by filename_column.
                  If list, should contain filenames (with or without extension).
            patch_iter: Function to iterate over patches (from MONAI PatchIterd).
            img_folder: Path to the folder containing NIfTI images.
            preprocess: Preprocessing transforms to apply.
            transform: Additional transforms to apply.
            patch_transform: Transforms to apply to patches.
            with_coordinates: Whether to include patch coordinates.
            extension: File extension (default: ".nii.gz").
            filename_column: Column name in DataFrame containing filenames.
        """
        if isinstance(data, list):
            self.df = pd.DataFrame({filename_column: data})
        else:
            self.df = data.copy()

        self.filename_column = filename_column
        self.patch_iter = patch_iter
        self.img_folder = img_folder
        self.preprocess = preprocess
        self.transform = transform
        self.patch_transform = patch_transform
        self.with_coordinates = with_coordinates
        self.extension = extension

        if not self.extension.startswith("."):
            self.extension = "." + self.extension

        assert self.preprocess is not None, "preprocess must be defined for inference"

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return len(self.df)

    def _get_filename(self, idx: int) -> str:
        """Get the filename for a given index.

        Args:
            idx: Index of the sample.

        Returns:
            Filename without extension.
        """
        row = self.df.iloc[idx]
        filename = row[self.filename_column]

        # Remove extension if present
        for ext in [".nii.gz", ".nii", ".gz"]:
            if filename.endswith(ext):
                filename = filename[: -len(ext)]
                break

        return filename

    def _get_img_path(self, filename: str) -> str:
        """Get the full image path for a filename.

        Args:
            filename: Base filename without extension.

        Returns:
            Full path to the image file.
        """
        return os.path.join(self.img_folder, f"{filename}{self.extension}")

    def load(self, img_path: str) -> TensorTupleWithOriginalInfo:
        """Load a NIfTI image and prepare it for inference.

        Args:
            img_path: Path to the NIfTI image file.

        Returns:
            Tuple containing:
                - img: Image tensor
                - mask: Placeholder mask tensor (zeros)
                - img_shape: Shape of processed image
                - mask_shape: Shape of mask
                - img_nii: Processed NIfTI object
                - mask_nii: Placeholder mask NIfTI object
                - original_shape: Original image shape before processing
                - original_spacing: Original voxel spacing
                - original_orientation: Original image orientation as axis codes (e.g., ('L', 'A', 'S'))
        """
        _, img_nii = nutil.load(img_path)

        # Extract original orientation from header extensions first
        original_orientation = None
        for extension in img_nii.header.extensions:
            if extension.get_code() == 40:  # Custom extension code
                try:
                    metadata_json = extension.get_content().decode("utf-8")
                    metadata = json.loads(metadata_json)
                    original_orientation = metadata.get("original_orientation")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

        # If no orientation in header extensions, compute from affine matrix
        if original_orientation is None:
            original_orientation = nib.aff2axcodes(img_nii.affine)

        img = img_nii.get_fdata()

        # Store original image properties before preprocessing
        original_shape = img.shape
        original_spacing = img_nii.header.get_zooms()[:3]
        original_affine = img_nii.affine.copy()

        # Validate image orientation - skip abnormal acquisitions
        if len(original_spacing) >= 3:
            max_spacing_idx = np.argmax(original_spacing[:3])
            if max_spacing_idx != 2:  # Expect thickest spacing in 3rd dimension
                raise ValueError(
                    f"Abnormal image orientation detected: thickest slice spacing "
                    f"({original_spacing[max_spacing_idx]:.2f}mm) is in dimension {max_spacing_idx}, "
                    f"expected in dimension 2. Shape: {original_shape}, Spacing: {original_spacing}. "
                    f"This may be a sagittal/coronal acquisition or corrupted header. Skipping."
                )

        img = np.expand_dims(img, axis=0)
        img_nii = img_nii.__class__(img, img_nii.affine, img_nii.header)
        img_nii.header.set_zooms((1.0,) + img_nii.header.get_zooms()[:3])
        img_nii.header.set_data_shape(img.shape)

        # Create placeholder mask (zeros with 3 channels for WP, PZ, TZ)
        mask = np.zeros((3,) + img.shape[-3:], dtype=np.float32)
        mask_nii = img_nii.__class__(mask, img_nii.affine, img_nii.header)
        mask_nii.header.set_zooms(img_nii.header.get_zooms())
        mask_nii.header.set_data_shape(mask.shape)

        # Apply preprocessing
        if self.preprocess is not None:
            data = self.preprocess({"image": img_nii, "mask": mask_nii})
        else:
            data = {"image": img_nii, "mask": mask_nii}

        img_nii = data["image"]
        mask_nii = data["mask"]

        img = torch.tensor(data["image"].get_fdata().copy(), dtype=torch.float32)
        mask = torch.tensor(data["mask"].get_fdata().copy(), dtype=torch.float16)

        img_shape = img.shape
        mask_shape = mask.shape

        return (
            img,
            mask,
            img_shape,
            mask_shape,
            img_nii,
            mask_nii,
            original_shape,
            original_spacing,
            original_orientation,  # Return orientation tuple, not affine
        )

    def __getitem__(self, idx: int) -> GetItemReturnType:
        """Get a single item from the dataset.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Tuple containing filename, patches, shapes, and NIfTI metadata.
            Returns None for problematic files that should be skipped.
        """
        filename = self._get_filename(idx)
        img_path = self._get_img_path(filename)

        try:
            (
                img,
                mask,
                img_shape,
                mask_shape,
                img_nii,
                mask_nii,
                original_shape,
                original_spacing,
                original_affine,
            ) = self.load(img_path)
        except ValueError as e:
            # Handle validation errors (e.g., abnormal orientations)
            # Return None to signal this item should be skipped
            error_msg = str(e)
            if "Abnormal image orientation" in error_msg:
                print(f"Skipping {filename}: {error_msg[:150]}")
                return None
            else:
                print(f"Error loading {img_path}: {e}")
                raise
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            raise

        data = {"image": img, "mask": mask}

        if self.transform is not None:
            data = self.transform(data)

        patches = []

        if self.patch_iter is not None:
            for ret, others in self.patch_iter(data):
                img_patch, mask_patch = ret["image"], ret["mask"]
                if self.patch_transform is not None:
                    img_patch, mask_patch = apply_transform(
                        (img_patch, mask_patch), self.patch_transform, map_items=False
                    )
                if self.with_coordinates and len(others) > 0:
                    patches.append((img_patch, mask_patch, others, others))
                else:
                    patches.append(
                        (img_patch, mask_patch, np.asarray([0, 0]), np.asarray([0, 0]))
                    )

            del data

            # if patch count isn't exactly 2, drop the last patch
            if len(patches) / 2 != 1:
                patches = patches[:-1]

            return (
                filename,
                patches,
                img_shape,
                mask_shape,
                img_nii,
                mask_nii,
                original_shape,
                original_spacing,
                original_affine,
            )
        else:
            return data


def create_dataframe_from_directory(
    img_folder: str,
    extension: str = ".nii.gz",
    pattern: Optional[str] = None,
) -> pd.DataFrame:
    """Create a DataFrame from all NIfTI files in a directory.

    Args:
        img_folder: Path to the folder containing NIfTI images.
        extension: File extension to look for.
        pattern: Optional regex pattern to filter filenames.

    Returns:
        DataFrame with 'filename' column containing base filenames.
    """

    files = []
    for f in os.listdir(img_folder):
        if f.endswith(extension):
            base_name = f.replace(extension, "")
            if pattern is None or re.match(pattern, base_name):
                files.append(base_name)

    files.sort()
    return pd.DataFrame({"filename": files})
