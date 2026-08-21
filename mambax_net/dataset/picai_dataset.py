import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import monai
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from monai.transforms import apply_transform
from tqdm.auto import tqdm

import mambax_net.utilities.nifti_utilities as nutil

# Type definitions
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
LoadReturnType = Union[
    Dict[str, Any], TensorTuple, TensorTupleWithNifti, TensorTupleWithOriginalInfo
]
GetItemReturnType = Union[
    DataDict,
    PatchList,
    Tuple[str, PatchList, Tuple[int, ...], Tuple[int, ...], Any, Any],
    Tuple[
        str,
        PatchList,
        Tuple[int, ...],
        Tuple[int, ...],
        Any,
        Any,
        Tuple[int, ...],
        Tuple[float, ...],
        np.ndarray,
    ],
]


class PicSegDataset(monai.data.Dataset):
    """Dataset for prostate segmentation."""

    def __init__(
        self,
        df: pd.DataFrame,
        patch_iter: Optional[Callable],
        img_folder: str,
        wp_mask_folder: str,
        pz_tz_mask_folder: str,
        preprocess: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        patch_transform: Optional[Callable] = None,
        with_coordinates: bool = False,
        cache: bool = False,
        cache_dir: str = "cache",
        mode: str = "train",
        dataset_name: str = "picai",
    ) -> None:
        """Dataset for prostate segmentation.

        Args:
            df (pd.DataFrame): DataFrame containing the dataset information.
            patch_iter (callable): Function to iterate over patches.
            img_folder (str): Path to the image folder.
            wp_mask_folder (str): Path to the white matter mask folder.
            pz_tz_mask_folder (str): Path to the tumor mask folder.
            preprocess (callable, optional): Preprocessing function.
            transform (callable, optional): Transformation function.
            patch_transform (callable, optional): Patch transformation function.
            with_coordinates (bool, optional): Whether to include coordinates.
            cache (bool, optional): Whether to cache the dataset.
            cache_dir (str, optional): Directory for caching.
            mode (str, optional): Mode of the dataset (train, valid, test).
            dataset_name (str, optional): Name of the dataset.
        """
        self.df = df
        self.patch_iter = patch_iter
        self.img_folder = img_folder
        self.wp_mask_folder = wp_mask_folder
        self.pz_tz_mask_folder = pz_tz_mask_folder
        self.preprocess = preprocess
        self.transform = transform
        self.patch_transform = patch_transform
        self.with_coordinates = with_coordinates
        self.cache = cache
        self.cache_dir = cache_dir
        self.mode = mode
        self.dataset_name = dataset_name

        os.makedirs(cache_dir, exist_ok=True)

        assert (self.preprocess is not None) or (
            self.mode == "fingerprint"
        ), "preprocess must be defined"
        assert self.mode in [
            "train",
            "valid",
            "test",
            "infer",
            "train_seg_maps",
            "fingerprint",
        ], "mode must be either train, valid, test, infer, train_seg_maps, or fingerprint"
        assert self.dataset_name in [
            "picai",
            "AS",
        ], "dataset_name must be either picai or AS"

        self.extenstion = ".nii.gz"

        if self.cache:
            self.preprocess_and_cache_all()

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.df)

    def load(
        self, return_nifti: bool = False, return_original_info: bool = False
    ) -> LoadReturnType:
        """Load the image and mask NIfTI files.

        Args:
            return_nifti (bool, optional): Whether to return the NIfTI objects. Defaults to False.
            return_original_info (bool, optional): Whether to return original image shape and spacing. Defaults to False.

        Raises:
            ValueError: If the image or mask cannot be loaded.

        Returns:
            dict: A dictionary containing the loaded image and mask.
        """
        _, img_nii = nutil.load(self.img_path)

        # Extract original orientation from header extensions
        original_orientation = None
        for extension in img_nii.header.extensions:
            if extension.get_code() == 40:  # custom extension code
                try:
                    metadata_json = extension.get_content().decode("utf-8")
                    metadata = json.loads(metadata_json)
                    original_orientation = metadata.get("original_orientation")
                    break
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue

        img = img_nii.get_fdata()

        # Store original image properties before preprocessing
        original_shape = img.shape
        original_spacing = img_nii.header.get_zooms()[:3]
        original_affine = img_nii.affine.copy()

        img = np.expand_dims(img, axis=0)
        img_nii = img_nii.__class__(img, img_nii.affine, img_nii.header)
        img_nii.header.set_zooms((1.0,) + img_nii.header.get_zooms()[:3])
        img_nii.header.set_data_shape(img.shape)

        # Check if masks are available (check both folder and path attributes)
        masks_not_available = (
            self.wp_mask_folder is None
            or self.pz_tz_mask_folder is None
            or self.wp_mask_folder == "None"
            or self.pz_tz_mask_folder == "None"
            or self.wp_mask_path is None
            or self.pz_tz_mask_path is None
        )

        if masks_not_available:
            # running inference mode without masks. we replace mask with
            # zeros of same shape as image, but with 3 channels
            # we take the shape from img_nii (last 3 dimensions) and add a channel dimension of 3
            mask = np.zeros((3,) + img.shape[-3:], dtype=np.float32)
            mask_nii = img_nii.__class__(
                mask, img_nii.affine.copy(), img_nii.header.copy()
            )
            mask_nii.header.set_zooms(img_nii.header.get_zooms())
            mask_nii.header.set_data_shape(mask.shape)
        else:
            try:
                mask = self.combine_masks(self.wp_mask_path, self.pz_tz_mask_path)
            except Exception as e:
                try:
                    # check if mask name has 4 digits after the '-', if not subtract a 0 from the start of the string
                    pattern = r"(ProstateX-0)(\d{3})"

                    self.wp_mask_path = re.sub(
                        pattern, r"ProstateX-\2", self.wp_mask_path
                    )
                    mask = self.combine_masks(self.wp_mask_path, self.pz_tz_mask_path)
                except Exception as e:
                    raise ValueError(f"Could not load mask for {self.img_path}")

            mask = rearrange(mask, "h w d c -> c h w d")
            mask_nii = img_nii.__class__(mask, img_nii.affine, img_nii.header)
            mask_nii.header.set_zooms(img_nii.header.get_zooms())
            mask_nii.header.set_data_shape(mask.shape)

        if self.preprocess is not None:
            data = self.preprocess({"image": img_nii, "mask": mask_nii})
        else:
            data = {"image": img_nii, "mask": mask_nii}

        img_nii = data["image"]
        mask_nii = data["mask"]

        img, mask = torch.tensor(
            data["image"].get_fdata().copy(), dtype=torch.float32
        ), torch.tensor(data["mask"].get_fdata().copy(), dtype=torch.float16)

        # Capture shape after preprocessing - this should be the cropped/padded size
        img_shape = img.shape
        mask_shape = mask.shape

        # Validate dimensions - skip images with insufficient depth
        if self.mode == "infer":
            if img_shape[-1] < 1 or mask_shape[-1] < 1:
                raise ValueError(
                    f"Image has insufficient depth: img_shape={img_shape}, mask_shape={mask_shape}. File: {self.img_path}"
                )
            if img_shape[1] != 384 or img_shape[2] != 384:
                print(
                    f"Warning: Image shape after preprocessing is {img_shape}, expected (1, 384, 384, D)"
                )
                print(f"File: {self.img_path}")

        if self.mode == "fingerprint":
            return data
        else:
            if return_nifti and return_original_info:
                return (
                    img,
                    mask,
                    img_shape,
                    mask_shape,
                    img_nii,
                    mask_nii,
                    original_shape,
                    original_spacing,
                    original_orientation,
                )
            elif return_nifti:
                return img, mask, img_shape, mask_shape, img_nii, mask_nii
            else:
                return img, mask, img_shape, mask_shape

    def preprocess_and_cache_all(self) -> None:
        """Preprocess and cache all data."""
        print("Preprocessing and caching all data")
        if self.dataset_name == "picai":
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
                if self.mode in ("train", "valid"):
                    patient_id = row["patient_id"]
                    study_id = row["study_id"]
                    self.img_path = (
                        f"{self.img_folder}/{patient_id}_{study_id}{self.extenstion}"
                    )
                    self.wp_mask_path = f"{self.wp_mask_folder}/{patient_id}_{study_id}{self.extenstion}"
                    self.pz_tz_mask_path = f"{self.pz_tz_mask_folder}/{patient_id}_{study_id}{self.extenstion}"
                else:
                    filename = row["prostateX_filename"]
                    self.img_path = f"{self.img_folder}/{filename}{self.extenstion}"
                    self.wp_mask_path = (
                        f"{self.wp_mask_folder}/{filename}{self.extenstion}"
                    )
                    self.pz_tz_mask_path = (
                        f"{self.pz_tz_mask_folder}/{filename}{self.extenstion}"
                    )

                try:
                    img, mask, _, _ = self.load()
                except Exception as e:
                    print(e)
                    pass

                data = {"image": img, "mask": mask}

                if self.mode in ("train", "valid"):
                    cache_filename = (
                        f"{self.cache_dir}/{patient_id}_{study_id}_preprocessed.joblib"
                    )
                    with open(cache_filename, "wb") as f:
                        joblib.dump(data, f, compress="zlib")
                else:
                    cache_filename = f"{self.cache_dir}/{filename}_preprocessed.joblib"
                    with open(cache_filename, "wb") as f:
                        joblib.dump(data, f, compress="zlib")
        elif self.dataset_name == "AS":
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
                study_id = row["study_id"]
                self.img_path = f"{self.img_folder}/{study_id}{self.extenstion}"
                self.wp_mask_path = f"{self.wp_mask_folder}/{study_id}{self.extenstion}"

                if self.pz_tz_mask_folder is not None:
                    self.pz_tz_mask_path = (
                        f"{self.pz_tz_mask_folder}/{study_id}{self.extenstion}"
                    )

                try:
                    img, mask, _, _ = self.load()
                except Exception as e:
                    print(e)
                    pass

                data = {"image": img, "mask": mask}

                cache_filename = f"{self.cache_dir}/{study_id}_preprocessed.joblib"
                with open(cache_filename, "wb") as f:
                    joblib.dump(data, f, compress="zlib")

    def __getitem__(self, idx: int) -> GetItemReturnType:
        """Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the image and mask tensors.
        """
        if self.mode in ("train", "valid"):
            row = self.df.iloc[idx]

            if self.dataset_name == "picai":
                filename = f"{row['patient_id']}_{row['study_id']}"
            elif self.dataset_name == "AS":
                filename = row["study_id"]

            cache_filename = f"{self.cache_dir}/{filename}_preprocessed.joblib"

            with open(cache_filename, "rb") as f:
                data = joblib.load(f)
            data["study_id"] = filename
            if "mask_noise" in self.df.columns:
                data["mask_noise"] = row["mask_noise"]
        elif self.mode == "test":
            row = self.df.iloc[idx]

            if self.dataset_name == "picai":
                filename = row["prostateX_filename"]
            elif self.dataset_name == "AS":
                filename = row["study_id"]

            cache_filename = f"{self.cache_dir}/{filename}_preprocessed.joblib"

            with open(cache_filename, "rb") as f:
                data = joblib.load(f)
            data["study_id"] = filename
        elif self.mode == "fingerprint":
            assert (
                self.dataset_name == "AS"
            ), "fingerprint mode is only supported for AS dataset"
            row = self.df.iloc[idx]
            patient_id = row["patient_id"]
            study_id = row["study_id"]
            self.img_path = (
                f"{self.img_folder}/{patient_id}_{study_id}{self.extenstion}"
            )
            self.wp_mask_path = (
                f"{self.wp_mask_folder}/{patient_id}_{study_id}{self.extenstion}"
            )
            self.pz_tz_mask_path = (
                f"{self.pz_tz_mask_folder}/{patient_id}_{study_id}{self.extenstion}"
            )

            try:
                data = self.load()
            except Exception as e:
                print(e)
        elif self.mode in ("infer", "train_seg_maps"):
            row = self.df.iloc[idx]
            if self.mode == "infer":
                if self.dataset_name == "picai":
                    filename = row["prostateX_filename"]
                elif self.dataset_name == "AS":
                    filename = row["study_id"]

            elif self.mode == "train_seg_maps":
                if self.dataset_name == "picai":
                    filename = f"{row["patient_id"]}_{row["study_id"]}"
                elif self.dataset_name == "AS":
                    filename = row["study_id"]

            self.img_path = f"{self.img_folder}/{filename}{self.extenstion}"

            # Handle optional mask paths
            if self.wp_mask_folder is not None and self.wp_mask_folder != "None":
                self.wp_mask_path = f"{self.wp_mask_folder}/{filename}{self.extenstion}"
            else:
                self.wp_mask_path = None

            if self.pz_tz_mask_folder is not None and self.pz_tz_mask_folder != "None":
                self.pz_tz_mask_path = (
                    f"{self.pz_tz_mask_folder}/{filename}{self.extenstion}"
                )
            else:
                self.pz_tz_mask_path = None

            try:
                if self.mode == "infer":
                    # For infer mode, get original shape and spacing info
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
                    ) = self.load(return_nifti=True, return_original_info=True)
                else:
                    # For train_seg_maps mode, keep existing behavior
                    img, mask, img_shape, mask_shape, img_nii, mask_nii = self.load(
                        return_nifti=True
                    )
                    # Initialize placeholders for consistency (won't be used)
                    original_shape = None
                    original_spacing = None
                    original_affine = None

            except Exception as e:
                print(f"Error loading data for {filename}: {e}")
                raise e

            data = {"image": img, "mask": mask}
            data["study_id"] = filename

        if self.transform is not None:
            data = self.transform(data)
        else:
            pass
        data.pop("study_id", None)
        data.pop("mask_noise", None)

        patches = []

        if self.patch_iter is not None:
            for ret, others in self.patch_iter(data):
                img_patch, mask_patch = ret["image"], ret["mask"]
                if self.patch_transform is not None:
                    img_patch, mask_patch = apply_transform(
                        (img_patch, mask_patch), self.transform, map_items=False
                    )
                if self.with_coordinates and len(others) > 0:
                    patches.append((img_patch, mask_patch, others, others))
                else:
                    patches.append(
                        (img_patch, mask_patch, np.asarray([0, 0]), np.asarray([0, 0]))
                    )
            del data

            if self.mode in ("infer", "train_seg_maps"):
                if len(patches) == 0:
                    raise ValueError(
                        f"Image too small to generate patches: shape={img_shape}. File: {filename}"
                    )

                if self.mode == "infer":
                    # Return original shape and spacing for inference mode
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
                    # Keep existing return format for train_seg_maps mode
                    return (filename, patches, img_shape, mask_shape, img_nii, mask_nii)
            else:
                return patches
        else:
            return data

    def combine_masks(self, wp_data: Any, pz_tz_data: Any) -> np.ndarray:
        """Combine the prostate and peripheral zone masks into a single mask.

        Args:
            wp_data (np.ndarray): The whole prostate mask data.
            pz_tz_data (np.ndarray): The peripheral zone and transition zone mask data.

        Returns:
            np.ndarray: The combined mask data.
        """
        _, wp_nii = nutil.load(self.wp_mask_path)
        wp_data = np.round(wp_nii.get_fdata()).astype(np.int8)

        _, pz_tz_nii = nutil.load(self.pz_tz_mask_path)
        pz_tz_data = np.round(pz_tz_nii.get_fdata()).astype(np.int8)

        mask = np.zeros(wp_data.shape + (3,))
        mask[wp_data == 1, 0] = 1
        mask[pz_tz_data == 1, 1] = 1  # pz
        mask[pz_tz_data == 2, 2] = 1  # tz
        return mask
