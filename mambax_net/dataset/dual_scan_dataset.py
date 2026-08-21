import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import joblib
import monai
import numpy as np
import pandas as pd
import torch
from einops import rearrange
from monai.transforms import apply_transform
from torch.nn.functional import pad
from tqdm.auto import tqdm

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


class DualScanDataset(monai.data.Dataset):
    """Dataset that pairs each study's current scan with its previous timepoint scan.

    For every row in `df`, loads the current scan/mask and the nearest prior
    scan/mask for the same patient, concatenates them along the channel
    dimension, and (optionally) preprocesses and caches the combined tensors
    to disk for reuse across epochs.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        patch_iter: Optional[Callable],
        img_folder: str,
        img_past_folder: str,
        wp_mask_folder: str,
        wp_mask_folder_past: str,
        pz_tz_mask_folder: str,
        pz_tz_mask_folder_past: str,
        preprocess: Callable,
        transform: Optional[Callable] = None,
        patch_transform: Optional[Callable] = None,
        with_coordinates: bool = False,
        cache: bool = True,
        cache_dir: str = "xnat_cache",
        mode: str = "train",
    ) -> None:
        """Dataset that pairs each study's current scan with its previous timepoint scan.

        Args:
            df (pd.DataFrame): DataFrame with a `study_id` column identifying each scan.
            patch_iter (callable, optional): Function to iterate over patches of the loaded data.
            img_folder (str): Path to the current scan image folder.
            img_past_folder (str): Path to the previous timepoint image folder.
            wp_mask_folder (str): Path to the whole prostate mask folder for the current scan.
            wp_mask_folder_past (str): Path to the whole prostate mask folder for the previous timepoint.
            pz_tz_mask_folder (str): Path to the peripheral/transition zone mask folder for the current scan, or None.
            pz_tz_mask_folder_past (str): Path to the peripheral/transition zone mask folder for the previous timepoint.
            preprocess (callable): Preprocessing function applied to image/mask pairs.
            transform (callable, optional): Transformation function applied to the loaded data dict.
            patch_transform (callable, optional): Transformation function applied to each patch.
            with_coordinates (bool, optional): Whether to keep patch coordinates in the output. Defaults to False.
            cache (bool, optional): Whether to preprocess and cache all samples to disk on init. Defaults to True.
            cache_dir (str, optional): Directory used for caching preprocessed samples. Defaults to "xnat_cache".
            mode (str, optional): One of "train", "valid", "test", "infer", or "train_seg_maps". Defaults to "train".
        """
        self.df = df
        self.patch_iter = patch_iter
        self.img_folder = img_folder
        self.img_past_folder = img_past_folder
        self.wp_mask_folder = wp_mask_folder
        self.wp_mask_folder_past = wp_mask_folder_past
        self.pz_tz_mask_folder = pz_tz_mask_folder
        self.pz_tz_mask_past_folder = pz_tz_mask_folder_past
        self.preprocess = preprocess
        self.transform = transform
        self.patch_transform = patch_transform
        self.with_coordinates = with_coordinates
        self.cache = cache
        self.cache_dir = cache_dir
        self.mode = mode

        os.makedirs(cache_dir, exist_ok=True)

        assert self.preprocess is not None, "preprocess must be defined"
        assert self.mode in [
            "train",
            "valid",
            "test",
            "infer",
            "train_seg_maps",
        ], "mode must be either train, valid, test, infer or train_seg_maps"

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
        self,
        return_nifti: bool = False,
        return_original_info: bool = False,
    ) -> LoadReturnType:
        """Load, preprocess and concatenate the current and past scan/mask pairs.

        Loads the current scan and its mask (built from the whole-prostate mask
        alone, or combined with the PZ/TZ mask via `combine_masks` when
        `pz_tz_mask_folder` is set) as well as the corresponding previous
        timepoint scan and mask, using `self.img_path`, `self.img_past_path`,
        `self.wp_mask_path`, `self.wp_mask_past_path`, `self.pz_tz_mask_path`
        and `self.pz_tz_mask_path_past` set by the caller. Both pairs are run
        through `self.preprocess`, converted to tensors, aligned along the
        z-dimension (truncating or zero-padding the past scan/mask to match
        the current scan), and concatenated along the channel dimension.

        Args:
            return_nifti (bool, optional): Whether to also return the preprocessed NIfTI image/mask objects. Defaults to False.
            return_original_info (bool, optional): Whether to also return the original (pre-preprocessing) shape, spacing and orientation of the current scan. Defaults to False.

        Returns:
            tuple: `(img, mask, img_shape, mask_shape)` by default; additionally
            includes `(img_nii, mask_nii)` when `return_nifti` is True, and
            `(original_shape, original_spacing, original_orientation)` when
            `return_original_info` is also True.
        """
        _, img_nii = nutil.load(self.img_path)

        # Extract original orientation from header extensions
        original_orientation = None
        for extension in img_nii.header.extensions:
            if extension.get_code() == 40:  # Our custom extension code
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

        _, img_past_nii = nutil.load(self.img_past_path)
        img_past = img_past_nii.get_fdata()

        img_past = np.expand_dims(img_past, axis=0)
        img_past_nii = img_past_nii.__class__(
            img_past, img_past_nii.affine, img_past_nii.header
        )
        img_past_nii.header.set_zooms((1.0,) + img_past_nii.header.get_zooms()[:3])
        img_past_nii.header.set_data_shape(img_past.shape)

        if self.pz_tz_mask_folder is not None:
            mask = self.combine_masks(self.wp_mask_path, self.pz_tz_mask_path)
            mask = rearrange(mask, "h w d c -> c h w d")
            mask_nii = img_nii.__class__(mask, img_nii.affine, img_nii.header)
            mask_nii.header.set_zooms(img_nii.header.get_zooms())
            mask_nii.header.set_data_shape(mask.shape)

            mask_past = self.combine_masks(
                self.wp_mask_past_path, self.pz_tz_mask_path_past
            )
            mask_past = rearrange(mask_past, "h w d c -> c h w d")
            mask_past_nii = img_past_nii.__class__(
                mask_past, img_past_nii.affine, img_past_nii.header
            )
            mask_past_nii.header.set_zooms(img_past_nii.header.get_zooms())
            mask_past_nii.header.set_data_shape(mask_past.shape)
        else:
            try:
                _, mask_nii = nutil.load(self.wp_mask_path)
                mask = mask_nii.get_fdata()
                mask = np.expand_dims(mask, axis=0)
                mask = np.repeat(mask, 3, axis=0)
                mask_nii = img_nii.__class__(mask, img_nii.affine, img_nii.header)
                mask_nii.header.set_zooms(img_nii.header.get_zooms())
                mask_nii.header.set_data_shape(mask.shape)

                _, mask_past_nii = nutil.load(self.wp_mask_past_path)
                mask_past = mask_past_nii.get_fdata()
                mask_past = np.expand_dims(mask_past, axis=0)
                mask_past = np.repeat(mask_past, 3, axis=0)
                mask_past_nii = img_past_nii.__class__(
                    mask_past, img_past_nii.affine, img_past_nii.header
                )
                mask_past_nii.header.set_zooms(img_past_nii.header.get_zooms())
                mask_past_nii.header.set_data_shape(mask_past.shape)
            except Exception as e:
                mask = np.zeros_like(img_nii.get_fdata())
                mask = np.expand_dims(mask, axis=0)
                mask = np.repeat(mask, 3, axis=0)
                mask_nii = img_nii.__class__(mask, img_nii.affine, img_nii.header)
                mask_nii.header.set_zooms(img_nii.header.get_zooms())
                mask_nii.header.set_data_shape(mask.shape)

                mask_past = np.zeros_like(img_past_nii.get_fdata())
                mask_past = np.expand_dims(mask_past, axis=0)
                mask_past = np.repeat(mask_past, 3, axis=0)
                mask_past_nii = img_past_nii.__class__(
                    mask_past, img_past_nii.affine, img_past_nii.header
                )
                mask_past_nii.header.set_zooms(img_past_nii.header.get_zooms())
                mask_past_nii.header.set_data_shape(mask_past.shape)

        data = self.preprocess({"image": img_nii, "mask": mask_nii})
        data1 = self.preprocess({"image": img_past_nii, "mask": mask_past_nii})

        img_nii = data["image"]
        mask_nii = data["mask"]

        img, mask = torch.tensor(
            data["image"].get_fdata().copy(), dtype=torch.float32
        ), torch.tensor(data["mask"].get_fdata().copy(), dtype=torch.float16)
        img_past, mask_past = torch.tensor(
            data1["image"].get_fdata().copy(), dtype=torch.float32
        ), torch.tensor(data1["mask"].get_fdata().copy(), dtype=torch.float16)

        # Ensure current and past scans have same z-dimension
        if img_past.shape[-1] != img.shape[-1]:
            if img_past.shape[-1] > img.shape[-1]:
                # Past scan has more slices, truncate to match current
                img_past = img_past[..., : img.shape[-1]]
                mask_past = mask_past[..., : img.shape[-1]]
            else:
                # Past scan has fewer slices, pad to match current
                img_past = pad(
                    img_past,
                    (0, img.shape[-1] - img_past.shape[-1]),
                    mode="constant",
                    value=0,
                )
                mask_past = pad(
                    mask_past,
                    (0, mask.shape[-1] - mask_past.shape[-1]),
                    mode="constant",
                    value=0,
                )

        img = torch.cat((img, img_past), dim=0)
        mask = torch.cat((mask, mask_past), dim=0)
        assert not torch.equal(mask, mask_past), "mask and mask_past are the same"

        img_shape = img.shape
        mask_shape = mask.shape

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
        """Preprocess and cache every study in `self.df` to disk.

        For each row, resolves the current and previous timepoint scan/mask
        paths, skips the row if the previous timepoint image is missing or
        loading fails, and otherwise loads/preprocesses the pair via `load`
        and writes the resulting image/mask dict to a joblib file in
        `self.cache_dir`. After processing, `self.df` is filtered down to
        only the rows whose cache file was successfully written.
        """
        print("Preprocessing and caching all data")
        for idx, row in tqdm(self.df.iterrows(), total=len(self.df)):
            study_id = row["study_id"]
            self.img_path = f"{self.img_folder}/{study_id}{self.extenstion}"

            try:
                p_id, s_id = study_id.split("_")
                int_study_id = int(s_id) - 1
            except Exception as e:
                print(e)
                continue

            self.img_past_path = (
                f"{self.img_past_folder}/{p_id}_{int_study_id}{self.extenstion}"
            )
            if not os.path.exists(self.img_past_path):
                continue

            self.wp_mask_path = f"{self.wp_mask_folder}/{study_id}{self.extenstion}"
            self.wp_mask_past_path = (
                f"{self.wp_mask_folder_past}/{p_id}_{int_study_id}{self.extenstion}"
            )

            if self.pz_tz_mask_folder is not None:
                self.pz_tz_mask_path = (
                    f"{self.pz_tz_mask_folder}/{study_id}{self.extenstion}"
                )
                self.pz_tz_mask_path_past = f"{self.pz_tz_mask_past_folder}/{p_id}_{int_study_id}{self.extenstion}"

            try:
                img, mask, _, _ = self.load()
            except Exception as e:
                print(e)
                continue

            data = {"image": img, "mask": mask}

            cache_filename = f"{self.cache_dir}/{study_id}_preprocessed.joblib"
            with open(cache_filename, "wb") as f:
                joblib.dump(data, f, compress="zlib")

        # Filter df to only include rows that were successfully cached
        self.df = self.df[
            self.df["study_id"].apply(
                lambda sid: os.path.exists(
                    f"{self.cache_dir}/{sid}_preprocessed.joblib"
                )
            )
        ].reset_index(drop=True)

    def __getitem__(self, idx: int) -> GetItemReturnType:
        """Get the item at `idx`, behavior depending on `self.mode`.

        In "train"/"valid"/"test" mode, loads the sample from the joblib
        cache (if `self.cache` is True) or on the fly by searching backwards
        for the nearest existing previous timepoint, and attaches "noise"/
        "mask_noise" columns from `self.df` when present. In "infer" or
        "train_seg_maps" mode, always loads on the fly, also searching
        backwards for the nearest previous timepoint, and additionally
        returns the study id and (for "infer") the original scan shape,
        spacing and orientation. In all modes, applies `self.transform` if
        set, then either splits the data into patches via `self.patch_iter`
        (optionally applying `self.patch_transform` and coordinates, and
        dropping the last patch in "infer"/"train_seg_maps" mode when the
        patch count is uneven) or returns the data dict unchanged.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            In "train"/"valid"/"test" mode: the patch list (or the data dict
            unchanged if `self.patch_iter` is not set).

            In "infer" mode: `(study_id, patches, img_shape, mask_shape,
            img_nii, mask_nii, original_shape, original_spacing,
            original_affine)` — includes the pre-resample shape/spacing/
            affine needed to map predictions back onto the original scan.

            In "train_seg_maps" mode: `(study_id, patches, img_shape,
            mask_shape, img_nii, mask_nii)` — same as "infer" but without
            the original shape/spacing/affine.

            Returns an empty list if on-the-fly loading fails.
        """
        if self.mode in ("train", "valid", "test"):
            row = self.df.iloc[idx]

            if self.cache:
                # load images from cache
                cache_filename = (
                    f"{self.cache_dir}/{row['study_id']}_preprocessed.joblib"
                )
                with open(cache_filename, "rb") as f:
                    data = joblib.load(f)
            else:
                # load images on the fly — ensures label paths are always respected
                study_id = row["study_id"]
                p_id, s_id = study_id.split("_")
                int_study_id = int(s_id) - 1

                # search backwards for the nearest previous timepoint
                while int_study_id >= 0:
                    self.img_past_path = (
                        f"{self.img_past_folder}/{p_id}_{int_study_id}{self.extenstion}"
                    )
                    if os.path.exists(self.img_past_path):
                        break
                    int_study_id -= 1

                self.img_path = f"{self.img_folder}/{study_id}{self.extenstion}"
                self.wp_mask_path = f"{self.wp_mask_folder}/{study_id}{self.extenstion}"
                self.wp_mask_past_path = (
                    f"{self.wp_mask_folder_past}/{p_id}_{int_study_id}{self.extenstion}"
                )

                if self.pz_tz_mask_folder is not None:
                    self.pz_tz_mask_path = (
                        f"{self.pz_tz_mask_folder}/{study_id}{self.extenstion}"
                    )
                    self.pz_tz_mask_path_past = f"{self.pz_tz_mask_past_folder}/{p_id}_{int_study_id}{self.extenstion}"

                try:
                    img, mask, _, _ = self.load()
                except Exception as e:
                    print(
                        f"[DualScanDataset] Failed to load {study_id} on-the-fly: {e}"
                    )
                    return []

                data = {"image": img, "mask": mask}

            # change noise quantity in data
            if "noise" in self.df.columns:
                data["noise"] = row["noise"]
            if "mask_noise" in self.df.columns:
                data["mask_noise"] = row["mask_noise"]

        elif self.mode in ("infer", "train_seg_maps"):
            # load images on the fly
            row = self.df.iloc[idx]
            study_id = row["study_id"]

            p_id, s_id = study_id.split("_")
            int_study_id = int(s_id) - 1

            # Loop to check for the existence of the previous timepoint
            while int_study_id >= 0:
                self.img_past_path = (
                    f"{self.img_past_folder}/{p_id}_{int_study_id}{self.extenstion}"
                )
                if os.path.exists(self.img_past_path):
                    break
                int_study_id -= 1

            self.img_path = f"{self.img_folder}/{study_id}{self.extenstion}"
            self.img_past_path = (
                f"{self.img_past_folder}/{p_id}_{int_study_id}{self.extenstion}"
            )

            self.wp_mask_path = f"{self.wp_mask_folder}/{study_id}{self.extenstion}"
            self.wp_mask_past_path = (
                f"{self.wp_mask_folder_past}/{p_id}_{int_study_id}{self.extenstion}"
            )

            if self.pz_tz_mask_folder is not None:
                self.pz_tz_mask_path = (
                    f"{self.pz_tz_mask_folder}/{study_id}{self.extenstion}"
                )
                self.pz_tz_mask_path_past = f"{self.pz_tz_mask_past_folder}/{p_id}_{int_study_id}{self.extenstion}"

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
                    original_shape = None
                    original_spacing = None
                    original_affine = None
            except Exception as e:
                print(e)

            data = {"image": img, "mask": mask}

        if self.transform is not None:
            data = self.transform(data)
        else:
            pass

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
                # if patch count isn't exactly 2, drop the last patch
                if len(patches) / 2 != 1:
                    # print(f"Batch size is not equal to length of patches: {len(patches)}. Dropping last patch.")
                    patches = patches[:-1]

                if self.mode == "infer":
                    return (
                        study_id,
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
                    return (study_id, patches, img_shape, mask_shape, img_nii, mask_nii)
            else:
                return patches
        else:
            return data

    def combine_masks(self, wp_mask_path, pz_tz_mask_path):
        """Combine whole-prostate and PZ/TZ masks into a single 3-channel mask.

        Args:
            wp_mask_path (str): Path to the whole prostate mask NIfTI file.
            pz_tz_mask_path (str): Path to the peripheral/transition zone mask NIfTI file, with label 1 for PZ and 2 for TZ.

        Returns:
            np.ndarray: Array of shape `wp_data.shape + (3,)` where channel 0
            is the whole prostate mask, channel 1 is the peripheral zone
            mask, and channel 2 is the transition zone mask.
        """
        wp_name, wp_nii = nutil.load(wp_mask_path)
        wp_data = np.round(wp_nii.get_fdata()).astype(int)

        pz_tz_name, pz_tz_nii = nutil.load(pz_tz_mask_path)
        pz_tz_data = np.round(pz_tz_nii.get_fdata()).astype(int)

        mask = np.zeros(wp_data.shape + (3,))
        mask[wp_data == 1, 0] = 1
        mask[pz_tz_data == 1, 1] = 1  # pz
        mask[pz_tz_data == 2, 2] = 1  # tz
        return mask
