import argparse
import os

import wandb
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
from monai.data import PatchIterd
from monai.transforms import Compose
from monai.utils import set_determinism
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import mambax_net.utilities.nifti_utilities as nutil
from mambax_net.dataset.collate.custom_collate import (patch_collate_infer,
                                                       patch_collate_seg)
from mambax_net.dataset.csv_load import active_surveillance_data_load
from mambax_net.dataset.dual_scan_dataset import DualScanDataset
from mambax_net.dataset.picai_dataset import PicSegDataset
from mambax_net.utilities.crop import InPlaneCrop
from mambax_net.utilities.normalize import NormalizeData
from mambax_net.utilities.resample import MONAIResample
from mambax_net.utilities.train_helpers import seed_torch


def process():
    """Build and cache train/val/test segmentation datasets from CLI args.

    Parses CLI arguments, loads nnUNet plans and W&B config, then loads the
    active-surveillance train/val/test splits via ``active_surveillance_data_load``.
    Depending on ``--dataset_name``, builds either ``PicSegDataset`` (AS,
    single-scan) or ``DualScanDataset`` (dual-scan) instances for each split,
    wraps each in a ``DataLoader`` with disk caching enabled, and iterates
    every loader once to populate the cache on disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t2", "--T2_path", type=str, required=True)
    parser.add_argument("-wp", "--WP_label_path", type=str, required=True)
    parser.add_argument("-pz", "--PZ_TZ_label_path", type=str, required=True)
    parser.add_argument("-val_t2", "--val_T2_path", type=str, required=True)
    parser.add_argument("-val_wp", "--val_WP_label_path", type=str, required=True)
    parser.add_argument("-val_pz", "--val_PZ_TZ_label_path", type=str, required=True)
    parser.add_argument("-test_t2", "--test_T2_path", type=str, required=True)
    parser.add_argument("-test_wp", "--test_WP_label_path", type=str, required=True)
    parser.add_argument("-test_pz", "--test_PZ_TZ_label_path", type=str, required=True)
    parser.add_argument(
        "-name",
        "--dataset_name",
        type=str,
        default="dual",
        choices=["dual", "AS"],
        help="Dataset class: 'dual' for DualScanDataset (dual-scan), 'AS' for PicSegDataset (single-scan, nnunet-style)",
    )
    parser.add_argument("-best_preds", "--best_preds_path", type=str, required=False)
    parser.add_argument("-conf", "--config", type=str, required=True)
    parser.add_argument("-nw", "--num_workers", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--sample_sz", type=int, default=300)
    parser.add_argument(
        "--time_point",
        type=str,
        default="_2",
        help="Time-point suffix filter (e.g. '_2'). Pass '-1' to include all time points.",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Limit number of images for processing (for testing)",
    )

    args = parser.parse_args()

    load_dotenv()

    os.getenv("WANDB_API_KEY")
    if not os.getenv("WANDB_API_KEY"):
        raise EnvironmentError(
            "The environment variable 'WANDB_API_KEY' is not set. Please set it before running the script."
        )

    ENTITY = os.getenv("ENTITY")
    if ENTITY is None:
        raise EnvironmentError(
            "The environment variable 'ENTITY' is not set. Please set it before running the script."
        )

    PROJECT = os.getenv("PROJECT")
    if PROJECT is None:
        raise EnvironmentError(
            "The environment variable 'PROJECT' is not set. Please set it before running the script."
        )

    os.environ["WANDB_NOTEBOOK_NAME"] = "seg_AS_cache_dataset.py"

    T2_path = args.T2_path
    T2_past_path = T2_path
    wp_label_path = args.WP_label_path
    wp_label_path_past = wp_label_path
    pz_tz_label_path = args.PZ_TZ_label_path
    pz_tz_label_path_past = pz_tz_label_path

    val_T2_path = args.val_T2_path
    val_T2_past_path = val_T2_path
    val_wp_label_path = args.val_WP_label_path
    val_wp_label_path_past = wp_label_path
    val_pz_tz_label_path = args.val_PZ_TZ_label_path
    val_pz_tz_label_path_past = pz_tz_label_path

    test_T2_path = args.test_T2_path
    test_T2_past_path = test_T2_path
    test_wp_label_path = args.test_WP_label_path
    test_wp_label_path_past = wp_label_path
    test_pz_tz_label_path = args.test_PZ_TZ_label_path
    test_pz_tz_label_path_past = pz_tz_label_path

    config = load_json(args.config)

    nnunet_plan_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "configs",
        "nnUNetPlans_segmentation.json",
    )
    nnunet_plan_path = os.path.normpath(nnunet_plan_path)
    if os.path.exists(nnunet_plan_path):
        plans_manager = PlansManager(nnunet_plan_path)
        configuration_manager = plans_manager.get_configuration("3d_fullres")
    else:
        raise FileNotFoundError(f"nnUNet plans file not found at {nnunet_plan_path}")

    img_mean = plans_manager.foreground_intensity_properties_per_channel["0"]["mean"]
    img_std = plans_manager.foreground_intensity_properties_per_channel["0"]["std"]
    suggested_patch_size = configuration_manager.patch_size

    median_size = plans_manager.original_median_shape_after_transp  # [::-1]
    crop_sz, patch_sz = nutil.possible_patch_size(median_size, suggested_patch_size)
    spacing = plans_manager.original_median_spacing_after_transp[::-1]

    seed_torch(seed=config["seed"])
    set_determinism(seed=config["seed"])
    print(f"Seed set to: {os.getenv('PYTHONHASHSEED')}")

    wandb.init(project=PROJECT, entity=ENTITY, config=config)

    rem_ids_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "configs",
        "ignore_ids.yaml",
    )
    rem_ids_config_path = os.path.normpath(rem_ids_config_path)

    train_df, val_df, test_df = active_surveillance_data_load(
        args,
        config,
        T2_path,
        val_wp_label_path,
        test_wp_label_path,
        test_time_point=args.time_point,
        best_preds_path=args.best_preds_path,
        coarse_preds_path=args.WP_label_path,
        rem_ids_config_path=rem_ids_config_path,
    )

    preprocess = Compose(
        [
            MONAIResample(
                (spacing[0], spacing[1], spacing[2]), skip_mask_preprocess=False
            ),
            InPlaneCrop(crop_sz[0], crop_sz[1]),
            NormalizeData(mean=img_mean, std=img_std),
        ]
    )

    transforms = None

    if config["custom_patch"]:
        print(f"Using custom patch size {patch_sz[-1]}")
        patch_iter = PatchIterd(
            keys=["image", "mask"],
            patch_size=(patch_sz[-1]),
            start_pos=(0, 0),
            mode="wrap",
        )
    else:
        print(f"Using suggested patch size {configuration_manager.patch_size[::-1]}")
        patch_iter = PatchIterd(
            keys=["image", "mask"],
            patch_size=(configuration_manager.patch_size[::-1]),
            start_pos=(0, 0),
            mode="wrap",
        )

    if args.dataset_name == "AS":
        cache_subdir = "AS_seg_cache"
    else:
        cache_subdir = "mambaX_net_cache"

    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        cache_subdir,
    )
    cache_dir = os.path.normpath(cache_dir)

    print("caching train dataset...")
    if args.dataset_name == "AS":
        train_dataset = PicSegDataset(
            train_df,
            patch_iter,
            T2_path,
            wp_label_path,
            pz_tz_label_path,
            preprocess=preprocess,
            transform=transforms,
            patch_transform=None,
            with_coordinates=True,
            cache=True,
            cache_dir=cache_dir,
            mode="train",
            dataset_name="AS",
        )
    else:
        train_dataset = DualScanDataset(
            train_df,
            patch_iter,
            T2_path,
            T2_past_path,
            wp_label_path,
            wp_label_path_past,
            pz_tz_mask_folder=pz_tz_label_path,
            pz_tz_mask_folder_past=pz_tz_label_path_past,
            preprocess=preprocess,
            transform=transforms,
            patch_transform=None,
            with_coordinates=True,
            cache=True,
            cache_dir=cache_dir,
            mode="test",
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=patch_collate_seg,
    )

    print("caching val dataset...")
    if args.dataset_name == "AS":
        valid_dataset = PicSegDataset(
            val_df,
            patch_iter,
            val_T2_path,
            val_wp_label_path,
            val_pz_tz_label_path,
            preprocess=preprocess,
            transform=None,
            patch_transform=None,
            with_coordinates=False,
            cache=True,
            cache_dir=cache_dir,
            mode="valid",
            dataset_name="AS",
        )
    else:
        valid_dataset = DualScanDataset(
            val_df,
            patch_iter,
            val_T2_path,
            val_T2_past_path,
            val_wp_label_path,
            val_wp_label_path_past,
            pz_tz_mask_folder=val_pz_tz_label_path,
            pz_tz_mask_folder_past=val_pz_tz_label_path_past,
            preprocess=preprocess,
            transform=None,
            patch_transform=None,
            with_coordinates=False,
            cache=True,
            cache_dir=cache_dir,
            mode="valid",
        )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=patch_collate_seg,
    )

    print("caching test dataset...")
    if args.dataset_name == "AS":
        test_dataset = PicSegDataset(
            test_df,
            patch_iter,
            test_T2_path,
            test_wp_label_path,
            test_pz_tz_label_path,
            preprocess=preprocess,
            transform=None,
            patch_transform=None,
            with_coordinates=True,
            cache=True,
            cache_dir=cache_dir,
            mode="infer",
            dataset_name="AS",
        )
    else:
        test_dataset = DualScanDataset(
            test_df,
            patch_iter,
            test_T2_path,
            test_T2_past_path,
            test_wp_label_path,
            test_wp_label_path_past,
            pz_tz_mask_folder=test_pz_tz_label_path,
            pz_tz_mask_folder_past=test_pz_tz_label_path_past,
            preprocess=preprocess,
            transform=None,
            patch_transform=None,
            with_coordinates=True,
            cache=True,
            cache_dir=cache_dir,
            mode="infer",
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=patch_collate_infer,
    )

    print("Testing train_loader...")
    for i, (image, mask, _, _) in tqdm(
        enumerate(train_loader), total=len(train_loader)
    ):
        pass

    print("Testing valid_loader...")
    for i, (image, mask, _, _) in tqdm(
        enumerate(valid_loader), total=len(valid_loader)
    ):
        pass

    print("Testing test_loader...")
    if args.dataset_name == "AS":
        for i, batch in tqdm(enumerate(test_loader), total=len(test_loader)):
            pass
    else:
        for i, (_, image, mask, _, _, im_shape, mask_shape, _, _, _, _, _) in tqdm(
            enumerate(test_loader), total=len(test_loader)
        ):
            pass


if __name__ == "__main__":
    process()
    wandb.finish()
