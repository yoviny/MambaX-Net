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
from mambax_net.dataset.collate.custom_collate import (
    patch_collate_infer,
    patch_collate_seg,
)
from mambax_net.dataset.csv_load import picai_data_load
from mambax_net.dataset.picai_dataset import PicSegDataset
from mambax_net.utilities.crop import InPlaneCrop
from mambax_net.utilities.normalize import NormalizeData
from mambax_net.utilities.resample import MONAIResample
from mambax_net.utilities.train_helpers import seed_torch


def process():
    """Build and cache PICAI train/test segmentation datasets from CLI args.

    Parses CLI arguments, loads nnUNet plans and W&B config, then loads the
    PICAI train/test split via ``picai_data_load`` (optionally subsampling
    when ``--debug`` is set). Builds a ``PicSegDataset`` for train and one
    for test (using the ProstateX paths), wraps each in a ``DataLoader``
    with disk caching enabled, and iterates both loaders once to populate
    the cache on disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--df_path", type=str, required=True)
    parser.add_argument("-p", "--prostateX_mapping", type=str, required=True)
    parser.add_argument("-t2", "--T2_path", type=str, required=True)
    parser.add_argument("-wp", "--WP_label_path", type=str, required=True)
    parser.add_argument("-pz", "--PZ_TZ_label_path", type=str, required=True)
    parser.add_argument("-px", "--px_T2", type=str, required=True)
    parser.add_argument("-px_wp", "--px_WP_label_path", type=str, required=True)
    parser.add_argument("-px_pz", "--px_PZ_TZ_label_path", type=str, required=True)
    parser.add_argument("-conf", "--config", type=str, required=True)
    parser.add_argument("-nw", "--num_workers", type=int, required=True)
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

    os.environ["WANDB_NOTEBOOK_NAME"] = "seg_cache_dataset.py"

    df_path = args.df_path
    prostateX_mapping = args.prostateX_mapping
    T2_path = args.T2_path
    WP_label_path = args.WP_label_path
    PZ_TZ_label_path = args.PZ_TZ_label_path

    px_T2 = args.px_T2
    px_WP_label_path = args.px_WP_label_path
    px_PZ_TZ_label_path = args.px_PZ_TZ_label_path

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

    wandb.init(project=PROJECT, entity=ENTITY, config=config)

    train_df, test_df = picai_data_load(df_path, prostateX_mapping)

    if args.debug is not None:
        train_df = train_df.sample(20).reset_index(drop=True)
        test_df = test_df.sample(20).reset_index(drop=True)

    preprocess = Compose(
        [
            MONAIResample((spacing[0], spacing[1], spacing[2])),
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

    cache_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "data",
        "seg_cache",
    )
    cache_dir = os.path.normpath(cache_dir)

    print("caching train dataset...")
    train_dataset = PicSegDataset(
        train_df,
        patch_iter,
        T2_path,
        WP_label_path,
        PZ_TZ_label_path,
        preprocess=preprocess,
        transform=transforms,
        patch_transform=None,
        with_coordinates=False,
        cache=True,
        cache_dir=cache_dir,
        mode="train",
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

    print("caching test dataset...")
    test_dataset = PicSegDataset(
        test_df,
        patch_iter,
        px_T2,
        px_WP_label_path,
        px_PZ_TZ_label_path,
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

    print("Testing test_loader...")
    for i, (_, image, mask, _, _, im_shape, mask_shape, _, _, _, _, _) in tqdm(
        enumerate(test_loader), total=len(test_loader)
    ):
        pass


if __name__ == "__main__":
    process()
    wandb.finish()
