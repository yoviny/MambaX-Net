import argparse
import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import wandb
from batchgenerators.utilities.file_and_folder_operations import load_json
from carbontracker import parser as tracker_parser
from carbontracker.tracker import CarbonTracker
from dotenv import load_dotenv
from monai.data import PatchIterd
from monai.transforms import (
    Compose,
    Rand3DElasticd,
    RandAxisFlipd,
    RandCoarseDropoutd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandShiftIntensityd,
    RandZoomd,
)
from monai.utils import set_determinism
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch.amp import GradScaler
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
    StepLR,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mambax_net.utilities.nifti_utilities as nutil
from mambax_net.dataset.collate.custom_collate import patch_collate_seg
from mambax_net.dataset.csv_load import active_surveillance_data_load
from mambax_net.dataset.dual_scan_dataset import DualScanDataset
from mambax_net.loss.compound_loss import FocalTverskyDiceCE
from mambax_net.metrics.utils import EarlyStopping
from mambax_net.network.mx_net import MambaXNet
from mambax_net.network.nnunet_arch import set_deep_supervision_enabled
from mambax_net.scheduler.warmup import GradualWarmupSchedulerV2
from mambax_net.utilities.crop import InPlaneCrop
from mambax_net.utilities.normalize import NormalizeData
from mambax_net.utilities.random_noise import (
    MaskDropout,
    RandomMaskNoise,
    mark_mask_noise_by_patient,
)
from mambax_net.utilities.resample import MONAIResample
from mambax_net.utilities.train_helpers import (
    get_scheduler,
    init_logger,
    seed_torch,
    train_seg,
)


def train_loop(tb_writer):
    """Run the full MambaXNet training pipeline: parse args, build data/model/optimizer, train, and log to wandb.

    Parses CLI arguments (data paths, config, experiment name, noise options),
    validates required environment variables (`WANDB_API_KEY`, `ENTITY`,
    `PROJECT`), loads the nnUNet plans and experiment config, builds the
    train/val `DualScanDataset`s and dataloaders (with optional noise
    injection into the prior masks), constructs the `MambaXNet` model,
    optimizer, LR scheduler, loss (`FocalTverskyDiceCE`, optionally wrapped
    with deep supervision), and mixed-precision scaler. Runs the epoch loop
    via `train_seg`, steps the scheduler according to its type, saves the
    best model and a per-epoch checkpoint, applies early stopping, tracks
    carbon emissions, and logs metrics/artifacts to Weights & Biases.

    Args:
        tb_writer (torch.utils.tensorboard.SummaryWriter): TensorBoard writer passed through to `train_seg`.

    Returns:
        torch.utils.tensorboard.SummaryWriter: The same `tb_writer` passed in.
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
    parser.add_argument("-best_preds", "--best_preds_path", type=str, required=True)
    parser.add_argument("-conf", "--config", type=str, required=True)
    parser.add_argument("-nw", "--num_workers", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--both_mask_noise", type=float, default=0.0)
    parser.add_argument("--sample_sz", type=int, default=5)
    parser.add_argument(
        "--debug",
        action="store_true",
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

    os.environ["WANDB_NOTEBOOK_NAME"] = "mx_net_train.py"

    exp_name = args.exp_name

    T2_path = args.T2_path
    T2_past_path = T2_path
    wp_label_path = args.WP_label_path
    wp_label_path_past = wp_label_path
    pz_tz_label_path = args.PZ_TZ_label_path
    pz_tz_label_path_past = pz_tz_label_path

    val_T2_path = args.val_T2_path
    val_T2_past_path = val_T2_path
    val_wp_label_path = args.val_WP_label_path
    # prior fed to the model must match inference (AI-predicted), not GT -
    # pull from the same AI-labeled pool used for training, not val GT itself.
    val_wp_label_path_past = wp_label_path
    val_pz_tz_label_path = args.val_PZ_TZ_label_path
    val_pz_tz_label_path_past = pz_tz_label_path

    test_wp_label_path = args.test_WP_label_path

    # Create base directory path and ensure consistency - pointing to project root
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(
        current_file_dir, "..", ".."
    )  # Go up two levels to reach project root
    base_dir = os.path.normpath(base_dir)
    logs_dir = os.path.join(base_dir, "logs")
    model_weights_dir = os.path.join(base_dir, "model_weights")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    runs_dir = os.path.join(base_dir, "runs")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(model_weights_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    LOG_FILE = os.path.join(logs_dir, f"{exp_name}_train.log")
    LOGGER = init_logger(LOG_FILE)
    LOGGER.info(f"Experiment name: {exp_name}")

    config = load_json(args.config)

    carbon_log_path = os.path.join(logs_dir, f"carbon_{exp_name}_lesion_train.log")

    tracker = CarbonTracker(
        epochs=config["epochs"],
        log_dir=carbon_log_path,
        api_keys={
            "electricitymaps": os.getenv("ElectricityMaps"),
        },
    )

    nnunet_plan_path = os.path.join(
        base_dir,
        "mambax_net",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    wandb.init(project=PROJECT, entity=ENTITY, config=config, name=exp_name)
    wandb.define_metric("custom_train_step")
    wandb.define_metric("custom_val_step")
    wandb.define_metric("custom_lr_step")

    wandb.define_metric("train/*", step_metric="custom_train_step")
    wandb.define_metric("val/*", step_metric="custom_val_step")

    rem_ids_config_path = os.path.join(
        base_dir,
        "mambax_net",
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
        rem_ids_config_path=rem_ids_config_path,
    )

    if args.debug:
        train_df = train_df.sample(min(20, len(train_df))).reset_index(drop=True)
        test_df = test_df.sample(min(20, len(test_df))).reset_index(drop=True)
        config["epochs"] = 5

    if args.noise > 0:
        LOGGER.info(f"Adding noise to {args.noise}% of the training data")

        noise_indexes = np.random.choice(
            train_df.index, size=int(len(train_df) * args.noise / 100), replace=False
        )
        train_df["noise"] = 0
        train_df.loc[noise_indexes, "noise"] = 1
        train_df["noise"] = train_df["noise"].astype(int)
        LOGGER.info(
            f"Total number of patients with noise: {len(train_df[train_df['noise'] == 1])}"
        )
    if args.both_mask_noise > 0:
        train_df, noisy_mask_patients = mark_mask_noise_by_patient(
            train_df, args.both_mask_noise, config["seed"]
        )
        n_train_patients = (
            train_df["study_id"].astype(str).str.rsplit("_", n=1).str[0].nunique()
        )
        LOGGER.info(
            f"Perturbing current and prior masks for {len(noisy_mask_patients)}/{n_train_patients} training patients "
            f"({args.both_mask_noise}%)"
        )
        wandb.config.update(
            {
                "both_mask_noise": args.both_mask_noise,
                "both_mask_noise_patients": len(noisy_mask_patients),
            },
            allow_val_change=True,
        )
    LOGGER.info(f"Training...")

    net_num_pool_op_kernel_sizes = plans_manager.get_configuration(
        "3d_fullres"
    ).pool_op_kernel_sizes
    deep_supervision_scales = list(
        list(i) for i in 1 / np.cumprod(np.vstack(net_num_pool_op_kernel_sizes), axis=0)
    )[:-1]

    weights = np.array([1 / (2**i) for i in range(len(deep_supervision_scales))])
    ds_loss_weights = weights / weights.sum()

    LOGGER.info("Loading model...")

    model_path = (
        os.path.join(model_weights_dir, "nnunet_model.pt")
        if config.get("pretrained", True)
        else None
    )

    model = MambaXNet(
        config,
        configuration_manager,
        model_path,
        gradient_checkpointing=config.get("gradient_checkpointing", False),
        num_attention_layers=3,
        shape_out_channels=64,
        crossattention_dim=256,
        crossattention_heads=4,
        crossattention_num_patches=4,
    )

    if config["deep_supervision"]:
        LOGGER.info("Using deep supervision...")
        model = set_deep_supervision_enabled(True, False, model)
    model.to(device)

    LOGGER.info("Loading dataset...")

    preprocess = Compose(
        [
            MONAIResample(
                (spacing[0], spacing[1], spacing[2]), skip_mask_preprocess=False
            ),
            InPlaneCrop(crop_sz[0], crop_sz[1]),
            NormalizeData(mean=img_mean, std=img_std),
        ]
    )

    transforms = Compose(
        [
            RandAxisFlipd(prob=0.3, keys=["image", "mask"]),
            RandRotate90d(prob=0.4, keys=["image", "mask"]),
            RandGaussianNoised(keys=["image"], prob=0.35),
            RandShiftIntensityd(keys=["image"], offsets=(10, 20), prob=0.35),
            RandZoomd(
                prob=0.20,
                min_zoom=0.8,
                max_zoom=1.2,
                keep_size=True,
                keys=["image", "mask"],
            ),
            RandGaussianSmoothd(
                keys=["image"],
                sigma_x=(0.25, 1.5),
                sigma_y=(0.25, 1.5),
                sigma_z=(0.25, 1.5),
                approx="erf",
                prob=0.15,
            ),
            RandCoarseDropoutd(
                keys=["image"],
                holes=8,
                max_holes=15,
                spatial_size=(30, 30, 5),
                prob=0.15,
            ),
            Rand3DElasticd(
                keys=["image", "mask"],
                sigma_range=(3, 5),
                magnitude_range=(50, 150),
                prob=0.20,
                padding_mode="reflection",
                mode=["bilinear", "nearest"],
            ),
        ]
        + (
            [RandomMaskNoise(keys=["image", "mask"], prob=float(args.noise / 100.0))]
            if args.noise > 0
            else []
        )
        + (
            [MaskDropout(keys=["image", "mask"], drop_prob=1.0)]
            if args.both_mask_noise > 0
            else []
        )
    )
    if config["custom_patch"]:
        LOGGER.info(f"Using custom patch size {patch_sz[-1]}")
        patch_iter = PatchIterd(
            keys=["image", "mask"],
            patch_size=(patch_sz[-1]),
            start_pos=(0, 0),
            mode="wrap",
        )
    else:
        LOGGER.info(
            f"Using suggested patch size {configuration_manager.patch_size[::-1]}"
        )
        patch_iter = PatchIterd(
            keys=["image", "mask"],
            patch_size=(configuration_manager.patch_size[::-1]),
            start_pos=(0, 0),
            mode="wrap",
        )

    cache_dir = os.path.join(
        base_dir,
        "mambax_net",
        "data",
        "mambaX_net_cache",
    )
    cache_dir = os.path.normpath(cache_dir)

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
        mode="train",
    )

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=patch_collate_seg,
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

    LOGGER.info("Loading dataset complete")

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["lr"],
        weight_decay=3e-5,
        momentum=0.99,
    )

    scheduler = get_scheduler(
        config, optimizer, config["scheduler"], train_loader, config["step_size"]
    )

    early_stopping = EarlyStopping(patience=config["patience"], verbose=False)

    criterion = FocalTverskyDiceCE(
        alpha=0.7,
        beta=0.4,
        gamma=0.5,
        focal_gamma=2.5,
        background=True,
        sigmoid=True,
        softmax=False,
        to_onehot_y=False,
    )

    if config["deep_supervision"]:
        criterion = DeepSupervisionWrapper(criterion, ds_loss_weights)

    if config["bf16"]:
        scaler = GradScaler(enabled=False)
        LOGGER.info("Using mixed precision training...")
    else:
        scaler = None

    best_dice = 0
    best_wp = 0
    best_pz = 0
    best_tz = 0

    LOGGER.info("Starting training...")

    wandb.watch(model, log="all")

    for epoch in range(1, config["epochs"] + 1):
        tracker.epoch_start()
        start_time = time.time()
        torch.cuda.empty_cache()
        gc.collect()

        LOGGER.info(f"starting epoch {epoch}...")

        avg_train_loss, avg_val_loss, metrics = train_seg(
            epoch,
            config,
            model,
            optimizer,
            scheduler,
            train_loader,
            valid_loader,
            tb_writer,
            scaler,
            criterion,
            device,
            dual_scan=True,
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(metrics["val/mean_dice"])
        if isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        if isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()
        if isinstance(scheduler, GradualWarmupSchedulerV2):
            scheduler.step()
        if isinstance(scheduler, StepLR):
            scheduler.step()
        if isinstance(scheduler, PolyLRScheduler):
            scheduler.step()

        elapsed = time.time() - start_time
        LOGGER.info(
            f"  Epoch {epoch} - train/avg_loss: {avg_train_loss:.4f} | val/avg_loss: {avg_val_loss:.4f} | time: {elapsed:.0f}s"
        )
        LOGGER.info(
            f"  Epoch {epoch} - train/mean_dice_avg: {metrics['train/mean_dice_avg']} | val/mean_dice_avg: {metrics['val/mean_dice_avg']}"
        )

        if metrics["val/mean_dice_avg"] > best_dice:
            model_path = os.path.join(model_weights_dir, f"{exp_name}_model.pt")
            torch.save(model.state_dict(), model_path)
            best_dice = metrics["val/mean_dice_avg"]
            best_wp = metrics["val/wp_dice_avg"]
            best_pz = metrics["val/pz_dice_avg"]
            best_tz = metrics["val/tz_dice_avg"]

        checkpoint = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }

        checkpoint_path = os.path.join(checkpoints_dir, f"{exp_name}_checkpoint.pt")
        torch.save(checkpoint, checkpoint_path)

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            tracker.stop()
            break

        tracker.epoch_end()

    wandb.unwatch(model)

    LOGGER.info(f"  Best Dice score {best_dice}")
    wandb.run.summary["Best Dice score"] = best_dice
    wandb.run.summary["Best WP Dice"] = best_wp
    wandb.run.summary["Best PZ Dice"] = best_pz
    wandb.run.summary["Best TZ Dice"] = best_tz

    model_path = os.path.join(model_weights_dir, f"{exp_name}_model.pt")
    wandb.run.log_model(
        path=model_path,
        name=f"{exp_name}_model",
    )

    logs = tracker_parser.parse_all_logs(
        log_dir=f"logs/carbon_{exp_name}_lesion_train.log"
    )
    last_log = logs[-1]

    wandb.run.summary["Actual consumption"] = last_log["actual"]
    wandb.run.summary["Predicted emissions"] = last_log["pred"]

    return tb_writer


if __name__ == "__main__":
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(current_file_dir, "..", "..")
    project_root = os.path.normpath(project_root)
    runs_dir = os.path.join(project_root, "runs")
    tb_writer = SummaryWriter(runs_dir)

    tb_writer = train_loop(tb_writer)
    print("Training complete")

    tb_writer.close()
    wandb.finish()
