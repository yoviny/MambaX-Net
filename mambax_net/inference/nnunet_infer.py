import argparse
import gc
import os
import shutil
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import wandb
from batchgenerators.utilities.file_and_folder_operations import load_json
from dotenv import load_dotenv
from monai.data import PatchIterd
from monai.transforms import Compose
from monai.utils import set_determinism
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from torch.utils.data import DataLoader

from mambax_net.dataset.collate.custom_collate import patch_collate_infer
from mambax_net.dataset.csv_load import (active_surveillance_infer_data_load,
                                         picai_data_load)
from mambax_net.dataset.picai_dataset import PicSegDataset
from mambax_net.loss.compound_loss import FocalTverskyDiceCE
from mambax_net.network.nnunet_arch import build_network_architecture
from mambax_net.utilities.crop import InPlaneCrop
from mambax_net.utilities.normalize import NormalizeData
from mambax_net.utilities.resample import MONAIResample
from mambax_net.utilities.train_helpers import (generate_predictions,
                                                inference_func, init_logger,
                                                log_ci_bounds_to_wandb,
                                                seed_torch)


def infer_loop():
    """Run nnUNet-architecture inference on Active Surveillance or ProstateX data.

    Parses CLI args and, based on whether ``--prostateX_mapping`` is given,
    loads the test dataframe with either the Active Surveillance loader or
    the ProstateX/PI-CAI loader. Loads the nnUNet plans/config for
    preprocessing parameters, builds the network via
    ``build_network_architecture``, and constructs a PicSegDataset/DataLoader
    for inference. If both WP and PZ/TZ ground-truth label paths are
    provided, runs evaluation via ``inference_func`` and logs Dice/Hausdorff
    metrics and regional (apex/mid-gland/base) statistics to Weights &
    Biases. Always generates and saves prediction volumes/probability maps
    via ``generate_predictions``, regardless of whether ground truth is
    available.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--df_path", type=str, required=False)
    parser.add_argument("-map", "--prostateX_mapping", type=str, required=False)
    parser.add_argument("-t2", "--T2_path", type=str, required=True)
    parser.add_argument("-wp", "--WP_label_path", type=str, required=False)
    parser.add_argument("-pz", "--PZ_TZ_label_path", type=str, required=False)
    parser.add_argument(
        "-val_wp",
        "--val_WP_label_path",
        type=str,
        required=False,
        help="Deprecated for AS inference; validation WP labels are not needed during inference",
    )
    parser.add_argument(
        "-test_wp",
        "--test_WP_label_path",
        type=str,
        required=False,
        help="Optional AS subset path when running without mask evaluation",
    )
    parser.add_argument("-conf", "--config", type=str, required=True)
    parser.add_argument("-name", "--dataset_name", type=str, required=True)
    parser.add_argument("-nw", "--num_workers", type=int, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("-test_t_point", "--test_time_point", type=str, default="_2")
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--results_subdir",
        type=str,
        default=None,
        help="Optional subdirectory under Results for inference outputs",
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

    os.environ["WANDB_NOTEBOOK_NAME"] = "nnunet_infer.py"

    exp_name = args.exp_name
    dataset_name = args.dataset_name

    if args.df_path is not None:
        df_path = args.df_path

    if args.prostateX_mapping is None:
        prostateX_mapping = None
    else:
        prostateX_mapping = args.prostateX_mapping

    T2_path = args.T2_path

    # Handle mask paths - set to None if not provided
    WP_label_path = args.WP_label_path if args.WP_label_path is not None else None
    PZ_TZ_label_path = (
        args.PZ_TZ_label_path if args.PZ_TZ_label_path is not None else None
    )

    # Create base directory path and ensure consistency - pointing to project root
    current_file_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(
        current_file_dir, "..", ".."
    )  # Go up two levels to reach project root
    base_dir = os.path.normpath(base_dir)
    logs_dir = os.path.join(base_dir, "logs")
    results_root = os.path.join(base_dir, "Results")
    if args.results_subdir:
        results_root = os.path.join(results_root, args.results_subdir)
    csv_results_dir = os.path.join(results_root, "csv_results", exp_name)
    predictions_dir = os.path.join(results_root, "predictions", exp_name)
    prob_maps_dir = os.path.join(results_root, "prob_maps", exp_name)
    model_weights_dir = os.path.join(base_dir, "model_weights")

    os.makedirs(csv_results_dir, exist_ok=True)
    os.makedirs(predictions_dir, exist_ok=True)
    os.makedirs(prob_maps_dir, exist_ok=True)

    LOG_FILE = os.path.join(logs_dir, f"{exp_name}_train.log")
    LOGGER = init_logger(LOG_FILE)

    config = load_json(args.config)

    # batch_size can be > 1 - the collate function now tracks patches_per_image
    # and recreate_image handles variable patch counts per image
    config["batch_size"] = args.batch_size

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

    # For inference, use fixed crop size of 384x384 to match training
    crop_sz = [384, 384]
    patch_sz = suggested_patch_size
    spacing = plans_manager.original_median_spacing_after_transp[::-1]

    seed_torch(seed=config["seed"])
    set_determinism(seed=config["seed"])
    print(f"Seed set to: {os.getenv('PYTHONHASHSEED')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    wandb.init(project=PROJECT, entity=ENTITY, config=config)
    wandb.define_metric("custom_train_step")
    wandb.define_metric("custom_val_step")
    wandb.define_metric("custom_lr_step")

    # Associating the step metric with the train/val loss
    wandb.define_metric("train/*", step_metric="custom_train_step")
    wandb.define_metric("val/*", step_metric="custom_val_step")

    LOGGER.info(f"Experiment name: {exp_name}_inference")

    if prostateX_mapping is None:
        print("Using Active Surveillance data loader")
        rem_ids_config_path = os.path.join(
            base_dir,
            "mambax_net",
            "configs",
            "ignore_ids.yaml",
        )
        rem_ids_config_path = os.path.normpath(rem_ids_config_path)
        test_df = active_surveillance_infer_data_load(
            T2_path,
            wp_label_path=WP_label_path,
            pz_tz_label_path=PZ_TZ_label_path,
            test_wp_label_path=args.test_WP_label_path,
            test_time_point=args.test_time_point,
            rem_ids_config_path=rem_ids_config_path,
        )

        if WP_label_path is None or PZ_TZ_label_path is None:
            print(f"Found {len(test_df)} T2 images for inference without masks")
        else:
            print(f"Found {len(test_df)} AS studies for inference with masks")
    else:
        print("Using ProstateX data loader")
        _, test_df = picai_data_load(df_path, prostateX_mapping)

    if args.debug:
        test_df = test_df.sample(20).reset_index(drop=True)
        config["epochs"] = 1

    config["deep_supervision"] = False

    model = build_network_architecture(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels=config["in_channels"],
        num_output_channels=config["out_channels"],
        enable_deep_supervision=config["deep_supervision"],
    ).to(device)

    if config["bf16"]:
        LOGGER.info("Using mixed precision inference...")

    preprocess = Compose(
        [
            MONAIResample(
                (spacing[0], spacing[1], spacing[2]), skip_mask_preprocess=False
            ),
            InPlaneCrop(crop_sz[0], crop_sz[1]),
            NormalizeData(mean=img_mean, std=img_std),
        ]
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
        "seg_cache",
    )
    cache_dir = os.path.normpath(cache_dir)

    test_dataset = PicSegDataset(
        test_df,
        patch_iter,
        T2_path,
        WP_label_path,
        PZ_TZ_label_path,
        preprocess=preprocess,
        transform=None,
        patch_transform=None,
        with_coordinates=True,
        cache=False,
        cache_dir=cache_dir,
        mode="infer",
        dataset_name=dataset_name,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=patch_collate_infer,
    )

    LOGGER.info("Loading dataset complete")

    criterion = FocalTverskyDiceCE(
        alpha=0.5,
        beta=0.5,
        gamma=0.8,
        focal_gamma=2.0,
        background=True,
        sigmoid=True,
        softmax=False,
        to_onehot_y=False,
        mode="binary",
    )

    LOGGER.info("Running Inference")

    # Check if we have masks for evaluation
    has_masks = WP_label_path is not None and PZ_TZ_label_path is not None

    if has_masks:
        # Run inference with evaluation metrics
        test_metrics = inference_func(
            config,
            exp_name,
            model_weights_dir,
            model,
            test_loader,
            device,
            num_samples=config["samples"],
            criterion=criterion,
            dataset_name=dataset_name,
        )

        LOGGER.info(
            f"  Mean Loss: {test_metrics['test_loss_mean']}, \
                    Mean Dice: {test_metrics['test_dice_mean']}, Mean WP: {test_metrics['test_wp_mean']},\
                        Mean PZ: {test_metrics['test_pz_mean']}, Mean TZ: {test_metrics['test_tz_mean']}"
        )

        LOGGER.info(
            f"  Std Loss: {test_metrics['test_std_loss']}, \
                    Std Dice: {test_metrics['test_std_dice']}, Std WP: {test_metrics['test_std_wp']},\
                        Std PZ: {test_metrics['test_std_pz']}, Std TZ: {test_metrics['test_std_tz']}"
        )

        LOGGER.info(
            f"  95CI Loss: {test_metrics['test_95ci_loss']}, "
            f"95CI Dice: {test_metrics['test_95ci_dice']}, 95CI WP: {test_metrics['test_95ci_wp']}, "
            f"95CI PZ: {test_metrics['test_95ci_pz']}, 95CI TZ: {test_metrics['test_95ci_tz']}"
        )

        # Move bootstrap results
        bootstrap_file = f"{exp_name}_bootstrap.csv"
        if os.path.exists(bootstrap_file):
            shutil.move(bootstrap_file, f"{csv_results_dir}/{exp_name}_bootstrap.csv")

        # Log metrics to wandb
        wandb.run.summary["Mean Loss"] = test_metrics["test_loss_mean"]
        wandb.run.summary["Mean Dice"] = test_metrics["test_dice_mean"]
        wandb.run.summary["Mean WP"] = test_metrics["test_wp_mean"]
        wandb.run.summary["Mean PZ"] = test_metrics["test_pz_mean"]
        wandb.run.summary["Mean TZ"] = test_metrics["test_tz_mean"]
        wandb.run.summary["Std Loss"] = test_metrics["test_std_loss"]
        wandb.run.summary["Std Dice"] = test_metrics["test_std_dice"]
        wandb.run.summary["Std WP"] = test_metrics["test_std_wp"]
        wandb.run.summary["Std PZ"] = test_metrics["test_std_pz"]
        wandb.run.summary["Std TZ"] = test_metrics["test_std_tz"]
        wandb.run.summary["Mean WP HDF"] = test_metrics["test_wp_hdf_mean"]
        wandb.run.summary["Mean PZ HDF"] = test_metrics["test_pz_hdf_mean"]
        wandb.run.summary["Mean TZ HDF"] = test_metrics["test_tz_hdf_mean"]
        wandb.run.summary["Std WP HDF"] = test_metrics["test_std_wp_hdf"]
        wandb.run.summary["Std PZ HDF"] = test_metrics["test_std_pz_hdf"]
        wandb.run.summary["Std TZ HDF"] = test_metrics["test_std_tz_hdf"]
        wandb.run.summary["95CI Loss"] = test_metrics["test_95ci_loss"]
        wandb.run.summary["95CI Dice"] = test_metrics["test_95ci_dice"]
        wandb.run.summary["95CI WP"] = test_metrics["test_95ci_wp"]
        wandb.run.summary["95CI PZ"] = test_metrics["test_95ci_pz"]
        wandb.run.summary["95CI TZ"] = test_metrics["test_95ci_tz"]
        wandb.run.summary["95CI WP HDF"] = test_metrics["test_95ci_wp_hdf"]
        wandb.run.summary["95CI PZ HDF"] = test_metrics["test_95ci_pz_hdf"]
        wandb.run.summary["95CI TZ HDF"] = test_metrics["test_95ci_tz_hdf"]
        log_ci_bounds_to_wandb(test_metrics)
        # ── Regional (apex / mid-gland / base) gland metrics ──
        if "region_metrics" in test_metrics:
            for _reg, _zone_data in test_metrics["region_metrics"].items():
                for _zone, _metric_data in _zone_data.items():
                    for _metric, _stats in _metric_data.items():
                        for _stat in ("mean", "std", "ci95"):
                            wandb.run.summary[f"{_reg}/{_zone}/{_metric}/{_stat}"] = (
                                _stats[_stat]
                            )

    # Generate predictions (works with or without masks)
    generate_predictions(
        prob_maps_dir,
        predictions_dir,
        config,
        exp_name,
        model_weights_dir,
        model,
        test_loader,
        device,
        dataset_name=dataset_name,
        log_preds_to_wandb=False,
    )

    if not has_masks:
        LOGGER.info(
            "Inference completed without masks - predictions saved to output directories"
        )

    del model
    gc.collect()


if __name__ == "__main__":
    infer_loop()
    print("Inference complete")
    wandb.finish()
