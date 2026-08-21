import argparse
import os
import warnings

warnings.filterwarnings("ignore")

import monai
import pandas as pd

from mambax_net.dataset.collate.custom_collate import patch_collate_fingerprint
from mambax_net.dataset.picai_dataset import PicSegDataset
from mambax_net.utilities.dataset_fingerprint import DatasetFingerprintExtractor
from mambax_net.utilities.experiment_planner import ExperimentPlanner


def run_fingerprint_extractor():
    """Run the fingerprint extractor on the dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--df_path", type=str, required=True)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-i", "--image_path", type=str, required=True)
    parser.add_argument("-wp", "--wp_mask_path", type=str, required=True)
    parser.add_argument("-pz", "--pz_tz_path", type=str, required=True)
    parser.add_argument("-np", "--num_processes", type=int, required=True)
    parser.add_argument("-mem", "--gpu_memory_GB", type=int, required=True)
    parser.add_argument("-v", "--verbose", type=bool, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images for processing (for testing)",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.df_path)
    if args.limit is not None:
        df = df.iloc[: args.limit]

    piccai_dataset = PicSegDataset(
        df,
        None,
        args.image_path,
        args.wp_mask_path,
        args.pz_tz_path,
        mode="fingerprint",
        dataset_name="AS",
    )

    dataloader = monai.data.DataLoader(
        piccai_dataset,
        batch_size=1,
        collate_fn=patch_collate_fingerprint,
        num_workers=args.num_processes,
        pin_memory=True,
        shuffle=False,
    )

    fingerprint_extractor = DatasetFingerprintExtractor(
        output_folder=args.output_dir,
        dataloader=dataloader,
        channels=1,
        num_processes=args.num_processes,
        verbose=args.verbose,
    )

    fingerprint_extractor.run(overwrite_existing=True)

    print("Fingerprint extraction complete")

    print("Planning experiment")
    planner = ExperimentPlanner(
        fingerprint_dir=args.output_dir,
        output_folder=args.output_dir,
        dataloader=dataloader,
        num_channels=1,
        gpu_memory_target_in_gb=args.gpu_memory_GB,
    )
    ret = planner.plan_experiment()
    print(ret)

    # rename the dataset fingerprint file
    os.rename(
        os.path.join(args.output_dir, "dataset_fingerprint.json"),
        os.path.join(args.output_dir, "dataset_fingerprint_segmentation.json"),
    )

    # rename the experiment plan file
    os.rename(
        os.path.join(args.output_dir, "nnUNetPlans.json"),
        os.path.join(args.output_dir, "nnUNetPlans_segmentation.json"),
    )


if __name__ == "__main__":
    run_fingerprint_extractor()
