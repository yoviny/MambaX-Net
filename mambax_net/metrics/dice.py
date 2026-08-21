from typing import Any, Dict, List, Tuple

import torch
from monai.metrics import DiceMetric



def calculate_metrics(
    dice_metric: DiceMetric, dice_metric_batch: DiceMetric, conf: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate Dice metrics for segmentation tasks.

    Aggregates Dice coefficients from both overall and batch-specific metrics,
    extracting per-zone metrics for segmentation tasks (whole prostate, peripheral zone,
    transition zone).

    Args:
        dice_metric: MONAI DiceMetric instance for overall computation
        dice_metric_batch: MONAI DiceMetric instance for batch-wise computation
        conf: Configuration dictionary containing task information

    Returns:
        Tuple containing:
            - metric: Overall aggregated Dice metric
            - metric_wp: Whole prostate Dice metric (or 0.0 for non-segmentation tasks)
            - metric_pz: Peripheral zone Dice metric (or 0.0 for non-segmentation tasks)
            - metric_tz: Transition zone Dice metric (or 0.0 for non-segmentation tasks)
    """
    metric = dice_metric.aggregate()
    metric_batch = dice_metric_batch.aggregate()

    if conf.get("task") == "segmentation":
        metric_wp = metric_batch[0].detach()
        metric_pz = metric_batch[1].detach()
        metric_tz = metric_batch[2].detach()
    else:
        metric_wp = torch.tensor(0.0).cuda()
        metric_pz = torch.tensor(0.0).cuda()
        metric_tz = torch.tensor(0.0).cuda()

    dice_metric.reset()
    dice_metric_batch.reset()

    return metric, metric_wp, metric_pz, metric_tz


def calculate_test_metrics(
    dice_metric: DiceMetric,
    dice_metric_batch: DiceMetric,
    hd95_metric_batch: Any,
    conf: Dict[str, Any],
) -> Tuple[
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    """Calculate Dice and Hausdorff 95th percentile metrics for inference.

    Aggregates both Dice coefficients and Hausdorff distance metrics from batch computations,
    converting tensors to lists for statistical analysis.

    Args:
        dice_metric: MONAI DiceMetric instance for overall computation
        dice_metric_batch: MONAI DiceMetric instance for batch-wise Dice computation
        hd95_metric_batch: MONAI metric instance for Hausdorff 95th percentile computation
        conf: Configuration dictionary containing task information

    Returns:
        Tuple containing lists of:
            - metric: Overall Dice coefficients as list of floats
            - metric_wp: Whole prostate Dice coefficients
            - metric_pz: Peripheral zone Dice coefficients
            - metric_tz: Transition zone Dice coefficients
            - metric_wp_hdf: Whole prostate Hausdorff distances
            - metric_pz_hdf: Peripheral zone Hausdorff distances
            - metric_tz_hdf: Transition zone Hausdorff distances
    """
    metric = dice_metric.aggregate()

    metric_batch = dice_metric_batch.aggregate()
    metric_batch = metric_batch.T

    hd95 = hd95_metric_batch.aggregate()
    hd95 = hd95.T

    metric_wp, metric_pz, metric_tz = [], [], []
    metric_wp_hdf, metric_pz_hdf, metric_tz_hdf = [], [], []

    if conf.get("task") == "segmentation":
        metric_wp = metric_batch[0].detach()
        metric_pz = metric_batch[1].detach()
        metric_tz = metric_batch[2].detach()
        metric_wp_hdf = hd95[0].detach()
        metric_pz_hdf = hd95[1].detach()
        metric_tz_hdf = hd95[2].detach()
    else:
        zeros = torch.zeros(len(metric_batch[0])).cuda()
        metric_wp = zeros.clone()
        metric_pz = zeros.clone()
        metric_tz = zeros.clone()
        metric_wp_hdf = zeros.clone()
        metric_pz_hdf = zeros.clone()
        metric_tz_hdf = zeros.clone()

    metric = metric.detach().cpu().numpy().flatten().tolist()
    metric_wp = [t.cpu().numpy().item() for t in metric_wp]
    metric_pz = [t.cpu().numpy().item() for t in metric_pz]
    metric_tz = [t.cpu().numpy().item() for t in metric_tz]
    metric_wp_hdf = [t.cpu().numpy().item() for t in metric_wp_hdf]
    metric_pz_hdf = [t.cpu().numpy().item() for t in metric_pz_hdf]
    metric_tz_hdf = [t.cpu().numpy().item() for t in metric_tz_hdf]

    dice_metric.reset()
    dice_metric_batch.reset()
    hd95_metric_batch.reset()

    return (
        metric,
        metric_wp,
        metric_pz,
        metric_tz,
        metric_wp_hdf,
        metric_pz_hdf,
        metric_tz_hdf,
    )
