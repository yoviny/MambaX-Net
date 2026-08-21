# MambaX-Net

A dual-input, Mamba-enhanced cross-attention network for longitudinal prostate MRI segmentation. Given a patient's current T2-weighted scan plus their prior scan and segmentation mask, the model segments the whole prostate (WP) and the peripheral/transition zones (PZ/TZ) on the current scan.

![MambaX-Net architecture](https://ars.els-cdn.com/content/image/1-s2.0-S1361841526003208-gr1_lrg.jpg)

Read the paper:

🗎 [Yahathugoda, Y., Prezzi, D., Gutierrez, P.A., Ittichaiwong, P., Goh, V., Ourselin, S., Antonelli, M. **MambaX-Net: Dual-input Mamba-enhanced Cross-Attention network for longitudinal prostate MRI segmentation.** Medical Image Analysis 114, 104251 (2026). https://doi.org/10.1016/j.media.2026.104251](https://www.sciencedirect.com/science/article/pii/S1361841526003208)

# Table of Contents

  * [Layout](#layout)
  * [Setup](#setup)
  * [Running it](#running-it)
  * [Data](#data)
  * [Citation](#citation)
  * [License](#license)
  * [Disclaimer](#disclaimer)

# Layout

```
mambax_net/
├── dataset/     # CSV/NIfTI loading, dual-scan pairing (DualScanDataset)
├── network/     # MambaXNet + nnU-Net backbone
├── training/    # train loops (mx_net_train.py, nnunet_train.py)
├── inference/   # inference loops (mx_net_infer.py, nnunet_infer.py)
├── loss/        # Tversky / Focal-Tversky-Dice-CE compound losses
├── metrics/     # early stopping, running averages
├── preprocess/  # dataset fingerprinting, patch caching
├── postprocess/ # segmentation cleanup
├── scheduler/   # LR warmup scheduler
├── utilities/   # normalization, resampling, cropping, misc helpers
├── configs/     # dataset fingerprint / nnU-Net plans JSON, ignore lists
├── scripts/     # shell launchers for the above
└── data/mambaX_net_cache/  # DualScanDataset's on-disk cache (auto-created)
```

The first time you train or run inference, a few more directories get created at the repo root (all gitignored):

```
model_weights/  # put your pretrained nnU-Net fold checkpoint(s) here before training
checkpoints/    # saved training checkpoints
logs/           # train/infer log files
runs/           # TensorBoard runs
Results/        # inference output: csv_results/, predictions/, prob_maps/ (per exp_name)
cache/          # PicSegDataset's default cache dir (fingerprinting/PI-CAI path)
wandb/          # local W&B run data
```

# Setup

Tested with Python 3.12.6.

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu130 -r requirements.txt
```

You'll also need a `.env` file at the repo root for training and logging:

```
WANDB_API_KEY=
ENTITY=
PROJECT=
ElectricityMaps=      # optional, for carbon tracking
```

# Running it

**1. Compute a dataset fingerprint.** This has to run before training — it tells the network what patch size, normalization, and spacing to expect.

```bash
python -m mambax_net.preprocess.calculate_dataset_fingerprint_segmentation \
  -df <marksheet.csv> -o mambax_net/configs/ \
  -i <T2_dir> -wp <whole_prostate_masks> -pz <pz_tz_masks> \
  -np 4 -mem 20 -v True
```

**2. Train.**

By default, `MambaXNet` initializes its encoder/decoder from a pretrained nnU-Net fold checkpoint — you'll need a model weights file under `model_weights/` at the repo root first.

Set `"pretrained": false` in the config to skip this and train from scratch instead.

```bash
python -m mambax_net.training.mx_net_train \
  -t2 <T2_dir> -wp <wp_labels> -pz <pz_tz_labels> \
  -val_t2 <val_T2_dir> -val_wp <val_wp_labels> -val_pz <val_pz_tz_labels> \
  -test_t2 <test_T2_dir> -test_wp <test_wp_labels> -test_pz <test_pz_tz_labels> \
  -best_preds <path_for_best_predictions> \
  -conf mambax_net/configs/train_xnat_mx.json \
  -nw 4 --exp_name my-run --sample_sz 50
```

`mambax_net/scripts/mambaX_net_train.sh` wraps this command for the active-surveillance fine-tuning sweep — it trains at sample sizes 5, 10, 50, 100, and 250, and waits for a free GPU before starting. Set the `GPU_ID` / `MEM_THRESHOLD_MB` environment variables to change which GPU it watches and how much free memory it waits for.

Both training and inference log to Weights & Biases (see [Setup](#setup)), so running either starts a real W&B run under your account.

**3. Infer.**

```bash
python -m mambax_net.inference.mx_net_infer \
  -t2 <T2_dir> -ai_wp <prior_wp_preds> -ai_pz <prior_pz_tz_preds> \
  -wp <gt_wp_labels> -pz <gt_pz_tz_labels> \
  -conf mambax_net/configs/train_xnat_mx.json \
  -nw 4 -bs 1 --exp_name my-run -test_t_point _2
```

`-wp`/`-pz` are optional — pass them if you want evaluation metrics against ground truth, or omit them to just generate predictions.

# Data

`-t2`, `-wp`, and `-pz` each point to a folder of `.nii.gz` files named `<patient_id>_<timepoint>.nii.gz`. A patient's current and prior scans live in the **same** folder — the loader looks up the previous timepoint's file automatically:

```
t2/
├── 001_1.nii.gz   # patient 001, timepoint 1 (prior)
├── 001_2.nii.gz   # patient 001, timepoint 2 (current)
└── 002_1.nii.gz
wp/                # same naming, for whole-prostate masks
pz_tz/             # same naming, for PZ/TZ masks
```

The train/val/test split is just whichever `study_id`s show up in each folder — no CSV needed for this path.

At **inference**, `-ai_wp`/`-ai_pz` (the prior prediction) are always what gets fed to the model as the prior mask — the ground-truth prior is never used for this. `-wp`/`-pz` are optional and, when given, are only used as the evaluation target — never fed to the model.

`configs/ignore_ids.yaml` lets you exclude a handful of known-bad `study_id`s from training.

Note this CSV-free split only applies to training/inference. The fingerprinting step (`calculate_dataset_fingerprint_segmentation`, used for PI-CAI/nnU-Net pretraining) is a separate data-loading path (`picai_data_load`) and does read a CSV file.

# Citation

If you use this code, please cite the paper:

```bibtex
@article{YAHATHUGODA2026104251,
title = {MambaX-Net: Dual-input Mamba-enhanced Cross-Attention network for longitudinal prostate MRI segmentation},
journal = {Medical Image Analysis},
volume = {114},
pages = {104251},
year = {2026},
issn = {1361-8415},
doi = {https://doi.org/10.1016/j.media.2026.104251},
url = {https://www.sciencedirect.com/science/article/pii/S1361841526003208},
author = {Yovin Yahathugoda and Davide Prezzi and Patricia A. Gutierrez and Piyalitt Ittichaiwong and Vicky Goh and Sebastien Ourselin and Michela Antonelli},
}
```

# License

MIT — see [`LICENSE`](LICENSE). Note the exception in `mx_net.py`'s header: portions of that file (Mamba block, attention block) carry their own upstream licenses (Apache 2.0 / MIT).

# Disclaimer

The docstrings throughout this codebase and this README were generated with Claude Sonnet 5, reviewed against the actual code for accuracy.
