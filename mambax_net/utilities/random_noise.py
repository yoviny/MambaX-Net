import numpy as np
import torch
from monai.transforms import MapTransform


def mark_mask_noise_by_patient(train_df, percent, seed, flag_col="mask_noise"):
    """Randomly flags a percentage of patients' rows as noisy

    Args:
        train_df (pandas.DataFrame): training dataframe with a "study_id" column
        percent (float): percentage (0-100) of unique patients to flag
        seed (int): seed for the random number generator
        flag_col (str): name of the column to store the noise flag in

    Returns:
        df (pandas.DataFrame): copy of train_df with flag_col set to 1 for
            rows belonging to the selected patients, 0 otherwise
        selected_patients (list): sorted list of patient ids flagged as noisy
    """
    df = train_df.copy()
    df[flag_col] = 0

    percent = max(0.0, min(100.0, float(percent)))
    patient_ids = df["study_id"].astype(str).str.rsplit("_", n=1).str[0]
    unique_patients = patient_ids.drop_duplicates().sort_values().to_numpy()
    n_noisy = int(len(unique_patients) * percent / 100.0)
    if n_noisy == 0:
        return df, []

    rng = np.random.default_rng(int(seed))
    selected_patients = sorted(
        rng.choice(unique_patients, size=n_noisy, replace=False).tolist()
    )
    noisy_rows = patient_ids.isin(selected_patients)

    df.loc[noisy_rows, flag_col] = 1
    return df, selected_patients


def _flag_enabled(value):
    """Checks whether a noise flag value is truthy

    Args:
        value: flag value, may be None, a tensor, or a number

    Returns:
        bool: False if value is None, otherwise value converted to int and
            cast to bool
    """
    if value is None:
        return False
    if torch.is_tensor(value):
        value = value.item()
    return bool(int(value))


class RandomMaskNoise(MapTransform):
    """Replaces the mask with random binary noise with a given probability."""

    def __init__(self, keys, prob=0.5):
        """
        Initialize the object with specified keys and probability.
        Args:
            keys (iterable): The keys to be used by the object.
            prob (float, optional): The probability value, defaults to 0.5.
        """

        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        """
        Applies random noise to the 'mask' field in the input data with a given probability.

        Args:
            data (dict or Mapping): Input data containing a 'mask' key, where 'mask' is a tensor or a sequence of tensors.

        Returns:
            dict: A copy of the input data with the 'mask' potentially replaced by random noise, clamped to [0, 1], and thresholded to binary values.
        """
        d = dict(data)

        if torch.rand(1).item() < self.prob:
            for i in range(len(d["mask"])):
                # add random noise to the mask
                d["mask"][i] = torch.randn_like(d["mask"][i])
                # smooth the mask
                d["mask"][i] = torch.clamp(d["mask"][i], 0, 1)
                d["mask"] = d["mask"].to(torch.float32)
                # threshold the mask to binary
                d["mask"] = (d["mask"] > 0.5).float()
        return d


# Not currently used, but could be useful for future experiments with stratified noise injection
class StratifiedRandomMaskNoise(MapTransform):
    """Replaces the mask with random binary noise for samples flagged with data["noise"] == 1."""

    def __init__(self, keys):
        """
        Initializes the object with the specified keys.

        Args:
            keys: The keys to be used for initialization. The expected type and structure of keys should be specified in the class documentation.
        """

        super().__init__(keys)

    def __call__(self, data):
        """
        Applies random noise to the 'mask' in the input data if noise is enabled.

        Args:
            data (dict or Mapping): Input data containing at least the keys 'noise' and 'mask'.
                - 'noise' (int): If set to 1, random noise will be added to the mask.
                - 'mask' (Tensor or list of Tensors): The mask(s) to which noise will be applied.
        Returns:
            dict: A copy of the input data with the 'mask' modified if noise is enabled.
                The mask is replaced with random noise, clamped between 0 and 1, converted to float32,
                and thresholded to a binary mask.
        """
        d = dict(data)

        if d["noise"] == 1:
            for i in range(len(d["mask"])):
                # add random noise to the mask
                d["mask"][i] = torch.randn_like(d["mask"][i])
                d["mask"][i] = torch.clamp(d["mask"][i], 0, 1)
                d["mask"] = d["mask"].to(torch.float32)
                d["mask"] = (d["mask"] > 0.5).float()
        return d


class MaskDropout(MapTransform):
    """Randomly drops foreground mask voxels for samples flagged as noisy."""

    def __init__(self, keys, drop_prob=0.0, flag_key="mask_noise"):
        """Initialize the MaskDropout transform.

        Args:
            keys (iterable): keys to apply the transform to.
            drop_prob (float, optional): probability of dropping each
                foreground voxel, clamped to [0, 1]. Defaults to 0.0.
            flag_key (str, optional): key in the input data used to check
                whether dropout should be applied. Defaults to "mask_noise".
        """
        super().__init__(keys)
        self.drop_prob = max(0.0, min(1.0, float(drop_prob)))
        self.flag_key = flag_key

    def __call__(self, data):
        """Randomly zeroes out foreground mask voxels if the noise flag is set.

        Args:
            data (dict or Mapping): input data containing a "mask" key and
                the flag_key used to determine whether dropout is applied.

        Returns:
            dict: copy of the input data with foreground mask voxels randomly
                dropped according to drop_prob, unless drop_prob is 0 or the
                flag_key is not enabled, in which case the data is returned
                unchanged.
        """
        d = dict(data)
        if self.drop_prob <= 0:
            return d
        if not _flag_enabled(d.get(self.flag_key)):
            return d

        mask = d["mask"]
        output = mask.clone()
        foreground = output > 0.5
        keep = torch.rand_like(output.float()) >= self.drop_prob
        output = (foreground & keep).to(output.dtype)
        d["mask"] = output
        return d
