"""utility functions for data loading, normalization, plotting, etc."""

from __future__ import annotations

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from path import Path
from PIL import Image
from torch import nn

UTCI_TEMPORAL_KEYS = [
    "Ta",
    "Wind",
    "Wd",
    "Kdown",
    # "rain",
    "RH",
    "press_hPa",
    "ElevationAngle",
    "AzimuthAngle",
]

# (mean, std, min, max, out) all values outside min/max (if provided) are set to out
UTCI_STATISTICS: dict = {
    # temporal inputs
    "Ta": (11.278715686274511, 8.657770759593202),  # R
    "Wind": (2.5414812745098043, 1.483173046904121),  # R>=0
    "Wd": (163.03941869783196, 94.08611881090562),  # [0, 360]
    "Kdown": (147.97507069679543, 223.5762460698097),  # R>=0
    "rain": (0.1853888360642635, 0.37900680322066316),  # R>=0
    "RH": (76.03840743388773, 16.65996528626173),  # R>=0
    "press_hPa": (976.9546348230699, 8.272637165034709),  # R>=0
    "ElevationAngle": (14.173846108184303, 18.911825936233367),  # [0,90]
    "AzimuthAngle": (180.13989122920333, 102.24888121470248),  # [0, 360]
    # mlp spatial predictors
    "popdens": (45.588235294117645, 33.270999139693735),  # N>=0
    "Paved": (0.21297647058823532, 0.11874602514355989),  # [0, 1]
    "Buildings": (0.1759764705882353, 0.09583082520162041),  # [0, 1]
    "EvergreenTrees": (0.13984705882352944, 0.11977096256350854),  # [0, 1]
    "DecidiousTrees": (0.2801882352941176, 0.1351201840794271),  # [0, 1]
    "Grass": (0.18031764705882353, 0.17684208800406045),  # [0, 1]
    "Baresoil": (0.0009176470588235294, 0.004323831426788389),  # [0, 1]
    "Water": (0.009776470588235294, 0.037158404955319725),  # [0, 1]
    "fai_b": (0.1805411764705882, 0.1609853995850302),  # R>=0
    "zd_b": (10.59109411764706, 4.172255817910023),  # R>=0
    "z0_b": (0.8964352941176471, 0.5652058084294189),  # [0,1]
    "zH_b": (9.639564705882353, 2.391176615367914),  # R>=0
    "fai_v_d": (0.2670823529411765, 0.22823792993511055),  # [0, 1]
    "zH_v_d": (9.236635294117646, 3.368167990971674),  # R>=0
    "fai_v_e": (0.18584705882352942, 0.2457516412715795),  # [0,1]
    "zH_v_e": (9.347494117647058, 5.282355906185028),  # R>=0
    "alt": (270.30411286836505, 38.50535265879784),  # R>=0
    # spatial inputs
    "r.DEM": (267.00128173828125, 38.712276458740234),
    "r.DSM.GB": (268.712158203125, 38.72966766357422),
    "r.DSM.V": (2.1023178100585938, 6.042695045471191),
    "r.LCC": (3.275212049484253, 1.8797717094421387),
    "r.WA": (5.31844425201416, 34.83320617675781),
    "r.WH": (0.2617618143558502, 1.6696650981903076),
    "svf": (0.9085833430290222, 0.13685189187526703),
    "svfE": (0.8972461223602295, 0.15581455826759338),
    "svfEaveg": (0.9775575399398804, 0.03341229259967804),
    "svfEveg": (0.8098065853118896, 0.28681206703186035),
    "svfN": (0.9111116528511047, 0.15392698347568512),
    "svfNaveg": (0.9874547719955444, 0.03190629929304123),
    "svfNveg": (0.8173003196716309, 0.29323482513427734),
    "svfS": (0.895430862903595, 0.1540963053703308),
    "svfSaveg": (0.9742522239685059, 0.034386828541755676),
    "svfSveg": (0.8083451986312866, 0.2831850051879883),
    "svfW": (0.9092960953712463, 0.15255197882652283),
    "svfWaveg": (0.9843837022781372, 0.0325491726398468),
    "svfWveg": (0.8158380389213562, 0.28909674286842346),
    "svfaveg": (0.9868439435958862, 0.028343046084046364),
    "svfveg": (0.8175613880157471, 0.280950129032135),
    # wind inputs
    "cl_bl_ht": (6.7819414138793945, 4.4101057052612305),
    "dwd_e": (15.899197578430176, 20.88045883178711),
    "dwd_n": (15.715604782104492, 20.888029098510742),
    "dwd_s": (16.126678466796875, 21.058149337768555),
    "dwd_w": (15.850367546081543, 20.90506362915039),
    "horzontal_distance": (45.34073257446289, 80.68336486816406),
    "pc_ew": (0.8602695465087891, 0.27840062975883484),
    "pc_ns": (0.8603993654251099, 0.2781860828399658),
    "sli_e": (8.654397964477539, 5.380777359008789),
    "sli_n": (8.693970680236816, 5.370758056640625),
    "sli_s": (8.685530662536621, 5.362427711486816),
    "sli_w": (8.664311408996582, 5.387010097503662),
    "swi_ew": (5.800251483917236, 3.556208848953247),
    "swi_ns": (5.77887487411499, 3.5664608478546143),
    "uwd_e": (15.850367546081543, 20.90506362915039),
    "uwd_n": (16.126678466796875, 21.058149337768555),
    "uwd_s": (15.715604782104492, 20.888029098510742),
    "uwd_w": (15.899197578430176, 20.88045883178711),
    # UTCI outputs
    "utci": (12.02523422241211, 1.399465560913086),  # R
}

UTCI_SPATIAL_TIF_DIRPATH = Path("data/raw/Spatial_Predictors")
UTCI_MLP_SPATIAL_TIF_FILEPATH = Path(
    "data/raw/MLP_Spatial_Predictors/MLP_spatial_predictors.csv"
)
UTCI_WIND_PREDICTORS_TIF_DIRPATH = Path("dataraw/wind_predictors")
UTCI_OUTPUT_TIF_DIRPATH = Path("data/raw")


def normalize_array(  # pylint: disable=too-many-arguments, too-many-positional-arguments
    x, mean, std, min=None, max=None, replace=None  # pylint: disable=redefined-builtin
):
    """normalize array x with mean and std, clip values to min and max if provided"""
    if replace:
        return np.where((min <= x) & (x <= max), (x - mean) / std, replace)
    return (x - mean) / std


def denormalize_array(x, mean, std):
    """denormalize array x with mean and std"""
    return (x * std) + mean


def denormalize_multi_dim_array(x, means, stds):
    """denormalize multi-dimensional array x with means and stds"""
    arrs = []
    for index in range(x.shape[1]):
        arrs.append(denormalize_array(x[:, index], means[index], stds[index]))
    return torch.stack(arrs, dim=1)


def load_and_combine_images(img_list, normalize: bool = True):
    """load and combine images from list of paths"""

    normalize_info: dict = UTCI_STATISTICS

    arrays = []
    for img_path in img_list:
        if ".tif" == img_path[-4:]:
            with Image.open(img_path) as img:
                array = np.array(img).clip(min=0)
            statistics_name = os.path.basename(img_path).replace(".tif", "")
        elif ".npy" == img_path[-4:]:
            array = np.load(img_path).clip(min=0)
            statistics_name = os.path.basename(img_path).replace(".npy", "")
        else:
            raise NotImplementedError
        if normalize and statistics_name in normalize_info:
            mean, std = normalize_info[statistics_name]
            arrays.append(normalize_array(array, mean, std))
        else:
            arrays.append(array)
    return np.stack(arrays, axis=0)


def get_device(debug=False) -> str:
    """return 'cuda' if available, else 'cpu'"""
    return "cuda" if torch.cuda.is_available() and not debug else "cpu"


def set_seed(seed: int = 0) -> None:
    """set seed for reproducibility"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def plot(prediction: np.array, gt: np.array, save_path: str | Path, mask=None):
    """plot prediction, ground truth, and their difference"""
    fig, axes = plt.subplots(figsize=(12, 4), ncols=3)
    vmin = min(np.amin(prediction), np.amin(gt), np.amin(prediction - gt))
    vmax = max(np.amax(prediction), np.amax(gt), np.amax(prediction - gt))
    for ax, img, label in zip(
        axes,
        [prediction, gt, prediction - gt],
        ["prediction", "ground truth", "signed difference"],
    ):
        ax.axis("off")
        if mask is not None:
            im = np.ma.masked_where(
                ~mask, img  # pylint: disable=invalid-unary-operand-type
            )
        else:
            im = img
        if "difference" in label:
            # ensure symmetry of diff colorbar
            diff_vmax = max(abs(np.amin(im)), abs(np.amax(im)))
            ax_img = ax.imshow(im, cmap="seismic", vmin=-diff_vmax, vmax=diff_vmax)
        else:
            ax_img = ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=12)
        fig.colorbar(
            ax_img,
            ax=ax,
            extend="neither",
            spacing="proportional",
            shrink=0.7,
            format="%.1f",
        )
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.02)
    plt.savefig(save_path, bbox_inches="tight")
    plt.clf()
    plt.close()


class InputPadder:
    """Pads images such that dimensions are divisible by factor=2^x"""

    def __init__(self, dims: tuple, factor: int = 8, pad_mode: str = "replicate"):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // factor) + 1) * factor - self.ht) % factor
        pad_wd = (((self.wd // factor) + 1) * factor - self.wd) % factor
        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            pad_ht // 2,
            pad_ht - pad_ht // 2,
        ]
        self.pad_mode = pad_mode

    def pad(self, x):
        """pad spatial input to be divisible by factor"""
        return F.pad(x, self._pad + ([0, 0] if x.dim() > 4 else []), mode=self.pad_mode)

    def unpad(self, x):
        """remove padding from spatial input"""
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0] : c[1], c[2] : c[3]]


class MaskedLoss(
    nn.modules.loss._Loss
):  # pylint: disable=protected-access, too-few-public-methods
    """Masked loss function"""

    def __init__(self, loss, size_average=None, reduce=None, reduction: str = "mean"):
        super().__init__(size_average, reduce, reduction)
        self._loss = loss

    def forward(
        self,
        input: torch.Tensor,  # pylint: disable=redefined-builtin
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """forward pass of masked loss function"""
        if mask is None:
            return self._loss(input, target)
        loss = self._loss(input, target) * mask
        if self.reduction == "mean":
            if torch.sum(mask) == 0.0:
                return torch.sum(loss)
            return torch.sum(loss) / torch.sum(mask)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss
