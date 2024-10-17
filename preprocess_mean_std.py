"""compute the means and stds"""
import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import UTCI_TEMPORAL_KEYS

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="data")
args = parser.parse_args()

args.output_dir = Path(args.output_dir)

# Train areas and dates
train_areas = pd.read_csv(args.output_dir / "Train_Areas.csv")
test_areas = pd.read_csv(args.output_dir / "Test_Areas.csv")
train_dates = pd.read_csv(args.output_dir / "Date_Train.csv")
test_dates = pd.read_csv(args.output_dir / "Date_Test.csv")

# Temporal inputs
era5 = pd.read_csv(args.output_dir / "Met_Data_ERA5.csv")
filtered_era5 = pd.concat(
    [era5[era5["date"].str.contains(date)] for date in list(train_dates["date"])]
)
for key in UTCI_TEMPORAL_KEYS:
    if key == "press":
        key = "press_hPa"  # pylint: disable=invalid-name
    mean = filtered_era5[key].mean()
    std = filtered_era5[key].std()
    print(f'"{key}": ({mean}, {std}),')

# MLP spatial predictors
mlp_spatial_predictors = pd.read_csv(args.output_dir / "MLP_spatial_predictors.csv")
filtered_mlp_spatial_predictors = mlp_spatial_predictors[
    mlp_spatial_predictors["prediction_areas"].isin(
        list(train_areas["prediction_areas"])
    )
]
for column_names in filtered_mlp_spatial_predictors.columns.tolist():
    if column_names in ["id", "prediction_areas"]:
        continue
    mean = filtered_mlp_spatial_predictors[column_names].mean()
    std = filtered_mlp_spatial_predictors[column_names].std()
    print(f'"{column_names}": ({mean}, {std}),')

# Spatial
spatial_base_path = args.output_dir / "input/spatial_meta_data"
spatial_order = [
    "r.DEM",
    "r.DSM.GB",
    "r.DSM.V",
    "r.LCC",
    "r.WA",
    "r.WH",
    "svf",
    "svfE",
    "svfEaveg",
    "svfEveg",
    "svfN",
    "svfNaveg",
    "svfNveg",
    "svfS",
    "svfSaveg",
    "svfSveg",
    "svfW",
    "svfWaveg",
    "svfWveg",
    "svfaveg",
    "svfveg",
]
data_list = []
for np_file in tqdm(glob(str(spatial_base_path / "*.npy")), leave=False):
    if "normalized" in os.path.basename(np_file):
        continue
    if os.path.basename(np_file)[:-4] not in list(train_areas["prediction_areas"]):
        continue
    # if os.path.basename(np_file)[:-4] not in list(test_areas["prediction_areas"]):
    #     continue
    data_list.append(np.load(np_file))
assert len(train_areas) == len(data_list)
# assert len(test_areas) == len(np_data)
np_data = np.stack(data_list, axis=0)
means = np_data.mean(axis=(0, 2, 3))
stds = np_data.std(axis=(0, 2, 3))
for name, mean, std in zip(spatial_order, means, stds):
    print(f'"{name}": ({mean}, {std}),')

# Wind
wind_base_path = args.output_dir / "input/wind_predictors"
wind_order = [
    "cl_bl_ht",
    "dwd_e",
    "dwd_n",
    "dwd_s",
    "dwd_w",
    "horzontal_distance",
    "pc_ew",
    "pc_ns",
    "sli_e",
    "sli_n",
    "sli_s",
    "sli_w",
    "swi_ew",
    "swi_ns",
    "uwd_e",
    "uwd_n",
    "uwd_s",
    "uwd_w",
]
np_data = []
for np_file in tqdm(glob(str(wind_base_path / "*.npy")), leave=False):
    if "normalized" in os.path.basename(np_file):
        continue
    if os.path.basename(np_file)[:-4] not in list(train_areas["prediction_areas"]):
        continue
    # if os.path.basename(np_file)[:-4] not in list(test_areas["prediction_areas"]):
    #     continue
    np_arr = np.load(np_file)
    if np_arr.shape[1] != np_arr.shape[2]:
        continue
    np_arr[6] = np.nan_to_num(np_arr[6], nan=0)
    np_arr[7] = np.nan_to_num(np_arr[7], nan=0)
    np_data.append(np_arr)
assert len(train_areas) == len(np_data)
# assert len(test_areas)-1 == len(np_data)
np_data = np.stack(np_data, axis=0)
means = np_data.mean(axis=(0, 2, 3))
stds = np_data.std(axis=(0, 2, 3))
for name, mean, std in zip(wind_order, means, stds):
    print(f'"{name}": ({mean}, {std}),')

# UTCI
utci_base_path = args.output_dir / "output"
means, stds = [], []
for area in tqdm(list(train_areas["prediction_areas"]), leave=False):
    # for area in tqdm(list(test_areas["prediction_areas"]), leave=False):
    for date in tqdm(list(train_dates["date"]), leave=False):
        # for date in tqdm(list(test_dates["date"]), leave=False):
        folder_path = utci_base_path / area / date
        for np_file in glob(str(folder_path / "*.npy")):
            utci = np.load(np_file)
            means.append(np.mean(utci))
            stds.append(np.std(utci))
mean = np.mean(means)
std = np.mean(stds)
print(f'"utci": ({mean}, {std}),')
