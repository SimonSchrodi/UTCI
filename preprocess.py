"""Preprocess the raw data that we use for training and evaluating the model."""

import argparse
import glob
import os
import shutil
import time

import numpy as np
from path import Path
from PIL import Image
from tqdm import tqdm

from utils import load_and_combine_images

BUILDING_CLASS_NO = 2

parser = argparse.ArgumentParser()
parser.add_argument("--raw_data", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--print_info", action="store_true")
args = parser.parse_args()

# assert os.path.isdir(args.raw_data)
args.raw_data = Path(args.raw_data)
if not args.output_dir:
    args.output_dir = os.path.dirname(args.raw_data)
args.output_dir = Path(args.output_dir)
args.output_dir.makedirs_p()


for folder in tqdm(
    [
        "Meteorological_Input",
        "Spatial_Input",
    ],
    leave=False,
):
    for csv_file in tqdm(
        glob.glob(args.raw_data + f"/Predictors_Training/{folder}/*.csv"), leave=False
    ):
        shutil.copy(csv_file, args.output_dir / os.path.basename(csv_file))

for csv_file in glob.glob(
    args.raw_data + "/Predictors_ClimateChangeScenarios/Spatial_Input/Prediction*.csv"
):
    shutil.copy(csv_file, args.output_dir / os.path.basename(csv_file))

climate_scenarios = args.output_dir / "scenarios"
climate_scenarios.makedirs_p()
for folder in tqdm(
    [
        "Meteorological_Input",
    ],
    leave=False,
):
    for csv_file in tqdm(
        glob.glob(args.raw_data + f"/Predictors_ClimateChangeScenarios/{folder}/*.csv"),
        leave=False,
    ):
        shutil.copy(csv_file, climate_scenarios / os.path.basename(csv_file))

# input
input_dir = args.output_dir / "input"
input_dir.makedirs_p()

masks_dir = input_dir / "spatial_masks"
masks_dir.makedirs_p()
if args.print_info:
    tif_load_time, np_load_time = [], []
for path_to_raw_areas in tqdm(
    glob.glob(
        args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data" / "*"
    ),
    leave=False,
):
    area = os.path.basename(path_to_raw_areas)
    mask_data_file = os.path.join(path_to_raw_areas, "r.LCC.tif")

    start = time.time()
    with Image.open(mask_data_file) as img:
        array = np.array(img).clip(min=0)
    end = time.time()
    if args.print_info:
        tif_load_time.append(end - start)

    mask = array == BUILDING_CLASS_NO
    np.save(masks_dir / f"{area}.npy", mask)

    if args.print_info:
        start = time.time()
        _ = np.load(masks_dir / f"{area}.npy")
        end = time.time()
        np_load_time.append(end - start)

if args.print_info:
    print(
        "Masks:",
        f"Tif {np.mean(tif_load_time)}",
        f"Np {np.mean(np_load_time)}",
    )

# spatial predictors
spatial_dir = input_dir / "spatial_meta_data"
spatial_dir.makedirs_p()

if args.print_info:
    tif_load_time, np_load_time = [], []

if not len(
    os.listdir(args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data")
) == len(
    os.listdir(
        args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data_Wind"
    )
):
    print(
        "Missing wind predictors:",
        set(
            os.listdir(
                args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data"
            )
        )
        - set(
            os.listdir(
                args.raw_data
                / "Predictors_Training"
                / "Spatial_Input"
                / "Spatial_Data_Wind"
            )
        ),
    )
    print(
        "Missing spatial predictors:",
        set(os.listdir(args.raw_data / "v"))
        - set(
            os.listdir(
                args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data"
            )
        ),
    )

areas = glob.glob(
    args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data" / "*"
)
for path_to_raw_areas in tqdm(areas, leave=False):
    area = os.path.basename(path_to_raw_areas)
    spatial_meta_data = glob.glob(path_to_raw_areas + "/*.tif")
    spatial_meta_data.extend(
        glob.glob(os.path.join(path_to_raw_areas, "svfs") + "/*.tif")
    )
    for normalize in [False, True]:
        start = time.time()
        np_arr = load_and_combine_images(sorted(spatial_meta_data), normalize=normalize)
        end = time.time()
        if args.print_info:
            tif_load_time.append(end - start)
        if normalize:
            np.save(spatial_dir / f"{area}_normalized.npy", np_arr)
        else:
            np.save(spatial_dir / f"{area}.npy", np_arr)

areas = glob.glob(
    args.raw_data
    / "Predictors_ClimateChangeScenarios"
    / "Spatial_Input"
    / "Spatial_Data"
    / "*"
)
for path_to_raw_areas in tqdm(areas, leave=False):
    area = os.path.basename(path_to_raw_areas)
    spatial_meta_data = glob.glob(path_to_raw_areas + "/*.tif")
    spatial_meta_data.extend(
        glob.glob(os.path.join(path_to_raw_areas, "svfs") + "/*.tif")
    )
    for normalize in [False, True]:
        start = time.time()
        np_arr = load_and_combine_images(sorted(spatial_meta_data), normalize=normalize)
        end = time.time()
        if args.print_info:
            tif_load_time.append(end - start)
        if normalize:
            np.save(spatial_dir / f"{area}_normalized.npy", np_arr)
        else:
            np.save(spatial_dir / f"{area}.npy", np_arr)

if args.print_info:
    start = time.time()
    _ = np.load(spatial_dir / f"{area}.npy")
    end = time.time()
    np_load_time.append(end - start)

if args.print_info:
    print(
        "Spatial meta data:",
        f"Tif {np.mean(tif_load_time)}",
        f"Np {np.mean(np_load_time)}",
    )

wind_dir = input_dir / "wind_predictors"
wind_dir.makedirs_p()

if args.print_info:
    tif_load_time, np_load_time = [], []

areas = glob.glob(
    args.raw_data / "Predictors_Training" / "Spatial_Input" / "Spatial_Data_Wind" / "*"
)
for path_to_raw_areas in tqdm(areas, leave=False):
    area = os.path.basename(path_to_raw_areas)
    wind_predictors = glob.glob(path_to_raw_areas + "/*.tif")
    for normalize in [False, True]:
        start = time.time()
        np_arr = load_and_combine_images(sorted(wind_predictors), normalize=normalize)
        end = time.time()
        if args.print_info:
            tif_load_time.append(end - start)
        if normalize:
            np.save(wind_dir / f"{area}_normalized.npy", np_arr)
        else:
            np.save(wind_dir / f"{area}.npy", np_arr)

    if args.print_info:
        start = time.time()
        _ = np.load(wind_dir / f"{area}.npy")
        end = time.time()
        np_load_time.append(end - start)

areas = glob.glob(
    args.raw_data
    / "Predictors_ClimateChangeScenarios"
    / "Spatial_Input"
    / "Spatial_Data_Wind"
    / "*"
)
for path_to_raw_areas in tqdm(areas, leave=False):
    area = os.path.basename(path_to_raw_areas)
    wind_predictors = glob.glob(path_to_raw_areas + "/*.tif")
    for normalize in [False, True]:
        start = time.time()
        np_arr = load_and_combine_images(sorted(wind_predictors), normalize=normalize)
        end = time.time()
        if args.print_info:
            tif_load_time.append(end - start)
        if normalize:
            np.save(wind_dir / f"{area}_normalized.npy", np_arr)
        else:
            np.save(wind_dir / f"{area}.npy", np_arr)

    if args.print_info:
        start = time.time()
        _ = np.load(wind_dir / f"{area}.npy")
        end = time.time()
        np_load_time.append(end - start)

if args.print_info:
    print(
        "Spatial meta data:",
        f"Tif {np.mean(tif_load_time)}",
        f"Np {np.mean(np_load_time)}",
    )


# output
output_dir = args.output_dir / "output"
output_dir.makedirs_p()

if args.print_info:
    tif_load_time, np_load_time = [], []
folders = list(
    filter(
        lambda x: all(split.isdigit() for split in str(os.path.basename(x)).split("_")),
        filter(
            os.path.isdir,
            [args.raw_data / folder for folder in os.listdir(args.raw_data)],
        ),
    )
)
folders = glob.glob(args.raw_data / "Predictions_UTCI" / "Tiff_ERA5_SUEWS" / "*")
for folder in tqdm(sorted(folders), leave=False):
    for path_to_raw_areas in tqdm(glob.glob(Path(folder) / "*"), leave=True):
        tif_files = glob.glob(Path(path_to_raw_areas) / "*.tif")
        tmp_dict: list = []
        for tif_file in tqdm(tif_files, leave=False):
            if "average" in tif_file:
                continue
            start = time.time()
            with Image.open(tif_file) as img:
                utci = np.array(img)
            end = time.time()
            if args.print_info:
                tif_load_time.append(end - start)

            output_path = (
                output_dir
                / path_to_raw_areas.split("/")[-2]
                / os.path.basename(path_to_raw_areas)
                / tif_file[len(path_to_raw_areas) + 1 :].replace(".tif", ".npy")
            )
            Path(os.path.dirname(output_path)).makedirs_p()
            np.save(output_path, utci)

            if args.print_info:
                start = time.time()
                _ = np.load(output_path)
                end = time.time()
                np_load_time.append(end - start)

if args.print_info:
    print(
        "Output data:", f"Tif {np.mean(tif_load_time)}", f"Np {np.mean(np_load_time)}"
    )
