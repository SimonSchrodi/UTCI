"""Evaluate the time period of the UTCI model."""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import suppress
from copy import deepcopy

import numpy as np  # pylint: disable=import-error
import pandas as pd  # pylint: disable=import-error
import torch  # pylint: disable=import-error
from path import Path  # pylint: disable=import-error
from torch.utils.data import DataLoader  # pylint: disable=import-error
from tqdm import tqdm, trange  # pylint: disable=import-error

from dataset import UTCIDataset
from model import ConvEncoderDecoder, UTCIModel
from utils import UTCI_STATISTICS, UTCI_TEMPORAL_KEYS, InputPadder, get_device


def process_temporal_data(data_dict: dict, ignored_keys: list = None):
    """Process the temporal data."""
    if ignored_keys is None:
        ignored_keys = []
    result = []
    for k in UTCI_TEMPORAL_KEYS:
        if k == "press":
            k = "press_hPa"
        if k in ignored_keys:
            continue
        if k == "dt":
            (h, m, _) = data_dict[k].split(" ")[-1].split(":")
            result.append(int(h) * 60 + int(m))
        else:
            result.append(data_dict[k])
    return result


def find_nth(haystack, needle, n):
    """Find the nth occurrence of a substring in a string."""
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start


parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--temporal_data", type=str)
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--lower_bound", type=int, default=-15)
parser.add_argument("--upper_bound", type=int, default=48)
parser.add_argument(
    "--pad_mode",
    default="replicate",
    choices=["constant", "reflect", "replicate", "circular"],
    type=str,
)
parser.add_argument("--regenerate", action="store_true")
parser.add_argument("--amp", action="store_true")
parser.add_argument("--verbose", action="store_true")
parser.add_argument("--DEBUG", action="store_true")
parser.add_argument("--device", default="cuda", type=str)
args = parser.parse_args()

assert args.upper_bound > args.lower_bound

args.output_dir = Path(os.path.dirname(args.model_path)) / "scenarios"
dirname = os.path.basename(args.temporal_data)[:-4]
args.output_dir = args.output_dir / dirname
args.output_dir.makedirs_p()

ARGS_PATH = f"{os.path.dirname(args.model_path)}/args.json"
with open(ARGS_PATH, encoding="utf-8") as json_data:
    model_args = argparse.Namespace()
    model_args.__dict__.update(json.load(json_data))
    model_args = parser.parse_args(namespace=model_args)

DEVICE = get_device()
if args.DEBUG:
    DEVICE = "cuda:0"

checkpoint = torch.load(args.model_path, map_location="cpu")
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
bias = any(".bias" in w for w in list(checkpoint.keys()))
utci_model = ConvEncoderDecoder(model_args, bias=False)

model = UTCIModel(utci_model=utci_model)
model.load_state_dict(checkpoint, strict=True)
model.eval()
model.to(DEVICE)
model_input_kwargs = model.forward_input_kwargs

scenario_areas = list(
    pd.read_csv(os.path.join(args.data_path, "Prediction_Areas_116.csv")).iloc[:, 1]
)
if args.DEBUG:
    scenario_areas = scenario_areas[:2]
test_dates = list(
    pd.read_csv(os.path.join(args.data_path, "Date_Test.csv")).iloc[:, 0]
)  # dummy dates
test_data = UTCIDataset(
    args.data_path,
    areas=scenario_areas,
    dates=test_dates,
    ignore_temporal_keys=model_args.ignore_temporal,
    return_building_mask=False,
    crop=False,
    return_identifier=True,
    forward_input_kwargs=model_input_kwargs,
    without_aveg=model_args.without_aveg if "without_aveg" in model_args else False,
    only_spatial=True,
    requires_utci=False,
    mlp_spatial_data="MLP_spatial_data_116",
)
test_loader = DataLoader(
    test_data,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)

df = pd.read_csv(args.temporal_data)
meteorological_inputs: dict = {
    f"{month}_{day_night}": []
    for month in range(1, 13)
    for day_night in ["day", "night"]
}

iterator = (
    tqdm(enumerate(df.iterrows()), leave=False, total=len(df))
    if args.verbose
    else enumerate(df.iterrows())
)
for idx, (row_index, row) in iterator:
    temporal_meta_t = process_temporal_data(
        row,
        ignored_keys=None,
    )
    temporal_meta_t = torch.tensor(temporal_meta_t).float().unsqueeze(0)
    temporal_meta_t[torch.isnan(temporal_meta_t)] = 0
    meteorological_inputs[f"{row['month']}_{row['day_night']}"].append(temporal_meta_t)

for month in range(1, 13):
    for day_night in ["day", "night"]:
        if len(meteorological_inputs[f"{month}_{day_night}"]) > 0:
            meteorological_inputs[f"{month}_{day_night}"] = torch.concat(
                meteorological_inputs[f"{month}_{day_night}"], dim=0
            ).to(DEVICE)
        else:
            print("WARNING:", f"{month}, {day_night} not available")
            del meteorological_inputs[f"{month}_{day_night}"]

amp_autocast = suppress  # pylint: disable=invalid-name
if args.amp:
    amp_autocast = torch.cuda.amp.autocast  # type: ignore[misc]

start_all = time.time()
with torch.inference_mode():
    data_iterator = (
        tqdm(test_loader, leave=False, total=len(test_loader))
        if args.verbose
        else test_loader
    )
    for data_blob in data_iterator:
        start_time = time.process_time()

        identifier = data_blob["identifier"][0]
        area = identifier[: find_nth(identifier, "_", 2)]

        model_inputs = {"statistics": UTCI_STATISTICS["utci"]}
        utci = data_blob["utci"]
        padder = InputPadder(utci.shape)
        # pylint: disable=R0801
        model_inputs.update(
            {
                k: (
                    padder.pad(data_blob[k].to(DEVICE))
                    if len(data_blob[k].shape) == 4
                    else data_blob[k].to(DEVICE)
                )
                for k in model_input_kwargs
                if k != "statistics"
            }
        )
        # pylint: enable=R0801
        statistics = model_inputs["statistics"]

        month_day_night_iterator = (
            tqdm(meteorological_inputs.items(), leave=False)
            if args.verbose
            else meteorological_inputs.items()
        )
        for month_day_night, input_temporal_t in month_day_night_iterator:
            copied_model_inputs = deepcopy(model_inputs)
            copied_model_inputs["spatial"] = torch.cat(
                [model_inputs["spatial"].clone() for _ in range(args.batch_size)],
                dim=0,
            )
            copied_model_inputs[
                "wind_predictors"
                if "wind_predictors" in model_inputs
                else "wind_precomputed"
            ] = torch.cat(
                [
                    (
                        model_inputs["wind_predictors"]
                        if "wind_predictors" in model_inputs
                        else model_inputs["wind_precomputed"]
                    ).clone()
                    for _ in range(args.batch_size)
                ],
                dim=0,
            )
            if hasattr(model, "forward_fast"):
                spatial_wind_predictors = torch.cat(
                    [
                        copied_model_inputs["spatial"],
                        copied_model_inputs[
                            "wind_predictors"
                            if "wind_predictors" in model_inputs
                            else "wind_precomputed"
                        ],
                    ],
                    dim=1,
                )

            if hasattr(model, "forward_fast"):
                mlp_spatial = torch.cat(
                    [
                        model_inputs["mlp_spatial"].clone()
                        for _ in range(args.batch_size)
                    ],
                    dim=0,
                )
            else:
                copied_model_inputs["mlp_spatial"] = torch.cat(
                    [
                        model_inputs["mlp_spatial"].clone()
                        for _ in range(args.batch_size)
                    ],
                    dim=0,
                )

            save_path = (
                args.output_dir
                / f"{area}_{month_day_night}_lower_bound_{args.lower_bound}_upper_bound_{args.upper_bound}.npy"  # pylint: disable=line-too-long
            )

            if os.path.isfile(save_path) and not args.regenerate and not args.DEBUG:
                continue

            aggregation_tensor = torch.zeros(
                data_blob["spatial"].shape[2:]
                + (abs(args.upper_bound) + abs(args.lower_bound) + 1,),
                dtype=torch.long,
                device=DEVICE,
            )

            temporal_iterator = (
                trange(0, input_temporal_t.size(0), args.batch_size, leave=False)
                if args.verbose
                else range(0, input_temporal_t.size(0), args.batch_size)
            )

            for outer in temporal_iterator:
                upper = min(outer + args.batch_size, input_temporal_t.size(0))

                with amp_autocast():
                    if hasattr(model, "forward_fast"):
                        if (
                            upper - outer < args.batch_size
                        ):  # if input_temporal_t % batch_size != 0
                            spatial_wind_predictors = spatial_wind_predictors[
                                : upper - outer
                            ]
                            mlp_spatial = mlp_spatial[: upper - outer]
                        outputs = model.forward_fast(
                            spatial_wind_predictors=spatial_wind_predictors,
                            temporal=input_temporal_t[outer:upper],
                            mlp_spatial=mlp_spatial,
                            statistics=statistics,
                        )
                    else:
                        if (
                            upper - outer < args.batch_size
                        ):  # if input_temporal_t % batch_size != 0
                            copied_model_inputs = {
                                k: (
                                    v[: upper - outer]
                                    if "spatial" in k
                                    or "wind" in k
                                    or "mlp_spatial" in k
                                    else v
                                )
                                for k, v in copied_model_inputs.items()
                            }
                        copied_model_inputs["temporal"] = input_temporal_t[outer:upper]
                        outputs = model.forward(**copied_model_inputs)
                outputs = padder.unpad(outputs).squeeze(dim=1)

                floored_outputs = (
                    torch.clamp(
                        outputs + abs(args.lower_bound),
                        min=0,
                        max=abs(args.upper_bound) + abs(args.lower_bound),
                    )
                    .floor()
                    .long()
                )

                aggregation_tensor += (
                    torch.nn.functional.one_hot(
                        floored_outputs,
                        num_classes=abs(args.upper_bound) + abs(args.lower_bound) + 1,
                    )
                    .long()
                    .sum(dim=0)
                )

            # save with as few bits as possible
            max_val = aggregation_tensor.max().item()
            if max_val <= np.iinfo(np.uint8).max:
                np.save(save_path, aggregation_tensor.cpu().numpy().astype(np.uint8))
            elif max_val <= np.iinfo(np.uint16).max:
                np.save(save_path, aggregation_tensor.cpu().numpy().astype(np.uint16))
            elif max_val <= np.iinfo(np.uint32).max:
                np.save(save_path, aggregation_tensor.cpu().numpy().astype(np.uint32))
            else:
                np.save(save_path, aggregation_tensor.cpu().numpy())

        end_time = time.process_time()
        print(f"Compute time for area {area}", end_time - start_time)
