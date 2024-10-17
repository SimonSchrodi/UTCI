"""Evaluate the UTCI model on the test set."""

from __future__ import annotations

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skill_metrics as sm
import torch
from path import Path
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import UTCIDataset
from model import ConvEncoderDecoder, UTCIModel
from utils import UTCI_STATISTICS, InputPadder, MaskedLoss, get_device, plot

parser = argparse.ArgumentParser()
parser.add_argument("model_path")
parser.add_argument("--data_path", type=str, default="data")
parser.add_argument("--plot", action="store_true")
parser.add_argument("--save_np", action="store_true")
parser.add_argument("--detailed", action="store_true")
parser.add_argument(
    "--pad_mode",
    default="replicate",
    choices=["constant", "reflect", "replicate", "circular"],
    type=str,
)
parser.add_argument("--apply_mask", action="store_true")
parser.add_argument("--DEBUG", action="store_true")
args = parser.parse_args()

if args.plot or args.detailed:
    args.output_dir = Path(os.path.dirname(args.model_path)) / (
        "outputs_mask" if args.apply_mask else "outputs"
    )
    args.output_dir.makedirs_p()
if args.save_np:
    args.np_out_dir = Path(os.path.dirname(args.model_path)) / (
        "nps_mask" if args.apply_mask else "nps"
    )
    args.np_out_dir.makedirs_p()

ARGS_PATH = f"{os.path.dirname(args.model_path)}/args.json"
with open(ARGS_PATH, encoding="utf-8") as json_data:
    model_args = argparse.Namespace()
    model_args.__dict__.update(json.load(json_data))
    model_args = parser.parse_args(namespace=model_args)

DEVICE = get_device()
utci_model = ConvEncoderDecoder(model_args)
model = UTCIModel(utci_model=utci_model)

checkpoint = torch.load(args.model_path, map_location="cpu")
if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]
model.load_state_dict(checkpoint, strict=True)

model.eval()
model.to(DEVICE)
model_input_kwargs = model.forward_input_kwargs

test_areas = list(
    pd.read_csv(os.path.join(args.data_path, "Test_Areas.csv")).iloc[:, 1]
)
test_dates = list(pd.read_csv(os.path.join(args.data_path, "Date_Test.csv")).iloc[:, 0])
test_data = UTCIDataset(
    args.data_path,
    areas=test_areas,
    dates=test_dates,
    ignore_temporal_keys=model_args.ignore_temporal,
    return_building_mask=args.apply_mask or model_args.apply_mask,
    temporal_data=model_args.temporal_data,
    crop=False,
    return_identifier=True,
    forward_input_kwargs=model_input_kwargs,
    without_aveg=model_args.without_aveg if "without_aveg" in model_args else False,
)
test_loader = DataLoader(
    test_data,
    batch_size=24,
    num_workers=1 if args.DEBUG else 20,
    shuffle=False,
)

criterion = MaskedLoss(
    nn.L1Loss(reduction="none")
    if model_args.apply_mask or args.apply_mask
    else nn.L1Loss()
)

test_error: dict = {
    "L1": [],
    "RMSD": [],
    "CRMSD": [],
    "t_SDEV": [],
    "o_SDEV": [],
    "correlation": [],
}
times = []
errors_per_dt: dict = {
    "L1": {dt: [] for dt in range(24)},
    "RMSD": {dt: [] for dt in range(24)},
    "CRMSD": {dt: [] for dt in range(24)},
    "t_SDEV": {dt: [] for dt in range(24)},
    "o_SDEV": {dt: [] for dt in range(24)},
    "correlation": {dt: [] for dt in range(24)},
}
start_all = time.time()
with torch.no_grad():
    pbar = tqdm(test_loader, leave=False, total=len(test_loader))
    for index, data_blob in enumerate(pbar):
        start = time.process_time()

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
        if args.apply_mask:
            building_mask = data_blob["mask"]
        else:
            building_mask = None  # pylint: disable=invalid-name
        identifier = data_blob["identifier"]

        output = model(**model_inputs)
        output = padder.unpad(output)
        times.append(time.process_time() - start)

        output = output.cpu().detach()

        if args.plot:
            for j in range(output.shape[0]):
                plot(
                    output[j].squeeze().numpy(),
                    utci[j].squeeze().numpy(),
                    args.output_dir / f"{identifier[j]}.jpg",
                    mask=building_mask[j].squeeze(0).numpy().astype(np.bool_)
                    if building_mask is not None
                    else None,
                )
        if args.detailed:
            for j in range(output.shape[0]):
                identifier_j = identifier[j]
                dt = int(identifier_j.split("_")[-1])
                if dt not in errors_per_dt["L1"].keys():
                    continue

                if model_args.apply_mask or args.apply_mask:
                    errors_per_dt["L1"][dt].append(
                        float(criterion(output[j], utci[j], building_mask[j]).item())
                    )
                else:
                    errors_per_dt["L1"][dt].append(
                        float(criterion(output[j], utci[j]).item())
                    )
                output_np = output[j].numpy().flatten()
                utci_np = utci[j].numpy().flatten()
                errors_per_dt["RMSD"][dt].append(float(sm.rmsd(output_np, utci_np)))
                errors_per_dt["CRMSD"][dt].append(
                    float(sm.centered_rms_dev(output_np, utci_np))
                )
                errors_per_dt["t_SDEV"][dt].append(float(np.std(utci_np)))
                errors_per_dt["o_SDEV"][dt].append(float(np.std(output_np)))
                errors_per_dt["correlation"][dt].append(
                    float(np.corrcoef(output_np, utci_np)[0, 1])
                )
        if args.save_np:
            for j in range(output.size(0)):
                np.save(
                    args.np_out_dir / f"{identifier[j]}.npy",
                    output[j].squeeze().numpy(),
                )

        output_np = output.numpy().flatten()
        utci_np = utci.numpy().flatten()
        if model_args.apply_mask or args.apply_mask:
            test_error["L1"].append(
                float(criterion(output, utci, building_mask).item())
            )
            building_mask_np = building_mask.numpy().flatten().astype(bool)
            output_np = output_np[building_mask_np]
            utci_np = utci_np[building_mask_np]
        else:
            test_error["L1"].append(float(criterion(output, utci).item()))
            output_np = output.numpy().flatten()
            utci_np = utci.numpy().flatten()
        test_error["RMSD"].append(float(sm.rmsd(output_np, utci_np)))
        test_error["CRMSD"].append(float(sm.centered_rms_dev(output_np, utci_np)))
        test_error["t_SDEV"].append(float(np.std(utci_np)))
        test_error["o_SDEV"].append(float(np.std(output_np)))
        test_error["correlation"].append(float(np.corrcoef(output_np, utci_np)[0, 1]))
        pbar.set_description(
            f"L1: {np.mean(test_error['L1']):.2f},"
            f"RMSD: {np.mean(test_error['RMSD']):.2f},"
            f"CRMSD: {np.mean(test_error['CRMSD']):.2f},"
            f"t_SDEV: {np.mean(test_error['t_SDEV']):.2f},"
            f"o_SDEV: {np.mean(test_error['o_SDEV']):.2f},"
            f"corr: {np.mean(test_error['correlation']):.2f}"
        )

full_time = time.time() - start_all
print("Entire time", full_time, len(test_data), full_time / len(test_data))

print(f"Time: {np.mean(times):.2f}")

data = {}
data["test_error_mean"] = {k: float(np.mean(v)) for k, v in test_error.items()}
if args.detailed:
    data["errors_per_dt_mean"] = {
        k: {k_: float(np.mean(v_)) for k_, v_ in v.items()}  # type: ignore[misc]
        for k, v in errors_per_dt.items()
    }
data["test_error"] = test_error
if args.detailed:
    data["errors_per_dt"] = errors_per_dt
with open(
    Path(os.path.dirname(args.model_path))
    / ("eval_mask.json" if args.apply_mask or model_args.apply_mask else "eval.json"),
    "w",
    encoding="utf-8",
) as f:
    json.dump(data, f, indent=4)

if args.detailed:
    os.makedirs(Path(os.path.dirname(args.model_path)) / "detailed", exist_ok=True)
    for k, v in errors_per_dt.items():
        dt_str = list(v.keys())
        plt.bar(dt_str, [np.mean(x) for x in v.values()], width=0.2)
        ax = plt.gca()
        _ = [
            l.set_visible(False)
            for (i, l) in enumerate(ax.xaxis.get_ticklabels())
            if i % 4 != 0
        ]
        plt.xticks(fontsize=7)
        plt.title(k)
        plt.savefig(Path(os.path.dirname(args.model_path)) / "detailed" / f"{k}.jpg")
        plt.clf()
