"""training script for UTCI model"""

import argparse
import json
import os
import random
from contextlib import suppress

import numpy as np
import pandas as pd
import torch
from path import Path
from tensorboardX import SummaryWriter
from timm.models import model_parameters
from timm.utils import NativeScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from dataset import UTCIDataset
from model import ConvEncoderDecoder, UTCIModel
from utils import (
    UTCI_STATISTICS,
    UTCI_TEMPORAL_KEYS,
    InputPadder,
    MaskedLoss,
    get_device,
    set_seed,
)


def parse_arguments():
    """python arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_path")
    parser.add_argument("--data_path", default="data", type=str)
    parser.add_argument("--temporal_data", default="Met_Data_ERA5.csv", type=str)
    parser.add_argument("--dimension", default=32, type=int)
    parser.add_argument("--n_epochs", default=10, type=int)
    parser.add_argument("--skip", nargs="*", type=int)
    parser.add_argument("--ignore_temporal", nargs="*", type=str)
    parser.add_argument("--restore_ckpt", action="store_true")
    parser.add_argument("--tmrt_ckpt", type=str)
    parser.add_argument("--check_tmrt", action="store_true")
    parser.add_argument("--subset_size", default=1, type=float)
    parser.add_argument("--apply_mask", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--clip_grad", action="store_true")
    parser.add_argument("--loss_fn", type=str, default="l1", choices=["l1", "l2"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--without_aveg", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--DEBUG", action="store_true")
    return parser.parse_args()


args = parse_arguments()
assert 0 < args.subset_size <= 1

SPATIAL_DIM = 21
if args.without_aveg:
    SPATIAL_DIM -= 5
WIND_PREDICTORS_DIM = 18

args.input_channels = SPATIAL_DIM + WIND_PREDICTORS_DIM  # spatial (R^21), wind (R^18)
args.output_channels = 1
args.global_channels = (
    len(UTCI_TEMPORAL_KEYS) + 17
)  # temporal (R^8), mlp_spatial (R^17)

if args.ignore_temporal is not None:
    if not all(temp in UTCI_TEMPORAL_KEYS for temp in args.ignore_temporal):
        missing_keys = [
            temp for temp in args.ignore_temporal if temp in UTCI_TEMPORAL_KEYS
        ]
        raise ValueError(
            f"Keys {missing_keys} not in temporal keys {UTCI_TEMPORAL_KEYS}"
        )
    args.global_channels -= len(args.ignore_temporal)
args.learning_rate = 0.001
args.gamma = 0.9999
args.data_parallel = torch.cuda.device_count() > 1 if not args.DEBUG else False
args.exp_path = Path(args.exp_path)
args.exp_path.makedirs_p()

set_seed(args.seed)

utci_statistics = UTCI_STATISTICS["utci"]

# set up model
DEVICE = get_device()
utci_model = ConvEncoderDecoder(args)
model = UTCIModel(utci_model=utci_model, channels_last=args.channels_last)
model.to(DEVICE)
model_input_kwargs = model.forward_input_kwargs

# load test data
test_areas = list(
    pd.read_csv(os.path.join(args.data_path, "Test_Areas.csv")).iloc[:, 1]
)
if args.DEBUG:
    test_areas = test_areas[:2]
test_dates = list(pd.read_csv(os.path.join(args.data_path, "Date_Test.csv")).iloc[:, 0])
if args.DEBUG:
    test_dates = test_dates[:2]
test_data = UTCIDataset(
    args.data_path,
    areas=test_areas,
    dates=test_dates,
    ignore_temporal_keys=args.ignore_temporal,
    return_building_mask=args.apply_mask,
    crop=False,
    temporal_data=args.temporal_data,
    forward_input_kwargs=model_input_kwargs,
    without_aveg=args.without_aveg,
)
assert len(test_data) == len(test_areas) * len(test_dates) * 24
test_loader = DataLoader(
    test_data,
    batch_size=24,
    num_workers=1 if args.DEBUG else 20,
    pin_memory=False,
    shuffle=False,
)

# load training data
train_areas = list(
    pd.read_csv(os.path.join(args.data_path, "Train_Areas.csv")).iloc[:, 1]
)
if args.DEBUG:
    train_areas = train_areas[:2]
assert set(train_areas).intersection(set(test_areas)) == set()

train_dates = list(
    pd.read_csv(os.path.join(args.data_path, "Date_Train.csv")).iloc[:, 0]
)
if args.DEBUG:
    train_dates = train_dates[:2]
assert set(train_dates).intersection(set(test_dates)) == set()

train_data = UTCIDataset(
    args.data_path,
    areas=train_areas,
    dates=train_dates,
    random=True,
    ignore_temporal_keys=args.ignore_temporal,
    return_building_mask=args.apply_mask,
    temporal_data=args.temporal_data,
    forward_input_kwargs=model_input_kwargs,
    without_aveg=args.without_aveg,
)
assert len(train_data) == len(train_areas) * len(train_dates) * 24
if args.subset_size != 1:
    subset = random.sample(
        range(len(train_data)), round(len(train_data) * args.subset_size)
    )
    train_data = torch.utils.data.Subset(train_data, subset)
train_loader = DataLoader(
    train_data,
    batch_size=2 if args.DEBUG else 32,
    num_workers=1 if args.DEBUG else 20,
    shuffle=True,
    pin_memory=False,
)

REDUCTION = "none" if args.apply_mask else "mean"
if args.loss_fn == "l1":
    base_loss = nn.L1Loss(reduction=REDUCTION)
elif args.loss_fn == "l2":
    base_loss = nn.MSELoss(reduction=REDUCTION)
else:
    raise NotImplementedError
criterion = MaskedLoss(base_loss)
eval_criterion = MaskedLoss(nn.L1Loss(reduction=REDUCTION))

optimizer = torch.optim.AdamW(
    model.parameters(), lr=args.learning_rate, weight_decay=1e-3
)
lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.n_epochs * len(train_loader)
)

amp_autocast = suppress  # pylint: disable=invalid-name
if args.amp:
    amp_autocast = torch.cuda.amp.autocast  # type: ignore[misc]
loss_scaler = NativeScaler() if args.amp else None

START_EPOCH = 1
curr_iter = 0  # pylint: disable=invalid-name
if args.restore_ckpt:
    if not os.path.isfile(args.exp_path / "checkpoint.pth"):
        print("WARNING: Cannot find checkpoint file -> train from scratch")
    else:
        checkpoint = torch.load(args.exp_path / "checkpoint.pth", map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if args.amp:
            loss_scaler.load_state_dict(checkpoint["scaler_state_dict"])
        START_EPOCH = checkpoint["epoch"] + 1
        curr_iter = checkpoint["curr_iter"]

with open(args.exp_path / "args.json", "w", encoding="utf-8") as f:
    json.dump(args.__dict__, f, indent=2)
log_writer = SummaryWriter(log_dir=args.exp_path)

if not args.DEBUG and args.data_parallel:
    model = torch.nn.DataParallel(model)

for epoch in range(START_EPOCH, args.n_epochs + 1):
    train_loss = 0.0  # pylint: disable=invalid-name
    model.train()
    for data_blob in train_loader:
        optimizer.zero_grad(set_to_none=True)

        model_inputs = {"statistics": utci_statistics}
        model_inputs.update(
            {
                k: data_blob[k].to(DEVICE)
                for k in model_input_kwargs
                if k != "statistics"
            }
        )
        utci = data_blob["utci"].to(DEVICE)
        if args.apply_mask:
            building_mask = data_blob["mask"].to(DEVICE)
        else:
            building_mask = None  # pylint: disable=invalid-name

        with amp_autocast():
            utci_pred = model(**model_inputs)
            loss = criterion(utci_pred, utci, mask=building_mask)
        if loss_scaler is not None:
            loss_scaler(
                loss=loss,
                optimizer=optimizer,
                clip_grad=1.0 if args.clip_grad else None,
                parameters=model_parameters(model),
            )
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        cur_loss = loss.item()
        train_loss += cur_loss
        log_writer.add_scalar("loss", cur_loss, curr_iter)
        log_writer.add_scalar("learning_rate", lr_scheduler.get_last_lr()[0], curr_iter)
        curr_iter += 1

    train_loss = train_loss / len(train_loader)
    print(f"Epoch {epoch}, train loss: {train_loss:.2f}")

    if args.data_parallel:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict(),
            "scaler_state_dict": loss_scaler.state_dict() if args.amp else None,
            "epoch": epoch,
            "curr_iter": curr_iter,
        },
        args.exp_path / "checkpoint.pth",
        _use_new_zipfile_serialization=False,
    )

test_error = []
model.eval()
with torch.no_grad():
    for data_blob in test_loader:
        utci = data_blob["utci"].to(DEVICE)
        padder = InputPadder(utci.shape)
        model_inputs = {"statistics": utci_statistics}
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
        if args.apply_mask:
            building_mask = data_blob["mask"].to(DEVICE)
        else:
            building_mask = None  # pylint: disable=invalid-name

        output = model(**model_inputs)
        output = padder.unpad(output)
        error = eval_criterion(output, utci, mask=building_mask)
        test_error.append(error.item())

print(f"Epoch {epoch}, val error: {np.mean(test_error):.2f}")
log_writer.add_scalar("val", np.mean(test_error), epoch)

if args.data_parallel:
    model_state_dict = model.module.state_dict()
else:
    model_state_dict = model.state_dict()
torch.save(model_state_dict, args.exp_path / "model.pth")
