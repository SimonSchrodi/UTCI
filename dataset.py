"""dataset for UTCI data"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

from utils import (
    UTCI_MLP_SPATIAL_TIF_FILEPATH,
    UTCI_OUTPUT_TIF_DIRPATH,
    UTCI_SPATIAL_TIF_DIRPATH,
    UTCI_STATISTICS,
    UTCI_TEMPORAL_KEYS,
    UTCI_WIND_PREDICTORS_TIF_DIRPATH,
    load_and_combine_images,
)


class BaseDataset(Dataset):
    """Base dataset class"""

    def __init__(
        self,
        random: bool = False,
        crop: int | tuple[int, int] | bool = 256,
    ):
        self.random = random
        if crop:
            self.transform = transforms.Compose(
                [
                    (
                        transforms.RandomCrop(
                            crop, pad_if_needed=True, padding_mode="edge"
                        )
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose([])
        self.data: list = []
        self.crop = crop

    def __len__(self):
        return len(self.data)

    def add_normalizations(self, mean, std):
        """add normalization to transforms"""
        self.transform.transforms.append(transforms.Normalize(mean, std))


class UTCIDataset(BaseDataset):  # pylint: disable=too-many-instance-attributes
    """dataset for UTCI data"""

    spatial_indices_wo_aveg = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 13, 15, 16, 18, 20]

    def __init__(  # pylint: disable=too-many-arguments, too-many-locals, too-many-positional-arguments
        self,
        data_path: str | Path,
        areas: list[str],
        dates: list[str],
        random: bool = False,
        crop: int = 256,
        ignore_temporal_keys: list = None,
        return_identifier: bool = False,
        return_building_mask: bool = True,
        precompute_tmrt: bool = False,
        only_load_utci: bool = False,
        temporal_data: str = "",
        return_intermediate_gt: bool = False,
        forward_input_kwargs: list = None,
        without_aveg: bool = False,
        only_spatial: bool = False,
        requires_utci: bool = True,
        tif: bool = False,
        mlp_spatial_data: str = None,
    ):
        super().__init__(random, crop)

        self.return_identifier = return_identifier
        self.return_building_mask = return_building_mask
        self.return_intermediate_gt = return_intermediate_gt
        self.without_aveg = without_aveg
        self.only_spatial = only_spatial
        self.precompute_tmrt = precompute_tmrt
        self.only_load_utci = only_load_utci
        self.requires_utci = requires_utci
        self.tif = tif

        self.forward_input_kwargs: list = (
            forward_input_kwargs if forward_input_kwargs is not None else []
        )

        ignore_temporal_keys = (
            [] if ignore_temporal_keys is None else ignore_temporal_keys
        )
        self.temporal_keys = [
            key for key in UTCI_TEMPORAL_KEYS if key not in ignore_temporal_keys
        ]

        # spatial inputs
        if "spatial" in forward_input_kwargs:  # type: ignore[operator]
            if tif:
                self.spatial_meta_data_path = UTCI_SPATIAL_TIF_DIRPATH
            else:
                self.spatial_meta_data_path = os.path.join(
                    data_path, "input/spatial_meta_data"
                )
            assert os.path.isdir(self.spatial_meta_data_path)

        # spatial masks
        self.spatial_masks_path = os.path.join(data_path, "input/spatial_masks")
        assert os.path.isdir(self.spatial_masks_path)

        # wind inputs
        if "wind_predictors" in forward_input_kwargs:  # type: ignore[operator]
            if tif:
                self.spatial_wind_predictors_path = UTCI_WIND_PREDICTORS_TIF_DIRPATH
            else:
                self.spatial_wind_predictors_path = os.path.join(
                    data_path, "input/wind_predictors"
                )
            assert os.path.isdir(self.spatial_wind_predictors_path)

        # temporal inputs
        self.temporal_meta_data_path = os.path.join(
            data_path, temporal_data if temporal_data else "Met_Data_ERA5.csv"
        )
        assert os.path.isfile(self.temporal_meta_data_path)

        # mlp spatial predictor inputs
        if tif:
            self.mlp_spatial_predictors_path = UTCI_MLP_SPATIAL_TIF_FILEPATH
        else:
            self.mlp_spatial_predictors_path = os.path.join(
                data_path,
                (
                    f"{mlp_spatial_data}.csv"
                    if mlp_spatial_data is not None
                    else "MLP_spatial_predictors.csv"
                ),
            )
        assert os.path.isfile(self.mlp_spatial_predictors_path)

        # UTCI outputs
        if tif:
            self.output_path = UTCI_OUTPUT_TIF_DIRPATH
        else:
            self.output_path = os.path.join(data_path, "output")
        assert os.path.isdir(self.output_path)

        self.load_data(areas, dates, ignore_temporal_keys)

    @staticmethod
    def _normalize(x, key):
        if key not in UTCI_STATISTICS:
            raise NotImplementedError
        mean, std = UTCI_STATISTICS[key]
        assert std != 0
        return (x - mean) / std

    def load_data(  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        self,
        areas: list[str],
        dates: list[str],
        ignore_temporal_keys: list[str] | None = None,
    ) -> None:
        """loads the data or add the paths to the data dict"""
        if ignore_temporal_keys is None:
            ignore_temporal_keys = []

        # read in temporal inputs
        tmp_csv = pd.read_csv(self.temporal_meta_data_path).to_dict()
        self.temporal_meta_data: dict = {}
        for idx, date in tmp_csv["date_CET"].items():
            date_split = date.split(" ")
            if len(date_split) == 1:
                date = date_split[0]
                hour = "00:00:00"
            elif len(date_split) == 2:
                date, hour = date.split(" ")
            else:
                raise NotImplementedError
            identifier_key = f"{date}_{hour[:2]}"
            assert identifier_key not in self.temporal_meta_data
            self.temporal_meta_data[identifier_key] = torch.tensor(
                [tmp_csv[k][idx] for k in self.temporal_keys]
            ).float()

        # MLP spatial predictors input
        tmp_csv = pd.read_csv(self.mlp_spatial_predictors_path).to_dict()
        self.mlp_spatial_predictors: dict = {}
        for idx, area in tmp_csv["prediction_areas"].items():
            if area in areas:
                assert area not in self.mlp_spatial_predictors
                self.mlp_spatial_predictors[area] = torch.from_numpy(
                    np.array(
                        [
                            tmp_csv[k][idx]
                            for k in sorted(tmp_csv.keys())
                            # if k != "prediction_areas" and k != "id"
                            if k not in ("prediction_areas", "id")
                        ]
                    )
                ).float()

        # get paths for spatial, wind and UTCI and put all in a data dict
        for area in tqdm(areas, leave=False):
            if self.tif:
                spatial_meta_data = os.path.join(self.spatial_meta_data_path, str(area))
                assert os.path.isdir(
                    spatial_meta_data
                ), f"Spatial information for area {area} is missing"
            else:
                filename = f"{area}.npy"
                spatial_meta_data = os.path.join(
                    self.spatial_meta_data_path, filename
                )  # pylint: disable=no-member
                assert os.path.isfile(
                    spatial_meta_data
                ), f"Spatial information for area {area} is missing"

            if self.return_building_mask:
                if self.tif:
                    mask_data = None
                else:
                    mask_data = os.path.join(self.spatial_masks_path, f"{area}.npy")
                    assert os.path.isfile(
                        mask_data
                    ), f"Building mask for area {area} is missing"
            else:
                mask_data = None

            if "wind_predictors" in self.forward_input_kwargs:
                if self.tif:
                    wind_predictors = os.path.join(
                        self.spatial_wind_predictors_path, str(area)
                    )
                    assert (
                        os.path.isdir(wind_predictors)
                        or "wind_predictors" not in self.forward_input_kwargs
                    ), f"Wind data for area {area} is missing"
                else:
                    filename = f"{area}.npy"
                    wind_predictors = os.path.join(
                        self.spatial_wind_predictors_path, filename
                    )
                    assert (
                        os.path.isfile(wind_predictors)
                        or "wind_predictors" not in self.forward_input_kwargs
                    ), f"Wind data for area {area} is missing"
            else:
                wind_predictors = None

            for date in tqdm(dates, leave=False):
                # check if date in area exists
                if self.requires_utci and not os.path.isfile(
                    os.path.join(
                        self.output_path,
                        area,
                        date,
                        f"UTCI_00.{'tif' if self.tif else 'npy'}",
                    )
                ):
                    continue
                for hour in range(24):  # type: ignore
                    utci_filepath = os.path.join(
                        self.output_path,
                        area,
                        date,
                        f"UTCI_{str(hour).zfill(2)}.{'tif' if self.tif else 'npy'}",
                    )
                    # assert not self.requires_utci or os.path.isfile(
                    #     utci_filepath
                    # ), f"UTCI data for area {area}, date {date}, and hour {hour} is missing"
                    if self.requires_utci and not os.path.isfile(utci_filepath):
                        print(
                            f"UTCI data for area {area}, date {date}, and hour {hour} is missing"
                        )
                        continue

                    identifier = f"{area}_{date}_{str(hour).zfill(2)}"
                    self.data.append(
                        {
                            "utci": utci_filepath,
                            "spatial_meta_data": spatial_meta_data,
                            "spatial_mask": mask_data,
                            "wind_predictors": wind_predictors,
                            "temporal_meta_data": self.temporal_meta_data[
                                f"{date}_{str(hour).zfill(2)}"
                            ],
                            "area": area,
                            "date": date,
                            "hour": str(hour).zfill(2),
                            "identifier": identifier,
                        }
                    )

                    if self.only_spatial:
                        break
                if self.only_spatial:
                    break

    def __getitem__(
        self, idx: int | torch.tensor
    ):  # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore[union-attr]

        if self.only_load_utci:
            utci = torch.from_numpy(np.load(self.data[idx]["utci"]))
            return utci

        ret_val = {}

        # add temporal inputs to ret_val
        if "temporal" in self.forward_input_kwargs:
            temporal_meta = self.data[idx]["temporal_meta_data"]
            temporal_meta[torch.isnan(temporal_meta)] = 0
            ret_val["temporal"] = temporal_meta

        # add mlp spatial predictor inputs to ret_val
        if "mlp_spatial" in self.forward_input_kwargs:
            mlp_spatial_predictors = self.mlp_spatial_predictors[self.data[idx]["area"]]
            mlp_spatial_predictors[torch.isnan(mlp_spatial_predictors)] = 0
            ret_val["mlp_spatial"] = mlp_spatial_predictors

        # load spatial data from disk
        if "spatial" in self.forward_input_kwargs:
            if self.tif:
                spatial_meta_tifs = glob.glob(
                    self.data[idx]["spatial_meta_data"] + "/*.tif"
                )
                spatial_meta_tifs.extend(
                    glob.glob(
                        os.path.join(self.data[idx]["spatial_meta_data"], "svfs")
                        + "/*.tif"
                    )
                )
                assert len(spatial_meta_tifs) == 21
                np_arr = load_and_combine_images(
                    sorted(spatial_meta_tifs), normalize=False
                )
                spatial_meta = torch.from_numpy(np_arr)
            else:
                spatial_meta = torch.from_numpy(
                    np.load(self.data[idx]["spatial_meta_data"])
                )
            if self.without_aveg:
                spatial_meta = spatial_meta[self.spatial_indices_wo_aveg]
            spatial_dim = spatial_meta.size(0)
        else:
            spatial_dim = 0

        if "wind_predictors" in self.forward_input_kwargs:
            if self.tif:
                wind_predictor_tifs = glob.glob(
                    self.data[idx]["wind_predictors"] + "/*.tif"
                )
                assert len(wind_predictor_tifs) == 18
                np_arr = load_and_combine_images(
                    sorted(wind_predictor_tifs), normalize=False
                )
                wind_predictors = torch.from_numpy(np_arr)
            else:
                wind_predictors = torch.from_numpy(
                    np.load(self.data[idx]["wind_predictors"])
                )
            wind_predictors_dim = wind_predictors.size(0)
        else:
            wind_predictors_dim = 0

        if self.requires_utci:
            if self.tif:
                utci = torch.from_numpy(
                    np.expand_dims(np.array(Image.open(self.data[idx]["utci"])), axis=0)
                )
            else:
                utci = torch.from_numpy(
                    np.expand_dims(np.load(self.data[idx]["utci"]), axis=0)
                )
        else:
            utci = torch.zeros((1,) + spatial_meta.shape[1:])

        if self.return_building_mask and (
            self.data[idx]["spatial_mask"] is not None or self.tif
        ):
            if self.tif:
                mask = spatial_meta[3] == 2  # 3 = lcc, 2 = building cls
                spatial_mask_np = (1 - mask.int()).numpy()
            else:
                spatial_mask_np = 1 - np.load(
                    self.data[idx]["spatial_mask"]
                )  # s.t. 1 is no building, 0 is building
            spatial_mask = torch.from_numpy(spatial_mask_np)[None, ...]
        else:
            spatial_mask = torch.ones_like(utci)

        # transform spatial data
        if self.random or self.crop:
            inputs = [utci]
            if "spatial" in self.forward_input_kwargs:
                inputs.append(spatial_meta)
            if "wind_predictors" in self.forward_input_kwargs:
                inputs.append(wind_predictors)
            if self.return_building_mask:
                inputs.append(spatial_mask)
            combined = torch.cat(inputs, dim=0).float()

        if self.random:
            # pylint: disable=possibly-used-before-assignment
            combined_cropped = self.transform(combined).unsqueeze(0)
            # pylint: enable=possibly-used-before-assignment
        else:
            if self.crop:
                combined_cropped = torch.nn.functional.interpolate(
                    combined.unsqueeze(0), self.crop
                )
            else:
                # combined_cropped = combined.unsqueeze(0)
                pass

        if self.random or self.crop:
            utci = combined_cropped[  # pylint: disable=possibly-used-before-assignment
                :, 0
            ]
            if "spatial" in self.forward_input_kwargs:
                spatial_meta = combined_cropped[:, 1 : spatial_dim + 1]
            if "wind_predictors" in self.forward_input_kwargs:
                wind_predictors = combined_cropped[
                    :, 1 + spatial_dim : 1 + spatial_dim + wind_predictors_dim
                ]
            if self.return_building_mask:
                spatial_mask = combined_cropped[
                    :, 1 + spatial_dim + wind_predictors_dim
                ]
                spatial_mask = spatial_mask.type(torch.int32)
        else:  # test
            if utci.shape[1] == 400:
                spatial_meta = spatial_meta[:, :400, :]
                spatial_mask = spatial_mask[:, :400, :]
            elif utci.shape[2] == 400:
                raise NotImplementedError

        utci[torch.isnan(utci)] = 0
        ret_val["utci"] = utci
        if "spatial" in self.forward_input_kwargs:
            spatial_meta[torch.isnan(spatial_meta)] = 0
            spatial_meta = spatial_meta.squeeze()
            ret_val["spatial"] = spatial_meta
        if "wind_predictors" in self.forward_input_kwargs:
            wind_predictors[torch.isnan(wind_predictors)] = 0
            wind_predictors = wind_predictors.squeeze()
            ret_val["wind_predictors"] = wind_predictors

        if self.return_building_mask:
            ret_val["mask"] = spatial_mask
        if self.return_identifier:
            ret_val["identifier"] = self.data[idx]["identifier"]

        return ret_val
