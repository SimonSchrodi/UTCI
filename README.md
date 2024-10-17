# UTCI

## Setup

1. Install the conda environment, via `conda env create --name utci --file env.yml
2. Activate the environemnt, via `conda activate utci'

## Prepare data

1. Download the raw data

2. Run `python preprocess.py --raw_data data/raw/UHTC_NN --output_dir data`

3. Run `python preprocess_mean_std.py --output_dir data` and copy the printed strings into `UTCI_STATISTICS` in `utils.py`

## Train

To train the UTCI model, run
```bash
python train.py exp/01 --data_path data --skip 1 2 3 --amp --clip_grad --without_aveg
```

## Evaluate

To test the previously trained UTCI model, run
```bash
python eval.py exp/01/model.pth
```

If you want to:
* save detailed information (per sample additionally to the aggregated information), add `--detailed`.
* plot the UTCI predictions, add `--plot`.
* save the UTCI predictions as numpy files for further processing, add `--save_np`.

## Predict UTCIs for a given time period

To predict UTCIs for a given time period and get the aggregated results, run
```bash
python eval_time_period.py exp/01/model.pth --amp --verbose --temporal_data $temporal_data
```
where `$temporal_data` is a placeholder for the path to the csv file with the meteorological data of the time period.
