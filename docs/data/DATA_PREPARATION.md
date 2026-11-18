# Data preparation

Benchmarks are supplied separately. The exact input files and fingerprints are
recorded in `configs/data_sources.yaml` and `reference/data/raw_manifest.json`;
`reference/data/processed_manifest.json` records all 116 processed entities.
The locations below are the recorded acquisition sources. Downloads were not
attempted while packaging, and source access terms still apply.

## Required layout

```text
data/raw/
  telemanom/labeled_anomalies.csv
  telemanom/train/*.npy
  telemanom/test/*.npy
  smd/train/machine-*.txt
  smd/test/machine-*.txt
  smd/test_label/machine-*.txt
  psm/train.csv
  psm/test.csv
  psm/test_label.csv
  tab/SWAT.csv
  tab/PUMP.csv
  tsb-ad/001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
  tsb-ad/018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv
  tsb-ad/173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv
```

## Acquisition

The commands use `curl` and `unzip`, without creating any Git repository.
Review the benchmark distributions' terms before downloading or redistributing.

```bash
mkdir -p downloads data/raw/{telemanom,smd,psm,tab,tsb-ad}

# MSL / SMAP: Telemanom metadata and its NASA telemetry distribution.
curl -L --fail \
  https://raw.githubusercontent.com/khundman/telemanom/2e6c5b6c3558e7835601519b7bdef37c649bdbdc/labeled_anomalies.csv \
  -o data/raw/telemanom/labeled_anomalies.csv
curl -L --fail \
  https://www.kaggle.com/api/v1/datasets/download/patrickfleith/nasa-anomaly-detection-dataset-smap-msl \
  -o downloads/telemetry.zip
unzip downloads/telemetry.zip -d downloads/telemetry
cp -R downloads/telemetry/data/data/train data/raw/telemanom/
cp -R downloads/telemetry/data/data/test data/raw/telemanom/

# SMD: the recorded OmniAnomaly data snapshot.
curl -L --fail \
  https://github.com/NetManAIOps/OmniAnomaly/archive/7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4.zip \
  -o downloads/smd.zip
unzip downloads/smd.zip -d downloads/smd
cp -R downloads/smd/OmniAnomaly-7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4/ServerMachineDataset/{train,test,test_label} data/raw/smd/

# PSM: the recorded RANSynCoders data snapshot.
curl -L --fail \
  https://github.com/eBay/RANSynCoders/archive/948e61eb579815ff2a6bce721332d8220d59ca47.zip \
  -o downloads/psm.zip
unzip downloads/psm.zip -d downloads/psm
cp downloads/psm/RANSynCoders-948e61eb579815ff2a6bce721332d8220d59ca47/data/{train,test,test_label}.csv data/raw/psm/

# TSB-AD-M: use the multivariate archive, not TSB-UAD or Public-v2.
curl -L --fail https://www.thedatum.org/datasets/TSB-AD-M.zip -o downloads/tsb-ad.zip
unzip downloads/tsb-ad.zip -d downloads/tsb-ad
cp downloads/tsb-ad/TSB-AD-M/001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv data/raw/tsb-ad/
cp downloads/tsb-ad/TSB-AD-M/018_Daphnet_id_1_HumanActivity_tr_9693_1st_20732.csv data/raw/tsb-ad/
cp downloads/tsb-ad/TSB-AD-M/173_GECCO_id_1_Sensor_tr_16165_1st_16265.csv data/raw/tsb-ad/
```

If Kaggle requires authentication, acquire the same archive manually from its
[distribution page](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl).
Metadata provenance is [Telemanom](https://github.com/khundman/telemanom).

For **SWAT/PUMP**, obtain `TAB_dataset.zip` from the dataset link in the
[TAB project](https://github.com/decisionintelligence/TAB). The recorded
[hosted archive](https://drive.google.com/file/d/1V5BAHWBKU8uih3hE1R7WdF6_crZlIbQT/view)
contains `dataset/anomaly_detect/data/SWAT.csv` and `PUMP.csv`.
Place those files in `data/raw/tab/`. The experiments use these exact
long-format files; substituting a separately downloaded SWaT spreadsheet
changes the data protocol. TAB hosting does not grant additional rights to
the underlying datasets. Follow the original access/licensing requirements.

## Validate and preprocess

```bash
python scripts/reproduce.py validate-data --raw-root data/raw
python scripts/reproduce.py prepare-data --raw-root data/raw --processed data/processed
```

Validation requires **257 actual input files** and the recorded bytes. The
historical raw manifest lists one extra interrupted-copy temporary file;
`excluded_legacy_snapshot_non_inputs` explicitly excludes it. Do not manufacture
that file. Missing, additional or changed inputs fail with their relative paths.

Preprocessing validates the raw inputs, writes a new processed directory, then
checks all entity identities, lengths, labels and value-level fingerprints.
An existing processed directory is accepted only after the same verification.
No source datasets are modified. Keep original benchmark splits and filenames.
The Telemanom anomaly intervals retain the recorded half-open interpretation;
source-only normalization, chronological source split, and constant-column
handling are detailed in the [protocol](../protocols/PROTOCOL.md).
