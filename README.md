# Plant Disease MLOps

This repository currently implements the **data ingestion + preprocessing** stage for a plant disease classification pipeline.

It supports:

- Downloading PlantVillage image data (pinned Hugging Face revisions)
- Saving raw data locally
- Preprocessing into `train/validation/test` class folders
- Syncing processed data to S3 (including resume-friendly upload-only mode)

## Project Structure

```text
plant-disease-mlops/
├── data/
│   ├── raw/                      # Local raw datasets (save_to_disk output)
│   ├── processed/                # Local processed JPEGs by split/class
│   └── scripts/
│       ├── download_data.py      # Download raw dataset + optional metadata check
│       └── preprocess_data.py    # Preprocess and optionally sync to S3
├── src/
│   └── plant_disease_mlops/
│       └── data/
│           ├── constants.py      # Pinned HF revisions and split file names
│           ├── hf_plantvillage.py# HF download/extract/build DatasetDict logic
│           ├── preprocess.py     # Preprocessing + aws s3 sync helper
│           ├── transforms.py     # Image transforms / resize utilities
│           ├── validation.py     # PIL + Great Expectations data checks
│           └── dataset.py        # PyTorch dataset over processed class folders
├── tests/
│   └── test_data_validation.py
├── requirements.txt
├── setup.py
└── pytest.ini
```

## Local Setup

From `plant-disease-mlops/`:

```bash
python -m pip install -e .
```

Optional (if not already installed in your environment):

```bash
python -m pip install -r requirements.txt
```

## Run: Data Ingestion

Download and build local raw dataset:

```bash
python data/scripts/download_data.py
```

Default output:

- `data/raw/plantvillage/`

Useful options:

```bash
python data/scripts/download_data.py --help
```

- `--force` to rebuild
- `--max-train-samples` / `--max-test-samples` for smaller debug runs

## Run: Preprocessing

Preprocess local raw data into class folders:

```bash
python data/scripts/preprocess_data.py
```

Default behavior:

- Reads: `data/raw/plantvillage/`
- Writes: `data/processed/{train,validation,test}/class_<label>/...`
- Validation split fraction defaults to `0.1`

### Preprocess only (skip S3)

```bash
python data/scripts/preprocess_data.py --no-s3
```

### Preprocess + sync to S3

Set environment variables first:

```bash
export PLANT_DISEASE_S3_BUCKET=your-bucket-name
export PLANT_DISEASE_S3_PREFIX=processed   # optional (default: processed)
export AWS_PROFILE=plant-mlops             # optional if you use profiles
```

Then run:

```bash
python data/scripts/preprocess_data.py
```

## Upload-Only / Resume-Friendly Sync

If local processed data already exists and you want to only continue/redo S3 sync:

```bash
python data/scripts/preprocess_data.py --upload-only
```

This uses:

```bash
aws s3 sync <local_dir>/ s3://<bucket>/<prefix>/
```

So unchanged files are skipped, which is faster for retries.

## Testing

Run tests:

```bash
python -m pytest tests/ -q
```
