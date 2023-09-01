## Sport news classification

This repository contains code for training a text classification model and model weights for the task of sport news classification with 13 categories (russian language): `['winter_sport', 'extreme', 'football', 'boardgames', 'hockey', 'esport', 'athletics', 'motosport', 'basketball', 'tennis', 'autosport', 'martial_arts', 'volleyball']`.

### Installation

```bash
git clone
cd simple_text_classification
```

### New model training

```bash
conda env create -f env_dev.yml
conda activate text_clf
pip install .
```
#### Train

Pre-requirement: dataset in csv format with columns `oid`, `text`.

```bash
python text_classifier/train.py --dataset_path "./data/interim/train_no_dup.csv" --model_filename "baseline.pkl"
```

#### Sample inference
```bash
python text_classifier/inference.py --text "Теннис спорт" --confidence_threshold 0.5 --checkpoint_path "./models/baseline.pkl"
# [{'category': 'tennis', 'probability': 0.5016982131569547}]
```

### Service for sport news category prediction
#### To get predictions for a batch of texts

1. Build docker image and start a service:
   - ```bash
     make build
     make run-service
     ```
2. Prepare testset (testset is in csv format with required column `text`) 
3. Specify path to the testset in `PATH_TO_TESTSET` and run:
   - ```bash
     make get-test-predictions PATH_TO_TESTSET=data/raw/test.csv
     ```
