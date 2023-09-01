from text_classifier.data.make_dataset import get_cleaned_dataset, get_splits
from text_classifier.common.evaluation import run_evaluation_report
from text_classifier.models.tf_idf_model import TfIdfModel

import pandas as pd


def run_training(dataset_path, output_dir: str, model_filename: str = "model.pkl"):
    print("Start training.")

    assert dataset_path.endswith(".csv"), "File containing dataset must be in csv format."
    dataset = pd.read_csv(dataset_path, index_col=0)
    assert "text" in dataset, f"Dataset must have column `text`. Columns provided: {dataset.columns}."

    dataset_clean = get_cleaned_dataset(dataset)
    print("Cleaning dataset: done.")

    df_train, y_train, df_test, y_test = get_splits(dataset_clean)

    model = TfIdfModel()
    model.fit(df_train, y_train)
    print("Training: done.")

    model.save_checkpoint(output_dir, model_filename)
    run_evaluation_report(model, df_test, y_test, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train text classifier.')
    parser.add_argument(
        '--dataset_path',
        type=str,
        help='Path to csv dataset with columns: oid, text.',
        default="./data/interim/train_no_dup.csv"
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Folder where artifacts will be saved during training.',
        default="./models"
    )
    parser.add_argument(
        '--model_filename',
        type=str,
        help='Model filename to be saved.',
        default="baseline.pkl"
    )
    args = parser.parse_args()

    run_training(
        args.dataset_path,
        output_dir=args.output_dir,
        model_filename=args.model_filename
    )
