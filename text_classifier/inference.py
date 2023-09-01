import pandas as pd

from text_classifier.models.artifacts import Artifacts
from text_classifier.common import SerializedPredictions, UNKNOWN_CATEGORY_NAME, CONFIDENCE_THRESHOLD
from typing import Iterable, Union
from text_classifier.data.output import transform_predictions2df


def run_inference(
        texts: Union[str, Iterable],
        artifacts: Artifacts,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        unknown_category_name: str = UNKNOWN_CATEGORY_NAME,
) -> SerializedPredictions:
    if isinstance(texts, str):
        texts = [texts]

    texts_preproc = artifacts.preprocessor(texts)
    labels, probas = artifacts.model.predict(texts_preproc)
    confidence = probas.max(axis=1)

    predictions = []
    for label, prob in zip(labels, confidence):
        prediction = {
            "category": label if prob > confidence_threshold else unknown_category_name,
            "probability": prob,
        }
        predictions.append(prediction)

    return predictions


def run_inference_on_test(
        dataset: pd.DataFrame,
        artifacts_path: str,
        output_path
):
    assert "text" in dataset, f"Dataset must have column `text`. Columns provided: {dataset.columns}."
    texts = dataset["text"].values
    artifacts_ = Artifacts(artifacts_path)
    predictions = run_inference(texts, artifacts_, confidence_threshold=0)
    df_predictions = transform_predictions2df(predictions)
    df_predictions.to_csv(output_path)

    return predictions


def test_inference():
    samples = "Играть в теннис кубок чемпионат мира 2023"
    artifacts_ = Artifacts("./experiment_baseline/model.pkl")

    preds = run_inference(samples, artifacts_, confidence_threshold=0)
    assert preds[0]["category"] == "tennis"

    preds = run_inference(samples, artifacts_, confidence_threshold=1)
    assert preds[0]["category"] == UNKNOWN_CATEGORY_NAME

    print(preds)
    df = transform_predictions2df(preds)


def test_batch_inference():
    dataset = pd.read_csv("../data/raw/test.csv")
    run_inference_on_test(
        dataset,
        artifacts_path="../models/baseline.pkl",
        output_path="../data/output.csv"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train text classifier.')
    parser.add_argument(
        '--text',
        type=str,
        help='Text to be classified.',
    )
    parser.add_argument(
        '--confidence_threshold',
        type=float,
        help='Threshold for a probability returned from the model, '
             'to filter texts that are out of expected distribution.',
        default=0
    )
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        help='Path to a checkpoint.',
        default="./models/baseline.pkl"
    )
    args = parser.parse_args()

    artifacts = Artifacts(args.checkpoint_path)

    prediction = run_inference(args.text, artifacts)
    print(prediction)

