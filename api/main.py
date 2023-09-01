from fastapi import FastAPI
from text_classifier.inference import run_inference
from text_classifier.models.artifacts import Artifacts
from data_format import str2df, list2df, df2str

from starlette.requests import Request
from starlette.responses import Response
import os


CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "../models/baseline.pkl")
app = FastAPI()
artifacts = Artifacts(CHECKPOINT_PATH)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.get("/predict")
def get_prediction(text: str):
    prediction = run_inference(text, artifacts, confidence_threshold=0)
    print(prediction)
    return prediction


@app.post("/test_predictions", response_class=Response)
async def test(request: Request):
    testset = await request.body()
    testset = testset.decode("utf-8")
    testset_df = str2df(testset)
    testset_df = testset_df

    assert "text" in testset_df, f"Dataset must have column `text`. Columns provided: {testset_df.columns}."
    texts = testset_df["text"].values
    predictions = run_inference(texts, artifacts, confidence_threshold=0)
    predictions_df = list2df(predictions)

    return Response(content=df2str(predictions_df), media_type="text/csv")

