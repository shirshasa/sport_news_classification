from text_classifier.models import BaseModel
from sklearn import feature_extraction, feature_selection, naive_bayes, pipeline
import numpy as np
import pickle
import os
import logging


class TfIdfModel(BaseModel):

    def __init__(self, checkpoint=None):
        super().__init__()
        if checkpoint:
            self.vectorizer = checkpoint["vectorizer"]
            self.model: pipeline.Pipeline = checkpoint
        else:
            self.vectorizer = feature_extraction.text.TfidfVectorizer(max_features=10000, ngram_range=(1, 2))

    @property
    def classes(self) -> list:
        return self.model.classes_

    def fit(self, X, y):
        assert "text_clean" in X
        features = self.build_features(X, y)
        classifier = naive_bayes.MultinomialNB()
        # prepare pipeline
        self.model = pipeline.Pipeline([("vectorizer", self.vectorizer), ("classifier", classifier)])
        # train classifier
        self.model["classifier"].fit(features, y)

    def build_features(self, X, y) -> np.array:
        # first run
        corpus = X["text_clean"]
        self.vectorizer.fit(corpus)
        X_train: np.array = self.vectorizer.transform(corpus)
        # select best tokens aka features
        selected_features = self.get_best_features(X_train, y)

        if selected_features:
            # final run
            self.vectorizer = feature_extraction.text.TfidfVectorizer(vocabulary=selected_features)
            print(corpus)
            self.vectorizer.fit(corpus)
            X_train = self.vectorizer.transform(corpus)
        else:
            logging.warning(
                "Dataset is too small to have good results using tf-idf features. "
                "Consider using another model."
            )

        return X_train

    def get_best_features(self, x_train: np.array, y_train):
        feat_names = self.vectorizer.get_feature_names_out()
        p_value_limit = 0.9

        selected_features = set()
        df_features = []

        for cat in np.unique(y_train):
            binary_target = y_train == cat
            chi2, p = feature_selection.chi2(x_train, binary_target)

            for feat_name, chi2_val, p_val in zip(feat_names, chi2, p):
                if 1 - p_val > p_value_limit:
                    selected_features.add(feat_name)
                    df_features.append({"y": cat, "feature": feat_name, "score": 1 - p_val})

        print(f"Features selected amount: {len(selected_features)}")
        return selected_features

    def save_checkpoint(self, output_dir, model_filename="model.pkl"):
        assert self.model, "No model was initialized, first run fit()."
        os.makedirs(output_dir, exist_ok=True)

        file_path = os.path.join(output_dir, model_filename)
        with open(file_path, "wb") as f:
            pickle.dump(self.model, f)

    @classmethod
    def load_from_checkpoint(cls, file_path) -> 'TfIdfModel':
        with open(file_path, "rb") as f:
            checkpoint = pickle.load(f)

        assert isinstance(checkpoint, pipeline.Pipeline)
        return cls(checkpoint)

    def predict(self, batch: tuple[str, np.array]):
        assert self.model, "No model was initialized, first run fit() or load_from_checkpoint()."

        if isinstance(batch, str):
            batch = [batch]

        labels = self.model.predict(batch)
        probas = self.model.predict_proba(batch)

        return labels, probas
