from text_classifier.models import BaseModel, TfIdfModel
from text_classifier.data.make_dataset import Preprocessor
from functools import cached_property


class Artifacts:
    def __init__(self, output_dir, model_cls=TfIdfModel):
        self.output_dir = output_dir
        self.model_cls = model_cls
        self.model: BaseModel = self.model_cls.load_from_checkpoint(self.output_dir)
        self.preprocessor = Preprocessor()
