from abc import ABC, abstractmethod


class BaseModel:
    def __init__(self):
        self.model = None

    def predict(self, *args, **kwargs):
        pass

    def save_checkpoint(self, *args, **kwargs):
        pass

    def load_from_checkpoint(self, *args, **kwarg) -> 'BaseModel':
        pass

    @property
    @abstractmethod
    def classes(self) -> list:
        pass


