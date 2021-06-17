import logging
from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self):
        self.predictor = None

    @abstractmethod
    def init_model(self, config):
        pass

    @abstractmethod
    def predict(self, input_image):
        pass

    @abstractmethod
    def format_prediction(self, prediction):
        pass

    @abstractmethod
    def get_visualization(self, input_image, outputs):
        pass
