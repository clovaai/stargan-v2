import logging
from abc import ABC, abstractmethod

from core.solver import Solver
from munch import Munch
from PIL import Image


class BaseModel(ABC):
    def __init__(self):
        self.predictor = None

    @abstractmethod
    def init_model(self, config: dict):
        pass

    @abstractmethod
    def predict(self, inputs: dict):
        pass

    @abstractmethod
    def format_prediction(self, prediction):
        pass

    @abstractmethod
    def get_visualization(self, inputs, outputs):
        pass


class StarGAN(BaseModel):
    def init_model(self, config: dict):
        self.predictor = Solver(Munch(config))
        self.predictor._load_checkpoint(config['resume_iter'])

    def predict(self, inputs: dict) -> Image.Image:
        src_img = inputs.get('src_img')
        ref_img = inputs.get('ref_img')
        ref_class = inputs.get('ref_class')
        return self.predictor.predict(src_img, ref_img, ref_class)

    def format_prediction(self, prediction):
        pass

    def get_visualization(self, inputs, outputs):
        pass
