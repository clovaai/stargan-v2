import logging
from abc import ABC, abstractmethod

import torch
from core import wing
from core.data_loader import get_test_transform
from core.solver import Solver
from core.utils import denormalize
from munch import Munch
from PIL import Image


class BaseModel(ABC):
    def __init__(self):
        self.predictor = None
        self.config = dict()

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
        self.config = config
        self.predictor = Solver(Munch(config))
        self.predictor._load_checkpoint(config['resume_iter'])

    def predict(self, inputs: dict) -> Image.Image:
        src_img = inputs.get('src_img')
        ref_img = inputs.get('ref_img')
        ref_class = inputs.get('ref_class')
        face_aligner = inputs.get('face_aligner')

        return self.predictor.predict(src_img, ref_img, ref_class, face_aligner)

    def format_prediction(self, prediction):
        pass

    def get_visualization(self, inputs, outputs):
        pass


class FaceAligner(BaseModel):
    def init_model(self, config: dict):
        self.config = config

        for key in ('wing_path', 'lm_path', 'img_size'):
            assert key in config, f"Key '{key}' not found in config"

        self.predictor = wing.FaceAligner(config['wing_path'],
                                          config['lm_path'],
                                          config['img_size'])

    def predict(self, inputs: dict):
        """ Performs face alignment using pre-trained model from StarGANv2 repository

        Parameters
        ----------
        inputs : dict
            Dictionary object with image to be aligned stored under key `img`

        Returns
        -------
        PIL.Image.Image
            Aligned image as a PIL image

        """
        img = inputs.get('img')

        transform = get_test_transform(self.config['img_size'])
        transformed_img = transform(img)
        transformed_img = transformed_img.unsqueeze(0)

        aligned_source = self.predictor.align(transformed_img)

        output = aligned_source.squeeze()
        output = denormalize(output)
        output = output.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
        output = output.to('cpu', torch.uint8).numpy()

        return Image.fromarray(output)

    def format_prediction(self, prediction):
        pass

    def get_visualization(self, inputs, outputs):
        pass
