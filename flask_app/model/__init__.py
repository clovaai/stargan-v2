import logging
import time

from flask_app.model.declarations import FaceAligner, StarGAN

model_store = {}


def init_model_store(app):
    logging.info('Loading model store...')
    start_time = time.time()

    model_store['stargan'] = StarGAN()
    logging.info('Initializing StarGAN...')
    model_store['stargan'].init_model(app.config.get('MODEL_ARGS'))

    model_store['face_aligner'] = FaceAligner()
    logging.info('Initializing FaceAligner...')
    model_store['face_aligner'].init_model(app.config.get('MODEL_ARGS'))

    end_time = time.time()

    logging.info('Total time taken to load model store: %.3fs.' %
                 (end_time - start_time))
