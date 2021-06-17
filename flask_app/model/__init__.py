import logging
import time

from flask_app.model.declarations import StarGAN

model_store = {}


def init_model_store(app):
    logging.info('Loading model store...')
    start_time = time.time()

    model_store['stargan'] = StarGAN()
    model_store['stargan'].init_model(app.config.get('MODEL_ARGS'))
    end_time = time.time()

    logging.info('Total time taken to load model store: %.3fs.' %
                 (end_time - start_time))
