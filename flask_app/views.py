import logging
import time

from flask import Blueprint, render_template, request

from flask_app.model import model_store

blueprint = Blueprint('blueprint', __name__)


@blueprint.route('/')
def index():
    logging.info("GET /")
    return render_template('index.html')


@blueprint.route('/predict', methods=['POST'])
def predict():
    logging.info("POST /predict")
    req = request.get_json()

    try:
        start_time = time.time()

        example_response = {'example': 'example response'}

        end_time = time.time()
        logging.info('Total prediction time: %.3fs.' % (end_time - start_time))

        return example_response

    except Exception as error:
        logging.exception(error)
        return {'error': 'Error in prediction'}
