import logging
import time

from flask import Blueprint, render_template, request

from flask_app.commons.util import base64_to_image, image_to_base64
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

        model = model_store.get('stargan')

        # convert images in base64 string format to PIL image
        output_img = model.predict({
            'src_img': base64_to_image(req['src_img']),
            'ref_img': base64_to_image(req['ref_img']),
            'ref_class': req['ref_class'],
            'face_aligner': model_store.get('face_aligner').predictor if req.get('align_face') else None
        })

        # convert output image from PIL image to base64 string
        response = {'output_img': image_to_base64(output_img)}

        end_time = time.time()
        logging.info('Total prediction time: %.3fs.' % (end_time - start_time))

        return response

    except Exception as error:
        logging.exception(error)
        return {'error': 'Error in prediction'}
