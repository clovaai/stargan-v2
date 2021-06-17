import logging

from flask_restful import Resource

FOO_ENDPOINT = '/foo'


class FooResouce(Resource):
    def get(self):
        logging.info('GET %s', FOO_ENDPOINT)
        pass

    def post(self):
        logging.info('POST %s', FOO_ENDPOINT)
        pass
