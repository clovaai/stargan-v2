import logging
from logging.config import dictConfig

from config import LOGGING_CONFIG
from flask import Flask

dictConfig(LOGGING_CONFIG)

from datetime import datetime

logging.getLogger().setLevel(0)  # Allows all messages to go through logger, msg to be filtered at handlers

ID = LOGGING_CONFIG['handlers']['instance']['filename'][2:-4]  # Get hexadecimal ID of instance

logging.info('\n' + '='*60 + '\nNew instance started at ' + datetime.now().strftime("%d/%m/%y, %H:%M:%S")[:-3] +
            '\nInstance ID: {}\n'.format(ID) + '='*60)  # Print instance start message and datetime as quick reference

def create_app():
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object('config')
    # override config from config.py in the instance folder if it exists
    app.config.from_pyfile('config.py', silent=True)

    # init model store
    from flask_app.model import init_model_store
    try:
        init_model_store(app)
    except Exception:
        logging.exception('Unable to init model store. Raising error.')
        raise

    # register blueprints
    from flask_app.views import blueprint
    app.register_blueprint(blueprint)

    # register db
    from flask_app.database import db
    db.init_app(app)

    # register serializer
    from flask_app.serialize import ma
    ma.init_app(app)

    # register api resources
    from flask_restful import Api

    from flask_app.resources.foo import FOO_ENDPOINT, FooResouce

    api = Api(app)
    api.add_resource(FooResouce, FOO_ENDPOINT)

    return app
