from flask_marshmallow import Marshmallow

from flask_app.database.database import Foo

ma = Marshmallow()


class FooSchema(ma.SQLAlchemyAutoSchema):
    class Meta:
        model = Foo


foo_schema = FooSchema()
foo_list_schema = FooSchema(many=True)
