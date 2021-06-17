from flask_app.database import db


class Foo(db.Model):
    foo_id = db.Column(db.Integer, primary_key=True, autoincrement=True)

    def __repr__(self):
        return '<Foo ID %r>' % self.foo_id
