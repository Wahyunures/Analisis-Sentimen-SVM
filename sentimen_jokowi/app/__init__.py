from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)
migrate = Migrate()
db = SQLAlchemy()

def create_app():
    from .config import Config

    app.config.from_object(Config)

    db.init_app(app)
    migrate.init_app(app,db)

    from .controller import controller
    
    from .model.model import Tweet,Preprocess

    app.register_blueprint(controller,url_prefix="/")
    return app