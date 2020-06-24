import os

from flask import Flask

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)

    @app.route("/hello")
    def hello():
        return "Hello, World!"

    from Flaskk import MealPrediction, AddMeal

    app.register_blueprint(MealPrediction.bp)
    app.register_blueprint(AddMeal.bp)
    app.add_url_rule("/", 'result')
    return app