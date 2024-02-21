import os

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS

from text2performer import Text2Performer

app = Flask(__name__)
CORS(app)

text2performer = Text2Performer()


@app.route("/")
def hello_world() -> str:
    return "<h1>Hello, World!</h1>"


@app.route('/favicon.ico')
def favicon() -> Response:
    return send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon',
    )


@app.route('/generate_appearance', methods=['POST'])
def generate_appearance():
    return text2performer.generate_appearance(request.json['input_appearance'])


@app.route('/generate_motion', methods=['POST'])
def generate_motion():
    return text2performer.generate_motion(request.json['input_motion'])


@app.route('/interpolate')
def interpolate():
    return text2performer.interpolate()


def main() -> None:
    app.run()


if __name__ == '__main__':
    main()
