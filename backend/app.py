import os

import flask
from flask import Flask, Response, request
from flask_cors import CORS

from text2performer import text2performer

app = Flask(__name__)
CORS(app)


@app.route("/")
def hello_world() -> str:
    """
    返回一个包含 "Hello, World!" 的 HTML 字符串。

    Returns:
        str: 包含 "Hello, World!" 的 HTML 字符串。
    """
    return "<div><h1>Hello, World!</h1></div>"


@app.route('/favicon.ico')
def favicon() -> Response:
    return flask.send_from_directory(
        os.path.join(app.root_path, 'static'),
        'favicon.ico',
        mimetype='image/vnd.microsoft.icon',
    )


@app.route('/text2performer/generate_appearance', methods=['POST'])
def text2performer_generate_appearance() -> str:
    return text2performer.generate_appearance(request.json['input_appearance'])


@app.route('/text2performer/generate_motion', methods=['POST'])
def text2performer_generate_motion() -> str:
    return text2performer.generate_motion(request.json['input_motion'])


@app.route('/text2performer/interpolate')
def text2performer_interpolate() -> str:
    return text2performer.interpolate()


def main() -> None:
    app.run()


if __name__ == '__main__':
    main()
