import os

import flask
from flask import Flask, Response, request
from flask_cors import CORS
from werkzeug.utils import secure_filename

from text2performer import Text2Performer

app = Flask(__name__)
CORS(app)

text2performer = Text2Performer()


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
    # 获取 favicon 文件的路径
    file_path = os.path.join(app.root_path, 'static', 'favicon.ico')
    # 使用 secure_filename 确保文件名是安全的
    secure_file_path = secure_filename(file_path)
    # 发送 favicon 文件给客户端
    return flask.send_from_directory(
        app.root_path, secure_file_path, mimetype='image/vnd.microsoft.icon'
    )


@app.route('/generate_appearance', methods=['POST'])
def generate_appearance() -> str:
    return text2performer.generate_appearance(request.json['input_appearance'])


@app.route('/generate_motion', methods=['POST'])
def generate_motion() -> str:
    return text2performer.generate_motion(request.json['input_motion'])


@app.route('/interpolate')
def interpolate() -> str:
    return text2performer.interpolate()


def main() -> None:
    app.run()


if __name__ == '__main__':
    main()
