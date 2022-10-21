import io

from io import BytesIO
from typing import Optional

from flask import Flask, request, send_file
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_pydantic import validate
from pydantic import BaseModel

from werkzeug.datastructures import FileStorage

from server_help import HTTPMethods, generate_midas_depth_map

app: Flask = Flask(__name__)
CORS(app, origins="*")
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50/minute"],
    storage_uri="memory://",
)


@app.route("/ping", methods=[HTTPMethods.get.value])
@limiter.limit("5/second")
def ping():
    return "pong"


class DepthMapQueryParams(BaseModel):
    name: str
    model_type: str
    attachment: bool = False


@app.route("/depthmap", methods=[HTTPMethods.post.value])
@validate(query=DepthMapQueryParams)
def create_depth_map(query: DepthMapQueryParams):
    img: FileStorage = request.files[query.name]
    base_img_buf: BytesIO = io.BytesIO(img.stream.read())

    data: BytesIO = generate_midas_depth_map(base_img_buf, query.model_type)
    return send_file(data, "image/png", as_attachment=query.attachment, download_name=query.name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8069, threaded=True)
