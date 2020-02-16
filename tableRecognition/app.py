import cv2
import flask
from flask import Flask

import base64

from flask_cors import cross_origin
import flask

from tableRecognition.service import ocr, parseTable
from tableRecognition.service.recogniseTable import recogniseTableFromImage

UPLOAD_FOLDER = '/images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/image', methods=["POST"])
@cross_origin()
def upload_image():
    f = flask.request.get_json()
    picture = flask.request.get_json()
    pictureBase64 = picture['pictureData']
    noise = picture['noise']
    table_image = recogniseTableFromImage(pictureBase64, noise)

    # parse cropped table and returns cropped images.
    cropped_matrix = parseTable.parseTable(table_image)
    text = ocr.table_ocr(cropped_matrix)
    print(text)
    return True


if __name__ == '__main__':
    app.run(threaded=False)

