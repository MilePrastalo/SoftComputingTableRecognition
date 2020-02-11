from flask import Flask
import base64

from service.recogniseTable import recogniseTableFromImage

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/image', methods=["POST"])
def upload_image():
    picture = Flask.request.get_json()
    pictureBase64 = picture['pictureData']
    imgdata = base64.b64decode(pictureBase64)
    table_image = recogniseTableFromImage(imgdata)

if __name__ == '__main__':
    app.run()
