from function import *

import json
import base64
import cv2
import os

from flask import Flask, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from werkzeug.utils import secure_filename


app = Flask(__name__)
api = Api(app)
CORS(app)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, "upload_files")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (0, 255, 255), (255, 0, 255)]


class Prediction(Resource):
    def post(self):
        file = request.files['file']
        filepath = ""
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
        print("Detecting...")
        # filepath = "data/car/images/00088.jpg"
        # result = detect(net, meta, b'data/car/images/00088.jpg')
        result = detect(net, meta, filepath)
        img = cv2.imread(filepath)

        prediction = ""
        draw_image = img
        for index, prediction in enumerate(result):
            name, accuracy, box = prediction
            box = map(int, box)
            font = cv2.FONT_HERSHEY_SIMPLEX

            x, y, w, h = box
            left = x - w/2
            top = y - h/2

            draw_image = cv2.rectangle(
                draw_image, (left, top), (left+w, top+h), colors[index], 2)
            cv2.putText(draw_image, name, (left, top), font, 0.8,
                        colors[index], 2, cv2.LINE_AA)
            # crop_img = img[top:top+h, left:left+w]
            retval, buffer = cv2.imencode('.jpg', draw_image)

            encoded_string = base64.b64encode(buffer)
            prediction = encoded_string

        print("Done!")
        return {'prediction': prediction}


api.add_resource(Prediction, '/prediction')


def main():
    print("abc")
    # # print(r)
    # print(json.dumps(r))
    # return (encoded_string)


if __name__ == "__main__":
    # net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    # im = load_image("data/wolf.jpg", 0, 0)
    # meta = load_meta("cfg/imagenet1k.data")
    # r = classify(net, meta, im)
    # print r[:10]

    net = load_net(b"cfg/yolov3.cfg", b"weights/yolov3.weights", 0)
    # meta = load_meta(b"cfg/yolov3.data")
    meta = load_meta(b"cfg/coco.data")
    # with open(b"data/dog.jpg", "rb") as image_file:
    #     encoded_string = base64.b64encode(image_file.read())
    # r = detect(net, meta, b"data/dog.jpg")
    app.run(host='0.0.0.0', port=5000)
