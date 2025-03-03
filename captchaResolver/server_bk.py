from flask import Flask, request
from flask_restful import Resource, Api
# from flask.ext.jsonpify import jsonify
from keras.models import load_model
import base64
from PIL import Image
import cv2
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
api = Api(app)

LETTERSTR = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
SCALE_PERCENT = 100

model = load_model("real_3_model.keras")


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


class CaptchaSolver (Resource):
    def get(self):
        base64String = request.args.get('base64')

        if base64String == None:
            return ''
        
        image = stringToImage(base64String)
        image = toRGB(image)
        
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        width = int(img.shape[1] * SCALE_PERCENT / 100)
        height = int(img.shape[0] * SCALE_PERCENT / 100)
        dim = (width, height)

        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img = cv2.medianBlur(img, 5)

        answer = ""
        prediction = model.predict(np.stack([np.array(img)/255.0]))
        for predict in prediction:
            answer += LETTERSTR[np.argmax(predict[0])]


        print(answer)
        return answer
    
    def readb64(self, encoded_data):
        nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img


class QRSolver (Resource):
    def get(self):
        base64String = request.args.get('base64')

        if base64String == None:
            return ''
        
        image = stringToImage(base64String)
        image = toRGB(image)
        
        detector = cv2.QRCodeDetector()
        data, vertices_array, binary_qrcode = detector.detectAndDecode(image)
        
        if vertices_array is not None:
            print("QRCode data:")
            print(data)
            return data
        else:
            print("There was some error")

            return ''



api.add_resource(CaptchaSolver, '/captcha') # Route_1
api.add_resource(QRSolver, '/qr') # Route_1

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)