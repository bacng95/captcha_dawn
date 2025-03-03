from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from keras.models import load_model
import base64
from PIL import Image
import cv2
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Áp dụng CORS cho toàn bộ ứng dụng
api = Api(app)

LETTERSTR = "ABCDEFGHJKMNPQRSTVWXY1234568@#$%&="
SCALE_PERCENT = 100

model = load_model("real_3_model.keras")


# Take in base64 string and return PIL image
def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(imgdata))

# convert PIL Image to an RGB image( technically a numpy array ) that's compatible with opencv
def toRGB(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def adjust_levels(image, in_black, in_white, gamma=1.0, out_black=0, out_white=255):
    """
    Adjust the levels of an image, mimicking Photoshop's Levels adjustment.
    
    Parameters:
        image (numpy.ndarray): Input image in BGR format.
        in_black (int): Input black point (0-255).
        in_white (int): Input white point (0-255).
        gamma (float): Gamma correction value (default is 1.0).
        out_black (int): Output black point (0-255).
        out_white (int): Output white point (0-255).
    
    Returns:
        numpy.ndarray: Image after levels adjustment.
    """
    # Ensure input is within valid range
    in_black = max(0, min(in_black, 255))
    in_white = max(0, min(in_white, 255))
    out_black = max(0, min(out_black, 255))
    out_white = max(0, min(out_white, 255))
    gamma = max(0.1, gamma)  # Prevent gamma values too close to 0
    
    # Normalize the image to the range [0, 1]
    img_normalized = image / 255.0
    
    # Apply input range adjustment
    img_adjusted = (img_normalized - (in_black / 255.0)) / ((in_white - in_black) / 255.0)
    img_adjusted = np.clip(img_adjusted, 0, 1)
    
    # Apply gamma correction
    img_gamma_corrected = np.power(img_adjusted, gamma)
    
    # Apply output range adjustment
    img_output = img_gamma_corrected * ((out_white - out_black) / 255.0) + (out_black / 255.0)
    img_output = np.clip(img_output, 0, 1)
    
    # Scale back to [0, 255] and convert to uint8
    img_final = (img_output * 255).astype(np.uint8)
    
    return img_final



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
        img = cv2.medianBlur(img, 3)

        adjusted = adjust_levels(img, 158, 200, 13, 0, 255)
        adjusted = cv2.GaussianBlur(adjusted,(5,5),0)
        adjusted = adjust_levels(img, 158, 200, 13, 0, 255)

        # v1
        # adjusted = adjust_levels(img, 158, 200, 110, 0, 255)
        # adjusted = cv2.GaussianBlur(adjusted,(5,5),0)
        # adjusted = adjust_levels(img, 158, 200, 110, 0, 255)

        grayscale_image = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('image', np.array(grayscale_image)/255.0)
        # cv2.waitKey(0)

        answer = ""
        prediction = model.predict(np.stack([np.array(grayscale_image)/255.0]))
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
    app.run(host='0.0.0.0', port=9998)