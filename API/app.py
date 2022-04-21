'''
/*
 * @Author: Pasindu Akalpa 
 * @Date: 2022-04-04 19:37:38 
 * @Last Modified by: Pasindu Akalpa
 * @Last Modified time: 2021-04-04 22:15:26
 */
'''

from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
from cv2 import cv2
from keras.models import load_model
import urllib.request
from os import environ

from sqlalchemy import true

app = Flask(__name__)

model = load_model('model/fruits_vegs_CNN_model.model')

im_w = 224
im_h = 224

vegetableDict = {0: "apple", 1: "banana", 2: "beetroot", 3: "bell pepper", 4: "cabbage", 5: "capsicum", 6: "carrot", 7: "cauliflower", 8: "chilli pepper", 9: "corn", 10: "cucumber", 11: "eggplant",
            12: "garlic", 13: "ginger", 14: "grapes", 15: "jalepeno", 16: "kiwi", 17: "lemon", 18: "lettuce", 19: "mango", 20: "onion", 21: "orange", 22: "paprika", 23: "pear", 24: "peas", 25: "pineapple",
            26:"pomegranate", 27:"potato", 28:"raddish", 29:"soy beans", 30:"spinach", 31:"sweetcorn", 32:"sweet potato", 33:"tomato", 34:"turnip", 35:"watermelon"}

def convert_to_array(img):
    im = cv2.imread(img)
    img = Image.fromarray(im, 'RGB')
    image = img.resize((im_w, im_h))
    return np.array(image)

def download_image_and_save(url):
    f = open('veg.jpg','wb')
    f.write(urllib.request.urlopen(url).read())
    f.close()

@app.route('/url', methods=['POST'])
def process_json():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.json
        download_image_and_save(json['url'])
        imageArray = convert_to_array("veg.jpg")
        imageArray = imageArray/255
        numpyImageArray = []
        numpyImageArray.append(imageArray)
        numpyImageArray = np.array(numpyImageArray)
        prediction = model.predict(numpyImageArray)
        label = np.argmax(prediction)
        return jsonify(Response = "True", Prediction = vegetableDict[label])
    else:
        return jsonify(Response = "False", Error = "Internal Error")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=environ.get("PORT",5000))