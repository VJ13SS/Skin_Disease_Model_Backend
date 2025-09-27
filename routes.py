from flask import Blueprint,jsonify,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from io import BytesIO

main = Blueprint('main',__name__)

@main.route('/predict',methods = ['POST'])
def perdict():

    try:
        data = request.files['file']
        class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']
        
        model = load_model("models/skin_cancer_model.h5",compile=False)
        img_path = BytesIO(data.read())
        img = image.load_img(img_path,target_size=(32,32))
        img_array = image.img_to_array(img)/255.0
        img_array = np.expand_dims(img_array,axis=0)
        
        predict = model.predict(img_array)
        res = class_names[np.argmax(predict)]

        print('helo',res)
        return jsonify({'res':'result'}),200
    except Exception as e:
        print(e)
        return jsonify({'error':'Some Error Occured'}),404
