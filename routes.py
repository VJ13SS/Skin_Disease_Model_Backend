from flask import Blueprint,jsonify,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np
from io import BytesIO
from PIL import Image

main = Blueprint('main',__name__)

"""Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)"""

SIZE = 32  # same as used in training

def preprocess_image(file_data):
    """
    Preprocess uploaded image for prediction.

    Args:
        file_data (BytesIO or file-like object): Uploaded image file

    Returns:
        np.array: Preprocessed image ready for model.predict, shape=(1, SIZE, SIZE, 3)
    """
    try:
        # Load image from BytesIO
        img = Image.open(file_data).convert('RGB')
        
        # Resize to match training
        img = img.resize((SIZE, SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.asarray(img) / 255.0
        
        # Expand dims to add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None

@main.route('/predict',methods = ['POST'])
def predict():
    try:
        file = request.files['file']
        class_names = ['Actinic keratoses','Basal cell carcinoma','Benign keratosis-like lesions',
                       'Dermatofibroma','Melanoma','Melanocytic nevi','Vascular lesions']
        
        model = load_model("models/skin_cancer_model.h5", compile=False)
        
        # Use the preprocess function
        img_array = preprocess_image(file)
        if img_array is None:
            return jsonify({'error':'Invalid image'}), 400
        
        pred = model.predict(img_array)
        res = class_names[np.argmax(pred)]
        
        return jsonify({'res': res}), 200
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500
