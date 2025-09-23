from flask import Blueprint,jsonify
#from tensorflow.keras.models import load_model
#from tensorflow.keras.utilsimport load_img,img_to_array
import numpy as np

main = Blueprint('main',__name__)

def predict_image(img_path,model,class_indices):
    img = load_img(img_path,target_size=(224,224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array,axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions,axis=1)[0]

    #Map class to label
    labels = dict((v,k) for k,v in class_indices.items())
    return labels[predicted_index]

@main.route('/predict')
def perdict():

    try:
        img_path = ""
        class_indices = {'Actinic keratosis': 0,
 'Atopic Dermatitis': 1,
 'Benign keratosis': 2,
 'Dermatofibroma': 3,
 'Melanocytic nevus': 4,
 'Melanoma': 5,
 'Squamous cell carcinoma': 6,
 'Tinea Ringworm Candidiasis': 7,
 'Vascular lesion': 8}
        model = load_model("skin_disease_model.keras")
        
        prediction = predict_image(img_path,model,class_indices)
        return jsonify({'res':prediction}),200
    except:
        return jsonify({'error':'Some Error Occured'}),404
