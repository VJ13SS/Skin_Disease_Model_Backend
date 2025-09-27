
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img,img_to_array
import numpy as np

Melanocytic nevi (nv)
Melanoma (mel)
Benign keratosis-like lesions (bkl)
Basal cell carcinoma (bcc)
Actinic keratoses (akiec)
Vascular lesions (vas)
Dermatofibroma (df)

model = load_model("models/skin_cancer_model.h5",compile=False)
img_path = 'ISIC_0000166.jpg'
img = image.load_img(img_path,target_size=(32,32))
img_array = image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array,axis=0)
class_names = ['akiec','bcc','bkl','df','mel','nv','vasc']

predict = model.predict(img_array)
print('helo',np.argmax(predict))
