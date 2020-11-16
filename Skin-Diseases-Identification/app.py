from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH ='Skin_Diseases.h5'

# Load your trained model
model = load_model(MODEL_PATH)




def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(64, 64))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x = np.expand_dims(x, axis=0)
   

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
   # x = preprocess_input(x)

    preds = model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="Suffered Skin Disease is : Acne  || Treatment : Washing once or twice a day with a mild cleansing bar or liquid (for example, Dove, Neutrogena) will keep the skin clean and minimize sensitivity and irritation, A variety of mild scrubs, exfoliants, and masks can be used.These products remove the outer layer of the skin and thus open pores, Retinol: Not to be confused with the prescription medication Retin-A, this derivative of vitamin A can help promote skin peeling. "
    elif preds==1:
        preds="Suffered Skin Disease is : Eczema  || Treatment : Treatment includes avoiding soap and other irritants. Certain creams or ointments may also provide relief from the itching, Ultraviolet light therapy for serious skin diseases. Used along with a special medication that increases light absorption, Use of Moisturizer hydrates and protects skin from damage. "
    elif preds==2:
        preds="Suffered Skin Disease is : Melanoma  || Treatment : Treatment may involve surgery, radiation, medication or in some cases, chemotherapy, Radiation therapy treatment that uses x-rays and other high-energy rays to kill abnormal cells, Immunotherapy Lowers or changes normal immune response to treat disease, especially cancer "
    elif preds==3:
        preds="Suffered Skin Disease is : Psoriasis  || Treatment : Treatment aims to remove scales and stop skin cells from growing so quickly. Topical ointments, light therapy and medication can offer relief, Medical procedure:Photodynamic therapy, Medications:Steroid, Vitamin A derivative, Anti-inflammatory, Immunosuppressive drug and Vitamin, Self-care:Stress management, Petroleum jelly, Light therapy, Ultraviolet light therapy, Moisturizer and Coal tar extract"
    elif preds==4:
        preds="Suffered Skin Disease is : Rosacea  || Treatment : Treatments such as antibiotics or anti-acne medication can control and reduce symptoms. Left untreated, it tends to worsen over time, Self-care:Sunblock, Light therapy, Moisturizer and Artificial tears"
    else:
        preds="Suffered Skin Disease is : Vitiligo || Treatment : Treatment may improve the appearance of the skin but doesn't cure the disease, Self-care:Sunblock, Ultraviolet light therapy, Covermark topical and Dermablend topical"
    
        
    
    
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None


if __name__ == '__main__':
    app.run(port=5001,debug=True)
