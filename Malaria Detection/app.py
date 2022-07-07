
import os
import sys
from flask import Flask, request, jsonify, send_file, render_template
from io import BytesIO
from PIL import Image, ImageOps
import base64
import urllib
import numpy as np
import scipy.misc
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route("/")
def login():
    return render_template('login.html')

@app.route("/homePage")
def homePage():
	return render_template('homePage.html')    

@app.route("/index",methods=['GET'])
def index():
	return render_template('index.html')

@app.route("/upload", methods=['POST'])
def upload_file():
	try:
		img = Image.open(BytesIO(request.files['imagefile'].read())).convert('RGB')
		img = ImageOps.fit(img, (224, 224), Image.ANTIALIAS)
	except:
		error_msg = "Please choose an image file!"
		return render_template('index.html', **locals())

	args = {'input' : img}
	out_pred, out_prob = predict(args)
	out_prob = out_prob * 100

	img_io = BytesIO()
	img.save(img_io, 'PNG')

	png_output = base64.b64encode(img_io.getvalue())
	processed_file = urllib.parse.quote(png_output)

	return render_template('result.html',**locals())

@app.route("/performance")
def performance():
	return render_template('performance.html')

def predict(args):
	img = np.array(args['input']) / 255.0
	img = np.expand_dims(img, axis = 0)

	model = 'model_vgg19.h5'
	model = load_model(model)
	pred = model.predict(img)
	if np.argmax(pred, axis=1)[0] == 0:
		out_pred = "Infected"
	elif np.argmax(pred, axis=1)[0] == 1:
		out_pred = "Uninfected"       
	return out_pred, float(np.max(pred))

if __name__ == '__main__':
    app.run()

