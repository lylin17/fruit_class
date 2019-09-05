import os

import tensorflow as tf
from flask import render_template,redirect,url_for
from flask import Flask
from flask import request
from keras.models import load_model

import utils
from utils import ReusableForm

#load model outside app
global model
model = load_model('model.h5')

global graph
graph = tf.get_default_graph()

#Flask App

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config["CACHE_TYPE"] = "null"

app.secret_key = 'super secret key'

filepath = os.path.join('static', 'predict.png')
errpath = os.path.join('static', 'error.png')
fontpath = os.path.join('static', 'arial.ttf')

# Home page
@app.route("/", methods=['GET', 'POST'])
def home():
    """Home page of app with form"""
   
    if os.path.exists(filepath): 
        os.remove(filepath)
    
    # Create form
    form = ReusableForm()        
    # On form entry and all conditions met
    if request.method == 'POST' and form.validate():
        # Extract information
        file = request.files['file']
        img, arr = utils.process_image(file, errpath)
        utils.pred_fruit(model, img, arr, graph, filepath, fontpath)
        return render_template('prediction.html', output = filepath)          
    
    if request.method == 'GET':    
        return render_template('index.html', form=form)
    
@app.route('/clear')
def clear():
    return redirect(url_for('home'))

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == "__main__":
    
    print(("* Loading Keras model and Flask starting server..."
           "\nplease wait until server has fully started *"))  

    app.run(host='127.0.0.1', port=5000)
