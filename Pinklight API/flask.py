#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, request, render_template, redirect, url_for
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load the trained model
model = load_model('my_model.h5')

# Define the class names
class_names = ['malignant', 'benign', 'normal']

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for uploading files
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            return redirect(url_for('predict', file_path=file_path))
    return render_template('upload.html')

# Route for making predictions
@app.route('/predict')
def predict():
    file_path = request.args.get('file_path')
    img = load_img(file_path, target_size=(150, 150), color_mode='grayscale')
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('predict.html', file_path=file_path, prediction=predicted_class)

# Route for visualizations
@app.route('/visualize')
def visualize():
    # Your code for visualizing images from the dataset
    return render_template('visualize.html')

if __name__ == '__main__':
    app.run(debug=True)

