from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import os

app = Flask(__name__)

# Load the CT scan model
ct_model = load_model('ct_scan_model.h5')

# Load the histo_image model
histo_model = load_model('models/histopathology_model.h5')  # Update with the correct path to your model file

# Define image size for preprocessing
ct_img_size = (150, 150)
histo_img_size = (150, 150)

# Define class labels for CT scan model
ct_labels = {0: 'Adenocarcinoma', 1: 'Benign', 2: 'Squamous Cell Carcinoma'}

# Define class labels for histo_image model
histo_class_labels = {0: 'Lung adenocarcinoma', 1: 'Lung benign Tissue', 2: 'Lung squamous cell carcinoma', 3: 'none'}

# Function to preprocess an image for CT scan prediction
def preprocess_ct_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, ct_img_size)
    img = img / 255.0  # Normalize pixel values
    return img.reshape(1, *ct_img_size, 3)  # Reshape for model input

# Function to preprocess an image for histo_image prediction
def preprocess_histo_image(img):
    img = img.resize(histo_img_size)
    img = np.array(img) / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Function to predict class for CT scan
def predict_ct_class(image_path):
    processed_image = preprocess_ct_image(image_path)
    predictions = ct_model.predict(processed_image)
    predicted_class = ct_labels[np.argmax(predictions)]
    return predicted_class

# Function to predict class for histo_image
def predict_histo_class(img):
    processed_img = preprocess_histo_image(img)
    prediction = histo_model.predict(processed_img)
    predicted_class = np.argmax(prediction)
    return histo_class_labels[predicted_class]

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle CT scan prediction
@app.route('/ct_scan', methods=['GET', 'POST'])
def ct_scan():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = 'temp/uploaded_ct_image.png'
            uploaded_file.save(image_path)
            predicted_class = predict_ct_class(image_path)
            return jsonify({'result': predicted_class})
    return render_template('ct_scan.html')

# Route to handle histo_image prediction
@app.route('/histo_image', methods=['GET', 'POST'])
def histo_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file uploaded!'
        
        file = request.files['file']

        if file.filename == '':
            return 'No file selected!'

        # Check if the file is of allowed type
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if file.filename.split('.')[-1].lower() not in allowed_extensions:
            return 'Invalid file type! Please upload an image.'

        # Create the 'temp' directory if it doesn't exist
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        # Save the uploaded file to a temporary location
        upload_path = os.path.join(temp_dir, 'temp_img.jpg')
        file.save(upload_path)

        # Read and predict class of the uploaded image
        img = image.load_img(upload_path, target_size=histo_img_size)
        os.remove(upload_path)  # Remove the temporary file
        predicted_class = predict_histo_class(img)

        return jsonify({'result': predicted_class})

    return render_template('histo_image.html')

if __name__ == '__main__':
    app.run(debug=True)
