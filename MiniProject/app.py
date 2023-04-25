import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template

# Load the trained BrailleNet model
model = tf.keras.models.load_model('BrailleNet.h5')

# Define a function to preprocess the image for prediction
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0
    return x

# Initialize the Flask application
app = Flask(__name__)

# Define a route to accept user-uploaded images
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file from the request
    img_file = request.files['image']

    # Save the image to a temporary file
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], img_file.filename)
    img_file.save(img_path)

    # Preprocess the image for prediction
    x = preprocess_image(img_path)

    # Use the model to make a prediction
    pred = model.predict(x)[0]

    # Get the predicted label
    label = chr(ord('a') + np.argmax(pred))

    # Return the predicted label as a response
    return label

# Run the Flask application
if __name__ == '__main__':
    app.config['UPLOAD_FOLDER'] = './uploads'
    app.run(debug=True)
