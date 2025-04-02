from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
from io import BytesIO
import os
import uuid
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Initialize Flask app
application = Flask(__name__)

# Set secret key for session management
application.secret_key = 'your_secret_key_here'  # Replace with a secure key

# Path and model setup
model_path = 'models/My_custom_resnet_model_BGD.keras'  # Update with your model path
model = tf.keras.models.load_model(model_path)

# Allowed file types
ALLOWED_FILETYPES = set(['.jpg', '.jpeg', '.gif', '.png', '.bmp'])

# Hardcoded class dictionary for Blood Group Detection
class_dictionary = {
    0: 'A+',
    1: 'A-',
    2: 'AB+',
    3: 'AB-',
    4: 'B+',
    5: 'B-',
    6: 'O+',
    7: 'O-'
}

# Image dimensions (update to match your model's input size)
IMG_HEIGHT, IMG_WIDTH = 64, 64

# LIME Image Explainer setup
explainer = lime_image.LimeImageExplainer()

# Image preprocessing and classification functions
def classify_image(image_path):
    try:
        # Open the image and convert to RGB
        img = Image.open(image_path).convert('RGB')
        
        # Resize image to model's input dimensions
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        
        # Convert image to numpy array
        img_array = image.img_to_array(img)
        
        # Add batch dimension and normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize to [0, 1]

        # Make prediction with the model
        probabilities = model.predict(img_array)

        # Get the predicted class and prediction probability
        prediction_probability = probabilities[0, probabilities.argmax(axis=1)][0]
        
        # Print raw model output (probabilities)
        print(f"Model raw output (probabilities): {prediction_probability}")

        class_predicted = np.argmax(probabilities, axis=1)  # Class index
        print(class_predicted)
        inID = class_predicted[0]

        # Invert the class dictionary to get the label for the predicted class
        #inv_map = {v: k for k, v in class_dictionary.items()}
        #label = inv_map.get(inID, "Unknown")  # Get the label from the dictionary
        label = class_dictionary[inID]

        print("[Info] Predicted: {}, Confidence: {}".format(label, prediction_probability))
        
        return label, prediction_probability, img_array[0]

    except Exception as e:
        print(f"Error occurred while processing the image: {e}")
        return None, None, None


def get_image_thumbnail(image):
    image.thumbnail((400, 400), resample=Image.LANCZOS)
    image = image.convert("RGB")
    with BytesIO() as buffer:
        image.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

def generate_lime_overlay(image_array):
    explanation = explainer.explain_instance(image_array, model.predict, top_labels=5, hide_color=0, num_samples=1000)
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)
    img_boundaries = mark_boundaries(temp / 2 + 0.5, mask)
    return img_boundaries

# Route for handling image upload and prediction
@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'blood_image' not in request.files:
            flash('No file uploaded.')
            return redirect(url_for('index'))

        file = request.files['blood_image']
        if file.filename == '':
            flash('No file selected to upload.')
            return redirect(url_for('index'))

        # Secure filename and check extension
        sec_filename = secure_filename(file.filename)
        file_extension = os.path.splitext(sec_filename)[1].lower()

        if file and file_extension in ALLOWED_FILETYPES:
            file_tempname = uuid.uuid4().hex
            image_path = './uploads/' + file_tempname + file_extension
            file.save(image_path)

            # Classify the image
            label, prediction_probability, img_array = classify_image(image_path)
            prediction_probability = np.around(prediction_probability, decimals=4)

            # Generate LIME explanation overlay
            lime_overlay = generate_lime_overlay(img_array)

            # Create the base64-encoded image thumbnail
            orig_image = Image.open(image_path)
            image_data = get_image_thumbnail(orig_image)

            # Convert LIME overlay to base64 for displaying on the frontend
            lime_overlay_image = Image.fromarray((lime_overlay * 255).astype(np.uint8))
            with BytesIO() as buffer:
                lime_overlay_image.save(buffer, 'PNG')
                lime_overlay_base64 = base64.b64encode(buffer.getvalue()).decode()

            return render_template('index.html', label=label, prob=prediction_probability, image=image_data, lime_overlay=lime_overlay_base64)

        else:
            flash(f"Invalid file type: {file_extension}. Only .jpg, .jpeg, .gif, .png, .bmp are allowed.")
            return redirect(url_for('index'))

    return render_template('index.html')

# Handle 'filesize too large' errors
@application.errorhandler(413)
def http_413(e):
    print("[Error] Uploaded file too large.")
    flash('Uploaded file too large.')
    return redirect(url_for('index'))

# Run the Flask application
if __name__ == "__main__":
    application.run(debug=True)
