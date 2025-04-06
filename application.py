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
import time
from datetime import datetime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from flask import jsonify


# Initialize Flask app
application = Flask(__name__)

# Set secret key for session management
application.secret_key = 'manas123xyz'  # Replace with a secure key

# Path and model setup
model_path = ''  # Update with your model path
model = None

analysis_summaries = []
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
IMG_HEIGHT, IMG_WIDTH = 128, 128

# Model map for selections
model_map = {
    'model1': 'models/My_low_acc_model_BGD.keras',
    'model2': 'models/My_high_acc_model_BGD.keras',
    'model3': 'models/My_custom_resnet_model_BGD.keras'
}

# Resize image based on model
model_input_sizes = {
    'model1': (128, 128),
    'model2': (128, 128),
    'model3': (128, 128),
    'hybrid': (128, 128)  # Default size for hybrid (optional override below)
}

# LIME Image Explainer setup
explainer = lime_image.LimeImageExplainer()

# Image classification with model choice
def classify_with_model(image_path, model_path, input_size):
    try:
        print(f"[INFO] Loading model from: {model_path}")
        img = Image.open(image_path).convert('RGB')
        img = img.resize(input_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        model = tf.keras.models.load_model(model_path)
        probs = model.predict(img_array)

        if probs.shape[1] != len(class_dictionary):
            raise ValueError(f"Model output shape {probs.shape} does not match class dictionary ({len(class_dictionary)} classes).")

        pred = np.argmax(probs, axis=1)[0]
        conf = probs[0, pred]
        return class_dictionary[pred], conf, img_array[0]

    except Exception as e:
        print(f"[ERROR] classify_with_model failed: {e}")
        return None, None, None

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
        start_time = time.time()
        
        if 'blood_image' not in request.files or 'model_name' not in request.form:
            flash('Missing file or model selection.')
            return redirect(url_for('index'))

        file = request.files['blood_image']
        model_name = request.form['model_name']
        
        if file.filename == '':
            flash('No file selected to upload.')
            return redirect(url_for('index'))

        sec_filename = secure_filename(file.filename)
        file_extension = os.path.splitext(sec_filename)[1].lower()

        if file and file_extension in ALLOWED_FILETYPES:
            file_tempname = uuid.uuid4().hex
            image_path = './uploads/' + file_tempname + file_extension
            file.save(image_path)

            # Classify image with selected model
            input_size = model_input_sizes.get(model_name, (128, 128))  # Default fallback

            if model_name != 'hybrid':
                model_path = model_map.get(model_name)
                input_size = model_input_sizes.get(model_name, (128, 128))
                label, prediction_probability, img_array = classify_with_model(image_path, model_path, input_size)
            else:
                # Hybrid: average prediction
                predictions = []
                probs = []
                for key, path in model_map.items():
                    input_size = model_input_sizes.get(model_name, (128, 128))
                    label_i, conf_i, _ = classify_with_model(image_path, path, input_size)
                    if label_i is not None:
                        predictions.append(label_i)
                        probs.append(conf_i)
    
                if predictions:
                    # Pick the label that appeared most often
                    label = max(set(predictions), key=predictions.count)
                    prediction_probability = round(np.mean(probs), 4)
                    img_array = image.img_to_array(Image.open(image_path).resize((IMG_HEIGHT, IMG_WIDTH))) / 255.0
                else:
                    label, prediction_probability, img_array = None, None, None
            
            prediction_probability = np.around(prediction_probability, decimals=4)

            #lime_overlay = generate_lime_overlay(img_array)
            orig_image = Image.open(image_path)
            image_data = get_image_thumbnail(orig_image)

            #lime_overlay_image = Image.fromarray((lime_overlay * 255).astype(np.uint8))
            #with BytesIO() as buffer:
            #    lime_overlay_image.save(buffer, 'PNG')
            #    lime_overlay_base64 = base64.b64encode(buffer.getvalue()).decode()

            time_submitted = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            time_taken = round(time.time() - start_time, 2)

            result_row = {
                'time_submitted': time_submitted,
                'time_taken': time_taken,
                'image': image_data,
                'model_used': model_name,
                'label': label,
                'confidence': prediction_probability
            }

            analysis_summaries.append(result_row)
            # Sort summaries by time submitted (latest first)
            analysis_summaries.sort(key=lambda x: x['time_submitted'], reverse=True)

            return render_template('index.html', label=label, prob=prediction_probability,
                       image=image_data,
                       result_table=analysis_summaries)
        else:
            flash(f"Invalid file type: {file_extension}. Only .jpg, .jpeg, .gif, .png, .bmp are allowed.")
            return redirect(url_for('index'))

    return render_template('index.html',result_table=analysis_summaries)


@application.route('/explain', methods=['POST'])
def explain_lime():
    try:
        if not analysis_summaries:
            return jsonify({'success': False, 'error': 'No prediction available to explain.'})

        last_entry = analysis_summaries[0]
        image_data = last_entry['image']

        image_bytes = base64.b64decode(image_data)
        image_pil = Image.open(BytesIO(image_bytes)).resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = image.img_to_array(image_pil) / 255.0

        lime_overlay = generate_lime_overlay(img_array)
        lime_overlay_image = Image.fromarray((lime_overlay * 255).astype(np.uint8))

        with BytesIO() as buffer:
            lime_overlay_image.save(buffer, 'PNG')
            lime_overlay_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({'success': True, 'lime_overlay': lime_overlay_base64})

    except Exception as e:
        print(f"[ERROR] explain_lime: {e}")
        return jsonify({'success': False, 'error': str(e)})

# Handle 'filesize too large' errors
@application.errorhandler(413)
def http_413(e):
    print("[Error] Uploaded file too large.")
    flash('Uploaded file too large.')
    return redirect(url_for('index'))

def download_models_if_missing():
    import requests

    base_github_url = "https://github.com/manas-p-pandey/Fingerprint_BGD_Models/raw/refs/heads/main/"
    model_filenames = [
        'My_low_acc_model_BGD.keras',
        'My_high_acc_model_BGD.keras',
        'My_custom_resnet_model_BGD.keras'
    ]

    os.makedirs('models', exist_ok=True)

    for filename in model_filenames:
        local_path = os.path.join('models', filename)
        if not os.path.exists(local_path):
            url = base_github_url + filename
            print(f"[INFO] Downloading {filename} from {url}")
            response = requests.get(url)
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"[INFO] Saved to {local_path}")
            else:
                print(f"[ERROR] Failed to download {filename}: HTTP {response.status_code}")


# Run the Flask application
if __name__ == "__main__":
    download_models_if_missing()
    model_path = 'models/My_custom_resnet_model_BGD.keras'
    model = tf.keras.models.load_model(model_path)
    application.run(host='0.0.0.0', port=5000, debug=True)
