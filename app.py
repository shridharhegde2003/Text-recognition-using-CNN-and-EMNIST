from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
try:
    model = tf.keras.models.load_model('htr_model.h5')
except (IOError, ImportError) as e:
    print("Error loading model: htr_model.h5 not found or corrupted.")
    model = None

# Define the labels (must match the training labels from the notebook)
labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

import cv2
import numpy as np
from PIL import Image


def preprocess_image(image_file):
    """Preprocess the uploaded image so the model sees a clean, centred, white-on-black glyph.

    Steps:
    1.  Read and convert to grayscale.
    2.  Apply Gaussian blur to reduce sensor noise.
    3.  Apply Otsu threshold to obtain a clean binary mask irrespective of lighting / background.
    4.  Extract the largest contour (assumed to be the character) and crop tightly with a small margin.
    5.  Resize to 28×28 keeping aspect ratio (padding with black where needed).
    6.  Invert so the final glyph is white on black – matches EMNIST training set.
    7.  Normalize to [0,1] and reshape to (1,28,28,1).
    """
    # 1-2. Read to OpenCV (numpy array) and blur
    img_cv = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img_cv is None:
        # Fallback to PIL if OpenCV fails (rare for small uploads)
        image_file.seek(0)
        img_cv = cv2.cvtColor(np.array(Image.open(image_file).convert('L')), cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_cv, (5, 5), 0)

    # 3. Otsu threshold (automatically handles uneven lighting)
    _, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 4. Find largest contour
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        # add small margin
        pad = 4
        x, y = max(x - pad, 0), max(y - pad, 0)
        w, h = w + 2 * pad, h + 2 * pad
        roi = thresh[y:y + h, x:x + w]
    else:
        roi = thresh  # fall back to whole image

    # 5. Resize keeping aspect ratio and padding
    h_roi, w_roi = roi.shape
    scale = 20.0 / max(h_roi, w_roi)  # leave margin inside 28x28
    resized = cv2.resize(roi, (int(w_roi * scale), int(h_roi * scale)), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - resized.shape[1]) // 2
    y_offset = (28 - resized.shape[0]) // 2
    canvas[y_offset:y_offset + resized.shape[0], x_offset:x_offset + resized.shape[1]] = resized

    # 6-7. Normalize and reshape
    img_array = canvas.astype('float32') / 255.0  # already white-on-black because of THRESH_BINARY_INV
    img_array = img_array.reshape(1, 28, 28, 1)
    return img_array

@app.route('/', methods=['GET'])
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request."""
    if model is None:
        return jsonify({'error': 'Model is not loaded.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    try:
        processed_image = preprocess_image(file)
        prediction_probs = model.predict(processed_image)
        confidence = np.max(prediction_probs) * 100
        predicted_class_index = np.argmax(prediction_probs)
        predicted_character = labels[predicted_class_index]
        return jsonify({
            'prediction': predicted_character,
            'confidence': f'{confidence:.2f}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
