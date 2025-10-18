import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load the model (ensure model.h5 is in your project folder)
model = load_model('plant_disease_model (1).keras')




# Define image input size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Enable dynamic port binding (Render provides port via environment variable)
port = int(os.environ.get('PORT', 5000))

@app.route('/')
def home():
    return "Flask ML API is running. Use /predict endpoint."

@app.route('/predict', methods=['POST'])
def predict():
    # Expect JSON with 'inputs' as list, or image file for image prediction
    data = request.get_json(force=True)
    
    # Example: expecting 'inputs' key with list of flattened image data
    if 'inputs' in data:
        inputs = np.array(data['inputs'])
        # Reshape if needed (assuming model expects a specific shape)
        # Example: inputs = inputs.reshape(1, IMG_HEIGHT, IMG_WIDTH, 3)
        prediction = model.predict(inputs)
        return jsonify({'prediction': prediction.tolist()})
    else:
        return jsonify({'error': 'No inputs provided'}), 400

@app.route('/predict_image', methods=['POST'])
def predict_image():
    # Suppose the client uploads an image file
    if 'image' not in request.files:
        return jsonify({'error': 'No image file uploaded'}), 400
    file = request.files['image']
    img = Image.open(file).convert('RGB')
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return jsonify({'prediction': prediction.tolist()})

import os
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))




