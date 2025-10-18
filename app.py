from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

# Load the saved model
model = load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Assuming the input image or data is sent as a flat list under 'inputs'
    inputs = np.array(data['inputs'])
    
    # Reshape inputs if needed as per your model's requirement
    # Example for a single sample:
    # inputs = inputs.reshape(1, -1)
    
    prediction = model.predict(inputs)
    
    # Convert prediction to list to return as JSON
    result = prediction.tolist()
    
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
