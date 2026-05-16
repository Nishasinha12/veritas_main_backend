from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import sys

app = Flask(__name__)

MODEL_PATH = os.path.join("models", "deepaudio.h5")

print(f"🚀 Attempting to load model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Audio model loaded successfully.")
    
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load model.")
    print(f"Error details: {e}")
    sys.exit(1)


@app.route('/predict/audio', methods=['POST'])
def predict_audio():
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'No input data provided'}), 400
            
        input_data = np.array(data['input'], dtype=np.float32)
        print(f"Input shape: {input_data.shape}")
        
        prediction = model.predict(input_data)
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    print("📢 Starting Flask server on 127.0.0.1:5001...") 
    app.run(host='127.0.0.1', port=5001, debug=False)