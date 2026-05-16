from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
import sys
import base64
import cv2

app = Flask(__name__)

# --- MODEL LOADING LOGIC ---
MODEL_PATH = os.path.join("models", "deepfake_image_model.h5")

print(f"🚀 Attempting to load image model from: {MODEL_PATH}")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Image deepfake model loaded successfully.")
    print("📋 Model layers (last 10):")
    for layer in model.layers[-10:]:
        print(f"   {layer.name} -> {type(layer).__name__}")
except Exception as e:
    print(f"❌ FATAL ERROR: Could not load image model. Check path and files.")
    print(f"Error details: {e}")
    sys.exit(1)


def get_last_conv_layer(model):
    """Use a mid-level conv layer for better spatial Grad-CAM coverage."""
    # Try to find block6 or block5 level in EfficientNet for good resolution
    preferred = ['block6a_expand_conv', 'block5a_expand_conv', 'block4a_expand_conv']
    layer_names = [l.name for l in model.layers]
    
    for name in preferred:
        if name in layer_names:
            return name
    
    # Fallback: find a conv layer that's NOT the last one (better spatial size)
    conv_layers = [l.name for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    if len(conv_layers) >= 2:
        return conv_layers[-len(conv_layers)//2]  # pick middle conv layer
    
    return conv_layers[-1]


def generate_gradcam_heatmap(img_array, model):
    """
    Generate a Grad-CAM heatmap.
    img_array: numpy array shape (1, 256, 256, 3), normalized 0-1
    Returns: base64-encoded PNG string of heatmap overlaid on original image.
    """
    try:
        last_conv_layer_name = get_last_conv_layer(model)
        print(f"🔍 Using layer for Grad-CAM: {last_conv_layer_name}")

        # Build grad model: inputs -> [last_conv_output, final_output]
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[
                model.get_layer(last_conv_layer_name).output,
                model.output
            ]
        )

        # Convert to tf.Tensor so GradientTape tracks it
        img_tensor = tf.cast(img_array, tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            outputs = grad_model(img_tensor, training=False)
            conv_outputs = outputs[0]
            predictions = outputs[1]
            predictions = tf.convert_to_tensor(predictions)
            loss = predictions[:, 0]

        # Gradients of loss w.r.t. conv layer output
        grads = tape.gradient(loss, conv_outputs)  # (1, h, w, filters)

        # Pool gradients over spatial dims
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (filters,)

        # Weight conv output channels by pooled grads
        conv_outputs = conv_outputs[0]  # (h, w, filters)
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (h, w, 1)
        heatmap = tf.squeeze(heatmap)  # (h, w)

        # ReLU + normalize
        heatmap = tf.nn.relu(heatmap).numpy()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        # Resize to 256x256
        heatmap_resized = cv2.resize(heatmap, (256, 256))

        # Apply JET colormap
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)  # BGR

        # Original image as uint8 BGR
        original_rgb = np.uint8(255 * img_array[0])
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)

        # Overlay: 60% original + 40% heatmap
        superimposed = cv2.addWeighted(original_bgr, 0.6, heatmap_colored, 0.4, 0)

        # Encode to base64 PNG
        _, buffer = cv2.imencode(".png", superimposed)
        heatmap_b64 = base64.b64encode(buffer).decode("utf-8")
        print("✅ Grad-CAM heatmap generated successfully.")
        return heatmap_b64

    except Exception as e:
        print(f"⚠️ Grad-CAM generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


@app.route('/predict/image', methods=['POST'])
def predict_image():
    try:
        data = request.get_json()
        input_data = np.array(data['input'], dtype=np.float32)

        # Add batch dimension if needed: (256, 256, 3) -> (1, 256, 256, 3)
        if input_data.ndim == 3:
            input_data = np.expand_dims(input_data, axis=0)

        # Run prediction
        prediction = model.predict(input_data)
        raw_score = float(prediction[0][0])

        if raw_score > 0.5:
            label = "Real"
            confidence = raw_score
        else:
            label = "Fake"
            confidence = 1.0 - raw_score

        # Generate Grad-CAM heatmap
        heatmap_b64 = generate_gradcam_heatmap(input_data, model)

        response_data = {
            "prediction": label,
            "confidence": round(confidence, 4)
        }

        if heatmap_b64:
            response_data["heatmap_base64"] = heatmap_b64

        return jsonify(response_data)

    except Exception as e:
        print(f"❌ Image prediction error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


if __name__ == '__main__':
    print("📢 Starting Image Deepfake Detection service on 127.0.0.1:5002...")
    app.run(host='127.0.0.1', port=5002, debug=False)