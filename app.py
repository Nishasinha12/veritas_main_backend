from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS 
import os

# Configure Gemini API
genai.configure(api_key="AIzaSyA-NpCrHLxZ0kdhDCfXb9PEUsJsG-3TCJU")


# Flask app
app = Flask(__name__)
CORS(app)

# Load models
flash_model = genai.GenerativeModel("models/gemini-2.5-flash")
pro_model = genai.GenerativeModel("models/gemini-2.5-pro")


# ---- Health check ----
@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Aegis AI Backend"})

# ---- Summarization ----
@app.route("/api/v1/summarize", methods=["POST"])
def summarize():
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        prompt = f"Summarize in 3-4 sentences:\n\n{text}"
        response = flash_model.generate_content(prompt)

        return jsonify({"summary": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Sentiment ----
@app.route("/api/v1/sentiment", methods=["POST"])
def sentiment():
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        prompt = f"Classify sentiment as Positive, Negative, or Neutral:\n\n{text}"
        response = flash_model.generate_content(prompt)

        return jsonify({"sentiment": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Misinformation ----
@app.route("/api/v1/misinfo", methods=["POST"])
def misinfo():
    try:
        text = request.json.get("text", "")
        if not text:
            return jsonify({"error": "No text provided"}), 400

        prompt = f"""
        Detect misinformation in the following text.
        Classify as:
        - Likely True
        - Possibly Misinformation
        - Needs Verification

        Text: {text}
        """
        response = pro_model.generate_content(prompt)

        return jsonify({"misinfo_analysis": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Image Detection ----
@app.route("/api/v1/detect/image", methods=["POST"])
def detect_image():
    try:
        print("===== Debugging Image Upload =====")
        print("request.files:", request.files)
        print("request.form:", request.form)
        print("request.content_type:", request.content_type)
        print("=================================")

        # Accept either key 'image' or 'file'
        file_key = "image" if "image" in request.files else "file" if "file" in request.files else None

        if not file_key:
            return jsonify({"error": "No image file provided"}), 400

        image = request.files[file_key]
        contents = image.read()

        # Dummy AI processing (replace with your pro_model call)
        response = {
            "result": "AI-generated",  # or "Real"
            "confidence": 0.95
        }

        return jsonify({"output": response, "filename": image.filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---- Audio Detection ----
@app.route("/api/v1/detect/audio", methods=["POST"])
def detect_audio():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        contents = file.read()

        response = pro_model.generate_content([
            "Classify this audio as AI-generated or Real human speech. "
            "Respond only in JSON with keys: result (AI-generated or Real), confidence (0-1).",
            {"mime_type": file.mimetype, "data": contents}
        ])

        return jsonify({"output": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Video Detection ----
@app.route("/api/v1/detect/video", methods=["POST"])
def detect_video():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        contents = file.read()

        response = pro_model.generate_content([
            "Classify this video as either AI-generated or Real. "
            "Respond only in JSON with keys: result (AI-generated or Real), confidence (0-1).",
            {"mime_type": file.mimetype, "data": contents}
        ])

        return jsonify({"output": response.text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---- Run app ----
if __name__ == "__main__":
    models = genai.list_models()
    for m in models:
        print(m.name, m.supported_generation_methods)

    # IMPORTANT: host=0.0.0.0 so other machines can reach it
    app.run(host="0.0.0.0", port=8080, debug=True)
