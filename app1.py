# backend/app1.py
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))

from news_client import query_news   # match your actual filename
from flask import Flask, request, jsonify
import google.generativeai as genai
from flask_cors import CORS 
import os
from PIL import Image # <--- ADD THIS IMPORT
import traceback # <--- ADD THIS IMPORT
from verify_api.entrypoint import verify_claim_logic 
# --- All Necessary Imports ---
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 
import secrets
import smtplib
from email.message import EmailMessage
import joblib
import pandas as pd
import re
import requests
from datetime import datetime,timedelta
from scipy.sparse import hstack
import psycopg2
from psycopg2.extras import RealDictCursor

from snowflake import SnowflakeGenerator
from werkzeug.security import generate_password_hash, check_password_hash
# import os # Already imported above
# import sqlite3 # Already imported above
# import secrets # Already imported above
#import tensorflow as tf
#from tensorflow.keras.models import load_model
import librosa
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import json  

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ---------------------------------------------------------
# ✅ Ensure Database Exists
# ---------------------------------------------------------
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL)
    return conn

app = Flask(__name__)
CORS(app)

def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            user_id BIGINT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            reset_token TEXT,
            reset_token_expiry TEXT
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database initialized")

init_db()

# ---------------------------------------------------------------------------
# Microservice URLs (set in .env / Render dashboard)
# ---------------------------------------------------------------------------
AUDIO_SERVICE_URL = os.environ.get("AUDIO_SERVICE_URL", "http://localhost:5001")
IMAGE_SERVICE_URL = os.environ.get("IMAGE_SERVICE_URL", "http://localhost:5002")

# Initialize the Snowflake ID Generator
snowflake_gen = SnowflakeGenerator(1)
# -----------------------------
# Load Model & Preprocessors
# -----------------------------
print(f"🔑 GNEWS KEY: {os.getenv('GNEWS_API_KEY')}")  # confirm it loads


#print("🚀 Loading model and transformers...")
#MODEL_DIR = "./models"
#model = joblib.load(os.path.join(MODEL_DIR, 'frontend_model.pkl'))
#vectorizer = joblib.load(os.path.join(MODEL_DIR, 'frontend_vectorizer.pkl'))
#scaler = joblib.load(os.path.join(MODEL_DIR, 'frontend_scaler.pkl'))
#numeric_features_list = joblib.load(os.path.join(MODEL_DIR, 'frontend_numeric_features_list.pkl'))
#print("✅ Model and transformers loaded successfully.")

# --- LOAD YOUR AUDIO MODEL ---
#print("🚀 Loading audio deepfake model...")
#try:
  ##  if os.path.exists(audio_model_path):
  ##      print(f"✅ Audio model '{audio_model_path}' loaded successfully.")
    #else:
      #   audio_model = None
      #   print(f"⚠️ WARNING: Audio model 'deepaudio.h5' not found at {audio_model_path}.")
#except Exception as e:
  #   audio_model = None
  #   print(f"❌ ERROR loading audio model 'deepaudio.h5': {e}") '''
# -----------------------------


# -----------------------------
# Prediction Helper
# -----------------------------
#def make_prediction(tweet_text, created_at_str):
    # This function is unchanged
#    data = {'text': [tweet_text], 'created_at': [pd.to_datetime(created_at_str)]}
#    df = pd.DataFrame(data)
    # ... (feature creation logic is unchanged) ...
#    df['text_str'] = df['text'].astype(str)
#    df['word_count'] = df['text_str'].apply(lambda x: len(x.split()))
#    df['capital_count'] = df['text_str'].apply(lambda x: len(re.findall(r'[A-Z]', x)))
#   df['digit_count'] = df['text_str'].apply(lambda x: len(re.findall(r'[0-9]', x)))
#    df['hashtag_count'] = df['text_str'].apply(lambda x: len(re.findall(r'#', x)))
#   df['url_count'] = df['text_str'].apply(lambda x: len(re.findall(r'http[s]?://', x)))
#   df['mention_count'] = df['text_str'].apply(lambda x: len(re.findall(r'@', x)))
#   df['exclamation_count'] = df['text_str'].apply(lambda x: len(re.findall(r'!', x)))
#    df['question_count'] = df['text_str'].apply(lambda x: len(re.findall(r'\?', x)))
#   df['hour_of_day'] = df['created_at'].dt.hour
#   df['day_of_week'] = df['created_at'].dt.dayofweek
#   X_text_features = vectorizer.transform(df['text'])
#   df_numeric = df[numeric_features_list]
#   X_numeric_features = scaler.transform(df_numeric)
#   X_final = hstack([X_text_features, X_numeric_features])
#   prediction = model.predict(X_final)
#   proba = model.predict_proba(X_final)
#   result = "Human" if prediction[0] == 1 else "Bot"
#   confidence = proba[0][prediction[0]]
#   return result, confidence


flash_model = genai.GenerativeModel("models/gemini-2.5-flash")
pro_model = genai.GenerativeModel("models/gemini-2.5-pro")


# --- REPLICATION OF YOUR NOTEBOOK'S PREPROCESSING ---

def extract_features_from_audio(file_stream, sr=22050):
    """
    Extracts the 26 features from an audio stream based on your notebook's dataset.
    """
    try:
        y, sr = librosa.load(file_stream, sr=sr)
        
        feature_vector = [
            np.mean(librosa.feature.chroma_stft(y=y, sr=sr)),
            np.mean(librosa.feature.rms(y=y)),
            np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
            np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
            np.mean(librosa.feature.zero_crossing_rate(y=y)),
            *np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1)
        ]
        return np.array(feature_vector)
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        return None

def prepare_data_for_prediction(features, window_size=5):
    """
    This is the 'prepare_data' function from your notebook, adapted for a single prediction.
    It creates the sliding windows required by the LSTM model.
    """
    windows = []
    # This loop now exactly matches the notebook's logic to produce 21 windows
    for j in range(len(features) - window_size):
        window = features[j : j + window_size]
        windows.append(window)
    
    # Reshape for the model: (1, num_windows, window_size) -> (1, 21, 5)
    return np.array(windows)[np.newaxis, ...]


#send_email to email for otp
def send_email(recipient_email, subject, body):
    SMTP_SERVER = 'smtp.gmail.com'
    SMTP_PORT = 587
    EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

    if not EMAIL_ADDRESS or not EMAIL_PASSWORD:
        print("ERROR: Email credentials are not set in the backend .env file.")
        return

    msg = EmailMessage()
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = recipient_email
    msg.set_content(body)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print(f"Email sent successfully to {recipient_email}")
    except Exception as e:
        print(f"Failed to send email: {e}")
# -----------------------------
# API Routes
# -----------------------------

@app.route('/')
def home():
    return {"message": "Flask API running successfully"}
# --- Predict (AI Bot Detection) ---
#@app.route('/predict', methods=['POST'])
#def predict():
#   data = request.get_json()
#   tweet_text = data.get('text', '')
#   if not tweet_text.strip():
#       return jsonify({'error': 'Text cannot be empty'}), 400
#   now = datetime.now().isoformat()
#   result, confidence = make_prediction(tweet_text, now)
#   return jsonify({'prediction': result, 'confidence': float(confidence)})

# --- Register with reCAPTCHA ---
@app.route('/register', methods=['POST'])
def register_user():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    captcha_token = data.get('captcha_token')

    secret_key = os.getenv("RECAPTCHA_SECRET_KEY")
    #make sure this matches your reCAPTCHA site pair

    # --- Validate captcha ---
    if not captcha_token:
        return jsonify({'status': 'error', 'message': 'CAPTCHA token is missing.'}), 400

    verification_url = "https://www.google.com/recaptcha/api/siteverify"
    payload = {'secret': secret_key, 'response': captcha_token}
    response = requests.post(verification_url, data=payload)
    result = response.json()
    print(f"Google reCAPTCHA response: {result}")

    if not result.get('success'):
        return jsonify({'status': 'error', 'message': 'CAPTCHA verification failed. Please try again.'}), 400

    # --- Validate inputs ---
    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password are required'}), 400

    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({'status': 'error', 'message': 'Invalid email format'}), 400

    # --- Hash password ---
    password_hash = generate_password_hash(password)
    new_user_id = next(snowflake_gen)
    conn = None
    # --- Save to DB ---
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (user_id, email, password_hash) VALUES (%s, %s, %s)',
            (new_user_id, email, password_hash)
        )
        conn.commit()
        return jsonify({'status': 'success', 'message': 'Account created successfully!', 'user_id': new_user_id}), 201
    except psycopg2.IntegrityError:
        return jsonify({'status': 'error', 'message': 'This email address is already registered'}), 409
    finally:
        if conn:
            conn.close()


# --- Login ---
@app.route('/login', methods=['POST'])
def login_user():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'status': 'error', 'message': 'Email and password are required'}), 400

    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_id, password_hash FROM users WHERE email = %s', (email,))
        user_record = cursor.fetchone()

        if user_record and check_password_hash(user_record[1], password):
            user_id = user_record[0]
            return jsonify({
                'status': 'success',
                'message': 'Login successful!',
                'user_id': user_id
            }), 200
        else:
            return jsonify({'status': 'error', 'message': 'Invalid email or password'}), 401

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'status': 'error', 'message': 'An internal error occurred'}), 500
    finally:
        if conn:
            conn.close()


# --- Forgot Password Step 1: Request OTP ---
@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get('email')
    if not email:
        return jsonify({'status': 'error', 'message': 'Email is required'}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT user_id FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    
    if user:
        otp = str(secrets.randbelow(9000) + 1000)
        hashed_otp = generate_password_hash(otp)
        expiry_time = datetime.now() + timedelta(minutes=10)
        cursor.execute(
            'UPDATE users SET reset_token = %s, reset_token_expiry = %s WHERE email = %s',
            (hashed_otp, expiry_time.isoformat(), email)
        )
        conn.commit()
        send_email(email, "Your Veritas OTP", f"Your verification code is: {otp}")
        print(f"DEV INFO: OTP for {email} is {otp}")
    
    cursor.close()
    conn.close()
    return jsonify({'status': 'success', 'message': 'If an account with that email exists, an OTP has been sent.'}), 200

# --- Forgot Password Step 2: Verify OTP ---
@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email')
    otp = data.get('otp')
    if not email or not otp:
        return jsonify({'status': 'error', 'message': 'Email and OTP are required'}), 400
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT reset_token, reset_token_expiry FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()
    if not user or not user[0]:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Invalid request or OTP already used.'}), 400
    expiry_time = datetime.fromisoformat(user[1])
    if datetime.now() > expiry_time:
        conn.close()
        return jsonify({'status': 'error', 'message': 'OTP has expired.'}), 410
    if check_password_hash(user[0], otp):
        conn.close()
        return jsonify({'status': 'success', 'message': 'OTP verified successfully.'}), 200
    else:
        conn.close()
        return jsonify({'status': 'error', 'message': 'Invalid OTP.'}), 400

# --- Forgot Password Step 3: Reset Password ---
@app.route('/reset-password', methods=['POST'])
def reset_password():
    data = request.get_json()
    email = data.get('email')
    new_password = data.get('newPassword')
    if not email or not new_password:
        return jsonify({'status': 'error', 'message': 'Email and new password are required'}), 400
    new_password_hash = generate_password_hash(new_password)
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET password_hash = %s, reset_token = NULL, reset_token_expiry = NULL WHERE email = %s', 
               (new_password_hash, email))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success', 'message': 'Password has been reset successfully.'}), 200

# ---- Health check ----
@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "Aegis AI Backend"})

# ---- Summarization ----
@app.route("/api/v1/summarize", methods=["POST"])
@app.route("/api/v1/summarise", methods=["POST"])
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

        prompt = f"""Analyze the following text for potential misinformation.

Text: {text}

Respond with ONLY a JSON object, no markdown, no explanation outside the JSON.
Use exactly these keys: verdict, explanation, confidence

Example:
{{"verdict": "Likely True", "explanation": "The claim is supported by...", "confidence": 0.85}}

Verdicts must be one of: "Likely True", "Likely False", "Misleading", "Unverifiable"
"""

        response = flash_model.generate_content(prompt)
        raw = response.text.strip()
        
        # Debug — remove this once working
        print(f"RAW GEMINI RESPONSE: {repr(raw)}")

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        import json as json_mod
        parsed = json_mod.loads(raw)
        return jsonify(parsed)

    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}, raw was: {repr(raw)}")
        return jsonify({"error": "Failed to parse AI response", "raw": raw}), 500
    except Exception as e:
        print(f"Misinfo error: {e}")
        return jsonify({"error": str(e)}), 500
    
# ---- News Verification ----
@app.route("/api/v1/news", methods=["POST"])
def news_check():
    try:
        claim = request.json.get("claim", "") or request.json.get("text", "")
        if not claim:
            return jsonify({"error": "No claim/text provided"}), 400

        articles = query_news(claim)
        
        if not articles:
            return jsonify({
                "claim": claim,
                "articles": [],
                "message": "No related articles found."
            })

        # Optional: calculate average credibility score
        avg_credibility = sum(a["credibility_score"] for a in articles) / len(articles)

        return jsonify({
            "claim": claim,
            "total_articles": len(articles),
            "average_credibility": round(avg_credibility, 2),
            "articles": articles[:10]  # limit to top 10 to avoid huge responses
        })

    except Exception as e:
        print(f"News check error: {e}")
        return jsonify({"error": str(e)}), 500
# ---- Image Detection ----
from PIL import Image as PILImage
import traceback

@app.route("/api/v1/detect/image", methods=["POST"])
def detect_image():
    try:
        file_key = "image" if "image" in request.files else "file" if "file" in request.files else None
        if not file_key:
            return jsonify({"error": "No image file provided"}), 400

        image_file = request.files[file_key]
        img = PILImage.open(image_file.stream).convert("RGB")
        img = img.resize((256, 256))
        image_array = np.array(img, dtype=np.float32) / 255.0

        response = requests.post(
            f"{IMAGE_SERVICE_URL}/predict/image",
            json={"input": image_array.tolist()}
        )
        response.raise_for_status()
        result = response.json()
        print(f"🔍 TF service response keys: {list(result.keys())}")
        print(f"🔍 heatmap present: {'heatmap_base64' in result}")
        print(f"🔍 heatmap length: {len(result.get('heatmap_base64', ''))}")

        response_data = {
            "prediction": result.get("prediction", "Unknown"),
            "confidence": result.get("confidence", None)
        }

        # Forward Grad-CAM heatmap if the TF service produced one
        if "heatmap_base64" in result:
            response_data["heatmap_base64"] = result["heatmap_base64"]

        return jsonify(response_data)
    
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Internal server error during image detection."}), 500
#----------------------------------------------------------------------------------------------------------


# --- Audio Detection Route ---

def get_tf_prediction(input_data):
    try:
        response = requests.post(f"{AUDIO_SERVICE_URL}/predict/audio", json={"input": input_data})
        return response.json()
    
    except Exception as e:
        print(f"❌ An unexpected error occurred during audio prediction: {e}")
        traceback.print_exc() # traceback is now defined
        return jsonify({"error": f"Internal server error: {e}"}), 500
    
# --- ADD THIS NEW ENDPOINT TO YOUR app.py FILE ---

@app.route("/api/v1/detect/audio", methods=["POST"])
def detect_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files["file"]

    try:
        # ✅ Forward file directly to audio microservice
        response = requests.post(
            f"{AUDIO_SERVICE_URL}/predict/audio",
            files={"file": (file.filename, file.stream, file.mimetype)}
        )
        response.raise_for_status()
        result = response.json()

        return jsonify({
            "prediction": result["prediction"],   # ✅ already processed
            "confidence": result["confidence"]    # ✅ already processed
        })

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Audio service unavailable"}), 503
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


#NEWS--------------------------------------------------------------------------------------------------
@app.route("/verify", methods=["POST"])
def verify_news():
    claim = request.form.get("claim", "")
    image_file = request.files.get("image")
    audio_file = request.files.get("audio")

    # Debug
    print("CLAIM:", claim)
    print("IMAGE:", image_file)
    print("AUDIO:", audio_file)

    result = verify_claim_logic(claim)  # Currently only text
    # Later we add image/video verification here

    return jsonify(result)

# --- News Headlines (GET) ---
@app.route("/api/v1/news", methods=["GET"])
def get_news():
    try:
        GNEWS_KEY = os.environ.get('GNEWS_API_KEY')  # ← headlines
        response = requests.get(
            f"https://gnews.io/api/v4/top-headlines?category=general&lang=en&country=in&max=10&apikey={GNEWS_KEY}"
        )
        data = response.json()
        articles = [
            {
                "title": a.get("title"),
                "urlToImage": a.get("image"),
                "source": {"name": a.get("source", {}).get("name", "")},
                "url": a.get("url"),
            }
            for a in data.get("articles", [])
        ]
        return jsonify({"articles": articles})
    except Exception as e:
        return jsonify({"error": str(e), "articles": []}), 500
#-----------------------------------------------------------------------------------------------------------
# ---- Run app ----
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Veritas backend server on port {port}...")
    app.run(host='0.0.0.0', port=port)
