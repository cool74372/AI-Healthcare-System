from flask import Flask, render_template, request, session, redirect, url_for, flash, jsonify
import datetime
import sqlite3
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import requests
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os
import uuid
import shutil
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
import mimetypes

load_dotenv() # Load variables from .env

# ✅ CORRECT GRADCAM IMPORTS
from gradcam_tb import make_gradcam_heatmap as make_gradcam_heatmap_tb, overlay_heatmap as overlay_heatmap_tb
from gradcam_xray import make_gradcam_heatmap, overlay_heatmap

from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib import colors

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")
TB_HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "tb_heatmaps")

REPORT_FOLDER = os.path.join(BASE_DIR, "static", "reports")

app = Flask(__name__)
app.secret_key = "very-secure-healthcare-secret" # Essential for sessions!

# ─────────────────────────────────────────
# DATABASE AUTHENTICATION FUNCTIONS
# ─────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "users.db")

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Migrate: add created_at to users table if it doesn't exist yet
        try:
            cursor.execute("ALTER TABLE users ADD COLUMN created_at TEXT")
            cursor.execute("UPDATE users SET created_at = datetime('now') WHERE created_at IS NULL")
        except Exception:
            pass  # Column already exists — safe to ignore
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS admins (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                user_name TEXT,
                disease TEXT NOT NULL,
                result TEXT NOT NULL,
                severity TEXT,
                confidence REAL,
                report_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS email_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_email TEXT NOT NULL,
                report_path TEXT,
                status TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        # Seed default admin if not exists
        existing = cursor.execute("SELECT id FROM admins WHERE email = ?", ("admin@healthcare.com",)).fetchone()
        if not existing:
            hashed = generate_password_hash("Admin@123", method='pbkdf2:sha256')
            cursor.execute("INSERT INTO admins (name, email, password) VALUES (?, ?, ?)",
                          ("Admin", "admin@healthcare.com", hashed))
            print("[ADMIN] Seeded admin account: admin@healthcare.com / Admin@123")
        conn.commit()

init_db()

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash("Please log in to access this page.", "error")
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────

vitals_model = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vitals_model.pkl'), 'rb'))
vitals_columns = pickle.load(open(os.path.join(BASE_DIR, 'models', 'vitals_columns.pkl'), 'rb'))

# ✅ UPDATED MODEL
cnn_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'pneumonia_mobilenet_best.keras'))
tb_model = tf.keras.models.load_model(os.path.join(BASE_DIR, 'models', 'tb_model.h5'))

# Warm-up (important for Grad-CAM)
dummy = np.zeros((1, 224, 224, 3))
cnn_model.predict(dummy)
tb_model.predict(dummy)

# ─────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/vitals')
@login_required
def vitals():
    return render_template('vitals.html')

@app.route('/xray')
@login_required
def xray():
    return render_template('xray.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/tb')
@login_required
def tb():
    return render_template('tb.html')

# ─────────────────────────────────────────
# AUTHENTICATION ROUTES
# ─────────────────────────────────────────
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
                conn.commit()
            flash("Registration successful! Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email address already registered.", "error")
            return redirect(url_for('register'))
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        with get_db() as conn:
            user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            
        if user and check_password_hash(user['password'], password):
            session['user_id'] = user['id']
            session['user_name'] = user['name']
            flash(f"Welcome back, {user['name']}!", "success")
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password.", "error")
            return redirect(url_for('login'))
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("You have been logged out.", "success")
    return redirect(url_for('home'))

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────

API_KEY = "AIzaSyDYKjzO0T7P-idClq2Fc8f7sbfEBPzaHKs"

def generate_pdf_report(filename, disease, result, confidence, insight, original_img_path=None, gradcam_img_path=None):
    os.makedirs(REPORT_FOLDER, exist_ok=True)
    report_filename = f"report_{filename}.pdf"
    path = os.path.join(REPORT_FOLDER, report_filename)
    
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    Story = []
    
    Story.append(Paragraph(f"Clinical AI Report: {disease}", styles['Title']))
    Story.append(Spacer(1, 0.2 * inch))
    Story.append(Paragraph(f"<b>Result:</b> {result}", styles['Heading2']))
    Story.append(Paragraph(f"<b>AI Confidence:</b> {confidence}%", styles['Normal']))
    Story.append(Spacer(1, 0.3 * inch))
    
    # ADD IMAGES SIDE BY SIDE
    if original_img_path and gradcam_img_path:
        try:
            img1 = Image(original_img_path, width=2.5*inch, height=2.5*inch)
            img2 = Image(gradcam_img_path, width=2.5*inch, height=2.5*inch)
            
            # Wrapping images in a table to align them horizontally
            data = [[img1, img2],
                    [Paragraph("<b>Original</b>", styles['Normal']), Paragraph("<b>AI Focus (Grad-CAM)</b>", styles['Normal'])]]
            
            t = Table(data, colWidths=[3*inch, 3*inch])
            t.setStyle(TableStyle([
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ]))
            Story.append(t)
            Story.append(Spacer(1, 0.3 * inch))
        except Exception as e:
            print("Could not add images to PDF:", e)

    Story.append(Paragraph("AI Insight Detail", styles['Heading2']))
    
    if isinstance(insight, dict):
        Story.append(Paragraph(f"<b>Severity:</b> {insight.get('severity', 'N/A')}", styles['Normal']))
        Story.append(Paragraph("<br/>", styles['Normal']))
        Story.append(Paragraph(f"<b>Impression:</b> {insight.get('impression', 'N/A')}", styles['Normal']))
        Story.append(Paragraph("<br/>", styles['Normal']))
        Story.append(Paragraph(f"<b>Key Observations:</b> {insight.get('key_observations', 'N/A')}", styles['Normal']))
        Story.append(Paragraph("<br/>", styles['Normal']))
        Story.append(Paragraph(f"<b>Clinical Interpretation:</b> {insight.get('clinical_interpretation', 'N/A')}", styles['Normal']))
        Story.append(Paragraph("<br/>", styles['Normal']))
        Story.append(Paragraph(f"<b>Recommendation:</b> {insight.get('recommendation', 'N/A')}", styles['Normal']))
    else:
        Story.append(Paragraph(str(insight), styles['Normal']))
        
    Story.append(Paragraph("<br/><br/><i>Disclaimer: This report is generated by Artificial Intelligence and is intended to assist medical professionals, not replace them. Always consult an expert medical practitioner for a verified diagnosis and treatment plan.</i>", styles['Normal']))

    doc.build(Story)
    return f"/static/reports/{report_filename}"

import json

def get_fallback_insight(result, disease):
    detected = 'DETECTED' in result.upper()
    
    if "Tuberculosis" in disease or "TB" in disease:
        if detected:
            return {
                "impression": "Findings suggest possible Tuberculosis infection.",
                "severity": "Severe",
                "key_observations": "Common symptoms include cough, fever, chest pain, and night sweats.",
                "clinical_interpretation": "The model indicates abnormal lung patterns consistent with active TB.",
                "recommendation": "Consult a doctor immediately to undergo confirmatory testing (sputum culture) and begin antibiotic therapy if confirmed."
            }
        else:
            return {
                "impression": "No clear signs of Tuberculosis detected.",
                "severity": "Mild",
                "key_observations": "Lung structures appear generally clear without prominent cavitations typical of TB.",
                "clinical_interpretation": "The scan does not exhibit standard features of Tuberculosis infection.",
                "recommendation": "Routine monitoring as needed. Seek consultation if symptoms persist."
            }
            
    elif "Pneumonia" in disease:
        if detected:
            return {
                "impression": "Findings indicate the presence of Pneumonia.",
                "severity": "Moderate",
                "key_observations": "Areas of increased opacity and consolidation in the lung fields.",
                "clinical_interpretation": "The model detects opacities typically associated with bacterial or viral pneumonia.",
                "recommendation": "Seek medical attention for an accurate diagnosis and possible antibiotic or antiviral treatment."
            }
        else:
            return {
                "impression": "Lungs appear clear of Pneumonia.",
                "severity": "Mild",
                "key_observations": "Normal lung aeration and clear lung margins.",
                "clinical_interpretation": "The model does not detect consolidation indicative of Pneumonia.",
                "recommendation": "Maintain regular health checkups. No immediate treatment for pneumonia required."
            }
            
    else:
        # Cardiovascular / Vitals Fallback
        if detected:
            return {
                "impression": "Vitals suggest a potential cardiovascular issue.",
                "severity": "Severe",
                "key_observations": "Anomalies in entered vital parameters (e.g., BP, cholesterol, heart rate).",
                "clinical_interpretation": "The profile strongly correlates with elevated risk for cardiovascular disease.",
                "recommendation": "Schedule an appointment with a cardiologist promptly for a comprehensive evaluation."
            }
        else:
            return {
                "impression": "Vitals are within a generally normal cardiovascular profile.",
                "severity": "Mild",
                "key_observations": "Parameters do not align with high-risk cardiovascular patterns.",
                "clinical_interpretation": "The entered health data suggests low immediate risk of heart disease.",
                "recommendation": "Maintain a healthy lifestyle and continue annual physical checkups."
            }

def generate_ai_insight(result, confidence, disease):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={API_KEY}"
    headers = {
        "Content-Type": "application/json"
    }

    prompt = f"""
    Disease Focus: {disease}
    Diagnostic Result: {result}
    Model Confidence: {confidence}%

    You are an expert AI radiologist/physician analyzing a diagnostic result.
    Provide a structured clinical report strictly based on the above result.
    Respond ONLY with a valid JSON object. Do not wrap in markdown (e.g., no ```json).
    The JSON must contain EXACTLY the following keys:
    - "impression": A short, 1-sentence overall impression.
    - "severity": Must be exactly one of: "Mild", "Moderate", or "Severe". (If Normal, use "Mild").
    - "key_observations": 1-2 sentences on what is typically clinically observed for this result.
    - "clinical_interpretation": Medical explanation of the condition in plain language.
    - "recommendation": Actionable next steps or advice for the patient.
    """

    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    for attempt in range(2):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=12)
            if response.status_code == 200:
                response_json = response.json()
                text_response = response_json["candidates"][0]["content"]["parts"][0]["text"]
                
                clean_json = text_response.strip()
                if clean_json.startswith("```json"): clean_json = clean_json[7:]
                if clean_json.startswith("```"): clean_json = clean_json[3:]
                if clean_json.endswith("```"): clean_json = clean_json[:-3]
                
                parsed_json = json.loads(clean_json.strip())
                
                required_keys = ["impression", "severity", "key_observations", "clinical_interpretation", "recommendation"]
                if all(k in parsed_json for k in required_keys):
                    return parsed_json
            else:
                print(f"API Attempt {attempt+1} failed with status {response.status_code}")
                
        except Exception as e:
            print(f"AI insight generating attempt {attempt+1} Exception:", str(e))
            
    print("AI insight generation repeatedly failed. Engaging rule-based fallback mechanism.")
    return get_fallback_insight(result, disease)


def preprocess_image(filepath):
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)


def save_upload(file):
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ext = os.path.splitext(file.filename)[1]
    filename = str(uuid.uuid4()) + ext
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    return filename, path


# ─────────────────────────────────────────
# GRAD-CAM FUNCTIONS
# ─────────────────────────────────────────

def run_gradcam_xray(img_array, model, filename):
    os.makedirs(HEATMAP_FOLDER, exist_ok=True)

    save_path = os.path.join(HEATMAP_FOLDER, "gradcam_" + filename)

    try:
        heatmap = make_gradcam_heatmap(img_array, model)

        if heatmap is not None:
            overlay_heatmap(heatmap, f"{UPLOAD_FOLDER}/{filename}", save_path)
            label = "AI Focus (Grad-CAM)"
        else:
            shutil.copy(f"{UPLOAD_FOLDER}/{filename}", save_path)
            label = "Low activation"

        return save_path, label

    except Exception as e:
        print("GradCAM Error:", e)
        shutil.copy(f"{UPLOAD_FOLDER}/{filename}", save_path)
        return save_path, "Grad-CAM error"


def run_gradcam_tb(img_array, model, filename):
    os.makedirs(TB_HEATMAP_FOLDER, exist_ok=True)

    save_path = os.path.join(TB_HEATMAP_FOLDER, "gradcam_" + filename)

    try:
        heatmap = make_gradcam_heatmap_tb(img_array, model)

        if heatmap is not None:
            overlay_heatmap_tb(heatmap, f"{UPLOAD_FOLDER}/{filename}", save_path)
            label = "AI Focus (Grad-CAM)"
        else:
            shutil.copy(f"{UPLOAD_FOLDER}/{filename}", save_path)
            label = "Low activation"

        return save_path, label

    except Exception as e:
        print("TB GradCAM Error:", e)
        shutil.copy(f"{UPLOAD_FOLDER}/{filename}", save_path)
        return save_path, "Grad-CAM error"

# ─────────────────────────────────────────
# PREDICTIONS
# ─────────────────────────────────────────

@app.route('/predict_xray', methods=['POST'])
@login_required
def predict_xray():

    file = request.files.get('file')
    if not file:
        return "No file"

    filename, filepath = save_upload(file)
    img_array = preprocess_image(filepath)

    prediction = cnn_model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "🫁 PNEUMONIA DETECTED ❗"
        confidence = round(float(prediction) * 100, 2)
    else:
        result = "NORMAL ✅"
        confidence = round((1 - float(prediction)) * 100, 2)

    insight = generate_ai_insight(result, confidence, "Pneumonia")
    
    gradcam_path, gradcam_label = run_gradcam_xray(img_array, cnn_model, filename)
    
    report_path = generate_pdf_report(filename, "Pneumonia", result, confidence, insight, filepath, gradcam_path)

    # Log report
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO reports (user_id, user_name, disease, result, severity, confidence, report_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.get('user_id'), session.get('user_name'), 'Pneumonia', result, insight.get('severity','N/A'), confidence, report_path))
            conn.commit()
    except Exception as e:
        print("Report logging error:", e)

    return render_template(
        'result.html',
        result=result,
        confidence=confidence,
        image_path=f"/static/uploads/{filename}",
        gradcam_path=f"/static/heatmaps/gradcam_{filename}",
        gradcam_label=gradcam_label,
        insight=insight,
        report_path=report_path
    )


@app.route('/predict_tb', methods=['POST'])
@login_required
def predict_tb():

    file = request.files.get('file')
    if not file:
        return "No file"

    filename, filepath = save_upload(file)
    img_array = preprocess_image(filepath)

    prediction = tb_model.predict(img_array)[0][0]

    if prediction > 0.5:
        result = "🦠 TB DETECTED ❗"
        confidence = round(float(prediction) * 100, 2)
    else:
        result = "NORMAL ✅"
        confidence = round((1 - float(prediction)) * 100, 2)

    insight = generate_ai_insight(result, confidence, "Tuberculosis")
    
    gradcam_path, gradcam_label = run_gradcam_tb(img_array, tb_model, filename)
    
    report_path = generate_pdf_report(filename, "Tuberculosis", result, confidence, insight, filepath, gradcam_path)

    # Log report
    try:
        with get_db() as conn:
            conn.execute("INSERT INTO reports (user_id, user_name, disease, result, severity, confidence, report_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.get('user_id'), session.get('user_name'), 'Tuberculosis', result, insight.get('severity','N/A'), confidence, report_path))
            conn.commit()
    except Exception as e:
        print("Report logging error:", e)

    return render_template(
        'result.html',
        result=result,
        confidence=confidence,
        image_path=f"/static/uploads/{filename}",
        gradcam_path=f"/static/tb_heatmaps/gradcam_{filename}",
        gradcam_label=gradcam_label,
        insight=insight,
        report_path=report_path
    )


@app.route('/predict_vitals', methods=['POST'])
@login_required
def predict_vitals():
    try:
        age_val = float(request.form.get('Age', 0))
        sex_val = float(request.form.get('Sex', 0))
        cpt = float(request.form.get('ChestPainType', 0))
        trestbps = float(request.form.get('RestingBP', 0))
        chol = float(request.form.get('Cholesterol', 0))
        fbs = float(request.form.get('FastingBS', 0))
        restecg = float(request.form.get('RestingECG', 0))
        thalach = float(request.form.get('MaxHR', 0))
        exang = float(request.form.get('ExerciseAngina', 0))
        oldpeak = float(request.form.get('Oldpeak', 0))

        input_data = np.array([[age_val, sex_val, cpt, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]])
        
        df = pd.DataFrame(input_data, columns=vitals_columns)
        
        prediction = vitals_model.predict(df)[0]
        
        if prediction == 1:
            result = "⚠️ CARDIOVASCULAR ISSUE DETECTED ❗"
            confidence = 88.0
            if hasattr(vitals_model, "predict_proba"):
                confidence = round(float(vitals_model.predict_proba(df)[0][1]) * 100, 2)
        else:
            result = "NORMAL ✅"
            confidence = 88.0
            if hasattr(vitals_model, "predict_proba"):
                confidence = round(float(vitals_model.predict_proba(df)[0][0]) * 100, 2)

        insight = generate_ai_insight(result, confidence, "Cardiovascular Disease / Heart Condition")
        
        report_path = generate_pdf_report(f"vitals_{uuid.uuid4().hex[:8]}", "Cardiovascular Disease", result, confidence, insight)

        # Log report
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO reports (user_id, user_name, disease, result, severity, confidence, report_path) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session.get('user_id'), session.get('user_name'), 'Cardiovascular Disease', result, insight.get('severity','N/A'), confidence, report_path))
                conn.commit()
        except Exception as log_err:
            print("Report logging error:", log_err)

        return render_template(
            'result.html',
            result=result,
            confidence=confidence,
            image_path=None,
            gradcam_path=None,
            gradcam_label=None,
            insight=insight,
            report_path=report_path
        )
    except Exception as e:
        print("Vitals Prediction Error:", e)
        return f"An error occurred: {e}"


# ─────────────────────────────────────────
# EMAIL FEATURE
# ─────────────────────────────────────────

def send_report_email(user_email, report_path):
    sender_email = os.getenv("MAIL_USERNAME")
    sender_password = os.getenv("MAIL_PASSWORD")
    
    if not sender_email or not sender_password:
        return False, "Email credentials not configured."
        
    msg = EmailMessage()
    msg['Subject'] = "Your Clinical AI Report"
    msg['From'] = f"Clinical AI <{sender_email}>"
    msg['To'] = user_email
    msg.set_content("Hello from the Clinical AI System,\n\nPlease find your generated report attached.\n\nBest regards,\nYour AI Diagnostics Team")
    
    # Attach the file
    abs_report_path = os.path.join(BASE_DIR, report_path.lstrip('/'))
    if os.path.exists(abs_report_path):
        ctype, encoding = mimetypes.guess_type(abs_report_path)
        if ctype is None or encoding is not None:
            ctype = 'application/octet-stream'
        maintype, subtype = ctype.split('/', 1)
        with open(abs_report_path, 'rb') as f:
            msg.add_attachment(f.read(), maintype=maintype, subtype=subtype, filename=os.path.basename(abs_report_path))
    else:
        return False, "Report file not found."
    
    try:
        # Assuming Gmail setup
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(sender_email, sender_password)
            smtp.send_message(msg)
        return True, "Email sent successfully!"
    except Exception as e:
        print("Email sending failed:", e)
        return False, str(e)

@app.route('/send_email', methods=['POST'])
@login_required
def send_email():
    report_path = request.form.get('report_path')
    if not report_path:
        return jsonify({"success": False, "message": "No report path provided."}), 400
        
    user_id = session.get('user_id')
    with get_db() as conn:
        user = conn.execute("SELECT email FROM users WHERE id = ?", (user_id,)).fetchone()
        
    if user and user['email']:
        success, message = send_report_email(user['email'], report_path)
        # Log email
        try:
            with get_db() as conn:
                conn.execute("INSERT INTO email_logs (user_email, report_path, status, message) VALUES (?, ?, ?, ?)",
                    (user['email'], report_path, 'sent' if success else 'failed', message))
                conn.commit()
        except Exception as log_err:
            print("Email log error:", log_err)
        return jsonify({"success": success, "message": message}), 200 if success else 500
    else:
        return jsonify({"success": False, "message": "User email not found."}), 404

# ─────────────────────────────────────────
# ADMIN BLUEPRINT
# ─────────────────────────────────────────
from admin import admin_bp
app.register_blueprint(admin_bp, url_prefix='/admin')

# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=True)