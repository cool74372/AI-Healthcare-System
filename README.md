# 🏥 Clinical AI Healthcare Diagnostic System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Framework-Flask-lightgrey.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow-orange.svg)](https://www.tensorflow.org/)

A professional-grade medical diagnostic platform that leverages Deep Learning and Explainable AI (XAI) to assist healthcare professionals in early disease detection and vital sign analysis.

---

## 🌟 Key Features

### 🩺 Advanced AI Diagnostics
- **Pneumonia & TB Detection**: Uses MobileNet-based CNN architectures for high-accuracy X-ray classification.
- **Cardiovascular Risk Assessment**: LSTM-based processing of vital signs (Resting BP, Cholesterol, Max HR, etc.) to predict heart risks.
- **Automated Preprocessing**: Images are automatically resized and normalized for consistent model inference.

### 🔍 Explainable AI (Grad-CAM Visualization)
- **Visual Evidence**: The system generates **Grad-CAM Heatmaps** that highlight specific regions in X-rays (e.g., lung opacities) contributing to the AI's decision.
- **Clinical Transparency**: Helps medical practitioners verify AI findings by visually localizing anomalies.

### 📄 Clinical Intelligence & Reporting
- **Generative AI Insights**: Integrated with **Google Gemini API** to provide structured clinical reports including medical impressions, severity assessment, and actionable recommendations.
- **Rule-Based Fallback**: A robust fallback mechanism ensures clinical insights are available even if API limits are exceeded.
- **Professional PDF Reports**: Generates downloadable clinical reports featuring side-by-side original and Grad-CAM visualizations.

### 🛡️ Enterprise-Grade Admin Dashboard
- **Data Analytics**: Real-time charts showing disease breakdown and system activity over time.
- **Reporting Logs**: Detailed audit trail of every diagnostic report generated.
- **User Management**: Searchable database of registered users.
- **Communication Monitoring**: Tracking system for sent/failed email reports.

### 📧 Email Integration
- Securely send generated diagnostic reports directly to users' inboxes using SMTP integration.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Modern CSS3 (Glassmorphism), JavaScript |
| **Backend** | Flask (Python), SQLite3 |
| **Deep Learning** | TensorFlow, Keras, OpenCV, NumPy, Pandas |
| **AI Insights** | Google Gemini API (GenAI) |
| **Reporting** | ReportLab (PDF), EmailMessage (SMTP) |

---

## 📁 Project Structure

```text
AI Healthcare System/
├── app.py                 # Core Flask Application Logic
├── admin.py               # Admin Dashboard Blueprint
├── gradcam_xray.py        # Grad-CAM XAI Logic (Pneumonia)
├── gradcam_tb.py          # Grad-CAM XAI Logic (Tuberculosis)
├── models/                # Trained CNN & LSTM Model Files
├── static/                # Uploads, Heatmaps, and User Reports
├── templates/             # UI Templates (Main & Admin)
├── users.db               # SQLite Database (Reports, Logs, Users)
└── .env                   # Configuration for Secrets
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- [Google Gemini API Key](https://aistudio.google.com/)

### Installation

1. **Clone the Repository**
   ```bash
   git clone <your-repo-link>
   cd AI-Healthcare-System
   ```

2. **Set Up Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Configuration**
   Create a `.env` file in the root directory:
   ```env
   # Gemini API Key
   API_KEY=your_gemini_api_key_here

   # Email Credentials (Optional for Email Reporting)
   MAIL_USERNAME=your_email@gmail.com
   MAIL_PASSWORD=your_app_password
   ```

5. **Run the Application**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your browser.

---

## 🔒 Security & Disclaimer
- **HIPAA/Privacy**: Ensure all sample data is anonymized before use.
- **Disclaimer**: This system is a **Clinical Decision Support Tool** designed to assist, not replace, professional medical judgment. Always consult a qualified physician for final diagnosis and treatment.

---

## 👨‍💻 Author Details
**[Allikanti Saikiran / PWDA22A16]**  
**[Gaikwad Pranay / PWDA22A16]**
**[G. Sridhar Goud / PWDA22A16]**
*Major Project — 2026*
