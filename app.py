import os
import requests
from dotenv import load_dotenv
load_dotenv()

from flask import Flask, redirect, render_template, request, session
from PIL import Image
from werkzeug.utils import secure_filename
import torchvision.transforms.functional as TF
import CNN
import numpy as np
import torch
import pandas as pd
from flask_mail import Mail, Message

# ================= MODEL DOWNLOAD =================

MODEL_PATH = "plant_disease_model_1_latest.pt"
MODEL_URL = "https://huggingface.co/pranalipimple/plant-disease-model/resolve/main/plant_disease_model_1_latest.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("⬇ Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded successfully")
    else:
        print("✅ Model already exists")

download_model()

# ================= FLASK CONFIG =================

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config["SESSION_PERMANENT"] = False

app.config['MAIL_SERVER'] = os.getenv("MAIL_SERVER", "")
app.config['MAIL_PORT'] = int(os.getenv("MAIL_PORT", "587"))
app.config['MAIL_USE_TLS'] = os.getenv("MAIL_USE_TLS", "False") == "True"
app.config['MAIL_USERNAME'] = os.getenv("MAIL_USERNAME", "")
app.config['MAIL_PASSWORD'] = os.getenv("MAIL_PASSWORD", "")


mail = Mail(app)

# ================= DATA =================

disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# ================= MODEL LOAD =================

model = CNN.CNN(39)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ================= PREDICTION =================

def prediction(image_path):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((224, 224))
    input_data = TF.to_tensor(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_data)

    return torch.argmax(output).item()

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        if 'image' not in request.files:
            return "ERROR: No image part"

        image = request.files['image']

        if image.filename == "":
            return "ERROR: No file selected"

        filename = secure_filename(image.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        image.save(file_path)

        pred = prediction(file_path)

        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]

        user_email = session.get("user_email")

        print("\n Inside /submit route")
        print(" Session email:", user_email)

        if user_email:
            print("Preparing to send email...")
            print("MAIL_SERVER:", app.config['MAIL_SERVER'])
            print("MAIL_USERNAME:", app.config['MAIL_USERNAME'])

            msg = Message(
                subject="Your Plant Health Report 🌿",
                sender="pranalipimple12@gmail.com",
                recipients=[user_email]
            )

            msg.body = f"""
Hello!

Here is your Plant Health Report:

🌱 Description:
{description}

 Prevention Steps:
{prevent}

 Supplement:
{supplement_name}
Buy Link: {supplement_buy_link}

Thank you for using Plant Disease Detector!
"""

            try:
                mail.send(msg)
                print(" Email sent successfully!")
            except Exception as e:
                print(" EMAIL ERROR:", e)
        else:
            print(" No user_email in session — not sending email")

        return render_template(
            'submit.html',
            title=title,
            desc=description,
            prevent=prevent,
            image_url=image_url,
            pred=pred,
            sname=supplement_name,
            simage=supplement_image_url,
            buy_link=supplement_buy_link
        )

@app.route('/market')
def market():
    return render_template(
        'market.html',
        supplement_image=list(supplement_info['supplement image']),
        supplement_name=list(supplement_info['supplement name']),
        disease=list(disease_info['disease_name']),
        buy=list(supplement_info['buy link'])
    )


@app.route('/save_email', methods=['POST'])
def save_email():
    data = request.get_json()
    email = data.get("email")

    print(" Email received from frontend:", email)

    if email:
        session["user_email"] = email
        print(" Email saved in session:", session["user_email"])
        return {"status": "success"}

    print(" No email received")
    return {"status": "error", "message": "No email provided"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)