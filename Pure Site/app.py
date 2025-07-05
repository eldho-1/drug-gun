from flask import Flask, request, render_template, redirect, url_for
import os
import pandas as pd
import joblib
from model import prepare_training_data, train_models, predict_sample

# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------------------
# Training Models Once at Startup
# ------------------------------
print("🔬 Preparing and training models...")

drug_files = {
    'cocaine': 'cocaine.csv',
    'heroin': 'heroin.csv',
    'methadone': 'methadone.csv',
    'morphine': 'morphine.csv',
    'meth': 'meth.csv'
}

non_drug_files = [
    'sucrose.csv', 'lactic.csv', 'citric.csv', 'urea.csv', 'ethanol.csv'
]

# Prepare training data and train models
features_df = prepare_training_data(drug_files, non_drug_files)
binary_model, multiclass_model, le = train_models(features_df)

print("✅ Models trained and ready.")

# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No file selected"

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Predict using the uploaded file
    result = predict_sample(file_path, binary_model, multiclass_model, le)

    return render_template('output.html', result=result, filename=file.filename)

# ------------------------------
# Run the App
# ------------------------------
if __name__ == '__main__':
    app.run(debug=True)
