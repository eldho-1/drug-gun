from flask import Flask, request, render_template, redirect, url_for, send_file
import os
import pandas as pd
import joblib
import numpy as np
from model import prepare_training_data, train_models, predict_sample
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt
import io
import base64
from xhtml2pdf import pisa
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Drug and non-drug peak data
DRUGS = {
    "heroin": [1745, 1245, 1035, 950, 820],
    "morphine": [3400, 1320, 1250, 930],
    "cocaine": [1705, 1275, 1100, 860],
    "meth": [2960, 2925, 1490, 1380],
    "methadone": [1715, 1285, 1125, 980]
}
NON_DRUGS = {
    "sucrose": [3500, 2900, 1640, 1410],
    "citric": [3500, 2950, 1700, 1400],
    "ethanol": [3400, 2900, 1650, 1450],
    "lactic": [3500, 2950, 1720, 1450],
    "none": []
}
wavenumbers = np.linspace(800, 5500, 1000)

# Train models on startup
print("🔬 Training models...")
drug_files = {
    'cocaine': 'cocaine.csv',
    'heroin': 'heroin.csv',
    'methadone': 'methadone.csv',
    'morphine': 'morphine.csv',
    'meth': 'meth.csv'
}
non_drug_files = ['sucrose.csv', 'lactic.csv', 'citric.csv', 'urea.csv', 'ethanol.csv']
features_df = prepare_training_data(drug_files, non_drug_files)
binary_model, multiclass_model, le = train_models(features_df)
print("✅ Models ready.")

# Utility Functions
def generate_spectrum(peaks, weight=1.0):
    spec = np.zeros_like(wavenumbers)
    for peak in peaks:
        spec += weight * np.exp(-0.5 * ((wavenumbers - peak) / (0.03 * peak))**2)
    return spec

def create_mixture(drug, dw, agent, aw):
    d_spec = generate_spectrum(DRUGS[drug], dw / 100)
    a_spec = generate_spectrum(NON_DRUGS[agent], aw / 100)
    noise = 0.01 * np.random.normal(size=len(wavenumbers))
    return savgol_filter(d_spec + a_spec + noise, 11, 3)

def detect_peaks(spectrum, drug):
    peaks, _ = find_peaks(spectrum, height=0.1, prominence=0.05)
    matched = []
    for target in DRUGS[drug]:
        idx = np.argmin(np.abs(wavenumbers[peaks] - target))
        peak_wv = wavenumbers[peaks][idx]
        if abs(peak_wv - target) < 15:
            matched.append((target, peak_wv, spectrum[peaks][idx]))
    return matched

def plot_spectrum(spectrum, peaks, drug):
    plt.figure(figsize=(10, 5))
    plt.plot(wavenumbers, spectrum, label='Mixture Spectrum', color='blue')
    for target, found, intensity in peaks:
        plt.plot(found, intensity, 'ro')
        plt.text(found, intensity + 0.02, f'{found:.0f}', ha='center', fontsize=8)
    for ref in DRUGS[drug]:
        plt.axvline(ref, color='gray', linestyle='--', alpha=0.3)
    plt.gca().invert_xaxis()
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.title(f"Spectrum of Mixture with {drug}")
    plt.grid(True)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_pdf(html_content):
    pdf_buffer = io.BytesIO()
    pisa_status = pisa.CreatePDF(io.StringIO(html_content), dest=pdf_buffer)
    if pisa_status.err:
        return None
    pdf_buffer.seek(0)
    return pdf_buffer

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pure')
def pure():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file uploaded"
    file = request.files['file']
    if file.filename == '':
        return "No file selected"
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    result = predict_sample(path, binary_model, multiclass_model, le)
    try:
        df = pd.read_csv(path)
        result['wavenumber_list'] = df['wavenumber'].tolist()
        result['absorbance_list'] = df['absorbance'].tolist()
    except:
        result['wavenumber_list'] = []
        result['absorbance_list'] = []
    return render_template('output.html', result=result, filename=file.filename)

@app.route('/mixture', methods=['GET', 'POST'])
def mixture():
    if request.method == 'POST':
        drug = request.form['drug']
        agent = request.form['agent']
        dw = float(request.form['drug_weight'])
        aw = float(request.form['agent_weight'])

        if dw + aw > 100:
            return render_template('mixture.html', error="Total weight cannot exceed 100",
                                   drugs=DRUGS.keys(), agents=NON_DRUGS.keys())

        spec = create_mixture(drug, dw, agent, aw)
        peaks = detect_peaks(spec, drug)
        match = round(100 * len(peaks) / len(DRUGS[drug]), 1)
        plot_data = plot_spectrum(spec, peaks, drug)
        return render_template('mixture_result.html', plot_data=plot_data, match=match,
                               peaks=peaks, drug=drug, agent=agent,
                               dw=dw, aw=aw)
    return render_template('mixture.html', drugs=DRUGS.keys(), agents=NON_DRUGS.keys())

# PDF download for pure compound report
@app.route('/download_report_output/<filename>')
def download_report_output(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    result = predict_sample(path, binary_model, multiclass_model, le)

    try:
        df = pd.read_csv(path)
        spectrum = df['absorbance'].values
        wavenumbers = df['wavenumber'].values
        matched_peaks = detect_peaks(spectrum, result['predicted_class'])
        result['matched_peaks'] = matched_peaks
        # Generate plot specifically for the PDF
        plt.figure(figsize=(10, 5))
        plt.plot(wavenumbers, spectrum, label='Spectrum', color='blue')
        for target, found, intensity in matched_peaks:
            plt.plot(found, intensity, 'ro')
            plt.text(found, intensity + 0.02, f'{found:.0f}', ha='center', fontsize=8)
        for ref in DRUGS.get(result['predicted_class'], []):
            plt.axvline(ref, color='gray', linestyle='--', alpha=0.3)
        plt.gca().invert_xaxis()
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.title(f"Spectrum of {result['predicted_class']}")
        plt.grid(True)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        plt.close()
        buf.seek(0)
        result['plot_data'] = base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Error generating plot: {e}")
        result['matched_peaks'] = []
        result['plot_data'] = ''

    html = render_template("report_output_pdf.html", result=result, now=datetime.now())  # <-- Add now here
    pdf_file = generate_pdf(html)
    if not pdf_file:
        return "PDF generation failed", 500
    return send_file(pdf_file, download_name="drug_report.pdf", as_attachment=True)
# PDF download for simulated mixture report
@app.route('/download_report_mixture/<drug>/<agent>/<dw>/<aw>/<match>')
def download_report_mixture(drug, agent, dw, aw, match):
    dw = float(dw)
    aw = float(aw)
    spec = create_mixture(drug, dw, agent, aw)
    peaks = detect_peaks(spec, drug)
    plot_data = plot_spectrum(spec, peaks, drug)
    html = render_template("report_mixture_pdf.html", drug=drug, agent=agent, dw=dw, aw=aw,
                           match=match, peaks=peaks, plot_data=plot_data)
    pdf_file = generate_pdf(html)
    if not pdf_file:
        return "PDF generation failed", 500
    return send_file(pdf_file, download_name="mixture_report.pdf", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
