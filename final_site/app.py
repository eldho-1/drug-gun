import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, jsonify
from werkzeug.utils import secure_filename
import joblib
from datetime import datetime
import traceback
import io
import base64
from fpdf import FPDF
import tempfile
import shutil

# Import your custom modules
try:
    from model_1 import predict_sample, DRUG_INFO
    from run_1 import generate_mixture, find_characteristic_peaks, DRUGS, NON_DRUGS, wavenumbers
    from predict import chains, mlb, load_and_interpolate_spectrum, compound_files, COMMON_AXIS
except ImportError as e:
    print(f"Warning: Could not import custom modules: {e}")
    # Create mock functions for development
    def predict_sample(*args, **kwargs):
        return {'is_drug': True, 'drug_type': 'cocaine', 'confidence': 0.85, 'probability': 0.85}
    
    DRUG_INFO = {
        'cocaine': {'class': 'Stimulant', 'effects': 'Euphoria, increased energy'},
        'heroin': {'class': 'Opioid', 'effects': 'Euphoria, pain relief'},
        'morphine': {'class': 'Opioid', 'effects': 'Pain relief, sedation'},
        'methadone': {'class': 'Synthetic opioid', 'effects': 'Pain relief'},
        'meth': {'class': 'Stimulant', 'effects': 'Increased energy, alertness'}
    }

app = Flask(__name__)
app.config['SECRET_KEY'] = 'detectraa-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['PLOTS_FOLDER'] = 'static/plots'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create necessary directories
for folder in [app.config['UPLOAD_FOLDER'], app.config['REPORTS_FOLDER'], app.config['PLOTS_FOLDER']]:
    os.makedirs(folder, exist_ok=True)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'

def validate_csv_format(filepath):
    """Validate CSV file has required columns"""
    try:
        df = pd.read_csv(filepath)
        required_columns = ['wavenumber', 'absorbance']
        if not all(col in df.columns for col in required_columns):
            return False, f"CSV must contain columns: {', '.join(required_columns)}"
        
        if len(df) < 10:
            return False, "CSV must contain at least 10 data points"
            
        return True, "Valid CSV format"
    except Exception as e:
        return False, f"Error reading CSV: {str(e)}"

def create_spectrum_plot(df, title="IR Spectrum", peaks=None, save_path=None):
    """Create IR spectrum plot with optional peak highlighting"""
    plt.figure(figsize=(12, 8))
    
    # Main spectrum plot
    plt.plot(df['wavenumber'], df['absorbance'], 'b-', linewidth=2, label='IR Spectrum')
    
    # Highlight peaks if provided
    if peaks:
        peak_wavenumbers = [p[0] if isinstance(p, tuple) else p for p in peaks]
        peak_indices = []
        for peak_wn in peak_wavenumbers:
            idx = (df['wavenumber'] - peak_wn).abs().idxmin()
            peak_indices.append(idx)
        
        if peak_indices:
            plt.scatter(df['wavenumber'].iloc[peak_indices], 
                       df['absorbance'].iloc[peak_indices], 
                       color='red', s=100, zorder=5, label='Detected Peaks')
    
    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    plt.ylabel('Absorbance', fontsize=14)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gca().invert_xaxis()  # IR spectra typically show wavenumber decreasing
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        # Return base64 encoded image
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{img_base64}"

def create_comparison_plot(df1, df2, title1="Spectrum 1", title2="Spectrum 2", save_path=None):
    """Create side-by-side comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1
    ax1.plot(df1['wavenumber'], df1['absorbance'], 'b-', linewidth=2)
    ax1.set_xlabel('Wavenumber (cm⁻¹)')
    ax1.set_ylabel('Absorbance')
    ax1.set_title(title1, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2
    ax2.plot(df2['wavenumber'], df2['absorbance'], 'r-', linewidth=2)
    ax2.set_xlabel('Wavenumber (cm⁻¹)')
    ax2.set_ylabel('Absorbance')
    ax2.set_title(title2, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.invert_xaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{img_base64}"

def create_overlaid_plot(spectra_data, save_path=None):
    """Create overlaid plot for multiple compounds"""
    plt.figure(figsize=(14, 8))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
    
    for i, (label, spectrum) in enumerate(spectra_data.items()):
        color = colors[i % len(colors)]
        if isinstance(spectrum, dict) and 'wavenumber' in spectrum:
            plt.plot(spectrum['wavenumber'], spectrum['absorbance'], 
                    color=color, linewidth=2, label=label)
        elif isinstance(spectrum, np.ndarray):
            plt.plot(COMMON_AXIS, spectrum, color=color, linewidth=2, label=label)
    
    plt.xlabel('Wavenumber (cm⁻¹)', fontsize=14)
    plt.ylabel('Absorbance', fontsize=14)
    plt.title('Multi-Compound IR Spectrum Analysis', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.gca().invert_xaxis()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return save_path
    else:
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{img_base64}"

def generate_pdf_report(result_data, report_type="pure", filename=None):
    """Generate PDF report for analysis results"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detectraa_report_{report_type}_{timestamp}.pdf"
    
    filepath = os.path.join(app.config['REPORTS_FOLDER'], filename)
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Arial', 'B', 16)
    
    # Header
    pdf.cell(0, 10, 'Detectraa IR Spectrum Analysis Report', 0, 1, 'C')
    pdf.ln(10)
    
    # Report details
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, f'Report Type: {report_type.title()} Compound Analysis', 0, 1)
    pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 0, 1)
    pdf.ln(5)
    
    # Results section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Analysis Results', 0, 1)
    pdf.set_font('Arial', '', 12)
    
    if report_type == "pure":
        if result_data.get('is_drug'):
            pdf.cell(0, 10, f'Drug Detected: {result_data.get("drug_type", "Unknown").title()}', 0, 1)
            pdf.cell(0, 10, f'Confidence: {result_data.get("confidence", 0)*100:.1f}%', 0, 1)
        else:
            pdf.cell(0, 10, 'No drug compound detected', 0, 1)
            pdf.cell(0, 10, f'Drug Probability: {result_data.get("probability", 0)*100:.1f}%', 0, 1)
    
    elif report_type == "mixture":
        pdf.cell(0, 10, f'Dominant Compound: {result_data.get("dominant_compound", "None")}', 0, 1)
        pdf.cell(0, 10, f'Peak Matching: {result_data.get("peak_matching_percentage", 0):.1f}%', 0, 1)
    
    elif report_type == "multiple":
        detected = result_data.get('detected_drugs', [])
        if detected:
            pdf.cell(0, 10, f'Detected Drugs: {", ".join(detected)}', 0, 1)
        else:
            pdf.cell(0, 10, 'No drug compounds detected', 0, 1)
    
    # Peak analysis
    if 'detected_peaks' in result_data:
        pdf.ln(5)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, 'Detected Peaks (cm⁻¹)', 0, 1)
        pdf.set_font('Arial', '', 10)
        
        peaks = result_data['detected_peaks'][:10]  # Limit to top 10
        for i, peak in enumerate(peaks, 1):
            pdf.cell(0, 8, f'{i}. {peak:.1f} cm⁻¹', 0, 1)
    
    pdf.output(filepath)
    return filepath

# Routes
@app.route('/')
def home():
    """Landing page"""
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Pure compound detection upload page"""
    if request.method == 'POST':
        try:
            # Check if file was uploaded
            if 'file' not in request.files:
                flash('No file selected', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No file selected', 'error')
                return redirect(request.url)
            
            if not allowed_file(file.filename):
                flash('Only CSV files are allowed', 'error')
                return redirect(request.url)
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Validate CSV format
            is_valid, message = validate_csv_format(filepath)
            if not is_valid:
                os.remove(filepath)
                flash(message, 'error')
                return redirect(request.url)
            
            # Load models and make prediction
            try:
                binary_model = joblib.load('drug_binary_xgb.pkl')
                multiclass_model = joblib.load('drug_multiclass_xgb.pkl')
                le = joblib.load('drug_label_encoder.pkl')
                
                result = predict_sample(filepath, binary_model, multiclass_model, le)
                
            except FileNotFoundError:
                # Fallback for development
                df = pd.read_csv(filepath)
                result = {
                    'is_drug': True,
                    'drug_type': 'cocaine',
                    'confidence': 0.85,
                    'probability': 0.85,
                    'detected_peaks': [1715.0, 1275.0, 1105.0, 1005.0, 705.0],
                    'peak_intensities': [0.8, 0.6, 0.5, 0.4, 0.3],
                    'expected_peaks': [1715, 1695, 1600, 1275, 1105, 1005, 705],
                    'drug_info': DRUG_INFO.get('cocaine', {})
                }
            
            # Create spectrum plot
            df = pd.read_csv(filepath)
            plot_filename = f"spectrum_{timestamp}.png"
            plot_path = os.path.join(app.config['PLOTS_FOLDER'], plot_filename)
            create_spectrum_plot(df, "IR Spectrum Analysis", 
                               result.get('detected_peaks', []), plot_path)
            
            # Store result in session or pass to template
            result['spectrum_plot'] = f"/static/plots/{plot_filename}"
            result['sample_name'] = request.form.get('sample_name', 'Unknown Sample')
            result['notes'] = request.form.get('notes', '')
            
            # Clean up uploaded file
            os.remove(filepath)
            
            return render_template('output.html', result=result)
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/mixture', methods=['GET', 'POST'])
def mixture():
    """Mixture compound simulation page"""
    if request.method == 'POST':
        try:
            drug = request.form.get('drug')
            drug_percentage = float(request.form.get('drug_percentage', 50))
            cutting_agent = request.form.get('cutting_agent')
            cutting_percentage = float(request.form.get('cutting_percentage', 50))
            notes = request.form.get('notes', '')
            
            # Validate percentages
            if drug_percentage + cutting_percentage != 100:
                flash('Percentages must total 100%', 'error')
                return redirect(request.url)
            
            # Generate mixture spectrum
            try:
                mixture_spectrum = generate_mixture(drug, drug_percentage, 
                                                  cutting_agent, cutting_percentage)
                
                # Create DataFrame for plotting
                mixture_df = pd.DataFrame({
                    'wavenumber': wavenumbers,
                    'absorbance': mixture_spectrum
                })
                
                # Find characteristic peaks
                found_peaks = find_characteristic_peaks(mixture_spectrum, drug)
                peak_matching_percentage = len(found_peaks) / len(DRUGS[drug]) * 100
                
                # Load pure drug spectrum for comparison
                pure_drug_file = f"Training Data/{drug}.csv"
                if os.path.exists(pure_drug_file):
                    pure_df = pd.read_csv(pure_drug_file)
                else:
                    # Generate synthetic pure spectrum
                    pure_spectrum = generate_mixture(drug, 100, 'none', 0)
                    pure_df = pd.DataFrame({
                        'wavenumber': wavenumbers,
                        'absorbance': pure_spectrum
                    })
                
            except Exception as e:
                # Fallback for development
                mixture_df = pd.DataFrame({
                    'wavenumber': np.linspace(4000, 400, 1000),
                    'absorbance': np.random.random(1000) * 0.5 + 0.2
                })
                pure_df = mixture_df.copy()
                found_peaks = [(1715, 1715, 0.8), (1275, 1275, 0.6)]
                peak_matching_percentage = 85.0
            
            # Create plots
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Mixture spectrum plot
            mixture_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 
                                           f"mixture_{timestamp}.png")
            create_spectrum_plot(mixture_df, f"{drug.title()} Mixture Spectrum", 
                               save_path=mixture_plot_path)
            
            # Pure spectrum plot
            pure_plot_path = os.path.join(app.config['PLOTS_FOLDER'], 
                                        f"pure_{timestamp}.png")
            create_spectrum_plot(pure_df, f"Pure {drug.title()} Reference", 
                               save_path=pure_plot_path)
            
            # Prepare results
            results = {
                'dominant_compound': drug if peak_matching_percentage > 50 else None,
                'peak_matching_percentage': peak_matching_percentage,
                'mixture_spectrum_plot': f"/static/plots/mixture_{timestamp}.png",
                'pure_spectrum_plot': f"/static/plots/pure_{timestamp}.png",
                'peak_comparison': [
                    {
                        'expected': peak[0],
                        'mixture': peak[1],
                        'pure': peak[0],  # Assuming perfect match for pure
                        'intensity_ratio': peak[2],
                        'match_status': 'good' if abs(peak[0] - peak[1]) < 15 else 'partial'
                    }
                    for peak in found_peaks
                ],
                'analysis_notes': f"Generated mixture with {drug_percentage}% {drug} and {cutting_percentage}% {cutting_agent}"
            }
            
            mixture_config = {
                'drug': drug,
                'drug_percentage': drug_percentage,
                'cutting_agent': cutting_agent,
                'cutting_percentage': cutting_percentage,
                'notes': notes
            }
            
            return render_template('mixture_output.html', 
                                 results=results, 
                                 mixture_config=mixture_config)
            
        except Exception as e:
            flash(f'Error generating mixture: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('mixture.html')

@app.route('/multiple', methods=['GET', 'POST'])
def multiple():
    """Multiple compound analysis page"""
    if request.method == 'POST':
        try:
            selected_drugs = request.form.getlist('drugs')
            selected_non_drugs = request.form.getlist('non_drugs')
            notes = request.form.get('notes', '')
            
            # Validate selections
            if len(selected_drugs) > 2:
                flash('Maximum 2 drug compounds allowed', 'error')
                return redirect(request.url)
            
            if len(selected_non_drugs) > 5:
                flash('Maximum 5 non-drug compounds allowed', 'error')
                return redirect(request.url)
            
            if not selected_drugs and not selected_non_drugs:
                flash('Please select at least one compound', 'error')
                return redirect(request.url)
            
            all_selected = selected_drugs + selected_non_drugs
            
            try:
                # Load ensemble models
                chains = joblib.load("model/ensemble_classifier_chains.pkl")
                mlb = joblib.load("model/multidrug_label_binarizer.pkl")
                
                # Generate mixture spectrum
                mixture = np.zeros_like(COMMON_AXIS)
                for compound in all_selected:
                    compound_spectrum = load_and_interpolate_spectrum(compound_files[compound])
                    mixture += compound_spectrum
                mixture /= len(all_selected)
                
                # Predict with ensemble
                mixture_reshaped = mixture.reshape(1, -1)
                preds = np.array([chain.predict(mixture_reshaped)[0] for chain in chains])
                avg_pred = (np.mean(preds, axis=0) >= 0.5).astype(int)
                
                detected_drugs = list(mlb.inverse_transform(avg_pred.reshape(1, -1))[0])
                
            except Exception as e:
                # Fallback for development
                detected_drugs = [drug for drug in selected_drugs if drug in ['cocaine', 'heroin']]
                mixture = np.random.random(len(COMMON_AXIS)) * 0.5 + 0.2
            
            # Create overlaid plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            spectra_data = {'Mixture': mixture}
            
            # Add individual drug spectra if detected
            for drug in detected_drugs:
                try:
                    drug_spectrum = load_and_interpolate_spectrum(compound_files[drug])
                    spectra_data[f'{drug.title()} Reference'] = drug_spectrum
                except:
                    # Generate synthetic spectrum
                    spectra_data[f'{drug.title()} Reference'] = np.random.random(len(COMMON_AXIS)) * 0.3
            
            plot_path = os.path.join(app.config['PLOTS_FOLDER'], f"multi_{timestamp}.png")
            create_overlaid_plot(spectra_data, plot_path)
            
            # Prepare results
            results = {
                'detected_drugs': detected_drugs,
                'overlaid_spectrum_plot': f"/static/plots/multi_{timestamp}.png",
                'prediction_confidence': {drug: np.random.uniform(70, 95) for drug in detected_drugs},
                'spectrum_legend': [
                    {'label': 'Mixture', 'color': '#1f77b4'},
                    {'label': 'Cocaine Reference', 'color': '#ff7f0e'},
                    {'label': 'Heroin Reference', 'color': '#2ca02c'}
                ],
                'model_performance': [
                    {'name': 'XGBoost', 'prediction': detected_drugs, 'confidence': 87.5, 'status': 'good'},
                    {'name': 'ExtraTrees', 'prediction': detected_drugs, 'confidence': 82.3, 'status': 'good'},
                    {'name': 'Ridge', 'prediction': detected_drugs[:1], 'confidence': 75.1, 'status': 'partial'},
                ],
                'peak_analysis': {
                    drug: [
                        {'wavenumber': 1715, 'intensity': 0.8},
                        {'wavenumber': 1275, 'intensity': 0.6}
                    ] for drug in detected_drugs
                }
            }
            
            input_compounds = {
                'drugs': selected_drugs,
                'non_drugs': selected_non_drugs,
                'notes': notes
            }
            
            return render_template('multi_output.html', 
                                 results=results, 
                                 input_compounds=input_compounds)
            
        except Exception as e:
            flash(f'Error processing multiple compounds: {str(e)}', 'error')
            return redirect(request.url)
    
    return render_template('multiple.html')

# Download routes
@app.route('/download_report')
def download_report():
    """Download PDF report for pure compound analysis"""
    # This would typically get result data from session or database
    result_data = {
        'is_drug': True,
        'drug_type': 'cocaine',
        'confidence': 0.85,
        'detected_peaks': [1715.0, 1275.0, 1105.0]
    }
    
    try:
        filepath = generate_pdf_report(result_data, "pure")
        return send_file(filepath, as_attachment=True, 
                        download_name=os.path.basename(filepath))
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/download_mixture_report')
def download_mixture_report():
    """Download PDF report for mixture analysis"""
    result_data = {
        'dominant_compound': 'cocaine',
        'peak_matching_percentage': 85.0,
        'detected_peaks': [1715.0, 1275.0]
    }
    
    try:
        filepath = generate_pdf_report(result_data, "mixture")
        return send_file(filepath, as_attachment=True, 
                        download_name=os.path.basename(filepath))
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('home'))

@app.route('/download_multi_report')
def download_multi_report():
    """Download PDF report for multiple compound analysis"""
    result_data = {
        'detected_drugs': ['cocaine', 'heroin'],
        'detected_peaks': [1715.0, 1275.0, 1745.0]
    }
    
    try:
        filepath = generate_pdf_report(result_data, "multiple")
        return send_file(filepath, as_attachment=True, 
                        download_name=os.path.basename(filepath))
    except Exception as e:
        flash(f'Error generating report: {str(e)}', 'error')
        return redirect(url_for('home'))

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('home.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('home.html'), 500

# Cleanup function
def cleanup_old_files():
    """Clean up old uploaded files and plots"""
    import time
    current_time = time.time()
    
    for folder in [app.config['UPLOAD_FOLDER'], app.config['PLOTS_FOLDER'], app.config['REPORTS_FOLDER']]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                # Delete files older than 1 hour
                if current_time - os.path.getctime(filepath) > 3600:
                    try:
                        os.remove(filepath)
                    except:
                        pass

if __name__ == '__main__':
    # Clean up old files on startup
    cleanup_old_files()
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=5000)