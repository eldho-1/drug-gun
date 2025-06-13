import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

# Known IR peak positions for cocaine (approximate values, in cm⁻¹)
KNOWN_COCAINE_PEAKS = [3480, 2930, 1760, 1605, 1500, 1260, 1030]

def load_spectrum(csv_file):
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        sys.exit(1)
    try:
        df = pd.read_csv(csv_file)
        df = df[['wavelength', 'absorbance']]  # Only use relevant columns
        df = df.sort_values('wavelength')
        return df
    except Exception as e:
        print(f"Error while reading the CSV file: {e}")
        sys.exit(1)

def detect_peaks(df, height_threshold=0.1):
    peaks, _ = find_peaks(df['absorbance'], height=height_threshold)
    return df.iloc[peaks]

def compare_with_known_peaks(detected_peaks, known_peaks, tolerance=20):
    matched = []
    for known in known_peaks:
        match = detected_peaks[(detected_peaks['wavelength'] >= known - tolerance) &
                               (detected_peaks['wavelength'] <= known + tolerance)]
        if not match.empty:
            matched.append((known, match['wavelength'].values[0]))
    return matched

def plot_spectrum(df, detected_peaks, matched_peaks):
    plt.figure(figsize=(14, 6))
    plt.plot(df['wavelength'], df['absorbance'], label='IR Spectrum', color='blue')
    plt.scatter(detected_peaks['wavelength'], detected_peaks['absorbance'],
                color='red', label='Detected Peaks')

    # Annotate detected peaks
    for i, row in detected_peaks.iterrows():
        # Position the text slightly above the peak
        y_pos = row['absorbance'] + (df['absorbance'].max() * 0.05)  # 5% of max absorbance above peak
        plt.text(row['wavelength'], y_pos,
                 f"{row['wavelength']:.0f}", 
                 color='red', fontsize=8,
                 rotation=90, ha='center', va='bottom')

    # Draw vertical lines and annotate known peaks
    for known, detected in matched_peaks:
        plt.axvline(detected, color='green', linestyle='--', alpha=0.5)
        plt.text(detected, df['absorbance'].max() * 0.85, f'{known} cm⁻¹',
                 rotation=90, ha='center', color='green', fontsize=9)

    plt.title("Cocaine IR Spectrum - Absorbance vs Wavelength")
    plt.xlabel("Wavelength (cm⁻¹)")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
def analyze_and_plot(csv_file):
    df = load_spectrum(csv_file)
    
    # Convert wavelength from nm to cm⁻¹ if needed (assuming your data is in nm)
    # df['wavelength'] = 1e7 / df['wavelength']  # Uncomment if your data is in nm
    
    # Use a much lower threshold appropriate for your data
    detected_peaks = detect_peaks(df, height_threshold=0.05)  # Adjusted threshold
    
    matched_peaks = compare_with_known_peaks(detected_peaks, KNOWN_COCAINE_PEAKS)

    print(f"\nDetected {len(detected_peaks)} peaks.")
    print("Detected peak positions:")
    for i, row in detected_peaks.iterrows():
        print(f"  {row['wavelength']:.1f} cm⁻¹ (Absorbance: {row['absorbance']:.4f})")

    print("\nMatched peaks with known cocaine peaks:")
    for known, detected in matched_peaks:
        print(f"  Known: {known} cm⁻¹ ↔ Detected: {detected:.1f} cm⁻¹")

    plot_spectrum(df, detected_peaks, matched_peaks)
if __name__ == "__main__":
    analyze_and_plot("i_heroin.csv") #ith your actual file
