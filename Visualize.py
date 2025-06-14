import pandas as pd
import matplotlib.pyplot as plt

def plot_spectrum(csv_path):
    # Load the CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[Error] File not found: {csv_path}")
        return
    except Exception as e:
        print(f"[Error] Failed to load file: {e}")
        return

    # Check required columns
    required_cols = {'wavelength', 'absorbance', 'drug_type'}
    if not required_cols.issubset(df.columns):
        print(f"[Error] CSV is missing required columns: {required_cols}")
        return

    # Plot
    plt.figure(figsize=(10, 5))

    # If multiple drug types are present, plot them separately
    for drug in df['drug_type'].unique():
        sub_df = df[df['drug_type'] == drug]
        plt.plot(sub_df['wavelength'], sub_df['absorbance'], label=drug)

    plt.xlabel('Wavelength')
    plt.ylabel('Absorbance')
    plt.title('Spectral Graph (Absorbance vs Wavelength)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Example usage
plot_spectrum('correct_cocaine.csv')
