import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------
# Configuration
# ------------------------------
FILENAME = "heroin.csv"          # Change this to any other file in data/
INPUT_PATH = os.path.join("data", FILENAME)
BASE_TEMP = 25  # Reference temperature (°C)

# ------------------------------
# Simulated absorbance adjustment
# ------------------------------
def adjust_absorbance(absorbance, temperature):
    scale_factor = 1 + 0.0006 * (temperature - BASE_TEMP)
    thermal_noise = np.random.normal(0, 0.0005 * abs(temperature - BASE_TEMP), size=len(absorbance))
    return absorbance * scale_factor + thermal_noise

# ------------------------------
# Process and Save
# ------------------------------
def process_csv(temp_celsius):
    try:
        df = pd.read_csv(INPUT_PATH)

        if not {'wavenumber', 'absorbance', 'drug_type'}.issubset(df.columns):
            print("CSV must contain 'wavenumber', 'absorbance', and 'drug_type' columns.")
            return

        # Adjust absorbance
        df['absorbance'] = adjust_absorbance(df['absorbance'], temp_celsius)

        # Save to adjusted file
        name_only, ext = os.path.splitext(FILENAME)
        output_filename = f"{name_only}_T{int(temp_celsius)}C{ext}"
        output_path = os.path.join("data", output_filename)
        df.to_csv(output_path, index=False)

        print(f"✅ Adjusted file saved as: {output_path}")

        # Plotting
        plt.figure(figsize=(10, 5))
        plt.plot(df['wavenumber'], df['absorbance'], color='red')
        plt.gca().invert_xaxis()
        plt.title(f"{df['drug_type'].iloc[0]} - IR Spectrum @ {temp_celsius}°C")
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Absorbance")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except FileNotFoundError:
        print(f"❌ File not found: {INPUT_PATH}")
    except Exception as e:
        print(f"⚠️ Error: {e}")

# ------------------------------
# Main Program
# ------------------------------
if __name__ == "__main__":
    try:
        temp = float(input("Enter temperature in °C: "))
        process_csv(temp)
    except ValueError:
        print("❌ Invalid temperature. Please enter a number.")
