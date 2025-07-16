# main_pipeline.py

import pandas as pd
import os
import matplotlib.pyplot as plt

from noiseSimulator import add_realistic_environmental_noise
from tempSimulator import adjust_absorbance
from simulate_advanced_chemical_noise import add_advanced_chemical_noise
from noiseModel import predict_sample
import joblib

def main():
    print("\n🧪 DRUG DETECTION PIPELINE")

    # Step 1: Input file
    input_path = input("Enter path to CSV file: ").strip()
    if not os.path.exists(input_path):
        print("❌ File not found.")
        return

    original_df = pd.read_csv(input_path)

    if not {'wavenumber', 'absorbance'}.issubset(original_df.columns):
        print("❌ CSV must contain 'wavenumber' and 'absorbance' columns.")
        return

    df = original_df.copy()

    # Step 2: Apply environmental noise?
    if input("Apply environmental noise? (y/n): ").strip().lower() == 'y':
        from noiseSimulator import add_realistic_environmental_noise
        df = add_realistic_environmental_noise(df)

    # Step 3: Apply temperature disturbance?
    if input("Apply temperature disturbance? (y/n): ").strip().lower() == 'y':
        from tempSimulator import adjust_absorbance
        try:
            temp = float(input("Enter temperature in °C: ").strip())
            df['absorbance'] = adjust_absorbance(df['absorbance'], temp)
        except ValueError:
            print("⚠️ Invalid temperature. Skipping temperature adjustment.")

    # Step 4: Apply chemical noise?
    if input("Apply chemical noise? (y/n): ").strip().lower() == 'y':
        from simulate_advanced_chemical_noise import add_advanced_chemical_noise
        temp_path = "temp_input.csv"
        df.to_csv(temp_path, index=False)
        add_advanced_chemical_noise(temp_path, output_csv=temp_path)
        df = pd.read_csv(temp_path)
        os.remove(temp_path)

    # Step 5: Save modified file
    output_dir = "Noisy Data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "modified_input.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Modified spectrum saved at: {output_path}")

    # Step 6: Plot original vs modified spectrum
    print("\n📈 Displaying spectrum comparison...")
    plt.figure(figsize=(10, 5))
    plt.plot(original_df['wavenumber'], original_df['absorbance'], label='Original', color='blue')
    plt.plot(df['wavenumber'], df['absorbance'], label='Modified', color='red', alpha=0.7)
    plt.gca().invert_xaxis()
    plt.title("Spectrum Comparison")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Absorbance")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Step 7: Run detection
    try:
        print("\n🔍 Running detection...")
        binary_model = joblib.load("drug_binary_xgb.pkl")
        multiclass_model = joblib.load("drug_multiclass_xgb.pkl")
        label_encoder = joblib.load("drug_label_encoder.pkl")

        from noiseModel import predict_sample
        result = predict_sample(output_path, binary_model, multiclass_model, label_encoder)

        print("\n📊 DETECTION RESULT:")
        print(result)

    except Exception as e:
        print(f"\n❌ Error during prediction: {e}")

if __name__ == "__main__":
    main()
