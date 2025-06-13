import pandas as pd
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import os
import matplotlib.pyplot as plt

def compress_ir_spectrum(input_file, output_file, compression_ratio=3):
    """
    Compress IR spectrum data while preserving:
    - All major peaks and valleys
    - Characteristic fingerprint region features
    - Overall spectral shape
    - Key absorption bands for compound identification
    
    Parameters:
        input_file (str): Path to input CSV file
        output_file (str): Path to output compressed CSV file
        compression_ratio (int): Target compression ratio (default 3)
    """
    
    # Read the input file
    df = pd.read_csv(input_file)
    
    # Extract drug type from filename (remove extension and path)
    drug_type = os.path.splitext(os.path.basename(input_file))[0]
    
    # Assuming first column is wavenumber (x) and second is absorbance (y)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    # Step 1: Identify major peaks (key absorption bands)
    peaks, _ = find_peaks(y, prominence=np.percentile(y, 95))
    valleys, _ = find_peaks(-y, prominence=np.percentile(-y, 95))
    
    # Step 2: Identify fingerprint region (typically 1500-500 cm^-1)
    fingerprint_mask = (x >= 500) & (x <= 1500)
    fingerprint_indices = np.where(fingerprint_mask)[0]
    
    # Step 3: Combine important points (peaks, valleys, fingerprint region)
    important_indices = np.unique(np.concatenate([peaks, valleys, fingerprint_indices]))
    
    # Step 4: Calculate target number of points after compression
    original_size = len(x)
    target_size = original_size // compression_ratio
    
    # Step 5: If we need more points, add points that maintain overall shape
    if len(important_indices) < target_size:
        # Use curvature to find additional important points
        smoothed = savgol_filter(y, window_length=11, polyorder=3)
        dy = np.gradient(smoothed)
        ddy = np.gradient(dy)
        curvature = np.abs(ddy) / (1 + dy**2)**1.5
        
        # Select points with highest curvature
        remaining_needed = target_size - len(important_indices)
        curvature_indices = np.argsort(curvature)[-remaining_needed:]
        important_indices = np.unique(np.concatenate([important_indices, curvature_indices]))
    
    # Step 6: If we still have too many points, prioritize the most important ones
    if len(important_indices) > target_size:
        # Sort by importance (peaks first, then valleys, then fingerprint region)
        importance = np.zeros_like(important_indices)
        for i, idx in enumerate(important_indices):
            if idx in peaks:
                importance[i] = 3  # Highest priority
            elif idx in valleys:
                importance[i] = 2
            elif idx in fingerprint_indices:
                importance[i] = 1
        important_indices = important_indices[np.argsort(importance)[-target_size:]]
    
    # Step 7: Sort the indices and select the points
    important_indices = np.sort(important_indices)
    compressed_x = x[important_indices]
    compressed_y = y[important_indices]
    
    # Step 8: Create new DataFrame with drug type column
    compressed_df = pd.DataFrame({
        df.columns[0]: compressed_x,
        df.columns[1]: compressed_y,
        'drug_type': drug_type
    })
    
    # Only create directory if output path contains directories
    output_dir = os.path.dirname(output_file)
    if output_dir != '':
        os.makedirs(output_dir, exist_ok=True)
    
    compressed_df.to_csv(output_file, index=False)
    
    # Step 9: Create comparison plot
    plt.figure(figsize=(12, 6))
    
    # Original spectrum
    plt.subplot(1, 2, 1)
    plt.plot(x, y, 'b-', label='Original')
    plt.title(f'Original Spectrum\n{drug_type}')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    
    # Compressed spectrum
    plt.subplot(1, 2, 2)
    plt.plot(x, y, 'b-', alpha=0.3, label='Original')
    plt.plot(compressed_x, compressed_y, 'r.', markersize=8, label='Compressed Points')
    plt.title(f'Compressed Spectrum\n{drug_type} (Ratio: {original_size/len(compressed_x):.1f}x)')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = os.path.splitext(output_file)[0] + '_comparison.png'
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Compression complete. Original size: {original_size}, Compressed size: {len(compressed_x)}")
    print(f"Compression ratio: {original_size/len(compressed_x):.2f}x")
    print(f"Saved compressed data to: {output_file}")
    print(f"Saved comparison plot to: {plot_filename}")

# Example usage
if __name__ == "__main__":
    input_csv = "sugar.csv"  # Replace with your input file
    output_csv = "c_sugar.csv"  # Output file
    
    # Use full paths to avoid any directory issues
    # Example:
    # input_csv = r"C:\Users\eldho\OneDrive\Desktop\input_spectrum.csv"
    # output_csv = r"C:\Users\eldho\OneDrive\Desktop\compressed_spectrum.csv"
    
    compress_ir_spectrum(input_csv, output_csv, compression_ratio=3)
    #new code update
