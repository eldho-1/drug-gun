import pandas as pd
import numpy as np

def convert_reflectance_to_absorbance(input_file, output_file):
    """
    Convert reflectance spectra to absorbance spectra
    
    Args:
        input_file: Path to input CSV file (wavenumber, reflectance, drug_type)
        output_file: Path to save output CSV file (wavenumber, absorbance, drug_type)
    """
    try:
        # Read input file
        df = pd.read_csv(input_file)
        
        # Validate columns
        required_columns = {'wavenumber', 'reflectance', 'drug_type'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert reflectance to absorbance
        # A = -log10(R) where R is reflectance (0-1 scale)
        df['absorbance'] = -np.log10(df['reflectance'])
        
        # Select and reorder columns
        output_df = df[['wavenumber', 'absorbance', 'drug_type']]
        
        # Save to new file
        output_df.to_csv(output_file, index=False)
        print(f"Successfully converted and saved to {output_file}")
        
        return output_df
    
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    input_csv = "polyethylene.csv"  # Replace with your input file path
    output_csv = "abs_poly.csv"  # Replace with desired output path
    
    converted_data = convert_reflectance_to_absorbance(input_csv, output_csv)
    
    if converted_data is not None:
        print("\nFirst 5 rows of converted data:")
        print(converted_data.head())
