import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the original CSV file
input_file = 'exoplanet_data_clean.csv'  # Replace with your actual filename
output_file = 'tess_exoplanet_sample.csv'

# Load the data
df = pd.read_csv(input_file)

print(f"Original dataset size: {len(df)} rows")

# Randomly sample 20% of the data
sample_df = df.sample(frac=0.2, random_state=42)

print(f"Sample dataset size: {len(sample_df)} rows")

# List of features to extract from the TESS CSV
FEATURES = [
    'toi',  # TESS Object of Interest ID
    'tfopwg_disp',  # Label (CP, FP, KP, PC, APC, FA)
    
    # Stellar Proper Motion
    'st_pmra',  # PMRA [mas/yr]
    'st_pmraerr1',  # PMRA Upper Unc [mas/yr]
    'st_pmraerr2',  # PMRA Lower Unc [mas/yr]
    'st_pmdec',  # PMDec [mas/yr]
    'st_pmdecerr1',  # PMDec Upper Unc [mas/yr]
    'st_pmdecerr2',  # PMDec Lower Unc [mas/yr]
    
    # Orbital and Transit Parameters
    'pl_tranmid',  # Planet Transit Midpoint Value [BJD]
    'pl_orbper',  # Planet Orbital Period Value [days]
    'pl_trandurh',  # Planet Transit Duration Value [hours]
    'pl_trandurherr1',  # Planet Transit Duration Upper Unc [hours]
    'pl_trandurherr2',  # Planet Transit Duration Lower Unc [hours]
    'pl_trandep',  # Planet Transit Depth Value [ppm]
    'pl_trandeperr1',  # Planet Transit Depth Upper Unc [ppm]
    'pl_trandeperr2',  # Planet Transit Depth Lower Unc [ppm]
    
    # Planetary Properties
    'pl_rade',  # Planet Radius Value [R_Earth]
    'pl_insol',  # Planet Insolation Value [Earth flux]
    'pl_eqt',  # Planet Equilibrium Temperature Value [K]
    
    # Stellar Properties
    'st_tmag',  # TESS Magnitude
    'st_tmagerr1',  # TESS Magnitude Upper Unc
    'st_tmagerr2',  # TESS Magnitude Lower Unc
    'st_dist',  # Stellar Distance [pc]
    'st_disterr1',  # Stellar Distance Upper Unc [pc]
    'st_disterr2',  # Stellar Distance Lower Unc [pc]
    'st_teff',  # Stellar Effective Temperature Value [K]
    'st_tefferr1',  # Stellar Effective Temperature Upper Unc [K]
    'st_tefferr2',  # Stellar Effective Temperature Lower Unc [K]
    'st_logg',  # Stellar log(g) Value [cm/s**2]
    'st_rad',  # Stellar Radius Value [R_Sun]

]

# Check which features are available in the dataframe
available_features = [col for col in FEATURES if col in sample_df.columns]
missing_features = [col for col in FEATURES if col not in sample_df.columns]

if missing_features:
    print(f"\nWarning: The following features are not in the CSV: {missing_features}")

# Filter to keep only the specified features (if they exist)
if available_features:
    sample_df = sample_df[available_features]
    print(f"\nFiltered to {len(available_features)} features")

# Save the sample to a new CSV file
sample_df.to_csv(output_file, index=False)

print(f"\nSample file saved as: {output_file}")

# Display basic statistics
print("\nSample Statistics:")
print(f"Total rows: {len(sample_df)}")
if 'disposition' in sample_df.columns:
    print(f"\nDisposition distribution:")
    print(sample_df['disposition'].value_counts())

# Show missing values per column
print("\nMissing values per feature:")
print(sample_df.isnull().sum())