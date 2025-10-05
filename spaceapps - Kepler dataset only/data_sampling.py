import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the original CSV file
input_file = 'exoplanet_data_clean.csv'  # Replace with your actual filename
output_file = 'kepler_exoplanet_sample.csv'

# Load the data
df = pd.read_csv(input_file)

print(f"Original dataset size: {len(df)} rows")

# Randomly sample 20% of the data
sample_df = df.sample(frac=0.2, random_state=42)

print(f"Sample dataset size: {len(sample_df)} rows")

# List of features to extract from the CSV (updated feature names)
FEATURES = [
    'pl_name',  # Planet Name (ID equivalent)
    'disposition',
    
    # Orbital and Transit Parameters
    'pl_orbper',  # Orbital Period [days]
    'pl_tranmid',  # Transit Midpoint [days]
    'pl_trandur',  # Transit Duration [hours]

    # Planet Radius (Earth)
    'pl_rade',  # Planet Radius [Earth Radius]

    
    # Planet Radius (Jupiter)
    'pl_radj',  # Planet Radius [Jupiter Radius]
    'pl_radjerr1',  # Planet Radius Upper Unc. [Jupiter Radius]
    'pl_radjerr2',  # Planet Radius Lower Unc. [Jupiter Radius]
    
    # Planet-Star Ratios
    'pl_ratror',  # Ratio of Planet to Stellar Radius

    # Stellar Properties - Radius
    'st_rad',  # Stellar Radius [Solar Radius]
    'st_raderr1',  # Stellar Radius Upper Unc. [Solar Radius]
    'st_raderr2',  # Stellar Radius Lower Unc. [Solar Radius]
    
    # System Properties - Distance
    'sy_dist',  # Distance [pc]
    'sy_disterr1',  # Distance Upper Unc. [pc]
    'sy_disterr2',  # Distance Lower Unc. [pc]
    
    # System Properties - Parallax
    'sy_plx',  # Parallax [mas]
    'sy_plxerr1',  # Parallax Upper Unc. [mas]
    'sy_plxerr2',  # Parallax Lower Unc. [mas]
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