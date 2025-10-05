import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Read the original CSV file
input_file = 'exoplanet_data_clean.csv'  # Replace with your actual filename
output_file = 'Fullymerged_data_sample.csv'

# Load the data
df = pd.read_csv(input_file)

print(f"Original dataset size: {len(df)} rows")

# Randomly sample 20% of the data
sample_df = df.sample(frac=0.2, random_state=42)

print(f"Sample dataset size: {len(sample_df)} rows")

# Features to keep (if you want to filter columns)
FEATURES = [
    'label',
    'orbital_period',
    'transit_duration',
    'transit_duration_err1',
    'transit_duration_err2',
    'transit_depth',
    'transit_depth_err1',
    'transit_depth_err2',
    'planet_radius',
    'planet_radius_err1',
    'planet_radius_err2',
    'equi_temp',
    'stellar_temp',
    'stellar_temp_err1',
    'stellar_temp_err2',
    'stellar_radius',
    'stellar_radius_err1',
    'stellar_radius_err2',
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
if 'label' in sample_df.columns:
    print(f"\nLabel distribution:")
    print(sample_df['label'].value_counts())