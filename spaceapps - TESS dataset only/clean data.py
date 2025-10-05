import pandas as pd
import numpy as np

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

# Critical features that must be present for a row to be valid
CRITICAL_FEATURES = [
    'tfopwg_disp',  # Must have a label
    'pl_orbper',  # Orbital period is critical
    'pl_trandurh',  # Transit duration is critical
    'pl_trandep',  # Transit depth is critical
]


def clean_data(df, missing_threshold=0.5, critical_features=None):
    """
    Clean the dataset by removing rows with inadequate data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input dataframe to clean
    missing_threshold : float
        Maximum proportion of missing values allowed per row (0 to 1)
        Default: 0.5 (50% of features can be missing)
    critical_features : list
        List of features that must not be missing. Rows missing any of these
        will be removed regardless of missing_threshold
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe with rows removed
    """
    print("\n" + "="*50)
    print("DATA CLEANING")
    print("="*50)
    
    initial_rows = len(df)
    print(f"\nInitial number of rows: {initial_rows}")
    
    # Step 1: Remove rows where critical features are missing
    if critical_features is None:
        critical_features = CRITICAL_FEATURES
    
    print(f"\nChecking critical features: {critical_features}")
    
    critical_cols = [col for col in critical_features if col in df.columns]
    if critical_cols:
        df_cleaned = df.dropna(subset=critical_cols).copy()
        removed_critical = initial_rows - len(df_cleaned)
        print(f"Rows removed due to missing critical features: {removed_critical}")
        print(f"Remaining rows: {len(df_cleaned)}")
    else:
        df_cleaned = df.copy()
        print("Warning: No critical features found in dataset")
    
    # Step 2: Remove rows with too many missing values
    print(f"\nApplying missing value threshold: {missing_threshold*100}%")
    
    # Calculate percentage of missing values per row (excluding ID and label columns)
    feature_cols = [col for col in df_cleaned.columns 
                   if col not in ['toi', 'tfopwg_disp']]
    
    missing_per_row = df_cleaned[feature_cols].isnull().sum(axis=1) / len(feature_cols)
    
    # Keep rows where missing percentage is below threshold
    rows_to_keep = missing_per_row <= missing_threshold
    df_cleaned = df_cleaned[rows_to_keep].copy()
    
    removed_threshold = len(df) - removed_critical - len(df_cleaned)
    print(f"Rows removed due to exceeding missing threshold: {removed_threshold}")
    
    # Final statistics
    final_rows = len(df_cleaned)
    total_removed = initial_rows - final_rows
    removal_pct = (total_removed / initial_rows) * 100
    
    print(f"\n" + "-"*50)
    print(f"Final number of rows: {final_rows}")
    print(f"Total rows removed: {total_removed} ({removal_pct:.2f}%)")
    print(f"Data retention rate: {(final_rows/initial_rows)*100:.2f}%")
    
    # Show class distribution after cleaning
    if 'tfopwg_disp' in df_cleaned.columns:
        print(f"\nClass distribution after cleaning:")
        print(df_cleaned['tfopwg_disp'].value_counts())
        print(f"\nPercentage distribution:")
        for label, count in df_cleaned['tfopwg_disp'].value_counts().items():
            pct = (count/final_rows)*100
            print(f"  {label}: {count} ({pct:.2f}%)")
    
    return df_cleaned


def load_and_extract_data(csv_path, output_path='tess_exoplanet_data_clean.csv',
                         missing_threshold=0.5, remove_inadequate=True):
    """
    Load the TESS dataset, extract relevant features, and optionally clean data.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    output_path : str
        Path where the cleaned data will be saved
    missing_threshold : float
        Maximum proportion of missing values allowed per row (0 to 1)
    remove_inadequate : bool
        Whether to remove rows with inadequate data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing only the selected features (cleaned if requested)
    """
    print(f"Loading data from {csv_path}...")
    
    # Load the CSV file (handle potential comment lines starting with #)
    df = pd.read_csv(csv_path, comment='#', skipinitialspace=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Total columns in original dataset: {len(df.columns)}")
    
    # Check which features are available in the dataset
    available_features = [f for f in FEATURES if f in df.columns]
    missing_features = [f for f in FEATURES if f not in df.columns]
    
    print(f"\nAvailable features: {len(available_features)}/{len(FEATURES)}")
    
    if missing_features:
        print(f"\nWarning: The following features are missing from the dataset:")
        for feat in missing_features:
            print(f"  - {feat}")
    
    # Extract only the available features
    df_extracted = df[available_features].copy()
    
    print(f"\nExtracted dataset shape: {df_extracted.shape}")
    
    # Display basic information about the target variable
    if 'tfopwg_disp' in df_extracted.columns:
        print("\nTarget variable distribution (before cleaning):")
        print(df_extracted['tfopwg_disp'].value_counts())
        print(f"\nPercentage distribution:")
        for label, count in df_extracted['tfopwg_disp'].value_counts().items():
            pct = (count/len(df_extracted))*100
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print("\nLabel definitions:")
        print("  APC = Ambiguous Planetary Candidate")
        print("  CP  = Confirmed Planet")
        print("  FA  = False Alarm")
        print("  FP  = False Positive")
        print("  KP  = Known Planet")
        print("  PC  = Planetary Candidate")
    
    # Display missing values information (before cleaning)
    print("\nMissing values per feature (before cleaning):")
    missing_counts = df_extracted.isnull().sum()
    missing_pct = (missing_counts / len(df_extracted)) * 100
    missing_info = pd.DataFrame({
        'Missing_Count': missing_counts,
        'Missing_Percentage': missing_pct.round(2)
    })
    print(missing_info[missing_info['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    # Clean the data if requested
    if remove_inadequate:
        df_extracted = clean_data(df_extracted, missing_threshold=missing_threshold)
        
        # Show missing values after cleaning
        print("\nMissing values per feature (after cleaning):")
        missing_counts_clean = df_extracted.isnull().sum()
        missing_pct_clean = (missing_counts_clean / len(df_extracted)) * 100
        missing_info_clean = pd.DataFrame({
            'Missing_Count': missing_counts_clean,
            'Missing_Percentage': missing_pct_clean.round(2)
        })
        print(missing_info_clean[missing_info_clean['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))
    
    # Save the extracted data
    df_extracted.to_csv(output_path, index=False)
    print(f"\nCleaned data saved to: {output_path}")
    
    return df_extracted


def get_data_statistics(df):
    """
    Display detailed statistics about the extracted dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The extracted dataset
    """
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Number of samples: {len(df)}")
    print(f"Number of features: {len(df.columns)}")
    
    # Numeric columns statistics
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNumeric features: {len(numeric_cols)}")
    
    print("\nBasic statistics for numeric features:")
    print(df[numeric_cols].describe())


if __name__ == "__main__":
    # Example usage
    # Replace 'tess_data.csv' with your actual CSV file path
    CSV_FILE_PATH = 'TOI_2025.10.03_23.24.18.csv'
    OUTPUT_FILE_PATH = 'exoplanet_data_clean.csv'
    
    # Extract and clean the data
    # Adjust missing_threshold as needed (0.5 means 50% of features can be missing)
    # Set remove_inadequate=False to skip data cleaning
    df = load_and_extract_data(
        CSV_FILE_PATH, 
        OUTPUT_FILE_PATH,
        missing_threshold=0.3,  # Adjust this value (0.0 to 1.0)
        remove_inadequate=True   # Set to False to keep all rows
    )
    
    # Display statistics
    get_data_statistics(df)
    
    print("\n" + "="*50)
    print("Data extraction and cleaning completed successfully!")
    print("="*50)