"""
PCA Analysis of Universal Feature Set Across Multiple Network Security Datasets
================================================================================
Author: Researcher
Date: 2024
Description: This script performs Principal Component Analysis (PCA) on the 
             6 universal features proposed for DDoS detection across four 
             different network datasets. The analysis visualizes the feature 
             space distribution and inter-dataset relationships.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')
# ============================================================================
# 1. CONFIGURATION SECTION - USER MUST SET THESE PATHS MANUALLY
# ============================================================================
# Please provide the full paths to your CSV dataset files
DATASET_PATHS = {
    'CIC-DDoS2019': r'E:\Cic-DDos2019 Original\03-11\NetBIOS_Pre.csv',  
    'SDN-DDoS Traffic': r'E:\SDN-DDoS_Traffic_Dataset from Mendeley\SDN-DDoS_With_CIC_Features_Pre.csv',  
    'CIC-IoT 2023': r'E:\CIC-IoT 2023\CIC-IoT2023 form kaggle\test_with_CIC_Features.csv',  
    'VeReMi': r'E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary.csv'
}
# Universal features as defined in the research paper
UNIVERSAL_FEATURES = [
    'Packet Length Mean',      # Primary feature 1
    'Average Packet Size',     # Primary feature 2  
    'Bwd Packet Length Min',   # Backward packet length minimum
    'Fwd Packets/s',           # Forward packets per second
    'Min Packet Length',       # Minimum packet length
    'Down/Up Ratio'           # Download/Upload ratio
]
# Alternative column name variations (handles dataset-specific naming conventions)
FEATURE_ALIASES = {
    'Packet Length Mean': ['Pkt Len Mean', 'Packet Length Mean', 'pkt_len_mean', 'Packet_Length_Mean'],
    'Average Packet Size': ['Pkt Size Avg', 'Average Packet Size', 'Average_Packet_Size', 'pkt_size_avg'],
    'Bwd Packet Length Min': ['Bwd Pkt Len Min', 'Bwd Packet Length Min', 'Bwd_Packet_Length_Min'],
    'Fwd Packets/s': ['Fwd Pkts/s', 'Fwd Packets/s', 'fwd_pkts_per_sec', 'Fwd_Packets_per_second'],
    'Min Packet Length': ['Pkt Len Min', 'Min Packet Length', 'Min_Packet_Length', 'pkt_len_min'],
    'Down/Up Ratio': ['Down/Up Ratio', 'down_up_ratio', 'Down_Up_Ratio']
}
# Visualization settings
COLORS = ['green', 'red', 'blue', 'black']
DATASET_COLORS = dict(zip(DATASET_PATHS.keys(), COLORS))
FIG_SIZE = (14, 12)  # Under 15 inches as requested
# ============================================================================
# 2. DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================
def find_column_match(target_feature, available_columns):
    """
    Find matching column name for a target feature with flexible matching.
    Parameters:
    -----------
    target_feature : str
        The canonical feature name from UNIVERSAL_FEATURES
    available_columns : list
        List of actual column names in the dataset
    Returns:
    --------
    str or None: Matched column name or None if not found
    """
    # Check for exact match first
    if target_feature in available_columns:
        return target_feature
    # Check for case-insensitive match
    lower_columns = [col.lower().replace(' ', '').replace('_', '').replace('-', '') 
                     for col in available_columns]
    target_simplified = target_feature.lower().replace(' ', '').replace('_', '').replace('-', '')
    for idx, col in enumerate(lower_columns):
        if target_simplified == col:
            return available_columns[idx]
    # Check aliases
    if target_feature in FEATURE_ALIASES:
        for alias in FEATURE_ALIASES[target_feature]:
            if alias in available_columns:
                return alias
            # Check case-insensitive for aliases
            for avail_col in available_columns:
                if alias.lower() == avail_col.lower():
                    return avail_col
    return None
def load_and_prepare_dataset(dataset_name, file_path):
    """
    Load dataset and extract the 6 universal features with robust column matching.
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset for reporting
    file_path : str
        Path to the CSV file
    Returns:
    --------
    pd.DataFrame or None: DataFrame with universal features or None if error
    """
    try:
        print(f"Loading {dataset_name}...")
        df = pd.read_csv(file_path, low_memory=False)
        print(f"  Original shape: {df.shape}")
        print(f"  Available columns: {len(df.columns)} columns")
        # Find matching columns for each universal feature
        selected_columns = []
        available_cols = df.columns.tolist()
        for feature in UNIVERSAL_FEATURES:
            matched_col = find_column_match(feature, available_cols)
            if matched_col:
                selected_columns.append(matched_col)
                print(f"  ✓ Found '{feature}' as '{matched_col}'")
            else:
                print(f"  ✗ Missing feature: {feature}")
                # Return None if any essential feature is missing
                return None
        # Extract selected features
        df_selected = df[selected_columns].copy()
        # Convert to numeric, coercing errors
        for col in df_selected.columns:
            df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')
        # Drop rows with NaN values (PCA requires complete cases)
        initial_rows = len(df_selected)
        df_selected = df_selected.dropna()
        final_rows = len(df_selected)
        print(f"  Cleaned shape: {df_selected.shape}")
        print(f"  Removed {initial_rows - final_rows} rows with missing values")
        # Rename columns to canonical names for consistency
        column_mapping = {}
        for idx, feature in enumerate(UNIVERSAL_FEATURES):
            if idx < len(selected_columns):
                column_mapping[selected_columns[idx]] = feature
        df_selected = df_selected.rename(columns=column_mapping)
        return df_selected
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading {dataset_name}: {str(e)}")
        return None
# ============================================================================
# 3. PCA ANALYSIS AND VISUALIZATION
# ============================================================================
def perform_pca_analysis(datasets_dict):
    """
    Perform PCA analysis on multiple datasets and create visualizations.
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary with dataset names as keys and DataFrames as values
    """
    # 3.1 Data Preparation and Standardization
    print("\n" + "="*60)
    print("STEP 1: DATA STANDARDIZATION")
    print("="*60)
    scaled_datasets = {}
    scaler = StandardScaler()
    for name, df in datasets_dict.items():
        if df is not None:
            # Standardize features (mean=0, variance=1)
            scaled_data = scaler.fit_transform(df)
            scaled_datasets[name] = {
                'data': scaled_data,
                'features': df.columns.tolist(),
                'original_df': df
            }
            print(f"Standardized {name}: {scaled_data.shape}")
    # 3.2 Individual Dataset PCA
    print("\n" + "="*60)
    print("STEP 2: INDIVIDUAL DATASET PCA ANALYSIS")
    print("="*60)
    fig_individual, axes_individual = plt.subplots(2, 2, figsize=FIG_SIZE)
    axes_individual = axes_individual.flatten()
    individual_results = {}
    for idx, (name, data_dict) in enumerate(scaled_datasets.items()):
        if idx >= 4:  # Safety check
            break
        # Perform PCA with 2 components
        pca = PCA(n_components=2, random_state=42)
        principal_components = pca.fit_transform(data_dict['data'])
        # Store results
        individual_results[name] = {
            'pca': pca,
            'components': principal_components,
            'explained_variance': pca.explained_variance_ratio_
        }
        # Plot individual dataset
        ax = axes_individual[idx]
        scatter = ax.scatter(principal_components[:, 0], 
                            principal_components[:, 1],
                            alpha=0.6, 
                            c=DATASET_COLORS[name],
                            s=20,
                            edgecolors='w',
                            linewidth=0.5)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
        ax.set_title(f'{name}\nTotal Variance: {sum(pca.explained_variance_ratio_):.2%}', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add statistics text box
        stats_text = (f'Samples: {len(principal_components):,}\n'
                     f'Features: {len(data_dict["features"])}\n'
                     f'PC1+PC2: {sum(pca.explained_variance_ratio_):.2%}')
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"{name}:")
        print(f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.4f}, "
              f"PC2={pca.explained_variance_ratio_[1]:.4f}")
        print(f"  Total variance (PC1+PC2): {sum(pca.explained_variance_ratio_):.4f}")
    fig_individual.suptitle('PCA Analysis: Individual Datasets (Universal Feature Set)', 
                           fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    # 3.3 Combined Dataset Visualization
    print("\n" + "="*60)
    print("STEP 3: COMBINED DATASET VISUALIZATION")
    print("="*60)
    fig_combined, ax_combined = plt.subplots(figsize=(12, 10))
    all_components = []
    all_labels = []
    for name, results in individual_results.items():
        components = results['components']
        all_components.append(components)
        all_labels.extend([name] * len(components))
        # Plot with dataset-specific color
        ax_combined.scatter(components[:, 0], components[:, 1],
                          alpha=0.5, 
                          c=DATASET_COLORS[name],
                          s=15,
                          label=name,
                          edgecolors='w',
                          linewidth=0.3)
    ax_combined.set_xlabel('Principal Component 1', fontsize=12)
    ax_combined.set_ylabel('Principal Component 2', fontsize=12)
    ax_combined.set_title('PCA: Combined View of All Datasets\n(6 Universal Features)', 
                         fontsize=14, fontweight='bold')
    ax_combined.legend(loc='best', fontsize=10, markerscale=2)
    ax_combined.grid(True, alpha=0.3)
    # Add overall statistics
    combined_text = "Dataset Distribution in Feature Space\n"
    for name in individual_results.keys():
        n_samples = len(individual_results[name]['components'])
        var_exp = sum(individual_results[name]['explained_variance'])
        combined_text += f"\n{name}: {n_samples:,} samples, {var_exp:.2%} variance"
    ax_combined.text(0.02, 0.98, combined_text, transform=ax_combined.transAxes,
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.9))
    plt.tight_layout()
    # 3.4 Feature Loading Visualization
    print("\n" + "="*60)
    print("STEP 4: FEATURE LOADINGS ANALYSIS")
    print("="*60)
    # Use first dataset for feature loadings (all should have same features)
    first_dataset = list(scaled_datasets.keys())[0]
    pca_model = individual_results[first_dataset]['pca']
    fig_loadings, ax_loadings = plt.subplots(figsize=(10, 8))
    # Get feature loadings (components)
    loadings = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    # Plot feature vectors
    for i, feature in enumerate(UNIVERSAL_FEATURES):
        ax_loadings.arrow(0, 0, loadings[i, 0], loadings[i, 1],
                         color='darkred', alpha=0.8,
                         head_width=0.05, head_length=0.05)
        ax_loadings.text(loadings[i, 0]*1.15, loadings[i, 1]*1.15,
                        feature, color='darkred',
                        fontsize=9, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    # Add unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='blue', alpha=0.3, linestyle='--')
    ax_loadings.add_artist(circle)
    ax_loadings.set_xlim(-1.5, 1.5)
    ax_loadings.set_ylim(-1.5, 1.5)
    ax_loadings.set_xlabel('PC1 Loading', fontsize=11)
    ax_loadings.set_ylabel('PC2 Loading', fontsize=11)
    ax_loadings.set_title('PCA Feature Loadings (Correlation Circle)\n' +
                         'How each feature contributes to principal components',
                         fontsize=12, fontweight='bold')
    ax_loadings.grid(True, alpha=0.3)
    ax_loadings.axhline(y=0, color='k', alpha=0.3, linestyle='-')
    ax_loadings.axvline(x=0, color='k', alpha=0.3, linestyle='-')
    ax_loadings.set_aspect('equal')
    plt.tight_layout()
    # 3.5 Display all plots
    print("\n" + "="*60)
    print("STEP 5: DISPLAYING VISUALIZATIONS")
    print("="*60)
    print("All PCA visualizations have been generated.")
    print("Close each plot window to continue to the next.")
    plt.show()
    return individual_results
# ============================================================================
# 4. MAIN EXECUTION
# ============================================================================
def main():
    """
    Main execution function for PCA analysis of universal features.
    """
    print("="*70)
    print("UNIVERSAL FEATURE SET PCA ANALYSIS")
    print("="*70)
    print("This script analyzes the 6 universal features across 4 datasets:")
    print("1. CIC-DDoS2019")
    print("2. SDN-DDoS Traffic")
    print("3. CIC-IoT 2023")
    print("4. VeReMi")
    print("\nUniversal Features:")
    for i, feature in enumerate(UNIVERSAL_FEATURES, 1):
        print(f"  {i}. {feature}")
    # Check if paths are set
    if "YOUR_PATH_HERE" in DATASET_PATHS["CIC-DDoS2019"]:
        print("\n⚠️  WARNING: Dataset paths are not configured!")
        print("Please edit the DATASET_PATHS dictionary at the beginning of the script.")
        print("Replace 'YOUR_PATH_HERE' with actual file paths to your CSV datasets.")
        return
    # Load all datasets
    print("\n" + "="*70)
    print("LOADING DATASETS")
    print("="*70)
    datasets = {}
    for name, path in DATASET_PATHS.items():
        datasets[name] = load_and_prepare_dataset(name, path)
        print("-" * 50)
    # Check if we have at least one dataset loaded successfully
    successful_loads = sum(1 for df in datasets.values() if df is not None)
    if successful_loads == 0:
        print("\n❌ ERROR: No datasets were successfully loaded.")
        print("Please check:")
        print("1. File paths are correct")
        print("2. CSV files contain the required features")
        print("3. Feature names match (check for typos)")
        return
    print(f"\n✅ Successfully loaded {successful_loads} out of {len(datasets)} datasets")
    # Remove None values (failed loads)
    datasets = {k: v for k, v in datasets.items() if v is not None}
    # Perform PCA analysis
    print("\n" + "="*70)
    print("STARTING PCA ANALYSIS")
    print("="*70)
    try:
        results = perform_pca_analysis(datasets)
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("Summary of findings:")
        print("- Each plot shows the distribution of samples in 2D PCA space")
        print("- Clusters indicate similar feature patterns")
        print("- Overlap between datasets suggests feature space similarity")
        print("- Isolated clusters indicate unique dataset characteristics")
    except Exception as e:
        print(f"\n❌ ERROR during PCA analysis: {str(e)}")
        print("This might be due to:")
        print("1. Insufficient data after cleaning")
        print("2. Numerical issues in the data")
        print("3. Memory limitations")
        import traceback
        traceback.print_exc()
# ============================================================================
# 5. EXECUTION GUARD
# ============================================================================
if __name__ == "__main__":
    main()
