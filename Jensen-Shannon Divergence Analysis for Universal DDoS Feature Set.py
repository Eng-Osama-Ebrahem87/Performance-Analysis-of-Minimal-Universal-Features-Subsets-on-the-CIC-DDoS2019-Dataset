"""
Jensen-Shannon Divergence Analysis for Universal DDoS Feature Set
Author: Research Assistant
Purpose: Compute JS divergence for six universal features across four cybersecurity datasets
Reference: Nielsen, F. (2020). "On the Jensen–Shannon Symmetrization of Distances Relying on Abstract Means"
"""
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import jensenshannon
from scipy.stats import gaussian_kde

import warnings
warnings.filterwarnings('ignore')
# ==================== CONFIGURATION ====================
# USER MUST UPDATE THESE PATHS TO THEIR DATASET LOCATIONS
DATASET_PATHS = {
    'CIC-DDoS2019': r'E:\Cic-DDos2019 Original\03-11\NetBIOS_Pre.csv',  
    'SDN-DDoS': r'E:\SDN-DDoS_Traffic_Dataset from Mendeley\SDN-DDoS_With_CIC_Features_Pre.csv',  
    'CIC-IoT-2023': r'E:\CIC-IoT 2023\CIC-IoT2023 form kaggle\test_with_CIC_Features.csv',  
    'VeReMi': r'E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary.csv'
}
# Universal feature set with possible column name variations
UNIVERSAL_FEATURES = {
    'Packet Length Mean': ['Pkt Len Mean', 'Packet Length Mean', 'pkt_len_mean', 'Packet_Length_Mean'],
    'Average Packet Size': ['Pkt Size Avg', 'Average Packet Size', 'Average_Packet_Size', 'pkt_size_avg'],
    'Bwd Packet Length Min': ['Bwd Pkt Len Min', 'Bwd Packet Length Min', 'Bwd_Packet_Length_Min'],
    'Fwd Packets/s': ['Fwd Pkts/s', 'Fwd Packets/s', 'fwd_pkts_per_sec', 'Fwd_Packets_per_second'],
    'Min Packet Length': ['Pkt Len Min', 'Min Packet Length', 'Min_Packet_Length', 'pkt_len_min'],
    'Down/Up Ratio': ['Down/Up Ratio', 'down_up_ratio', 'Down_Up_Ratio']
}
# Color scheme for visualization
DATASET_COLORS = {
    "CIC-DDoS2019": "#2E86AB",    # Blue
    "SDN-DDoS": "#A23B72",        # Red
    "CIC-IoT-2023": "#18A999",    # Green
    "VeReMi": "#2D2D2D"           # Black
}
# ==================== DATA LOADING & PREPROCESSING ====================
def load_and_preprocess_dataset(name, path):
    """
    Load and preprocess a single dataset with robust error handling.
    Parameters:
    -----------
    name : str
        Dataset name
    path : str
        Path to CSV file
    Returns:
    --------
    pandas.DataFrame
        Preprocessed dataframe with standardized column names
    """
    print(f"⏳ Loading {name} dataset...")
    try:
        df = pd.read_csv(path)
        print(f"   ✓ Loaded {len(df):,} rows with {df.shape[1]} columns")
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found: {path}")
    except Exception as e:
        raise Exception(f"Error loading {name}: {str(e)}")
    # Standardize column names (strip whitespace, lowercase)
    df.columns = df.columns.str.strip().str.lower()
    # Map universal features to actual column names
    feature_mapping = {}
    for canonical_name, possible_names in UNIVERSAL_FEATURES.items():
        found = False
        for possible in possible_names:
            possible_lower = possible.lower().strip()
            if possible_lower in df.columns:
                feature_mapping[canonical_name] = possible_lower
                found = True
                break
        if not found:
            print(f"   ⚠ Warning: {canonical_name} not found in {name}")
    # Keep only universal features
    available_features = list(feature_mapping.values())
    if not available_features:
        raise ValueError(f"No universal features found in {name}")
    df_universal = df[available_features].copy()
    # Handle missing values and infinite values
    df_universal.replace([np.inf, -np.inf], np.nan, inplace=True)
    missing_percent = df_universal.isnull().sum().sum() / (df_universal.shape[0] * df_universal.shape[1]) * 100
    if missing_percent > 0:
        print(f"   ⚠ {missing_percent:.2f}% missing values detected")
    # For simplicity, drop rows with any missing values
    original_rows = len(df_universal)
    df_universal.dropna(inplace=True)
    if original_rows != len(df_universal):
        print(f"   ✓ Removed {original_rows - len(df_universal):,} rows with missing values")
    # Rename columns to canonical names
    reverse_mapping = {v: k for k, v in feature_mapping.items()}
    df_universal.rename(columns=reverse_mapping, inplace=True)
    print(f"   ✓ Preprocessed dataset has {len(df_universal):,} rows")
    return df_universal
# ==================== KDE ESTIMATION WITH ROBUST HANDLING ====================
def compute_kde_robust(data, grid_points=1000):
    """
    Compute Kernel Density Estimation with robust handling of singular matrices.
    Parameters:
    -----------
    data : numpy.array
        1D array of feature values
    grid_points : int
        Number of points for PDF evaluation
    Returns:
    --------
    tuple
        (grid, pdf_values)
    """
    # Remove any remaining NaN/Inf
    data_clean = data[~np.isnan(data) & ~np.isinf(data)]
    if len(data_clean) < 2:
        raise ValueError("Insufficient data points for KDE")
    # Handle constant values (zero variance)
    if np.std(data_clean) == 0:
        # Create a simple uniform distribution around the constant value
        grid = np.linspace(data_clean[0] - 1, data_clean[0] + 1, grid_points)
        pdf = np.zeros_like(grid)
        center_idx = len(grid) // 2
        pdf[center_idx-10:center_idx+10] = 1.0  # Small uniform distribution
        pdf = pdf / pdf.sum()
        return grid, pdf
    try:
        # Try standard KDE first
        kde = gaussian_kde(data_clean)
        # Define grid range with padding
        data_min, data_max = np.min(data_clean), np.max(data_clean)
        padding = (data_max - data_min) * 0.1  # 10% padding
        grid = np.linspace(data_min - padding, data_max + padding, grid_points)
        pdf = kde(grid)
        # Normalize to ensure sum ~= 1
        pdf = pdf / (np.sum(pdf) * (grid[1] - grid[0]))
        return grid, pdf
    except np.linalg.LinAlgError:
        # Fallback: use histogram-based PDF estimation
        print("   ⚠ Singular covariance detected, using histogram estimation")
        hist, bin_edges = np.histogram(data_clean, bins=50, density=True)
        grid = (bin_edges[:-1] + bin_edges[1:]) / 2
        pdf = hist
        return grid, pdf
# ==================== JS DIVERGENCE COMPUTATION ====================
def compute_js_divergence_between_datasets(dataset1, dataset2, feature, grid_points=1000):
    """
    Compute Jensen-Shannon divergence for a single feature between two datasets.
    Parameters:
    -----------
    dataset1, dataset2 : pandas.DataFrame
        Dataframes containing the feature
    feature : str
        Feature name
    grid_points : int
        Number of grid points for PDF alignment
    Returns:
    --------
    float
        JS divergence value (0 to 1)
    """
    # Extract feature values
    values1 = dataset1[feature].values.astype(float)
    values2 = dataset2[feature].values.astype(float)
    # Compute KDEs on a common grid
    try:
        grid1, pdf1 = compute_kde_robust(values1, grid_points)
        grid2, pdf2 = compute_kde_robust(values2, grid_points)
        # Create common grid covering both ranges
        all_values = np.concatenate([values1, values2])
        min_val, max_val = np.min(all_values), np.max(all_values)
        padding = (max_val - min_val) * 0.1
        common_grid = np.linspace(min_val - padding, max_val + padding, grid_points)
        # Re-evaluate PDFs on common grid
        pdf1_common = np.interp(common_grid, grid1, pdf1, left=1e-10, right=1e-10)
        pdf2_common = np.interp(common_grid, grid2, pdf2, left=1e-10, right=1e-10)
        # Normalize to probability distributions
        pdf1_common = pdf1_common / pdf1_common.sum()
        pdf2_common = pdf2_common / pdf2_common.sum()
        # Compute JS divergence
        js_div = jensenshannon(pdf1_common, pdf2_common)
        return js_div
    except Exception as e:
        print(f"   ⚠ Error computing JS divergence for {feature}: {str(e)}")
        return np.nan
# ==================== VISUALIZATION FUNCTIONS ====================
def plot_js_divergence_heatmap(js_matrix, datasets, feature_name):
    """
    Create a heatmap visualization of JS divergence matrix.
    Parameters:
    -----------
    js_matrix : numpy.ndarray
        JS divergence matrix (n_datasets x n_datasets)
    datasets : list
        Dataset names
    feature_name : str
        Name of the feature being analyzed
    """
    plt.figure(figsize=(8, 6))
    # Create mask for upper triangle (optional)
    mask = np.triu(np.ones_like(js_matrix, dtype=bool), k=1)
    # Plot heatmap
    sns.heatmap(js_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='YlOrRd',
                center=0.5,
                square=True,
                linewidths=1,
                cbar_kws={'label': 'JS Divergence'},
                xticklabels=datasets,
                yticklabels=datasets)
    plt.title(f'JS Divergence: {feature_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
def plot_feature_distributions(datasets_dict, feature_name):
    """
    Plot feature distributions for all datasets with dataset-specific colors.
    Parameters:
    -----------
    datasets_dict : dict
        Dictionary of {dataset_name: dataframe}
    feature_name : str
        Feature to visualize
    """
    plt.figure(figsize=(10, 6))
    for name, df in datasets_dict.items():
        if feature_name in df.columns:
            values = df[feature_name].dropna().values
            if len(values) > 1:
                # Compute KDE
                try:
                    grid, pdf = compute_kde_robust(values, 500)
                    plt.plot(grid, pdf, 
                            label=name, 
                            color=DATASET_COLORS[name],
                            linewidth=2,
                            alpha=0.8)
                except:
                    # Fallback to histogram
                    plt.hist(values, bins=50, density=True, 
                            alpha=0.5, label=f"{name} (hist)",
                            color=DATASET_COLORS[name])
    plt.xlabel(feature_name, fontsize=12)
    plt.ylabel('Probability Density', fontsize=12)
    plt.title(f'Distribution Comparison: {feature_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
def plot_aggregated_js_divergence(all_js_results, datasets):
    """
    Create aggregated visualization of JS divergence across all features.
    Parameters:
    -----------
    all_js_results : dict
        Dictionary containing JS divergence matrices for each feature
    datasets : list
        List of dataset names
    """
    # Calculate average JS divergence for each dataset pair
    n_datasets = len(datasets)
    avg_js_matrix = np.zeros((n_datasets, n_datasets))
    count_matrix = np.zeros((n_datasets, n_datasets))
    for feature, js_matrix in all_js_results.items():
        for i in range(n_datasets):
            for j in range(i+1, n_datasets):  # Upper triangle only
                if not np.isnan(js_matrix[i, j]):
                    avg_js_matrix[i, j] += js_matrix[i, j]
                    count_matrix[i, j] += 1
    # Compute averages
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_js_matrix = np.divide(avg_js_matrix, count_matrix)
        avg_js_matrix = np.where(np.isnan(avg_js_matrix), 0, avg_js_matrix)
    # Make symmetric
    avg_js_matrix = avg_js_matrix + avg_js_matrix.T
    # Plot aggregated heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(avg_js_matrix, 
                annot=True, 
                fmt='.3f',
                cmap='RdYlBu_r',
                square=True,
                linewidths=1,
                cbar_kws={'label': 'Average JS Divergence'},
                xticklabels=datasets,
                yticklabels=datasets)
    plt.title('Aggregated JS Divergence Across All Universal Features', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
# ==================== MAIN EXECUTION PIPELINE ====================
def main():
    """
    Main execution pipeline for JS divergence analysis.
    """
    print("=" * 70)
    print("JENSEN-SHANNON DIVERGENCE ANALYSIS")
    print("Universal DDoS Feature Set Evaluation")
    print("=" * 70)
    print()
    # ========== PHASE 1: DATA LOADING ==========
    print("📊 PHASE 1: DATASET LOADING & PREPROCESSING")
    print("-" * 50)
    datasets = {}
    for name, path in DATASET_PATHS.items():
        if path == "YOUR_PATH_HERE/" + name.lower().replace(" ", "_") + ".csv":
            print(f"   ⚠ Please update the path for {name} in DATASET_PATHS dictionary")
            continue
        try:
            df = load_and_preprocess_dataset(name, path)
            datasets[name] = df
        except Exception as e:
            print(f"   ✗ Failed to load {name}: {str(e)}")
    if len(datasets) < 2:
        print("❌ Need at least 2 datasets to perform JS divergence analysis")
        return
    dataset_names = list(datasets.keys())
    print(f"✓ Successfully loaded {len(datasets)} datasets")
    print()
    # ========== PHASE 2: FEATURE VALIDATION ==========
    print("🔍 PHASE 2: FEATURE AVAILABILITY CHECK")
    print("-" * 50)
    # Check which features are available in each dataset
    available_features = {}
    for canonical_name in UNIVERSAL_FEATURES.keys():
        available_in = []
        for name, df in datasets.items():
            if canonical_name in df.columns:
                available_in.append(name)
        if available_in:
            available_features[canonical_name] = available_in
            print(f"   ✓ {canonical_name}: available in {len(available_in)} datasets")
        else:
            print(f"   ✗ {canonical_name}: not found in any dataset")
    if not available_features:
        print("❌ No universal features found across datasets")
        return
    print()
    # ========== PHASE 3: JS DIVERGENCE COMPUTATION ==========
    print("📈 PHASE 3: JS DIVERGENCE COMPUTATION")
    print("-" * 50)
    all_js_results = {}
    n_datasets = len(dataset_names)
    for feature_name, datasets_with_feature in available_features.items():
        print(f"   Processing: {feature_name}")
        if len(datasets_with_feature) < 2:
            print(f"     ⚠ Skipping - needs at least 2 datasets")
            continue
        # Initialize JS matrix
        js_matrix = np.full((n_datasets, n_datasets), np.nan)
        # Compute pairwise JS divergence
        for i, name1 in enumerate(dataset_names):
            if name1 not in datasets_with_feature:
                continue
            for j, name2 in enumerate(dataset_names[i+1:], i+1):
                if name2 not in datasets_with_feature:
                    continue
                js_div = compute_js_divergence_between_datasets(
                    datasets[name1], 
                    datasets[name2], 
                    feature_name
                )
                js_matrix[i, j] = js_div
                js_matrix[j, i] = js_div  # Symmetric
        all_js_results[feature_name] = js_matrix
        print(f"     ✓ Completed JS divergence matrix")
    print()
    # ========== PHASE 4: VISUALIZATION ==========
    print("🎨 PHASE 4: VISUALIZATION")
    print("-" * 50)
    # Plot 1: Individual feature JS divergence heatmaps
    print("   Generating individual feature heatmaps...")
    for feature_name, js_matrix in all_js_results.items():
        plot_js_divergence_heatmap(js_matrix, dataset_names, feature_name)
    # Plot 2: Feature distribution comparisons
    print("   Generating feature distribution plots...")
    for feature_name in available_features.keys():
        if len(available_features[feature_name]) >= 2:
            plot_feature_distributions(datasets, feature_name)
    # Plot 3: Aggregated JS divergence
    if len(all_js_results) > 0:
        print("   Generating aggregated divergence visualization...")
        plot_aggregated_js_divergence(all_js_results, dataset_names)
    # ========== PHASE 5: SUMMARY STATISTICS ==========
    print("📋 PHASE 5: SUMMARY STATISTICS")
    print("-" * 50)
    # Calculate and display summary statistics
    for feature_name, js_matrix in all_js_results.items():
        # Extract upper triangle values (excluding diagonal and NaN)
        upper_triangle = js_matrix[np.triu_indices_from(js_matrix, k=1)]
        valid_values = upper_triangle[~np.isnan(upper_triangle)]
        if len(valid_values) > 0:
            mean_js = np.mean(valid_values)
            std_js = np.std(valid_values)
            min_js = np.min(valid_values)
            max_js = np.max(valid_values)
            print(f"   {feature_name}:")
            print(f"     Mean JS: {mean_js:.4f} ± {std_js:.4f}")
            print(f"     Range: [{min_js:.4f}, {max_js:.4f}]")
            print(f"     Comparisons: {len(valid_values)} valid pairs")
            print()
    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
# ==================== EXECUTION ====================
if __name__ == "__main__":
    # Execute main analysis
    main()

                             
