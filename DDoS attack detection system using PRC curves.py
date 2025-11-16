

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.metrics import precision_recall_curve, average_precision_score, classification_report

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB


from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')
class DDOSEvaluator:
    """
    A comprehensive evaluator for DDoS attack detection systems
    Handles binary classification with PRC curve analysis
    """
    def __init__(self, random_state=42):
        """Initialize the evaluator with models and parameters"""
        self.random_state = random_state
        self.models = self._initialize_models()
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    def _initialize_models(self):
        """Initialize machine learning models with default parameters"""
        models = {
            'RF': RandomForestClassifier(),
            'KNN': KNeighborsClassifier(),
            'LR': LogisticRegression(),
            'CNB': ComplementNB()
        }
        return models
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess CIC-DDoS 2019 dataset
        Args:
            file_path (str): Path to the dataset file
        """
        print("Loading and preprocessing data...")
        # Load dataset
        try:
            self.data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
        # Select specified features
        required_features = [
            #'Packet Length Mean', #for Other Datasets
            #'Average Packet Size', 
            #'Bwd Packet Length Min',
            #'Fwd Packets/s',
            #'Min Packet Length',
            #'Down/Up Ratio',
            #'Label'
            'Packet_Length_Mean', ## FOR CIC-IoT 2023 Dataset 
            'Average_Packet_Size',
            'Bwd_Packet_Length_Min',
            'Fwd_Packets_per_second',
            'Min_Packet_Length',
            'Down_Up_Ratio',   
            'Label'
        ]
        # Check if required features exist
        missing_features = [f for f in required_features if f not in self.data.columns]
        if missing_features:
            print(f"Missing features: {missing_features}")
            return False
        self.data = self.data[required_features]
        # Handle missing values
        self._handle_missing_values()
        # Remove duplicates
        initial_size = len(self.data)
        self.data = self.data.drop_duplicates()
        print(f"Removed {initial_size - len(self.data)} duplicate rows")
        # Encode labels
        self._encode_labels()
        # Remove infinite values
        self._remove_infinite_values()
        print(f"Final dataset shape: {self.data.shape}")
        return True
    def _handle_missing_values(self):
        """Handle missing values in the dataset"""
        missing_count = self.data.isnull().sum().sum()
        if missing_count > 0:
            print(f"Found {missing_count} missing values. Removing...")
            self.data = self.data.dropna()
    def _encode_labels(self):
        """Encode target labels"""
        original_labels = self.data['Label'].unique()
        print(f"Original labels: {original_labels}")
        # Convert to binary classification (Attack vs Normal)
        # Assuming 'Benign' is normal traffic and others are attacks
        if 'Benign' in self.data['Label'].values:
            self.data['Label'] = self.data['Label'].apply(
                lambda x: 0 if x == 'Benign' else 1
            )
        else:
            # If no explicit 'Benign', use label encoding
            self.data['Label'] = self.label_encoder.fit_transform(self.data['Label'])
        print(f"Class distribution:\n{self.data['Label'].value_counts()}")
    def _remove_infinite_values(self):
        """Remove infinite values from the dataset"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(self.data[numeric_cols]).any(axis=1)
        if inf_mask.any():
            print(f"Removing {inf_mask.sum()} rows with infinite values")
            self.data = self.data[~inf_mask]
    def prepare_features(self):
        """Prepare features and target variables"""
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        return X, y
    def evaluate_models(self, test_size=0.3, cv_folds=5):
        """
        Evaluate all models using cross-validation and holdout test set
        Args:
            test_size (float): Proportion of test set
            cv_folds (int): Number of cross-validation folds
        """
        X, y = self.prepare_features()
        # Stratified train-test split to maintain class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        for model_name, model in self.models.items():
            print(f"\nEvaluating {model_name}...")
            # Create pipeline with scaling (except for CNB which handles scaling differently)
            if model_name != 'CNB':
                pipeline = Pipeline([
                    ('scaler', StandardScaler()),
                    ('classifier', model)
                ])
            else:
                pipeline = Pipeline([
                    ('classifier', model)
                ])
            # Cross-validation with stratified k-fold
            cv_scores = cross_val_score(
                pipeline, X_train, y_train,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state),
                scoring='f1_macro'
            )
            # Train final model
            pipeline.fit(X_train, y_train)
            # Predict probabilities
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
            # Calculate metrics
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            avg_precision = average_precision_score(y_test, y_pred_proba)
            # Store results
            self.results[model_name] = {
                'model': pipeline,
                'precision': precision,
                'recall': recall,
                'avg_precision': avg_precision,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            print(f"Average Precision: {avg_precision:.4f}")
            print(f"CV F1 Scores: {cv_scores}")
            print(f"CV Mean ± Std: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    def plot_prc_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(12, 8))
        colors = {'RF': 'blue', 'KNN': 'red', 'LR': 'green', 'CNB': 'orange'}
        for model_name, result in self.results.items():
            plt.plot(
                result['recall'], 
                result['precision'],
                label=f'{model_name} (AP = {result["avg_precision"]:.3f})',
                color=colors[model_name],
                linewidth=2
            )
        plt.xlabel('Recall', fontsize=12, fontweight='bold')
        plt.ylabel('Precision', fontsize=12, fontweight='bold')
        plt.title('Precision-Recall Curves for DDoS Attack Detection\n', 
                 fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        # Add no-skill line
        no_skill = len(self.data[self.data['Label'] == 1]) / len(self.data)
        plt.axhline(y=no_skill, color='black', linestyle='--', alpha=0.8, 
                   label=f'No Skill (AP = {no_skill:.3f})')
        plt.legend()
        plt.tight_layout()
        plt.show()
    def print_detailed_results(self):
        """Print detailed evaluation results"""
        print("\n" + "="*60)
        print("DETAILED EVALUATION SUMMARY")
        print("="*60)
        results_summary = []
        for model_name, result in self.results.items():
            results_summary.append({
                'Model': model_name,
                'Avg Precision': f"{result['avg_precision']:.4f}",
                'CV Mean F1': f"{result['cv_mean']:.4f}",
                'CV Std F1': f"{result['cv_std']:.4f}"
            })
        results_df = pd.DataFrame(results_summary)
        print(results_df.to_string(index=False))
        # Find best model
        best_model = max(self.results.items(), key=lambda x: x[1]['avg_precision'])
        print(f"\nBest Model: {best_model[0]} with Average Precision: {best_model[1]['avg_precision']:.4f}")
def main():
    """Main execution function"""
    # Initialize evaluator
    evaluator = DDOSEvaluator(random_state=42)
    # Load and preprocess data
    # Replace with actual path to your CIC-DDoS 2019 dataset

    #dataset_path = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_balanced.csv"
    
    #dataset_path = r"E:\SDN-DDoS_Traffic_Dataset from Mendeley\SDN-DDoS_With_CIC_Features.csv"

    dataset_path = r"E:\CIC-IoT 2023\CIC-IoT2023 form kaggle\test_Pre_to_binary.csv"

    #dataset_path = r"E:\VeReMi Dataset\VeReMi and BSMList\Main_data_shuffled_CIC_Features_to_binary.csv"


    
    if not evaluator.load_and_preprocess_data(dataset_path):
        print("Failed to load data. Please check the file path and format.")
        return
    # Evaluate models
    evaluator.evaluate_models(test_size=0.3, cv_folds=5)
    # Plot results
    evaluator.plot_prc_curves()
    # Print summary
    evaluator.print_detailed_results()
if __name__ == "__main__":
    main()


''''
This professional code includes:
Key Features:
1. Robust Data Handling:
   · Missing value treatment
   · Duplicate removal
   · Infinite value handling
   · Binary label encoding
2. Overfitting Prevention:
   · Stratified train-test split
   · Cross-validation with stratified k-fold
   · Proper data preprocessing pipeline
3. Model Evaluation:
   · Four ML models (RF, KNN, LR, CNB) with default parameters
   · Precision-Recall Curve analysis
   · Average Precision scores
   · Cross-validation results
4. Visualization:
   · Professional PRC curves
   · Model comparison with AP scores
   · Clear labeling and styling
Usage Instructions:
1. Replace dataset_path with your actual CIC-DDoS 2019 dataset path
2. Ensure the dataset contains the specified features
3. The code automatically handles binary classification (Benign vs Attack)
Outputs:
· PRC curves for all models
· Cross-validation scores
· Average precision metrics
· Model performance comparison
The code is specifically designed for the CIC-DDoS 2019 dataset characteristics and follows best practices for machine learning evaluation while avoiding common pitfalls like overfitting and poor generalization.

'''


