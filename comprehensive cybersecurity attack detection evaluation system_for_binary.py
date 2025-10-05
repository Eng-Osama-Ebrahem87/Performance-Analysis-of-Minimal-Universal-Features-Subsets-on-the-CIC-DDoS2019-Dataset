
'''
Professional Network Intrusion Detection System Evaluation
Test Strategy: Comprehensive evaluation of ML models for DDoS attack classification
Dataset: CIC-DDoS2019 compatible
Focus: Binary and Multi-class classification with robust evaluation metrics
'''

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.preprocessing import MinMaxScaler #fixed import


from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_curve, auc, 
                           roc_auc_score)


from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB

from itertools import cycle

import warnings

warnings.filterwarnings('ignore')

class NetworkIntrusionEvaluator:
    """
    Professional evaluator for network intrusion detection systems
    Test Strategy: Implements robust evaluation to prevent overfitting and ensure generalization
    """ 
    def __init__(self, test_size=0.3, random_state=42, cv_folds=5):
        """
        Initialize the evaluator with robust testing parameters
        Test Strategy:
        - Train-Test Split: Prevents data leakage
        - Stratified Sampling: Maintains class distribution
        - Cross-Validation: Ensures model generalization
        - Random State: Reproducible results
        """
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.models = self._initialize_models()
        #self.scaler = StandardScaler()
        #ValueError: Negative values in data passed to ComplementNB (input X)
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.results = {}
    def _initialize_models(self):
        """
        Initialize ML models with default parameters
        Models selected for their complementary strengths in classification
        """
        return {
            'KNN': KNeighborsClassifier(),
            'RF': RandomForestClassifier(),
            'LR': LogisticRegression(),
            'CNB': ComplementNB()
        }
    def load_and_preprocess_data(self, file_path):
        """
        Load and preprocess CIC-DDoS2019 compatible dataset
        Test Strategy:
        - Feature Selection: Based on network traffic characteristics
        - Handling Missing Values: Robust data cleaning
        - Label Encoding: Converts categorical labels to numerical
        - Feature Scaling: Standardization for better model performance
        """
        try:
            # Load dataset
            print("Loading dataset...")
            data = pd.read_csv(file_path)
            # Select relevant features for DDoS detection
            features = [
                'Packet Length Mean',
                'Average Packet Size', 
                'Bwd Packet Length Min',
                'Fwd Packets/s',
                'Min Packet Length',
                'Down/Up Ratio',
                'Label'
            ]
            # Filter available features
            available_features = [f for f in features if f in data.columns]
            if len(available_features) < 4:  # Minimum features required
                raise ValueError("Insufficient features in dataset")
            data = data[available_features]
            # Handle missing values
            data = data.dropna()
            # Encode labels
            self.labels = data['Label'].unique()
            data['Label'] = self.label_encoder.fit_transform(data['Label'])
            # Separate features and target
            X = data.drop('Label', axis=1)
            y = data['Label']
            print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
            print(f"Classes: {len(self.labels)} - {self.labels}")
            return X, y
        except Exception as e:
            print(f"Error loading data: {e}")
            return None, None
    def prepare_data(self, X, y):
        """
        Prepare data with robust train-test split and feature scaling
        Test Strategy:
        - Stratified Split: Maintains class distribution in splits
        - Feature Scaling: Prevents feature dominance
        - Data Leakage Prevention: Scaling fitted only on training data
        """
        # Stratified train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, 
            stratify=y, shuffle=True
        )
        # Scale features (fit only on training data to prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"Training set: {X_train_scaled.shape}")
        print(f"Test set: {X_test_scaled.shape}")
        return X_train_scaled, X_test_scaled, y_train, y_test
    def evaluate_model(self, model, X_train, X_test, y_train, y_test, model_name):
        """
        Comprehensive model evaluation with multiple metrics
        Test Strategy:
        - Cross-Validation: Assess generalization capability
        - Multiple Metrics: Comprehensive performance assessment
        - Confidence Intervals: Statistical significance
        """
        print(f"\n{'='*50}")
        print(f"Evaluating {model_name}")
        print(f"{'='*50}")
        # Cross-validation for generalization assessment
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        # Train model
        model.fit(X_train, y_train)
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        # Calculate False Positive Rate
        cm = confusion_matrix(y_test, y_pred)
        fp_rate = self.calculate_false_positive_rate(cm)
        # Store results
        model_results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positive_rate': fp_rate,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        self.print_metrics(model_name, accuracy, precision, recall, f1, fp_rate)
        return model_results
    def calculate_false_positive_rate(self, cm):
        """
        Calculate False Positive Rate for multi-class classification
        Test Strategy:
        - Macro-average FPR: Balanced assessment across all classes
        - Robust to class imbalance
        """
        fp_rates = []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            fp = np.sum(cm[:, i]) - cm[i, i]  # False positives for class i
            tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]  # True negatives
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fp_rates.append(fpr)
        return np.mean(fp_rates)  # Macro-average FPR
    def print_metrics(self, model_name, accuracy, precision, recall, f1, fp_rate):
        """Print comprehensive evaluation metrics"""
        print(f"\n{model_name} Performance Metrics:")
        print(f"Accuracy:    {accuracy:.4f}")
        print(f"Precision:   {precision:.4f}")
        print(f"Recall:      {recall:.4f}")
        print(f"F1-Score:    {f1:.4f}")
        print(f"FPR:         {fp_rate:.4f}")
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        Test Strategy:
        - Visual assessment of classification performance
        - Identification of misclassification patterns
        """
        n_models = len(self.results)
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.ravel()
        for idx, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_)
            axes[idx].set_title(f'{model_name} - Confusion Matrix')
            axes[idx].set_xlabel('Predicted Label')
            axes[idx].set_ylabel('True Label')
        # Remove empty subplots
        for idx in range(n_models, 4):
            fig.delaxes(axes[idx])
        plt.tight_layout()
        plt.show()
    def plot_roc_curves(self, X_test, y_test):
        """
        Plot ROC curves for all models and classes
        Test Strategy:
        - Multi-class ROC curves using One-vs-Rest approach
        - AUC calculation for comprehensive performance assessment
        - Visual comparison of model discrimination capability
        """
        n_classes = len(self.labels)
        # Set up the plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 6))
        axes = axes.ravel()
        colors = cycle(['blue', 'red', 'green', 'orange', 'purple', 'brown'])
        for idx, (model_name, results) in enumerate(self.results.items()):
            if results['y_pred_proba'] is None:
                continue
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test == i, results['y_pred_proba'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Plot ROC curves
            ax = axes[idx]
            for i, color in zip(range(n_classes), colors):
                ax.plot(fpr[i], tpr[i], color=color, lw=2,
                       label=f'Class {self.label_encoder.classes_[i]} (AUC = {roc_auc[i]:.2f})')
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model_name} - ROC Curves')
            ax.legend(loc="lower right")
            ax.grid(True)
        plt.tight_layout()
        plt.show()
    def plot_auc_comparison(self, X_test, y_test):
        """
        Plot AUC comparison across models and classes
        Test Strategy:
        - Quantitative comparison of model discrimination capability
        - Class-wise performance analysis
        """
        n_classes = len(self.labels)
        model_names = list(self.results.keys())
        # Calculate AUC for each model and class
        auc_scores = np.zeros((len(model_names), n_classes))
        for model_idx, (model_name, results) in enumerate(self.results.items()):
            if results['y_pred_proba'] is not None:
                for class_idx in range(n_classes):
                    auc_scores[model_idx, class_idx] = roc_auc_score(
                        y_test == class_idx, results['y_pred_proba'][:, class_idx]
                    )
        # Plot AUC comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(n_classes)
        width = 0.2
        for i, model_name in enumerate(model_names):
            offset = width * i
            rects = ax.bar(x + offset, auc_scores[i], width, label=model_name)
            ax.bar_label(rects, padding=3, fmt='%.2f')
        ax.set_xlabel('Classes')
        ax.set_ylabel('AUC Score')
        ax.set_title('AUC Score Comparison Across Models and Classes')
        ax.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax.set_xticklabels(self.label_encoder.classes_, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    def comprehensive_evaluation(self, file_path):
        """
        Execute comprehensive evaluation pipeline
        Test Strategy Implementation:
        1. Data Loading & Preprocessing: Ensures data quality
        2. Robust Train-Test Split: Prevents overfitting
        3. Multiple Model Evaluation: Comparative analysis
        4. Cross-Validation: Generalization assessment
        5. Comprehensive Metrics: Multi-faceted performance evaluation
        6. Visualization: Intuitive result interpretation
        """
        print("Starting Comprehensive Network Intrusion Detection Evaluation")
        print("Test Strategy: Preventing Overfitting and Ensuring Generalization")
        print("=" * 70)
        # Load and preprocess data
        X, y = self.load_and_preprocess_data(file_path)
        if X is None:
            return
        # Prepare data with robust splitting
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        # Evaluate all models
        for model_name, model in self.models.items():
            results = self.evaluate_model(model, X_train, X_test, y_train, y_test, model_name)
            self.results[model_name] = results
        # Generate comprehensive visualizations
        print("\nGenerating Comprehensive Visualizations...")
        self.plot_confusion_matrices()
        self.plot_roc_curves(X_test, y_test)
        self.plot_auc_comparison(X_test, y_test)
        # Print final comparison
        self.print_final_comparison()
    def print_final_comparison(self):
        """Print final comparative analysis of all models"""
        print("\n" + "="*70)
        print("FINAL MODEL COMPARISON")
        print("="*70)
        comparison_data = []
        for model_name, results in self.results.items():
            comparison_data.append({
                'Model': model_name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'FPR': f"{results['false_positive_rate']:.4f}",
                'CV Accuracy': f"{results['cv_scores'].mean():.4f} ± {results['cv_scores'].std():.4f}"
            })
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        # Identify best model based on F1-Score (balanced metric)
        best_model = max(self.results.items(), key=lambda x: x[1]['f1_score'])
        print(f"\nBest Performing Model: {best_model[0]} (F1-Score: {best_model[1]['f1_score']:.4f})")
# Example usage and test execution
if __name__ == "__main__":
    """
    Professional Test Execution Plan for Network Intrusion Detection System
    Test Strategy Summary:
    1. Data Quality Assurance: Proper preprocessing and feature selection
    2. Model Diversity: Multiple algorithms with complementary strengths
    3. Robust Validation: Cross-validation and stratified sampling
    4. Comprehensive Metrics: Multi-faceted performance assessment
    5. Visualization: Intuitive result interpretation
    6. Generalization Focus: Techniques to prevent overfitting
    """
    # Initialize the evaluator
    evaluator = NetworkIntrusionEvaluator(test_size=0.3, cv_folds=5)
    # Execute comprehensive evaluation


############################################################  CIC-DDoS2019_CSV  from original   - -  All files Available to test ................................

    #data_path = r"E:\Cic-DDos2019 Original\03-11\Portmap_Pre.csv"
    #data_path = r"E:\Cic-DDos2019 Original\03-11\Portmap_balanced.csv" 
    data_path = r"E:\Cic-DDos2019 Original\03-11\Portmap_undersampling.csv"
 
    #data_path = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_Pre.csv" 
    #data_path = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_balanced.csv"
    #data_path = r"E:\Cic-DDos2019 Original\03-11\NetBIOS_undersampling.csv"

    #data_path = r"E:\Cic-DDos2019 Original\03-11\Syn_Pre.csv"
    #data_path = r"E:\Cic-DDos2019 Original\03-11\Syn_balanced.csv"
    #data_path = r"E:\Cic-DDos2019 Original\03-11\Syn_undersampling.csv"
 
###########################################################################################



    print("PROFESSIONAL TEST STRATEGY FOR NETWORK INTRUSION DETECTION")
    print("=" * 70)
    print("Key Strategies to Prevent Overfitting and Ensure Generalization:")
    print("1. Stratified Train-Test Split: Maintains class distribution")
    print("2. Cross-Validation: 5-fold stratified CV for generalization assessment")
    print("3. Feature Scaling: Prevents feature dominance")
    print("4. Multiple Metrics: Comprehensive performance evaluation")
    print("5. Model Diversity: Complementary classification algorithms")
    print("6. Data Leakage Prevention: Proper preprocessing pipeline")
    print("=" * 70)
    # Run evaluation
    evaluator.comprehensive_evaluation(data_path)
'''
This professional implementation includes:
## **Comprehensive Test Strategy:**
### **1. Overfitting Prevention:**
- **Stratified K-Fold Cross-Validation**
- **Proper Train-Test Split** (no data leakage)
- **Feature Scaling** on training data only
- **Multiple Model Evaluation** for robustness
### **2. Generalization Assurance:**
- **Cross-Validation Scores** with confidence intervals
- **Stratified Sampling** maintains class distribution
- **Comprehensive Metrics** beyond just accuracy
- **Visual Validation** through multiple plots
### **3. CIC-DDoS2019 Compatibility:**
- **Relevant Feature Selection** for DDoS detection
- **Multi-class Handling** for various attack types
- **Robust Preprocessing** for real-world network data
### **4. Professional Evaluation:**
- **ROC Curves & AUC** for binary and multi-class
- **Confusion Matrices** for detailed analysis
- **False Positive Rate** calculation
- **Comparative Model Analysis**
### **Key Features:**
- **Dynamic** handling of binary and multi-class classification
- **Comprehensive** metric calculation
- **Professional** visualization
- **Robust** error handling
- **Clear** documentation and comments
To use this code, simply replace the `dataset_path` with your actual CIC-DDoS2019 dataset file path. The code will automatically handle both binary and multi-class scenarios and provide a complete professional evaluation of your intrusion detection system.
'''
