import os
import sys
import pandas as pd
import numpy as np
import warnings
import argparse
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

#current directory + '/data/'
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run classification with specified class imbalance ratios')
    parser.add_argument('--imbalance_ratios', nargs='+', type=float, default=[0.05, 0.1, 0.2],
                        help='List of minority class ratios (e.g., 0.05 for 5% minority class)')
    parser.add_argument('--betas', nargs='+', type=float, default=[1,2.5, 5, 7.5, 10],
                        help='List of beta values for F-beta score')
    parser.add_argument('--n_splits', type=int, default=10,
                        help='Number of cross-validation folds')
    parser.add_argument('--output_file', type=str, default='classification_results.csv',
                        help='Output file name for results')
    return parser.parse_args()

def load_data_for_modality(path, modality):
    """
    Loads train and validation data for a given modality.
    """
    try:
        all_files = os.listdir(path)
    except FileNotFoundError:
        return None, None

    train_file = next((os.path.join(path, f) for f in all_files if "train" in f and f.startswith(modality) and f.endswith('.xlsx')), None)
    val_file = next((os.path.join(path, f) for f in all_files if "val" in f and f.startswith(modality) and f.endswith('.xlsx')), None)

    if not train_file or not val_file:
        return None, None

    data_train = pd.read_excel(train_file).dropna(axis=1, how='all')
    data_val = pd.read_excel(val_file).dropna(axis=1, how='all')

    if modality != 'spo2':
        if 1093 in data_train.index:
            print(f"Removing row 1094 from {modality} data as it is an outlier.")
            data_train = data_train.drop(index=1093, errors='ignore')

    print(f"Loaded file: {train_file} with shape {data_train.shape} for modality '{modality}'")
    print(f"Loaded file: {val_file} with shape {data_val.shape} for modality '{modality}'")
    return data_train, data_val

# --- Main Data Aggregation ---
modalities = ['bvp', 'eda', 'resp', 'spo2']
print(f"Processing specified modalities for stacking: {modalities}")

# Dictionaries to hold the COMBINED (train + val) data
full_train_data = {}
full_train_pids = {}
y_full_train = None

for m in modalities:
    print(f"\nLoading data for {m}...")
    d_train, d_val = load_data_for_modality(DATA_PATH, m)

    if d_train is not None and d_val is not None:
        y_combined = pd.concat([d_train['class'], d_val['class']], ignore_index=True)
        pids_combined = pd.concat([d_train['participant_id'], d_val['participant_id']], ignore_index=True)
        X_combined = pd.concat([d_train.drop(columns=['participant_id', 'class']),
                              d_val.drop(columns=['participant_id', 'class'])], ignore_index=True)

        full_train_pids[m] = pids_combined
        full_train_data[m] = X_combined

        if y_full_train is None:
            y_full_train = y_combined.to_numpy()

# Merge all modalities
X_full_train = pd.concat([full_train_data[m] for m in modalities], axis=1)
print(f"\nFull training data shape after merging modalities: {X_full_train.shape}")

# --- Preprocessing Pipeline ---
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from smote_variants.oversampling import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

class DropHighNaNs(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.2):
        self.threshold = threshold
        self.cols_to_drop_ = []
    
    def fit(self, X, y=None):
        nan_percent = np.isnan(X).mean(axis=0)
        self.cols_to_drop_ = np.where(nan_percent >= self.threshold)[0]
        return self
    
    def transform(self, X):
        return np.delete(X, self.cols_to_drop_, axis=1)
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return []
        return np.delete(input_features, self.cols_to_drop_)

# Create the preprocessing pipeline
preprocessor = Pipeline([
    ('cat_processing', ColumnTransformer([
        ('encode_cats', Pipeline([
            ('impute_cat', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder())
        ]), make_column_selector(dtype_include=['object', 'category']))
    ], remainder='passthrough')),
    ('drop_high_nans', DropHighNaNs(threshold=0.2)),
    ('impute_num', KNNImputer(n_neighbors=5)),
    ('remove_low_variance', VarianceThreshold()),
     ('normalize', StandardScaler()),
    ('feature_selection', SelectFromModel(
        RandomForestClassifier(n_estimators=100),
        max_features=100,
        threshold=0.01
    ))
])

# Fit and transform the full training data
X = preprocessor.fit_transform(X_full_train, y_full_train)
print(f"\nProcessed full training data shape: {X.shape}")

# Process labels
y = y_full_train.copy()
indexes = y != 2
X = X[indexes]
y = y[indexes]

def create_class_imbalance(X, y, minority_ratio, random_seed=342):
    """Create precise class imbalance with specified minority class ratio."""
    np.random.seed(random_seed)
    minority_class = 1
    majority_class = 0
    
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Calculate target number of minority samples
    total_samples = len(y)
    target_minority = int(minority_ratio * total_samples)
    
    # If target_minority is greater than available minority samples, keep all
    if target_minority >= len(minority_indices):
        selected_minority = minority_indices
    else:
        selected_minority = np.random.choice(minority_indices, size=target_minority, replace=False)
    
    # Calculate required majority samples to maintain total size
    target_majority = total_samples - len(selected_minority)
    if target_majority >= len(majority_indices):
        selected_majority = majority_indices
    else:
        selected_majority = np.random.choice(majority_indices, size=target_majority, replace=False)
    
    # Combine indices and sort
    final_indices = np.concatenate([selected_minority, selected_majority])
    final_indices = np.sort(final_indices)
    
    X_balanced = X[final_indices]
    y_balanced = y[final_indices]
    
    print(f"Class distribution for minority ratio {minority_ratio}:")
    print(np.bincount(y_balanced))
    
    return X_balanced, y_balanced

def get_tpr_tnr(y_true, y_pred):
    """Calculate True Positive Rate (TPR) and True Negative Rate (TNR)."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp / (tp + fn), tn / (tn + fp)

def resample_data(X_train, y_train, method, beta=None):
    """Handle data resampling based on the specified method."""
    if method is None:
        return X_train, y_train, None
    if method == Genetic_BETA_resampling_framework_v2:
        sol, X_resampled, y_resampled = method(Beta=beta,func1_toremove=False,func2_toremove=False,func1_pmf=pmf_local_neighbourhood,func2_pmf=pmf_prowsyn,func1_generate_samples=generate_samples_local_neighboorhood,func2_generate_samples=generate_samples_prowsyn).resample(X_train, y_train)
        return X_resampled, y_resampled, sol
    X_resampled, y_resampled = method(n_jobs=16).fit_resample(X_train, y_train)
    return X_resampled, y_resampled, None

def train_and_evaluate_classifier(clf, X_train, y_train, X_test, y_test, beta, class_weights=None, is_adaboost=False):
    """Train a classifier and compute evaluation metrics."""
    if is_adaboost:
        base_model = DecisionTreeClassifier(max_depth=5, class_weight=class_weights)
        model = clf(estimator=base_model,n_estimators=10).fit(X_train, y_train)
    else:
        model = clf(class_weight=class_weights).fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return {
        'fbeta': fbeta_score(y_test, y_pred, beta=beta),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'tpr_tnr': get_tpr_tnr(y_test, y_pred)
    }

def main(X, y, imbalance_ratios, betas, n_splits):
    """Main function to run classification experiments with different imbalance ratios."""
    resampling_methods = [None, SMOTE, SMOTE_IPF, Genetic_BETA_resampling_framework_v2]
    classifiers = [RandomForestClassifier, SVC, ExtraTreesClassifier, AdaBoostClassifier]
    kf = StratifiedKFold(n_splits=n_splits)
    
    # Initialize results storage
    all_results = []
    
    for ratio in imbalance_ratios:
        # Create class imbalance
        X_balanced, y_balanced = create_class_imbalance(X, y, ratio)
        
        # Initialize result arrays for this ratio
        results = np.zeros((len(betas), n_splits, len(resampling_methods) + len(classifiers)))
        precision_results = np.zeros_like(results)
        recall_results = np.zeros_like(results)
        resulting_dist = np.zeros((n_splits, 3),dtype=object)  # Assuming 13 genes in the solution
        
        for b, beta in enumerate(betas):
            for fold, (train_idx, test_idx) in enumerate(kf.split(X_balanced, y_balanced)):
                X_train, X_test = X_balanced[train_idx], X_balanced[test_idx]
                y_train, y_test = y_balanced[train_idx], y_balanced[test_idx]
                
                # Get class weights
                counts = np.bincount(y_train.astype(np.int16))
                maj, min_class = np.argmax(counts), np.argmin(counts)
                class_weights = {maj: 1, min_class: beta**2}
                
                # Evaluate resampling methods
                for k, method in enumerate(resampling_methods):
                    X_resampled, y_resampled, sol = resample_data(X_train, y_train, method, beta)
                    #base_estimator = DecisionTreeClassifier(max_depth=5,random_state=1)
                    model = RandomForestClassifier(random_state=1)
                    model.fit(X_resampled, y_resampled)
                    y_pred = model.predict(X_test)
                    
                    results[b, fold, k] = fbeta_score(y_test, y_pred, beta=beta)
                    precision_results[b, fold, k] = precision_score(y_test, y_pred)
                    recall_results[b, fold, k] = recall_score(y_test, y_pred)
                    
                    if method == Genetic_BETA_resampling_framework_v2 and sol is not None:
                        resulting_dist[fold] = sol
                
                print(f"Ratio: {ratio} | Beta: {beta} | Fold: {fold} | Methods performance: {results[b, fold, :len(resampling_methods)]}")
                
                # Evaluate classifiers with class weights
                for c, clf in enumerate(classifiers):
                    try:
                        metrics = train_and_evaluate_classifier(
                            clf, X_train, y_train, X_test, y_test, beta,
                            class_weights=class_weights, is_adaboost=(c == 3)
                        )
                        idx = c + len(resampling_methods)
                        results[b, fold, idx] = metrics['fbeta']
                        precision_results[b, fold, idx] = metrics['precision']
                        recall_results[b, fold, idx] = metrics['recall']
                    except Exception as e:
                        print(f"Error in classifier {clf.__name__}: {e}")
            
            # Print all final results for this beta
            print(f"\nFinal results for ratio={ratio}, beta={beta}:")
            print("Method | Mean F-beta (Std) | Mean Precision (Std) | Mean Recall (Std) | Mean TPR (Std) | Mean TNR (Std)")
            print("-" * 80)
            for m, method in enumerate(resampling_methods + classifiers):
                method_name = method.__name__ if method is not None else 'NoResampling'
                mean_fbeta = np.mean(results[b, :, m])
                std_fbeta = np.std(results[b, :, m])
                mean_precision = np.mean(precision_results[b, :, m])
                std_precision = np.std(precision_results[b, :, m])
                mean_recall = np.mean(recall_results[b, :, m])
                std_recall = np.std(recall_results[b, :, m])
                print(f"{method_name:<25} | {mean_fbeta:.4f} ({std_fbeta:.4f}) | "
                      f"{mean_precision:.4f} ({std_precision:.4f}) | {mean_recall:.4f} ({std_recall:.4f}) | ")
            print("-" * 80)
        
        # Store results for this ratio
        for b, beta in enumerate(betas):
            for m, method in enumerate(resampling_methods + classifiers):
                method_name = method.__name__ if method is not None else 'NoResampling'
                all_results.append({
                    'imbalance_ratio': ratio,
                    'beta': beta,
                    'method': method_name,
                    'mean_fbeta': np.mean(results[b, :, m]),
                    'std_fbeta': np.std(results[b, :, m]),
                    'mean_precision': np.mean(precision_results[b, :, m]),
                    'std_precision': np.std(precision_results[b, :, m]),
                    'mean_recall': np.mean(recall_results[b, :, m]),
                    'std_recall': np.std(recall_results[b, :, m]),
                })
    
    # Save results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(args.output_file, index=False)
    print(f"Results saved to {args.output_file}")
    
    return results, precision_results, recall_results, resulting_dist

if __name__ == "__main__":
    args = parse_arguments()
    results, precision_results, recall_results, resulting_dist = main(
        X, y, args.imbalance_ratios, args.betas, args.n_splits
    )
    