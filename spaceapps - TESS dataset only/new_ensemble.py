import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import joblib
warnings.filterwarnings('ignore')


class AdvancedEnsembleTESSClassifier:
    """
    Advanced ensemble classification system for TESS exoplanet detection.
    Implements multiple ensemble strategies including voting, stacking, and weighted ensembles.
    Supports binary or multi-class classification based on the data.
    """
    
    def __init__(self, train_path='data/train.csv', 
                 val_path='data/validation.csv', 
                 test_path='data/test.csv'):
        """Initialize the classifier with data paths."""
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.base_models = {}
        self.ensemble_models = {}
        self.all_results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        self.n_classes = None
        
    def load_and_preprocess_data(self):
        """Load and preprocess the train, validation, and test datasets."""
        print("="*80)
        print("LOADING AND PREPROCESSING TESS DATA")
        print("="*80)
        
        train_df = pd.read_csv(self.train_path)
        val_df = pd.read_csv(self.val_path)
        test_df = pd.read_csv(self.test_path)
        
        print(f"\nTrain set: {train_df.shape}")
        print(f"Validation set: {val_df.shape}")
        print(f"Test set: {test_df.shape}")
        
        # Separate features and target
        feature_cols = [col for col in train_df.columns 
                       if col not in ['toi', 'disposition_category']]
        
        X_train = train_df[feature_cols]
        y_train = train_df['disposition_category']
        X_val = val_df[feature_cols]
        y_val = val_df['disposition_category']
        X_test = test_df[feature_cols]
        y_test = test_df['disposition_category']
        
        print(f"\nNumber of features: {len(feature_cols)}")
        print(f"\nClass distribution in training set:")
        for label, count in y_train.value_counts().items():
            print(f"  {label}: {count}")
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_val_encoded = self.label_encoder.transform(y_val)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Store number of classes
        self.n_classes = len(self.label_encoder.classes_)
        
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                 self.label_encoder.transform(self.label_encoder.classes_)))
        print(f"\nLabel encoding: {label_mapping}")
        print(f"Number of classes: {self.n_classes}")
        
        # Handle missing values and scale
        print("\nHandling missing values with median imputation...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        X_test_imputed = self.imputer.transform(X_test)
        
        print("Scaling features with StandardScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train_imputed)
        X_val_scaled = self.scaler.transform(X_val_imputed)
        X_test_scaled = self.scaler.transform(X_test_imputed)
        
        print("\nData preprocessing completed!")
        
        return (X_train_scaled, y_train_encoded, 
                X_val_scaled, y_val_encoded,
                X_test_scaled, y_test_encoded)
    
    def initialize_base_models(self):
        """Initialize base classification models."""
        print("\n" + "="*80)
        print("INITIALIZING BASE MODELS")
        print("="*80)
        
        self.base_models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=10,
                random_state=42,
                n_jobs=-1
            ),
            
            'XGBoost': XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                eval_metric='logloss' if self.n_classes == 2 else 'mlogloss',
                n_jobs=-1
            ),
            
            'LightGBM': LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=-1,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            
            'AdaBoost': AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            ),
            
            'Neural Network': MLPClassifier(
                hidden_layer_sizes=(128, 64, 32),
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        }
        
        print(f"\nInitialized {len(self.base_models)} base models:")
        for i, name in enumerate(self.base_models.keys(), 1):
            print(f"  {i}. {name}")
    
    def calculate_auc_score(self, y_true, y_pred_proba):
        """Calculate AUC score handling both binary and multi-class cases."""
        if self.n_classes == 2:
            # For binary classification, use probability of positive class
            return roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # For multi-class, use OVR strategy
            return roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
    
    def train_base_models(self, X_train, y_train, X_val, y_val):
        """Train all base models and evaluate on validation set."""
        print("\n" + "="*80)
        print("TRAINING BASE MODELS")
        print("="*80)
        
        base_results = {}
        
        for name, model in self.base_models.items():
            print(f"\nTraining: {name}")
            model.fit(X_train, y_train)
            
            # Validation predictions
            y_val_pred = model.predict(X_val)
            y_val_proba = model.predict_proba(X_val)
            
            val_acc = accuracy_score(y_val, y_val_pred)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted' if self.n_classes > 2 else 'binary')
            val_auc = self.calculate_auc_score(y_val, y_val_proba)
            
            base_results[name] = {
                'model': model,
                'val_accuracy': val_acc,
                'val_f1': val_f1,
                'val_auc': val_auc
            }
            
            print(f"  Validation Accuracy: {val_acc:.4f}")
            print(f"  Validation F1: {val_f1:.4f}")
            print(f"  Validation AUC: {val_auc:.4f}")
        
        return base_results
    
    def create_ensemble_models(self, base_results):
        """Create various ensemble models combining base models."""
        print("\n" + "="*80)
        print("CREATING ENSEMBLE MODELS")
        print("="*80)
        
        # Get top 5 models based on validation F1 score
        top_models = sorted(base_results.items(), 
                           key=lambda x: x[1]['val_f1'], 
                           reverse=True)[:5]
        
        print("\nTop 5 models selected for ensemble:")
        for i, (name, results) in enumerate(top_models, 1):
            print(f"  {i}. {name} (F1: {results['val_f1']:.4f})")
        
        # Create estimator list for ensembles
        estimators = [(name, results['model']) for name, results in top_models]
        
        # 1. Hard Voting Classifier
        self.ensemble_models['Voting (Hard)'] = VotingClassifier(
            estimators=estimators,
            voting='hard',
            n_jobs=-1
        )
        
        # 2. Soft Voting Classifier
        self.ensemble_models['Voting (Soft)'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        # 3. Weighted Voting (based on validation F1 scores)
        weights = [results['val_f1'] for _, results in top_models]
        self.ensemble_models['Voting (Weighted)'] = VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=weights,
            n_jobs=-1
        )
        
        # 4. Stacking Classifier with Logistic Regression
        self.ensemble_models['Stacking (LogReg)'] = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42, max_iter=1000),
            cv=5,
            n_jobs=-1
        )
        
        # 5. Stacking Classifier with Random Forest
        self.ensemble_models['Stacking (RF)'] = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=100, random_state=42),
            cv=5,
            n_jobs=-1
        )
        
        # 6. Stacking Classifier with XGBoost
        eval_metric = 'logloss' if self.n_classes == 2 else 'mlogloss'
        self.ensemble_models['Stacking (XGB)'] = StackingClassifier(
            estimators=estimators,
            final_estimator=XGBClassifier(n_estimators=100, random_state=42, eval_metric=eval_metric),
            cv=5,
            n_jobs=-1
        )
        
        print(f"\nCreated {len(self.ensemble_models)} ensemble models:")
        for i, name in enumerate(self.ensemble_models.keys(), 1):
            print(f"  {i}. {name}")
    
    def train_and_evaluate_all(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """Train and evaluate both base and ensemble models."""
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING ALL MODELS")
        print("="*80)
        
        # Train base models first
        base_results = self.train_base_models(X_train, y_train, X_val, y_val)
        
        # Create ensemble models based on base model performance
        self.create_ensemble_models(base_results)
        
        # Evaluate base models on test set
        print("\n" + "="*80)
        print("EVALUATING BASE MODELS ON TEST SET")
        print("="*80)
        
        for name, info in base_results.items():
            model = info['model']
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted' if self.n_classes > 2 else 'binary')
            auc = self.calculate_auc_score(y_test, y_proba)
            
            self.all_results[f"Base: {name}"] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_proba,
                'true_labels': y_test,
                'type': 'base'
            }
            
            print(f"\n{name}:")
            print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Test F1: {f1:.4f}")
            print(f"  Test AUC: {auc:.4f}")
        
        # Train and evaluate ensemble models
        print("\n" + "="*80)
        print("TRAINING AND EVALUATING ENSEMBLE MODELS")
        print("="*80)
        
        for name, model in self.ensemble_models.items():
            print(f"\nTraining: {name}")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            # Handle probability prediction based on voting type
            if 'Hard' in name:
                # For hard voting, use one-hot encoded predictions for AUC calculation
                from sklearn.preprocessing import label_binarize
                
                if self.n_classes == 2:
                    # For binary classification, label_binarize returns single column
                    # We need to create the probability-like representation
                    y_pred_binary = label_binarize(y_pred, classes=range(self.n_classes))
                    # Convert to 2-column format for consistency
                    y_proba = np.hstack([1 - y_pred_binary, y_pred_binary])
                    auc = roc_auc_score(y_test, y_pred_binary.ravel())
                else:
                    y_pred_binary = label_binarize(y_pred, classes=range(self.n_classes))
                    y_test_binary = label_binarize(y_test, classes=range(self.n_classes))
                    auc = roc_auc_score(y_test_binary, y_pred_binary, average='weighted')
                    y_proba = y_pred_binary
            else:
                y_proba = model.predict_proba(X_test)
                auc = self.calculate_auc_score(y_test, y_proba)
            
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted' if self.n_classes > 2 else 'binary')
            
            self.all_results[f"Ensemble: {name}"] = {
                'model': model,
                'accuracy': accuracy,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_proba,
                'true_labels': y_test,
                'type': 'ensemble'
            }
            
            print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"  Test F1: {f1:.4f}")
            print(f"  Test AUC: {auc:.4f}")
        
        print("\n" + "="*80)
        print("ALL MODELS TRAINED AND EVALUATED")
        print("="*80)
    
    def plot_comprehensive_comparison(self, save_path='tess_comprehensive_comparison.png'):
        """Create comprehensive comparison of all models."""
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE COMPARISON PLOT")
        print("="*80)
        
        # Prepare data
        model_names = []
        accuracies = []
        f1_scores = []
        auc_scores = []
        model_types = []
        
        for name, result in sorted(self.all_results.items(), 
                                   key=lambda x: x[1]['accuracy']):
            model_names.append(name.replace('Base: ', '').replace('Ensemble: ', ''))
            accuracies.append(result['accuracy'] * 100)
            f1_scores.append(result['f1'] * 100)
            auc_scores.append(result['auc'] * 100)
            model_types.append(result['type'])
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(20, 8))
        
        # Colors based on model type
        colors = ['#3498db' if t == 'base' else '#e74c3c' for t in model_types]
        
        # Plot 1: Accuracy
        bars1 = axes[0].barh(range(len(model_names)), accuracies, color=colors, edgecolor='black')
        axes[0].set_xlabel('Accuracy (%)', fontsize=12, fontweight='bold')
        axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        axes[0].set_yticks(range(len(model_names)))
        axes[0].set_yticklabels(model_names, fontsize=9)
        axes[0].grid(axis='x', alpha=0.3, linestyle='--')
        
        # Plot 2: F1 Score
        bars2 = axes[1].barh(range(len(model_names)), f1_scores, color=colors, edgecolor='black')
        axes[1].set_xlabel('F1 Score (%)', fontsize=12, fontweight='bold')
        axes[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        axes[1].set_yticks(range(len(model_names)))
        axes[1].set_yticklabels(model_names, fontsize=9)
        axes[1].grid(axis='x', alpha=0.3, linestyle='--')
        
        # Plot 3: AUC Score
        bars3 = axes[2].barh(range(len(model_names)), auc_scores, color=colors, edgecolor='black')
        axes[2].set_xlabel('AUC Score (%)', fontsize=12, fontweight='bold')
        axes[2].set_title('Model AUC Score Comparison', fontsize=14, fontweight='bold')
        axes[2].set_yticks(range(len(model_names)))
        axes[2].set_yticklabels(model_names, fontsize=9)
        axes[2].grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', edgecolor='black', label='Base Models'),
            Patch(facecolor='#e74c3c', edgecolor='black', label='Ensemble Models')
        ]
        fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11)
        
        plt.suptitle('TESS Exoplanet: Comprehensive Model Performance Comparison', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nComprehensive comparison plot saved to: {save_path}")
        plt.show()
    
    def plot_ensemble_comparison(self, save_path='tess_ensemble_comparison.png'):
        """Create detailed comparison of ensemble methods only."""
        print("\n" + "="*80)
        print("GENERATING ENSEMBLE-ONLY COMPARISON")
        print("="*80)
        
        # Filter ensemble models
        ensemble_results = {k: v for k, v in self.all_results.items() 
                          if v['type'] == 'ensemble'}
        
        model_names = [name.replace('Ensemble: ', '') for name in ensemble_results.keys()]
        metrics = {
            'Accuracy': [r['accuracy'] * 100 for r in ensemble_results.values()],
            'F1 Score': [r['f1'] * 100 for r in ensemble_results.values()],
            'AUC Score': [r['auc'] * 100 for r in ensemble_results.values()]
        }
        
        x = np.arange(len(model_names))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#3498db', '#2ecc71', '#f39c12']
        for i, (metric_name, values) in enumerate(metrics.items()):
            offset = width * (i - 1)
            bars = ax.bar(x + offset, values, width, label=metric_name, 
                         color=colors[i], edgecolor='black')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Ensemble Method', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
        ax.set_title('TESS: Ensemble Methods Performance Comparison', 
                    fontsize=15, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nEnsemble comparison plot saved to: {save_path}")
        plt.show()
    
    def print_summary_table(self):
        """Print comprehensive summary table of all results."""
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        
        sorted_results = sorted(self.all_results.items(), 
                               key=lambda x: x[1]['f1'], 
                               reverse=True)
        
        print(f"\n{'Rank':<6} {'Model Name':<35} {'Type':<10} {'Accuracy':<12} {'F1':<12} {'AUC':<12}")
        print("-"*90)
        
        for rank, (name, result) in enumerate(sorted_results, 1):
            model_name = name.replace('Base: ', '').replace('Ensemble: ', '')
            print(f"{rank:<6} {model_name:<35} {result['type'].upper():<10} "
                  f"{result['accuracy']*100:>10.2f}% {result['f1']*100:>10.2f}% "
                  f"{result['auc']*100:>10.2f}%")
        
        print("="*90)
        
        # Print best models
        best_model = sorted_results[0]
        print(f"\nBEST OVERALL MODEL: {best_model[0].replace('Base: ', '').replace('Ensemble: ', '')}")
        print(f"   Accuracy: {best_model[1]['accuracy']*100:.2f}%")
        print(f"   F1 Score: {best_model[1]['f1']*100:.2f}%")
        print(f"   AUC Score: {best_model[1]['auc']*100:.2f}%")
    
    def save_best_models(self, top_n=2):
        """Save the top N best performing ensemble models."""
        # Filter only ensemble models
        ensemble_results = {k: v for k, v in self.all_results.items() 
                           if v['type'] == 'ensemble'}
        
        if not ensemble_results:
            print("\nNo ensemble models found to save!")
            return
        
        # Sort ensemble models by F1 score
        sorted_ensembles = sorted(ensemble_results.items(), 
                                 key=lambda x: x[1]['f1'], 
                                 reverse=True)
        
        # Save top N models
        saved_models = []
        for rank, (model_name, result) in enumerate(sorted_ensembles[:top_n], 1):
            # Create filename
            clean_name = model_name.replace('Ensemble: ', '').replace(' ', '_').replace('(', '').replace(')', '').lower()
            output_path = f'tess_best_ensemble_{rank}_{clean_name}.pkl'
            
            model_package = {
                'model': result['model'],
                'scaler': self.scaler,
                'imputer': self.imputer,
                'label_encoder': self.label_encoder,
                'model_name': model_name,
                'rank': rank,
                'n_classes': self.n_classes,
                'performance': {
                    'accuracy': result['accuracy'],
                    'f1': result['f1'],
                    'auc': result['auc']
                },
                'classes': list(self.label_encoder.classes_)
            }
            
            joblib.dump(model_package, output_path)
            saved_models.append((rank, model_name, output_path, result))
            
        # Print summary
        print("\n" + "="*80)
        print(f"TOP {top_n} ENSEMBLE MODELS SAVED")
        print("="*80)
        
        for rank, model_name, path, result in saved_models:
            print(f"\nRank {rank}: {model_name.replace('Ensemble: ', '')}")
            print(f"   File: {path}")
            print(f"   Performance:")
            print(f"     - Accuracy: {result['accuracy']*100:.2f}%")
            print(f"     - F1 Score: {result['f1']*100:.2f}%")
            print(f"     - AUC Score: {result['auc']*100:.2f}%")
        
        print("\n" + "="*80)
    
    def run_complete_pipeline(self):
        """Run the complete ensemble classification pipeline."""
        print("\n" + "="*80)
        print("STARTING ADVANCED ENSEMBLE TESS EXOPLANET CLASSIFICATION PIPELINE")
        print("="*80)
        
        # Load and preprocess data
        X_train, y_train, X_val, y_val, X_test, y_test = self.load_and_preprocess_data()
        
        # Initialize base models
        self.initialize_base_models()
        
        # Train and evaluate all models
        self.train_and_evaluate_all(X_train, y_train, X_val, y_val, X_test, y_test)
        
        # Generate visualizations
        self.plot_comprehensive_comparison()
        self.plot_ensemble_comparison()
        
        # Print summary
        self.print_summary_table()
        
        # Save top 2 best ensemble models
        self.save_best_models(top_n=2)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return self.all_results


if __name__ == "__main__":
    # Create classifier instance
    classifier = AdvancedEnsembleTESSClassifier(
        train_path='data/train.csv',
        val_path='data/validation.csv',
        test_path='data/test.csv'
    )
    
    # Run complete pipeline
    results = classifier.run_complete_pipeline()
    
    print("\nAdvanced ensemble TESS exoplanet classification complete!")
    print("Multiple ensemble strategies evaluated and compared!")