import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import Dict, List, Union
import warnings
warnings.filterwarnings('ignore')


class ExoplanetModelTester:
    """
    Testing interface for trained exoplanet classification models.
    Supports single prediction and batch predictions with detailed output.
    """
    
    def __init__(self, model_path: str):
        """
        Initialize the tester with a saved model.
        
        Args:
            model_path: Path to the saved model pickle file
        """
        self.model_path = model_path
        self.model_package = None
        self.model = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.load_model()
        
    def load_model(self):
        """Load the saved model and preprocessing objects."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print("="*80)
        print("LOADING MODEL")
        print("="*80)
        print(f"Loading model from: {self.model_path}")
        
        self.model_package = joblib.load(self.model_path)
        self.model = self.model_package['model']
        self.scaler = self.model_package['scaler']
        self.imputer = self.model_package['imputer']
        self.label_encoder = self.model_package['label_encoder']
        
        print(f"\n✓ Model loaded successfully!")
        print(f"  Model type: {self.model_package.get('model_name', 'Unknown')}")
        print(f"  Rank: {self.model_package.get('rank', 'N/A')}")
        
        if 'performance' in self.model_package:
            perf = self.model_package['performance']
            print(f"\n  Training Performance:")
            print(f"    - Accuracy: {perf['accuracy']*100:.2f}%")
            print(f"    - F1 Score: {perf['f1']*100:.2f}%")
            if perf.get('auc', 0) > 0:
                print(f"    - AUC Score: {perf['auc']*100:.2f}%")
        print("="*80)
    
    def prepare_input(self, data: Union[Dict, List[Dict]]) -> pd.DataFrame:
        """
        Prepare input data for prediction.
        
        Args:
            data: Single dictionary or list of dictionaries with feature values
            
        Returns:
            DataFrame with prepared features
        """
        # Convert single dict to list for uniform processing
        if isinstance(data, dict):
            data = [data]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Get expected features from the scaler (it was fitted on the training data)
        expected_features = self.scaler.n_features_in_
        
        print(f"\nInput validation:")
        print(f"  - Expected features: {expected_features}")
        print(f"  - Provided features: {len(df.columns)}")
        
        if len(df.columns) != expected_features:
            print(f"\n⚠️  Warning: Feature count mismatch!")
            print(f"  Model expects {expected_features} features")
            print(f"  You provided {len(df.columns)} features")
            print(f"\n  Provided features: {list(df.columns)}")
        
        return df
    
    def predict_single(self, data: Dict, show_details: bool = True) -> Dict:
        """
        Make prediction for a single exoplanet candidate.
        
        Args:
            data: Dictionary with feature values
            show_details: Whether to print detailed output
            
        Returns:
            Dictionary with prediction results
        """
        if show_details:
            print("\n" + "="*80)
            print("MAKING PREDICTION")
            print("="*80)
        
        # Prepare input
        df = self.prepare_input(data)
        
        # Apply preprocessing pipeline
        X_imputed = self.imputer.transform(df)
        X_scaled = self.scaler.transform(X_imputed)
        
        # Make prediction
        prediction_encoded = self.model.predict(X_scaled)[0]
        prediction_label = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get probability if available
        try:
            probabilities = self.model.predict_proba(X_scaled)[0]
            confidence = float(np.max(probabilities))
            class_probabilities = {
                self.label_encoder.inverse_transform([i])[0]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        except AttributeError:
            confidence = None
            class_probabilities = None
        
        result = {
            'prediction': prediction_label,
            'confidence': confidence,
            'class_probabilities': class_probabilities,
            'input_features': data
        }
        
        if show_details:
            self.print_prediction_result(result)
        
        return result
    
    def predict_batch(self, data_list: List[Dict], show_summary: bool = True) -> List[Dict]:
        """
        Make predictions for multiple exoplanet candidates.
        
        Args:
            data_list: List of dictionaries with feature values
            show_summary: Whether to print summary statistics
            
        Returns:
            List of prediction result dictionaries
        """
        print("\n" + "="*80)
        print(f"BATCH PREDICTION - {len(data_list)} SAMPLES")
        print("="*80)
        
        results = []
        for i, data in enumerate(data_list, 1):
            print(f"\nProcessing sample {i}/{len(data_list)}...")
            result = self.predict_single(data, show_details=False)
            results.append(result)
        
        if show_summary:
            self.print_batch_summary(results)
        
        return results
    
    def print_prediction_result(self, result: Dict):
        """Print detailed prediction result."""
        print("\n" + "-"*80)
        print("PREDICTION RESULT")
        print("-"*80)
        
        print(f"\n🎯 Prediction: {result['prediction']}")
        
        if result['confidence'] is not None:
            print(f"📊 Confidence: {result['confidence']*100:.2f}%")
            
            print(f"\n📈 Class Probabilities:")
            for class_name, prob in sorted(result['class_probabilities'].items(), 
                                          key=lambda x: x[1], reverse=True):
                bar_length = int(prob * 50)
                bar = '█' * bar_length + '░' * (50 - bar_length)
                print(f"  {class_name:20s} {bar} {prob*100:6.2f}%")
        else:
            print(f"📊 Confidence: N/A (hard voting classifier)")
        
        print("\n" + "-"*80)
    
    def print_batch_summary(self, results: List[Dict]):
        """Print summary statistics for batch predictions."""
        print("\n" + "="*80)
        print("BATCH PREDICTION SUMMARY")
        print("="*80)
        
        # Count predictions by class
        predictions = [r['prediction'] for r in results]
        prediction_counts = pd.Series(predictions).value_counts()
        
        print(f"\n📊 Prediction Distribution:")
        print(f"  Total samples: {len(results)}")
        print()
        for class_name, count in prediction_counts.items():
            percentage = (count / len(results)) * 100
            print(f"  {class_name:20s}: {count:4d} ({percentage:5.1f}%)")
        
        # Average confidence if available
        confidences = [r['confidence'] for r in results if r['confidence'] is not None]
        if confidences:
            avg_confidence = np.mean(confidences)
            min_confidence = np.min(confidences)
            max_confidence = np.max(confidences)
            
            print(f"\n📈 Confidence Statistics:")
            print(f"  Average: {avg_confidence*100:.2f}%")
            print(f"  Minimum: {min_confidence*100:.2f}%")
            print(f"  Maximum: {max_confidence*100:.2f}%")
        
        print("\n" + "="*80)
    
    def export_results(self, results: Union[Dict, List[Dict]], 
                      output_path: str, format: str = 'json'):
        """
        Export prediction results to file.
        
        Args:
            results: Single result dict or list of result dicts
            output_path: Path to save the results
            format: 'json' or 'csv'
        """
        if isinstance(results, dict):
            results = [results]
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n✓ Results exported to: {output_path}")
            
        elif format == 'csv':
            # Flatten results for CSV
            flattened = []
            for r in results:
                flat_result = {
                    'prediction': r['prediction'],
                    'confidence': r['confidence']
                }
                # Add class probabilities if available
                if r['class_probabilities']:
                    for class_name, prob in r['class_probabilities'].items():
                        flat_result[f'prob_{class_name}'] = prob
                flattened.append(flat_result)
            
            df = pd.DataFrame(flattened)
            df.to_csv(output_path, index=False)
            print(f"\n✓ Results exported to: {output_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


def test_from_json_file(model_path: str, json_file_path: str):
    """
    Test model with data from a JSON file.
    
    Args:
        model_path: Path to saved model
        json_file_path: Path to JSON file with test data
    """
    tester = ExoplanetModelTester(model_path)
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        results = tester.predict_batch(data)
    else:
        results = tester.predict_single(data)
    
    return results


def interactive_test(model_path: str):
    """
    Interactive testing mode with manual input.
    
    Args:
        model_path: Path to saved model
    """
    tester = ExoplanetModelTester(model_path)
    
    print("\n" + "="*80)
    print("INTERACTIVE TESTING MODE")
    print("="*80)
    print("\nEnter exoplanet features as JSON (or 'quit' to exit)")
    print("Example format:")
    print('''{
    "pl_orbper": 7.49543,
    "pl_tranmid": 2456987.674,
    "pl_trandur": 3.62,
    "pl_rade": 2.45,
    "pl_radj": 0.21857436,
    "pl_radjerr1": 0.01873495,
    "pl_radjerr2": -0.01873495,
    "pl_ratror": 0.021,
    "st_rad": 1.06,
    "st_raderr1": 0.07,
    "st_raderr2": -0.07,
    "sy_dist": 269.183,
    "sy_disterr1": 2.539,
    "sy_disterr2": -2.493,
    "sy_plx": 3.68643,
    "sy_plxerr1": 0.0344589,
    "sy_plxerr2": -0.0344589
}''')
    
    while True:
        print("\n" + "-"*80)
        user_input = input("\nEnter JSON data (or 'quit'): ").strip()
        
        if user_input.lower() == 'quit':
            print("\nExiting interactive mode.")
            break
        
        try:
            data = json.loads(user_input)
            tester.predict_single(data)
        except json.JSONDecodeError:
            print("❌ Invalid JSON format. Please try again.")
        except Exception as e:
            print(f"❌ Error: {str(e)}")


# Example usage demonstrations
if __name__ == "__main__":
    print("="*80)
    print("EXOPLANET MODEL TESTING SCRIPT")
    print("="*80)
    
    # Example 1: Single prediction
    print("\n" + "="*80)
    print("EXAMPLE 1: SINGLE PREDICTION")
    print("="*80)
    
    # Your example data
    sample_data = {
        "pl_orbper": 1.5771832,
        "pl_tranmid": 2456772.583,
        "pl_trandur": 3.96,
        "pl_rade": 33.4,
        "pl_radj": 2.97974842,
        "pl_radjerr1": 0.02230354,
        "pl_radjerr2": -0.02408779,
        "pl_ratror": 0.2491,
        "st_rad": 1.94,
        "st_raderr1": 0.04015,
        "st_raderr2": -0.04,
        "sy_dist": 273.796,
        "sy_disterr1": 2.093,
        "sy_disterr2": -2.063,
        "sy_plx": 4.77093,
        "sy_plxerr1": 0.0476123,
        "sy_plxerr2": -0.0476123
    }
    
    # Specify your model path (adjust based on what was saved)
    model_path = 'best_ensemble_1_stacking_rf.pkl'  # or the path to your saved model
    
    try:
        # Initialize tester
        tester = ExoplanetModelTester(model_path)
        
        # Make single prediction
        result = tester.predict_single(sample_data)
        
        # Export result
        tester.export_results(result, 'single_prediction_result.json', format='json')
        
    except FileNotFoundError:
        print(f"\n❌ Model file not found: {model_path}")
        print("Please ensure you have run the training script and saved the models.")
        print("\nAvailable model files should be named like:")
        print("  - best_ensemble_1_stacking_xgb.pkl")
        print("  - best_ensemble_2_voting_weighted.pkl")
    
    # Example 2: Batch prediction
    print("\n\n" + "="*80)
    print("EXAMPLE 2: BATCH PREDICTION")
    print("="*80)
    
    batch_data = [
        sample_data,  # Your example
        {
            "pl_orbper": 10.5,
            "pl_tranmid": 2456900.0,
            "pl_trandur": 4.0,
            "pl_rade": 3.0,
            "pl_radj": 0.267,
            "pl_radjerr1": 0.02,
            "pl_radjerr2": -0.02,
            "pl_ratror": 0.025,
            "st_rad": 1.1,
            "st_raderr1": 0.08,
            "st_raderr2": -0.08,
            "sy_dist": 300.0,
            "sy_disterr1": 3.0,
            "sy_disterr2": -3.0,
            "sy_plx": 3.33,
            "sy_plxerr1": 0.03,
            "sy_plxerr2": -0.03
        }
    ]
    
    try:
        if os.path.exists(model_path):
            tester = ExoplanetModelTester(model_path)
            batch_results = tester.predict_batch(batch_data)
            tester.export_results(batch_results, 'batch_prediction_results.json', format='json')
            tester.export_results(batch_results, 'batch_prediction_results.csv', format='csv')
    except FileNotFoundError:
        pass
    
    # Example 3: Interactive mode (commented out by default)
    # Uncomment the following line to enable interactive testing
    # interactive_test(model_path)
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nTo use this script:")
    print("1. Ensure your trained model file exists")
    print("2. Update the 'model_path' variable with your model's filename")
    print("3. Run the script with your own data")
    print("\nFor interactive testing, uncomment the interactive_test() call at the end.")