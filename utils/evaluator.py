import pandas as pd
from sklearn.metrics import classification_report
import contextlib
import os
from glob import glob
from pathlib import Path

def generate_classification_reports(predictions_dir, 
                                  results_dir, 
                                  pattern):
    """
    Generate classification reports for all prediction files matching the pattern.
    
    Args:
        predictions_dir (str): Directory containing prediction CSV files
        results_dir (str): Directory to save classification reports
        pattern (str): File pattern to match
    """


    # Create results directory if it doesn't exist
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Find all matching files
    file_pattern = os.path.join(predictions_dir, pattern)
    files = glob(file_pattern)
    
    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    
    for file_path in files:
        try:
            filename = Path(file_path).stem
            
            df_pred = pd.read_csv(file_path)
            
            if "offensiveYN" not in df_pred.columns:
                print(f"Warning: 'offensiveYN' column not found in {filename}")
                continue
            if "prediction" not in df_pred.columns:
                print(f"Warning: 'prediction' column not found in {filename}")
                continue
            
            y_true = df_pred["offensiveYN"].astype(int).tolist()
            y_pred = df_pred["prediction"].astype(int).tolist()
            

            if len(y_true) != len(y_pred):
                print(f"Warning: Length mismatch in {filename}")
                # Align lengths by taking minimum
                min_len = min(len(y_true), len(y_pred))
                y_true = y_true[:min_len]
                y_pred = y_pred[:min_len]
            

            output_file = os.path.join(results_dir, f"{filename}.txt")
            with open(output_file, "w") as f:
                with contextlib.redirect_stdout(f):
                    print(f"Classification Report for {filename}")
                    print("=" * 50)
                    print(f"Total samples: {len(y_true)}")
                    print()
                    print(classification_report(y_true, y_pred, digits=3))
            
            print(f"✓ Processed {filename}")
            
        except Exception as e:
            print(f"✗ Error processing {file_path}: {str(e)}")

