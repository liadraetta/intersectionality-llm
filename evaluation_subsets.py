from utils.evaluator import *
from utils.clean_output import *
import pandas as pd 
import os 
from glob import glob
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def generate_classification_reports_confusion_matrix_subset_variables(predictions_dir, 
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
    additional_variables = ['isAAE', 'targetsBlackPeople', 'vulgar']
    # Find all matching files
    file_pattern = os.path.join(predictions_dir, pattern)
    files = glob(file_pattern)
    
    original_data_path = './dataset/AnnAttDataset.csv'
    original_data = pd.read_csv(original_data_path)

    if not files:
        print(f"No files found matching pattern: {file_pattern}")
        return
    
    for file_path in files:
        #try:
        filename = Path(file_path).stem
        
        df_pred = pd.read_csv(file_path)
        df_pred = df_pred.fillna(-1,)
       
        # Merge with original data to keep the additional variables required
        merged_df = df_pred.merge(original_data, on=['postId', 'annId'])
        output_file_base = results_dir
        
        
        for additional_variable in additional_variables:
            output_file=f"{output_file_base}{additional_variable}/{filename}.txt"
            # Initialize output file and root directories, if it exists clear it
            if not os.path.exists(os.path.dirname(output_file)):
                os.makedirs(os.path.dirname(output_file))
            else:
                # Clear the contents of the output file if it already exists
                open(output_file, 'w').close()

            # Split the merged df based on additional variable
            if set(merged_df[additional_variable]) != set([True, False]):
                print(set(merged_df[additional_variable]))
                raise ValueError(f"Variable {additional_variable} does not have exactly two unique values.")
            merged_df0 = merged_df[merged_df[additional_variable] == True]
            merged_df1 = merged_df[merged_df[additional_variable] == False]
            print(len(merged_df0), len(merged_df1))

            y_true0 = merged_df0["offensiveYN_x"].astype(int).tolist()
            y_pred0 = merged_df0["prediction"].astype(int).tolist()
            y_true1 = merged_df1["offensiveYN_x"].astype(int).tolist()
            y_pred1 = merged_df1["prediction"].astype(int).tolist()
            # Print confusion matrix instead of classification report
            
            cm0 = confusion_matrix(y_true0, y_pred0, labels=[0, 1], )
            disp = ConfusionMatrixDisplay(confusion_matrix=cm0, display_labels=["Not Offensive", "Offensive"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {additional_variable} = True")
            plt.savefig(f"{output_file_base}{additional_variable}/{filename}_cm_true.png")

            
            cm1 = confusion_matrix(y_true1, y_pred1, labels=[0, 1], )
            disp = ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=["Not Offensive", "Offensive"])
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for {additional_variable} = False")
            plt.savefig(f"{output_file_base}{additional_variable}/{filename}_cm_false.png")

                    

            print(f"Writing to output_file: {output_file}")
            with open(output_file, 'a') as f:
                with contextlib.redirect_stdout(f):
                    print(f"Classification report for {additional_variable} = True")
                    print(classification_report(y_true0, y_pred0, digits=3, labels=[0, 1], target_names=["Not Offensive", "Offensive"]))
                    print(f"Total samples: {len(y_true0)}")
                    print("=" * 50)
                    print()
                    print(f"Classification report for {additional_variable} = False")   
                    print(classification_report(y_true1, y_pred1, digits=3, labels=[0, 1], target_names=["Not Offensive", "Offensive"]))
                    print(f"Total samples: {len(y_true1)}")
                    print("=" * 50)
        print(f"✓ Processed {filename}")


demographics = True
list_models = ["deepseek", "gemma", "Llama", "Ministral", "Qwen2"]
ids_to_remove = [3768, 213, 4104, 1770]

if not demographics:
    dir_predictions_original = "./predictions/original"
    dir_predictions_cleaned = "./predictions/cleaned"

    results_dir="./results_subset_confusion/"
    pattern_cleaned = "cleaned_predictions_*_*_*.csv"
   
    generate_classification_reports_confusion_matrix_subset_variables(dir_predictions_cleaned, results_dir, pattern_cleaned)

else:
    for model_name in list_models:

        dir_predictions_original = f"./predictions_dem/{model_name}/original"
        dir_predictions_cleaned = f"./predictions_dem/{model_name}/cleaned"

        Path(dir_predictions_cleaned).mkdir(parents=True, exist_ok=True)
        results_dir=f"./results_dem_subset_confusion/{model_name}/"
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        
        pattern = f"predictions_{model_name}*.csv"
        pattern_cleaned = f"cleaned_predictions_{model_name}*.csv"    
        generate_classification_reports_confusion_matrix_subset_variables(dir_predictions_cleaned, results_dir, pattern_cleaned)



