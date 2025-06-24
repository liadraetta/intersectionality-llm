from utils.evaluator import generate_classification_reports

predictions_dir="./predictions/"
results_dir="./results/"
pattern="predictions_*_*_*.csv"

#evaluate
generate_classification_reports(predictions_dir, results_dir, pattern)
