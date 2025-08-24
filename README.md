# intersectionality-llm

## ⚙️ How to run the code

### 0. prepare_data.py
Reads the original dataset from `data` and prepares it for the experiments in `dataset`. 

### 1. classification.py 
Obtain the model's predictions and explanations without any postprocessing. Ensure that a valid `HF_TOKEN` to download the models is specified in a `.env` file in the home directory.\
The results without any socio-demographic prompting are saved to `predictions/original/predictions_<model_name>_<isCoT>_baseline.csv`.\
The results with socio-demographic prompting are saved to `predictions_dem/<model_name>/original/predictions_<model_name>_<isCoT>_<traits used>.csv`

**Usage:**
```
python classification.py \
  --model_id <HF_MODEL_ID> \
  [--batch_size <BATCH_SIZE>] \
  [--cot] \
  [--sociodemographic_traits]
```

Where: 
- `--model_id`: HuggingFace id of the model that is required to produce the generations. 
- `--cot`: if passed, the model using the Chain of Thought prompting strategy is used.
- `--batch_size`: batch size for classification task.
- `--sociodemographic_traits`: if passed, the classification is done using all combinations of socio-demographic traits.

### 2. evaluation.py
Postprocesses the output of all model results found in the directories where the output of `classification.py` is saved. It parses the prediction and the explanation from the produced text.
The results without socio-demographic prompting are saved to `predictions/cleaned/predictions_<model_name>_<isCoT>_baseline.csv`.\
The results with socio-demographic prompting are saved to `predictions_dem/<model_name>/cleaned/predictions_<model_name>_<isCoT>_<traits used>.csv`

**Usage:**
```
python evaluation.py \
  [--sociodemographic_traits]
```
Where: 
- `--sociodemographic_traits`: if passed, the results for socio-demographic analaysis are postprocessed otherwise the baselines.

### 3. analysis_by_textual_variable_split.py, analysis_label_distribution.py, analysis_model_performance_stat.py
Contain the code for the additional analyses which are conducted in the paper. `analysis_by_textual_variable_split.py` can be run in the same way as `evaluation.py` while the others can be run directly.
