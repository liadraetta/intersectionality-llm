# intersectionality-llm

## ⚙️ How to run the code
### 1. classification.py / classification_dem.py
- **Without Socio-Demographic Prompting**\
    Use `classification.py` to extract the classifications without any socio-demographic prompting specified in the input. Results are saved to `predictions/original/predictions_<model_name>_<isCoT>_baseline.csv`.\
- **With Socio-Demographic Prompting**\
    Use `classification_dem.py` to extract the classification with the socio-demographic prompting specified in the input. Results are saved to `predictions_dem/<model_name>/original/predictions_<model_name>_<isCoT>_baseline.csv`

**Usage:**
```
python classification.py / classification_dem.py \
  --model_id <HF_MODEL_ID> \
  [--batch_size <BATCH_SIZE>] \
  [--cot]
```

Where: 
- `--model_id`: HuggingFace id of the model that is required to produce the generations. 
- `--cot`: if passed, the model using the Chain of Thought prompting strategy is used.
- `--batch_size`: batch size for classification task.

### 2. evaluation.py
Cleans the outputs of the models ...