
# DTIGN for pEC50 Prediction

This repository contains the implementation of the DTIGN (Drug–Target Interaction Graph Network) model to predict the **pEC50** values based on molecular graphs of ligands.

## Setup Instructions

### 1. Create virtual environment (optional)
```bash
python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate
```

### 2. Install required packages
```bash
pip install -r requirements.txt
```

> ⚠️ Ensure CUDA version matches your system if using GPU.

### 3. Reconstruct graph structure in order to compatible with torch_geometric.data.Data
```bash
python fix_format_data.py
```
### 4. Run Instructions

### To train the model and evaluate on test set:
```bash
python train.py
```
Define a DTIGN model:   dtign_model.py

This script performs:
- 5-fold cross-validation
- Training loop for 10 epochs
- Model selection based on best validation Pearson value
- Final evaluation on test set

Trained models are saved under `saved_models/`.

### 5. Output

After training, resutls were obtained:
- `training_logs_fold{1–5}.csv`: logs of train loss, RMSE, Pearson, and Kendall over epochs
- `best_model_fold{n}.pt`: best model weights per fold
- Final test performance printed:
  ```
  Test RMSE: 1.72 | Pearson: 0.088 | Kendall τ: 0.304
  ```

### 6. Tuning Model Hyperparameters

| Parameter       | Value                             |
|-----------------|-----------------------------------|
| Hidden Dim      | 128 / 256                         |
| GNN Layers      | 3                                 |
| Attention Heads | 4                                 |
| Dropout         | 0.2/0/3                           |
| Optimizer       | Adam (lr=1e-3, weight_decay=1e-3) |

## 📌 Notes

- Standardization `(y - mean) / std` is applied using train set statistics.

