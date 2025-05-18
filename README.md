
# DTIGN for pEC50 Prediction

This repository contains the implementation of the DTIGN (Drug–Target Interaction Graph Network) model to predict the **pEC50** values based on molecular graphs of ligands.

## 📁 Folder Structure

```
DTIG/
├── pEC50/
│   ├── fixed_data/            # Preprocessed `.pyg` graph data for training & testing
│   │   ├── fixed_train_1/     # Training folds (1–5)
│   │   ├── fixed_test/        # Test set
│   ├── train_test_full_fixed.py  # Main training & evaluation script
├── dtign_model.py             # Model architecture
├── utils/                     # (Optional) utility functions
├── saved_models/              # Saved best models
```

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

### 3. Run Instructions

### To train the model and evaluate on test set:
```bash
python train_test_full_fixed.py
```

This script performs:
- 5-fold cross-validation
- Model selection based on best validation Pearson correlation
- Final evaluation on test set

Trained models are saved under `saved_models/`.

## 📊 Output

After training, you will get:
- `training_logs_fold{1–5}.csv`: logs of RMSE, Pearson, and Kendall over epochs
- `best_model_fold{n}.pt`: best model weights per fold
- Final test performance printed:
  ```
  Test RMSE: 1.74 | Pearson: 0.07 | Kendall τ: 0.01
  ```

## ⚙️ Key Model Parameters

| Parameter       | Value         |
|----------------|---------------|
| Hidden Dim      | 128 / 256      |
| GNN Layers      | 3             |
| Attention Heads | 4             |
| Dropout         | 0.2           |
| Optimizer       | Adam (lr=1e-3, weight_decay=5e-4) |

## 📌 Notes

- All `y` values (pEC50) are **kept as-is** (no sign flipping) to reflect that **higher pEC50 = stronger bioactivity**.
- Standardization `(y - mean) / std` is applied using train set statistics.
- The model uses `global_mean_pool` for graph-level embedding and multi-head self-attention.

