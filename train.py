import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import glob
import numpy as np
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, kendalltau
from dtign_model import DTIGN

#------------------------define functions----------------------------------------------
def load_py_from_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.pyg"))
    dataset = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    return dataset
#-----------------------------------------------------
def standardize_y(dataset, data_mean, data_std):
    for data in dataset:
        data.y = (data.y - data_mean) / data_std
    return dataset
#------------------------------------------------------
def training(loader, model, optimizer):
    model.train()
    total_loss = 0
    total_graphs = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch).view(-1)
        loss = F.mse_loss(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_graphs += batch.num_graphs
    train_avg_loss = total_loss / total_graphs
    return train_avg_loss, model
#------------------------------------------------------
@torch.no_grad()
def validation(loader, model, data_mean, data_std):
    model.eval()
    val_targets, val_preds = [], []
    for batch in loader:
        batch = batch.to(device)
        pred = model(batch).view(-1)
        val_preds.extend(pred.cpu().numpy().tolist())
        val_targets.extend(batch.y.cpu().numpy().tolist())
    val_preds = np.array(val_preds)
    val_targets = np.array(val_targets)
    # Inverse transform--------------------------------------------
    val_preds = val_preds * data_std.item() + data_mean.item()
    val_targets = val_targets * data_std.item() + data_mean.item()
    #--------------------------------------------------------------
    val_rmse = np.sqrt(mean_squared_error(val_targets, val_preds))
    try:
        val_pearson_r, _ = pearsonr(val_targets, val_preds)
    except:
        val_pearson_r = float('nan')
    try:
        val_kendall_tau, _ = kendalltau(val_targets, val_preds)
    except:
        val_kendall_tau = float('nan')

    return val_rmse, val_pearson_r, val_kendall_tau
#------------------------------------------------------
@torch.no_grad()
def testing(loader, model, data_mean, data_std):
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        out = model(batch.to(device)).view(-1)
        y_pred.extend(out.cpu().numpy().tolist())
        y_true.extend(batch.y.cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    y_true_inv = y_true * data_std.item() + data_mean.item()
    y_pred_inv = y_pred * data_std.item() + data_mean.item()

    test_rmse = np.sqrt(mean_squared_error(y_true_inv, y_pred_inv))
    try:
        test_pearson_r, _ = pearsonr(y_true_inv, y_pred_inv)
    except:
        test_pearson_r = float('nan')
    try:
        test_kendall_tau, _ = kendalltau(y_true_inv, y_pred_inv)
    except:
        test_kendall_tau = float('nan')

    return y_true_inv, y_pred_inv, test_rmse, test_pearson_r, test_kendall_tau
#-------------------------------------------------------------------------------------
def train_epochs(epochs, model, train_loader, val_loader, data_mean, data_std, fold):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
    logs = []
    best_epoch_val_pearson = -float('inf')
    best_model_fold_path = os.path.join("saved_models", f"best_model_fold{fold}.pt")

    for epoch in range(epochs):
        epoch_loss, model = training(train_loader, model, optimizer)
        val_rmse, val_pearson_r, val_kendall_tau = validation(val_loader, model, data_mean, data_std)
        logs.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_rmse": val_rmse,
            "val_pearson_r": val_pearson_r,
            "val_kendall_tau": val_kendall_tau
        })
        print(
            f"Epoch {epoch + 1:02d} | Train Loss: {epoch_loss:.4f} | Val RMSE: {val_rmse:.4f} | Pearson r: {val_pearson_r:.4f} | Kendall τ: {val_kendall_tau:.4f}")

        # save best model by val_pearson (per-epoch)
        if val_pearson_r > best_epoch_val_pearson:
            best_epoch_val_pearson = val_pearson_r
            torch.save(model.state_dict(), best_model_fold_path)

    return best_epoch_val_pearson, best_model_fold_path, logs

#------------------------run training for 5 folds---------------------------------
os.makedirs("saved_models", exist_ok=True)
best_fold_pearson = -float('inf')
best_model_path = None

train_dir = os.path.join(os.getcwd(),"DTIG","pEC50", "fixed_data")
train_folders = [os.path.join(train_dir, f"fixed_train_{i}") for i in range(1,6)]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

for fold in range(5):
    print(f"=== Fold {fold +1} ===")
    val_dataset = load_py_from_folder(train_folders[fold])
    train_dataset = []
    for i in range(5):
        if i != fold:
            train_dataset.extend(load_py_from_folder(train_folders[i]))

    # normalizing data
    train_y = torch.cat([data.y.view(-1) for data in train_dataset])
    data_mean = train_y.mean()
    data_std = train_y.std()
    train_dataset = standardize_y(train_dataset, data_mean, data_std)
    val_dataset = standardize_y(val_dataset, data_mean, data_std)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1)

    model = DTIGN().to(device)
    best_epoch_val_pearson, model_path, logs = train_epochs(10, model, train_loader, val_loader, data_mean, data_std,fold)
    # save the best model
    if best_epoch_val_pearson > best_fold_pearson:
        best_fold_pearson = best_epoch_val_pearson
        best_model_path = model_path
#save logs
    log_df = pd.DataFrame(logs)
    log_df.to_csv(f"training_logs_fold{fold+1}.csv", index=False)
#---------------evaluation model on testset------------------------------------
test_dir = os.path.join(os.getcwd(),"DTIG","pEC50", "fixed_data","fixed_test")
test_dataset = load_py_from_folder(test_dir)
test_dataset = standardize_y(test_dataset, data_mean, data_std)
test_loader = DataLoader(test_dataset, batch_size=1)

best_model = DTIGN().to(device)
best_model.load_state_dict(torch.load(best_model_path))
y_true, y_pred, test_rmse, test_pearson_r, test_kendall_tau = testing(test_loader, best_model, data_mean, data_std)
print(f"Test RMSE: {test_rmse:.4f} | Pearson: {test_pearson_r:.4f} | Kendall τ: {test_kendall_tau:.4f}")







