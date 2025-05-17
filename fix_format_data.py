from torch_geometric.data import Data
import torch
import glob
import os

def fix_pyg_files(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    files = glob.glob(os.path.join(input_folder, "*.pyg"))
    for f in files:
        try:
            raw = torch.load(f, weights_only=False)
            store = raw._store
            y = store['y']
            if y.ndim >1 or y.shape[0] > 1:
                y = y[0].unsqueeze(0)

            data = Data(
                x=store['x'],
                y=y,
                pos=store['pos'],
                x_bond=store['x_bond'],
                edge_index_intra=store['edge_index_intra'],
                edge_index_inter=store['edge_index_inter'],
                dock_software=store.get('dock_software'),
                pocket_or_pose=store.get('pocket_or_pose'),
                split=store.get('split'),
                batch=store.get('batch'),
                ptr=store.get('ptr')
            )
            filename = os.path.basename(f)
            torch.save(data, os.path.join(output_folder, filename))
            print(f"Fixed: {filename}")
        except Exception as e:
            print(f" Failed: {fpath} with error: {e}")


test_dir = os.path.join(os.getcwd(),"DTIG","pEC50", "train_5")
fix_pyg_files(test_dir, "fixed_train_5")


from torch_geometric.loader import DataLoader
data_fix = torch.load("CHEMBL1585091_FIXED.pyg", weights_only=False)
print(type(data_fix))

def load_py_from_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.pyg"))
    dataset = [torch.load(f, map_location='cpu', weights_only=False) for f in files]
    return dataset
test_dir = os.path.join(os.getcwd(),"DTIG","pEC50", "fixed_data","fixed_test")
test_dataset = load_py_from_folder(test_dir)
test_loader = DataLoader(test_dataset, batch_size=32)
for batch in test_loader:
    print("batch.y.shape:", batch.y.shape)
    print("y unique:", batch.y.unique())
    print(x.shape)





