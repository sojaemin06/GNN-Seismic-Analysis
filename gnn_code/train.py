import sys
import os
import glob
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Add the project root to the Python path to resolve module import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gnn_code.models import GNN_Pushover

class PushoverDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, file_list=None):
        self.root_dir = root_dir
        if file_list:
            self.data_files = file_list
        else:
            # os.walk is more robust for paths with special characters than glob
            self.data_files = []
            for root, _, files in os.walk(root_dir):
                for file in files:
                    if file.endswith('_graph_data.pt'):
                        self.data_files.append(os.path.join(root, file))
        super(PushoverDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.data_files]

    @property
    def processed_file_names(self):
        # We are loading pre-processed files, so processed_file_names can be the same as raw_file_names
        return self.raw_file_names

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        # The 'weights_only' parameter is deprecated and will be removed in a future version of PyTorch.
        # Using `map_location` to ensure tensors are loaded correctly, especially if saved on a different device.
        data = torch.load(self.data_files[idx], map_location=torch.device('cpu'), weights_only=False)
        return data

def train(model, device, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        target = data.y.view(data.num_graphs, -1)
        loss = loss_fn(out, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, device, loader, loss_fn):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)
            target = data.y.view(data.num_graphs, -1)
            loss = loss_fn(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Dataset and Dataloaders ---
    dataset_path = os.path.join(project_root, 'dataset_output_parallel')
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found at: {dataset_path}")
        print("Please run generate_dataset.py first.")
        return

    # Dynamically get the list of all data files
    all_files = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('_graph_data.pt'):
                all_files.append(os.path.join(root, file))

    if not all_files:
        print("No data files found in the dataset directory.")
        return

    print(f"Found {len(all_files)} total data samples.")

    # Split files into training and validation sets
    train_files, val_files = train_test_split(all_files, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")

    train_dataset = PushoverDataset(root_dir=dataset_path, file_list=train_files)
    val_dataset = PushoverDataset(root_dir=dataset_path, file_list=val_files)

    # Create DataLoaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # --- 3. Model, Optimizer, Loss ---
    # Node features: 3 (x, y, z coords)
    # Edge features: 10 (is_col, is_beam, w, h, fc, fy, cover, As, n_bar1, n_bar2)
    # Output: 200 (pushover curve points for X and Z directions)
    model = GNN_Pushover(in_channels=3, hidden_channels=64, out_channels=200, edge_dim=10, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # --- 4. Training Loop ---
    epochs = 300
    best_val_loss = float('inf')
    model_save_path = os.path.join(project_root, 'gnn_code', 'best_model.pt')

    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, loss_fn)
        val_loss = evaluate(model, device, val_loader, loss_fn)

        if (epoch % 10 == 0) or (epoch == 1):
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Train RMSE: {np.sqrt(train_loss):.4f} | Val RMSE: {np.sqrt(val_loss):.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            if (epoch % 10 == 0) or (epoch == 1):
                print(f"  -> New best model saved to {model_save_path} (Val Loss: {best_val_loss:.4f})")

    # --- 5. Final Evaluation ---
    print("\n--- Training Complete ---")
    print(f"Best validation loss (MSE): {best_val_loss:.4f}")
    print(f"Best validation loss (RMSE): {np.sqrt(best_val_loss):.4f}")
    print(f"Best model saved to '{model_save_path}'")

if __name__ == '__main__':
    main()
