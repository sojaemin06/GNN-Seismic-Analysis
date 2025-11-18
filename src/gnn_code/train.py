import sys
import os
# Add the project root to the Python path to resolve module import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np

from gnn_code.models import GNN_Pushover

class PushoverDataset(Dataset):
    def __init__(self, root_dir, transform=None, pre_transform=None, file_list=None):
        self.root_dir = root_dir
        if file_list is not None:
            self.data_files = file_list
        else:
            self.data_files = glob.glob(os.path.join(root_dir, 'Building_*', '*_graph_data.pt'))
        super(PushoverDataset, self).__init__(root_dir, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [os.path.basename(f) for f in self.data_files]

    @property
    def processed_file_names(self):
        return self.raw_file_names

    def len(self):
        return len(self.data_files)

    def get(self, idx):
        data = torch.load(self.data_files[idx], weights_only=False)
        return data

def train(model, device, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # Reshape data.y to match the output shape [batch_size, out_channels]
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
            # Reshape data.y to match the output shape [batch_size, out_channels]
            target = data.y.view(data.num_graphs, -1)
            loss = loss_fn(out, target)
            total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)

def main():
    # --- 0. Debug: Print CWD ---
    print(f"Current Working Directory: {os.getcwd()}")

    # --- 1. Setup ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 2. Dataset and Dataloaders ---
    dataset_path = os.path.join(project_root, 'dataset_output_parallel')
    if not os.path.exists(dataset_path):
        print(f"Dataset directory not found at: {dataset_path}")
        print("Please run generate_dataset.py first.")
        return

    # Manually provide the file list to bypass Python's glob issue with special characters
    file_list = [
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0008_S4_BX3_BZ4_CFalse\Building_0008_S4_BX3_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0004_S5_BX4_BZ4_CFalse\Building_0004_S5_BX4_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0009_S3_BX2_BZ4_CFalse\Building_0009_S3_BX2_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0006_S4_BX4_BZ4_CFalse\Building_0006_S4_BX4_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0007_S4_BX2_BZ2_CFalse\Building_0007_S4_BX2_BZ2_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0005_S5_BX3_BZ3_CFalse\Building_0005_S5_BX3_BZ3_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0000_S5_BX3_BZ4_CFalse\Building_0000_S5_BX3_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0003_S3_BX3_BZ4_CFalse\Building_0003_S3_BX3_BZ4_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0002_S3_BX4_BZ2_CFalse\Building_0002_S3_BX4_BZ2_CFalse_graph_data.pt",
        r"C:\Users\82105\OneDrive\Desktop\ERS\[논문작성]\SCI_GNN_pushover\GNN_Project\dataset_output_parallel\Building_0001_S3_BX3_BZ2_CFalse\Building_0001_S3_BX3_BZ2_CFalse_graph_data.pt"
    ]

    dataset = PushoverDataset(root_dir=dataset_path, file_list=file_list)
    
    if len(dataset) == 0:
        print("No data found in the dataset directory.")
        return

    # Use the entire dataset for training and evaluation due to small sample size
    print(f"Dataset size: {len(dataset)} (using all for training and testing)")

    # Create a single loader for the entire dataset
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # --- 3. Model, Optimizer, Loss ---
    # Node features: 3 (x, y, z coords)
    # Edge features: 10 (is_col, is_beam, w, h, fc, fy, cover, As, n_bar1, n_bar2)
    # Output: 100 (pushover curve points)
    model = GNN_Pushover(in_channels=3, hidden_channels=64, out_channels=100, edge_dim=10, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    # --- 4. Training Loop ---
    epochs = 300
    print("\n--- Starting Training ---")
    for epoch in range(1, epochs + 1):
        loss = train(model, device, loader, optimizer, loss_fn)
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}: Train Loss: {loss:.4f} | RMSE: {np.sqrt(loss):.4f}")
            
    # --- 5. Final Evaluation ---
    print("\n--- Final Evaluation on Training Data ---")
    final_loss = evaluate(model, device, loader, loss_fn)
    print(f"Final Loss (MSE): {final_loss:.4f}")
    print(f"Final Loss (RMSE): {np.sqrt(final_loss):.4f}")
    
    # Save the final model
    model_path = os.path.join(project_root, 'final_model_10_samples.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved final model to '{model_path}'")

if __name__ == '__main__':
    main()
