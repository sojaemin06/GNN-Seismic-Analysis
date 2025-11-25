import sys
import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# Add the project root to the Python path to resolve module import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gnn_code.models import GNN_Pushover

def plot_prediction_vs_actual(model, data_loader, device, building_name, displacement_data, output_path):
    """
    Plots the predicted vs. actual pushover curve for a single building.
    """
    model.eval()
    
    # Find the specific building data
    target_data = None
    for data in data_loader:
        # The file_path is an attribute I need to add to the data object
        # For now, I'll just assume the first data object is the one we want for this example
        # A better implementation would be to match the building_name
        target_data = data
        break

    if target_data is None:
        print(f"Data for {building_name} not found.")
        return

    target_data = target_data.to(device)
    
    with torch.no_grad():
        prediction = model(target_data)

    # The model outputs predictions for the whole batch. 
    # Since we are evaluating one building at a time here, we take the first one.
    predicted_curve = prediction[0].cpu().numpy()
    
    # The target 'y' is also a batch, get the first element
    actual_curve = target_data.y.view(target_data.num_graphs, -1)[0].cpu().numpy()

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(displacement_data, actual_curve, 'b-', label='Actual (FEM)', linewidth=2)
    plt.plot(displacement_data, predicted_curve, 'r--', label='Predicted (GNN)', linewidth=2)
    
    plt.title(f'Pushover Curve: Prediction vs. Actual for {building_name}', fontsize=16)
    plt.xlabel('Roof Displacement (m)', fontsize=12)
    plt.ylabel('Base Shear (N)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")
    return output_path


def main():
    # --- 1. Setup ---
    device = torch.device('cpu')
    
    # --- 2. Load Model ---
    model_path = os.path.join(project_root, 'final_model_10_samples.pt')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
        
    model = GNN_Pushover(in_channels=3, hidden_channels=64, out_channels=100, edge_dim=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    print("Model loaded successfully.")

    # --- 3. Load Data for a specific building ---
    building_name = 'Building_0000_S5_BX3_BZ4_CFalse'
    data_dir = os.path.join(project_root, 'dataset_output_parallel', building_name)
    
    graph_data_path = os.path.join(data_dir, f'{building_name}_graph_data.pt')
    if not os.path.exists(graph_data_path):
        print(f"Graph data not found: {graph_data_path}")
        return
        
    # We load a single data point and put it in a list to simulate a batch of size 1
    graph_data = torch.load(graph_data_path, weights_only=False)
    
    # The DataLoader expects a Dataset object. We'll create a simple one on the fly.
    class SingleItemDataset:
        def __init__(self, item):
            self.item = item
        def __len__(self):
            return 1
        def __getitem__(self, idx):
            return self.item
            
    single_item_dataset = SingleItemDataset(graph_data)
    # Use PyG's DataLoader
    from torch_geometric.loader import DataLoader
    data_loader = DataLoader(single_item_dataset, batch_size=1)

    # --- 4. Load Displacement Data for X-axis ---
    pushover_curve_path = os.path.join(data_dir, f'{building_name}_pushover_curve.csv')
    if not os.path.exists(pushover_curve_path):
        print(f"Pushover curve CSV not found: {pushover_curve_path}")
        return
        
    pushover_df = pd.read_csv(pushover_curve_path)
    # The model predicts 100 points. We need to ensure the displacement data matches this.
    # The 'y' in graph_data.pt should have 100 points.
    # We assume the displacement values in the CSV correspond to these points.
    # If not, we might need to interpolate or select the first 100.
    displacement_data = pushover_df['Roof_Displacement_m'].iloc[:100]

    # --- 5. Generate and Save Plot ---
    output_path = os.path.join(project_root, 'results', f'prediction_{building_name}.png')
    plot_prediction_vs_actual(model, data_loader, device, building_name, displacement_data, output_path)


if __name__ == '__main__':
    main()
