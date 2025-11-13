import sys
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
import argparse

# Add the project root to the Python path to resolve module import errors
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from gnn_code.models import GNN_Pushover

def plot_prediction_vs_actual(building_name, pred_x, actual_x, disp_x, pred_z, actual_z, disp_z, output_path):
    """
    Plots the predicted vs. actual pushover curves for a single building in both X and Z directions.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # X-Direction Plot
    ax1.plot(disp_x, actual_x, 'b-', label='Actual (FEM)', linewidth=2)
    ax1.plot(disp_x, pred_x, 'r--', label='Predicted (GNN)', linewidth=2)
    ax1.set_title(f'Pushover Curve (X-Direction) for {building_name}', fontsize=16)
    ax1.set_xlabel('Roof Displacement (m)', fontsize=12)
    ax1.set_ylabel('Base Shear (N)', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Z-Direction Plot
    ax2.plot(disp_z, actual_z, 'b-', label='Actual (FEM)', linewidth=2)
    ax2.plot(disp_z, pred_z, 'r--', label='Predicted (GNN)', linewidth=2)
    ax2.set_title(f'Pushover Curve (Z-Direction) for {building_name}', fontsize=16)
    ax2.set_xlabel('Roof Displacement (m)', fontsize=12)
    ax2.set_ylabel('Base Shear (N)', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Ensure the results directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

def main(building_name):
    """
    Main function to load model, data, and generate prediction plot for a specific building.
    """
    # --- 1. Setup ---
    device = torch.device('cpu')
    
    # --- 2. Load Model ---
    model_path = os.path.join(project_root, 'gnn_code', 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
        
    # Model output is now 200
    model = GNN_Pushover(in_channels=3, hidden_channels=64, out_channels=200, edge_dim=10)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. Load Data for the specific building ---
    summary_path = os.path.join(project_root, 'dataset_output_parallel', 'dataset_summary.csv')
    if not os.path.exists(summary_path):
        print(f"Dataset summary not found at: {summary_path}")
        return
    
    summary_df = pd.read_csv(summary_path)
    building_info = summary_df[summary_df['analysis_name'] == building_name]

    if building_info.empty:
        print(f"Building '{building_name}' not found in dataset_summary.csv")
        return
        
    # Get max displacement for both directions
    max_disp_x = building_info.iloc[0]['max_roof_disp_x']
    max_disp_z = building_info.iloc[0]['max_roof_disp_z']
    displacement_data_x = np.linspace(0, max_disp_x, 100)
    displacement_data_z = np.linspace(0, max_disp_z, 100)

    # Load the graph data
    graph_data_path = os.path.join(project_root, 'dataset_output_parallel', building_name, f'{building_name}_graph_data.pt')
    if not os.path.exists(graph_data_path):
        print(f"Graph data not found: {graph_data_path}")
        return
        
    graph_data = torch.load(graph_data_path, weights_only=False)
    graph_data = graph_data.to(device)

    # --- 4. Make Prediction ---
    with torch.no_grad():
        data_loader = DataLoader([graph_data], batch_size=1)
        for batch in data_loader:
            prediction = model(batch)
            break

    # Split predicted and actual curves into X and Z
    predicted_curve_x = prediction[0][:100].cpu().numpy()
    predicted_curve_z = prediction[0][100:].cpu().numpy()
    
    actual_curve_x = graph_data.y[:100].cpu().numpy()
    actual_curve_z = graph_data.y[100:].cpu().numpy()

    # --- 5. Generate and Save Plot ---
    output_path = os.path.join(project_root, 'results', f'prediction_{building_name}_biaxial.png')
    plot_prediction_vs_actual(
        building_name, 
        predicted_curve_x, actual_curve_x, displacement_data_x,
        predicted_curve_z, actual_curve_z, displacement_data_z,
        output_path
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot GNN prediction vs. actual FEM result for a given building.')
    parser.add_argument(
        '--building_name', 
        type=str, 
        default='Building_0052_S5_BX4_BZ4_CFalse',
        help='The name of the building to plot (must exist in the dataset summary).'
    )
    args = parser.parse_args()
    
    main(args.building_name)
