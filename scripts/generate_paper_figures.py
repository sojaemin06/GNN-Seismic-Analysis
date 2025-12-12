import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.models import PushoverGNN
from src.gnn1.train import PushoverDataset 

def generate_figures():
    # --- 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(project_root) / 'results' / 'paper_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(project_root) / 'results' / 'models' / 'best_pushover_gnn_model.pt'
    scaler_path = Path(project_root) / 'results' / 'models' / 'scaler.pt'
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    # --- 데이터 로드 ---
    dataset_processed_dir = Path(project_root) / 'data' / 'processed'
    
    class CustomPushoverDataset(PushoverDataset):
        @property
        def raw_dir(self): return self.root 

    dataset = CustomPushoverDataset(root=str(dataset_processed_dir))
    data_list = [dataset[i] for i in range(len(dataset))]
    
    # Train/Test Split (train.py와 동일한 시드 사용)
    _, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False) # Batch size 1 for sample-wise analysis
    
    print(f"Generating figures using {len(test_data)} test samples...")

    # --- Scaler 로드 ---
    checkpoint = torch.load(scaler_path)
    y_mean = checkpoint['mean'].to(device)
    y_std = checkpoint['std'].to(device)

    # --- 모델 로드 ---
    first_data = data_list[0]
    model = PushoverGNN(
        node_dim=first_data.x.shape[1],
        edge_dim=first_data.edge_attr.shape[1] if first_data.edge_attr is not None else 0,
        global_dim=first_data.u.shape[1] if first_data.u is not None else 0,
        hidden_dim=64,
        output_dim=100,
        num_layers=3,
        heads=2,
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # --- 예측 수행 ---
    all_preds = []
    all_targets = []
    sample_r2_scores = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch, data.u)
            
            # 스케일 복원
            pred_original = out * y_std + y_mean
            target_original = data.y
            
            # Numpy 변환
            pred_np = pred_original.cpu().numpy().flatten()
            target_np = target_original.cpu().numpy().flatten()
            
            all_preds.extend(pred_np)
            all_targets.extend(target_np)
            
            # 개별 샘플 R2 계산
            if len(target_np) > 1:
                r2 = r2_score(target_np, pred_np)
                sample_r2_scores.append(r2)
            else:
                sample_r2_scores.append(0.0)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # --- Figure 1: Parity Plot (All Points) ---
    plt.figure(figsize=(7, 7))
    plt.scatter(all_targets, all_preds, alpha=0.3, s=10, c='blue', edgecolors='none')
    
    # Reference Line (y=x)
    min_val = min(all_targets.min(), all_preds.min())
    max_val = max(all_targets.max(), all_preds.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Global Metrics
    global_r2 = r2_score(all_targets, all_preds)
    plt.text(min_val + 0.05 * (max_val - min_val), max_val - 0.1 * (max_val - min_val), 
             f'$R^2 = {global_r2:.4f}$', fontsize=14, fontweight='bold')
    
    plt.xlabel('Ground Truth (Normalized Base Shear)', fontsize=12)
    plt.ylabel('Predicted (Normalized Base Shear)', fontsize=12)
    plt.title('Parity Plot: Actual vs Predicted Pushover Curve Points', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_parity_plot.png', dpi=300)
    print(f"Saved Parity Plot to {output_dir / 'fig_parity_plot.png'}")

    # --- Figure 2: Representative Curves (Best, Median, Worst) ---
    # 샘플별 R2 정렬
    sorted_indices = np.argsort(sample_r2_scores)
    
    best_idx = sorted_indices[-1]
    median_idx = sorted_indices[len(sorted_indices)//2]
    worst_idx = sorted_indices[0] # R2가 가장 낮은 것 (음수일 수도 있음)
    
    indices_to_plot = [
        ('Best', best_idx, 'green'),
        ('Median', median_idx, 'blue'),
        ('Worst', worst_idx, 'red')
    ]
    
    plt.figure(figsize=(15, 5))
    
    for i, (label, idx, color) in enumerate(indices_to_plot):
        data = test_loader.dataset[idx]
        data = data.to(device)
        
        with torch.no_grad():
            # Batch 차원 추가 (Unsqueeze)가 필요할 수 있음. DataLoader batch=1이라 data 자체가 Batch임.
            # 하지만 DataLoader에서 꺼낸 게 아니라 dataset에서 바로 가져왔으므로 Batch 처리가 안 되어 있음.
            # 수동으로 Batch=1 만들기
            from torch_geometric.data import Batch
            batch_data = Batch.from_data_list([data])
            
            out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, batch_data.u)
            pred = (out * y_std + y_mean).cpu().numpy().flatten()
            target = data.y.cpu().numpy().flatten()
            
        plt.subplot(1, 3, i+1)
        steps = np.arange(len(target)) # X축은 Step (또는 Drift인데 여기선 Step으로 근사)
        
        plt.plot(steps, target, 'k-', lw=2, label='Actual')
        plt.plot(steps, pred, color=color, linestyle='--', lw=2, label=f'Predicted ($R^2$={sample_r2_scores[idx]:.2f})')
        
        plt.title(f'{label} Case Sample', fontsize=12)
        plt.xlabel('Displacement Step')
        plt.ylabel('Base Shear Coefficient')
        plt.legend()
        plt.grid(True, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'fig_representative_curves.png', dpi=300)
    print(f"Saved Representative Curves to {output_dir / 'fig_representative_curves.png'}")

    # --- Figure 3: Error Histogram ---
    errors = all_preds - all_targets
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=50, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    
    # Normal Distribution Fit Curve (Optional comparison)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = (1 / (np.sqrt(2 * np.pi) * std_error)) * np.exp(-0.5 * ((x - mean_error) / std_error) ** 2)
    plt.plot(x, p, 'r--', linewidth=2, label=f'Fit ($\mu={mean_error:.4f}, \sigma={std_error:.4f}$)')
    
    plt.axvline(mean_error, color='k', linestyle='dashed', linewidth=1)
    plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / 'fig_error_histogram.png', dpi=300)
    print(f"Saved Error Histogram to {output_dir / 'fig_error_histogram.png'}")

if __name__ == '__main__':
    generate_figures()
