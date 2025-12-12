import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import pandas as pd

# --- 프로젝트 루트 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn.models import PushoverGNN
from src.gnn.predict import CustomPushoverDataset

def calculate_structural_metrics(y_true, y_pred):
    """
    단일 곡선에 대한 구조적 성능 지표를 계산합니다.
    """
    # 1. Max Base Shear (최대 강도)
    max_shear_true = np.max(y_true)
    max_shear_pred = np.max(y_pred)
    
    # 2. Initial Stiffness (초기 강성) - 초기 10% 구간의 기울기 (단순화)
    # x축은 0~1로 정규화되어 있다고 가정 (데이터 포인트 100개 중 10번째 인덱스)
    idx_stiff = 10 
    stiff_true = y_true[idx_stiff] / (idx_stiff / 100.0)
    stiff_pred = y_pred[idx_stiff] / (idx_stiff / 100.0)
    
    # 3. Energy Dissipation (에너지 소산) - 곡선 아래 면적 (Trapezoidal rule)
    x_axis = np.linspace(0, 1, len(y_true))
    energy_true = np.trapz(y_true, x_axis)
    energy_pred = np.trapz(y_pred, x_axis)
    
    return {
        'max_shear_true': max_shear_true, 'max_shear_pred': max_shear_pred,
        'stiff_true': stiff_true, 'stiff_pred': stiff_pred,
        'energy_true': energy_true, 'energy_pred': energy_pred
    }

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 데이터 로드
    dataset_root = Path(project_root) / 'data' / 'processed'
    dataset = CustomPushoverDataset(root=str(dataset_root))
    data_list = [dataset[i] for i in range(len(dataset))]
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # 2. 모델 로드
    first_data = data_list[0]
    model = PushoverGNN(
        node_dim=first_data.x.shape[1],
        edge_dim=first_data.edge_attr.shape[1],
        global_dim=first_data.u.shape[1],
        hidden_dim=64,
        output_dim=first_data.y.shape[1],
        num_layers=3, heads=2, dropout=0.0
    ).to(device)
    
    model_path = Path(project_root) / 'results' / 'models' / 'best_pushover_gnn_model.pt'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # [NEW] Scaler 로드 (역변환용)
    scaler_path = Path(project_root) / 'results' / 'models' / 'scaler.pt'
    scaler = torch.load(scaler_path, map_location=device)
    y_mean = scaler['mean'].cpu().numpy()
    y_std = scaler['std'].cpu().numpy()
    print(f"Loaded Scaler: Mean={y_mean:.4f}, Std={y_std:.4f}")

    # 3. 전체 테스트 셋 평가
    curve_metrics = {'r2': [], 'mae': [], 'mse': []}
    struct_data = []
    
    filtered_count = 0

    print(f"\nEvaluating on {len(test_data)} test samples...")
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = batch.to(device)
            
            y_true = batch.y.cpu().numpy().flatten()
            
            # [NEW] Filter out failed analysis (Flat curves)
            if np.var(y_true) < 1e-4:
                filtered_count += 1
                continue

            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
            
            # [NEW] 예측값 역변환 (Standardized -> Original Scale)
            y_pred_std = pred.cpu().numpy().flatten()
            y_pred = y_pred_std * y_std + y_mean
            
            # --- DEBUG: Print stats for the first sample ---
            if i == 0:
                print(f"\n--- DEBUG (Sample 0) ---")
                print(f"y_true (Coeff)    : Min={y_true.min():.4f}, Max={y_true.max():.4f}, Mean={y_true.mean():.4f}")
                print(f"y_pred (Coeff)    : Min={y_pred.min():.4f}, Max={y_pred.max():.4f}, Mean={y_pred.mean():.4f}")
                print(f"------------------------\n")
            
            # --- 곡선 적합도 메트릭 ---
            curve_metrics['r2'].append(r2_score(y_true, y_pred))
            curve_metrics['mae'].append(mean_absolute_error(y_true, y_pred))
            curve_metrics['mse'].append(mean_squared_error(y_true, y_pred))
            
            # --- 구조 성능 메트릭 ---
            s_metrics = calculate_structural_metrics(y_true, y_pred)
            struct_data.append(s_metrics)

    # 4. 결과 집계 및 출력
    avg_r2 = np.mean(curve_metrics['r2'])
    median_r2 = np.median(curve_metrics['r2'])
    avg_mae = np.mean(curve_metrics['mae'])
    
    neg_r2_count = sum(1 for r in curve_metrics['r2'] if r < 0)
    
    print("\n" + "="*40)
    print("   MODEL EVALUATION REPORT (Test Set)")
    print("="*40)
    print(f"Overall Curve Fit:")
    print(f"  - Mean R2 Score   : {avg_r2:.4f}")
    print(f"  - Median R2 Score : {median_r2:.4f} (Robust to outliers)")
    print(f"  - Mean MAE        : {avg_mae:.4f}")
    print(f"  - Negative R2 Cnt : {neg_r2_count} / {len(test_data)} samples")
    print("-" * 40)
    
    df_struct = pd.DataFrame(struct_data)
    
    metrics_to_plot = [
        ('Max Base Shear', 'max_shear_true', 'max_shear_pred'),
        ('Initial Stiffness', 'stiff_true', 'stiff_pred'),
        ('Energy Dissipation', 'energy_true', 'energy_pred')
    ]
    
    print(f"Structural Parameter Accuracy (R2 Score):")
    for name, true_col, pred_col in metrics_to_plot:
        param_r2 = r2_score(df_struct[true_col], df_struct[pred_col])
        param_corr, _ = pearsonr(df_struct[true_col], df_struct[pred_col])
        error_rate = np.mean(np.abs((df_struct[true_col] - df_struct[pred_col]) / df_struct[true_col])) * 100
        
        print(f"  - {name:<18} : R2={param_r2:.4f}, Corr={param_corr:.4f}, MAPE={error_rate:.2f}%")

    # 5. Scatter Plot 생성
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # 2x2 grid
    
    # R2 Histogram
    ax_hist = axes[0, 0]
    ax_hist.hist(curve_metrics['r2'], bins=30, range=(0, 1), color='skyblue', edgecolor='black')
    ax_hist.set_title("Distribution of Curve R2 Scores")
    ax_hist.set_xlabel("R2 Score")
    ax_hist.set_ylabel("Count")
    ax_hist.text(0.05, 0.95, f"Median R2: {median_r2:.4f}", transform=ax_hist.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Structural Plots (Map 3 plots to remaining 3 axes)
    struct_axes = [axes[0, 1], axes[1, 0], axes[1, 1]]
    
    for ax, (name, true_col, pred_col) in zip(struct_axes, metrics_to_plot):
        y_t = df_struct[true_col]
        y_p = df_struct[pred_col]
        
        # 산점도
        ax.scatter(y_t, y_p, alpha=0.7, edgecolors='b')
        
        # 이상적인 1:1 선
        min_val = min(y_t.min(), y_p.min())
        max_val = max(y_t.max(), y_p.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1')
        
        ax.set_title(name)
        ax.set_xlabel("Ground Truth (OpenSees)")
        ax.set_ylabel("GNN Prediction")
        ax.legend()
        ax.grid(True, linestyle=':', alpha=0.6)
        
        # R2 표시
        r2 = r2_score(y_t, y_p)
        ax.text(0.05, 0.95, f"R² = {r2:.4f}", transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = Path(project_root) / 'results' / 'evaluation_dashboard.png'
    plt.savefig(output_path)
    print(f"\nEvaluation dashboard saved to: {output_path}")

if __name__ == "__main__":
    evaluate()
