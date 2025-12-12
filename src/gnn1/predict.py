import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset

# --- 프로젝트 루트 설정 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.models import PushoverGNN

# --- Dataset 클래스 (train.py와 동일) ---
class PushoverDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pt')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self): pass
    def process(self): pass

class CustomPushoverDataset(PushoverDataset):
    @property
    def raw_dir(self):
        return self.root

def predict_and_plot():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. 데이터 로드 및 Test Set 분리
    dataset_root = Path(project_root) / 'data' / 'processed'
    dataset = CustomPushoverDataset(root=str(dataset_root))
    
    data_list = [dataset[i] for i in range(len(dataset))]
    
    # Random State 42로 고정하여 Train/Val/Test 분할 재현
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    print(f"Test Set Samples: {len(test_data)}")
    
    # 시각화를 위해 Shuffle하여 랜덤하게 샘플 선택
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # 2. 모델 설정 및 가중치 로드
    if not data_list:
        print("No data found.")
        return

    first_data = data_list[0]
    node_dim = first_data.x.shape[1]
    edge_dim = first_data.edge_attr.shape[1]
    global_dim = first_data.u.shape[1]
    output_dim = first_data.y.shape[1]

    model = PushoverGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=64,
        output_dim=output_dim,
        num_layers=3,
        heads=2,
        dropout=0.0 # Inference 시에는 Dropout 끔
    ).to(device)

    model_path = Path(project_root) / 'results' / 'models' / 'best_pushover_gnn_model.pt'
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        return

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # [NEW] Scaler 로드 (역변환용)
    scaler_path = Path(project_root) / 'results' / 'models' / 'scaler.pt'
    scaler = torch.load(scaler_path, map_location=device)
    y_mean = scaler['mean'].cpu().numpy()
    y_std = scaler['std'].cpu().numpy()
    print(f"Loaded Scaler: Mean={y_mean:.4f}, Std={y_std:.4f}")

    # 3. 예측 수행 및 시각화
    num_plots = 5
    fig, axes = plt.subplots(1, num_plots, figsize=(25, 5))
    
    # 폰트 설정 (한글 깨짐 방지용 - 영문으로 표기하므로 기본값 사용)
    plt.rcParams['font.family'] = 'sans-serif'

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_plots: break
            
            batch = batch.to(device)
            
            # 예측
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
            
            # 데이터 변환 (Tensor -> Numpy)
            y_true = batch.y.cpu().numpy().flatten()
            
            # [NEW] 예측값 역변환 (Standardized -> Original Scale)
            y_pred_std = pred.cpu().numpy().flatten()
            y_pred = y_pred_std * y_std + y_mean
            
            # 정규화된 X축 (0 ~ 1) - 실제 변위 값은 스케일링 복원이 필요할 수 있으나, 패턴 비교를 위해 정규화된 값 사용
            x_axis = np.linspace(0, 1, len(y_true))
            
            # 해석 방향 확인
            direction = "X" if batch.u[0, 0] > 0.5 else "Z"
            
            # Plot
            ax = axes[i]
            ax.plot(x_axis, y_true, label='Ground Truth (OpenSees)', color='black', linewidth=2, alpha=0.7)
            ax.plot(x_axis, y_pred, label='GNN Prediction', color='red', linestyle='--', linewidth=2)
            
            ax.set_title(f"Test Sample {i+1} ({direction}-Direction)")
            ax.set_xlabel("Normalized Roof Displacement")
            ax.set_ylabel("Base Shear Coefficient (V/W)")
            ax.legend()
            ax.grid(True, linestyle=':', alpha=0.6)
            
            # 오차(MSE) 표시
            mse = ((y_true - y_pred)**2).mean()
            ax.text(0.05, 0.95, f"MSE: {mse:.4e}", transform=ax.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    output_path = Path(project_root) / 'results' / 'prediction_samples.png'
    plt.savefig(output_path)
    print(f"\nPrediction visualization saved to: {output_path}")

if __name__ == "__main__":
    predict_and_plot()
