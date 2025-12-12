import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import os
import sys
import argparse

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.gnn1.models import PushoverGNN

class PushoverDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        # raw_file_names는 모든 .pt 파일 목록
        return [f for f in os.listdir(self.raw_dir) if f.endswith('.pt')]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 데이터가 이미 로컬에 있으므로 다운로드할 필요 없음
        pass

    def process(self):
        data_list = []
        for raw_file_name in tqdm(self.raw_file_names, desc="Processing raw data"):
            raw_path = os.path.join(self.raw_dir, raw_file_name)
            data = torch.load(raw_path, weights_only=False)
            
            # Extract structure_id from filename (e.g., 'data_0_X_neg.pt' -> 0)
            try:
                structure_id_str = raw_file_name.split('_')[1]
                data.structure_id = int(structure_id_str)
            except (IndexError, ValueError):
                data.structure_id = -1 # Assign a default or handle error
            
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def train_model(dataset_dir_name='processed_csm', model_name='best_pushover_gnn_model.pt', sample_count=None, epochs=200, silent=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not silent:
        print(f"Using device: {device}")
        print(f"Dataset: {dataset_dir_name}")
        print(f"Model Output: {model_name}")

    # 모델 및 스케일러 저장 경로 정의
    model_save_dir = Path(project_root) / 'results' / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True) # 디렉토리가 없으면 생성
    model_path = model_save_dir / model_name
    scaler_path = model_save_dir / "scaler.pt"

    # --- 1. 데이터셋 로드 ---
    dataset_root = Path(project_root) / 'data'
    dataset_processed_dir = dataset_root / dataset_dir_name # .pt 파일들이 있는 실제 경로
    
    # Check if dataset dir exists
    if not dataset_processed_dir.exists():
        print(f"Error: Dataset directory not found: {dataset_processed_dir}")
        return

    class CustomPushoverDataset(PushoverDataset):
        @property
        def raw_dir(self):
            return self.root 

    # processed_dir needs to be unique per dataset to avoid caching conflicts
    # We can force re-processing or use a unique suffix
    # Here, PyG uses `root/processed` by default. 
    # If we change `root`, `processed` will be `root/processed`.
    # Since `dataset_processed_dir` changes, the cache will be separate. Safe.
    
    dataset = CustomPushoverDataset(root=str(dataset_processed_dir))
    
    # --- 2. 데이터 분할 및 샘플링 ---
    # `data_list`의 각 Data 객체는 `structure_id` 속성을 가지고 있음
    data_list = [dataset[i] for i in range(len(dataset))]
    
    # [NEW] 데이터 수 조절 (실험용) - sample_count는 구조물의 개수를 의미
    if sample_count is not None:
        import random
        random.seed(42) # 재현성을 위해 seed 고정
        
        all_structure_ids = list(set([data.structure_id for data in data_list if hasattr(data, 'structure_id')]))
        
        if sample_count > len(all_structure_ids):
            if not silent: print(f"Warning: Requested sample_count (structures) {sample_count} > total unique structures {len(all_structure_ids)}. Using all unique structures.")
            selected_structure_ids = all_structure_ids
        else:
            selected_structure_ids = random.sample(all_structure_ids, sample_count)
            if not silent: print(f"Using subset of data: {len(selected_structure_ids)} unique structures")

        # 선택된 구조물 ID에 해당하는 모든 데이터 파일 포함
        filtered_data_list = [data for data in data_list if hasattr(data, 'structure_id') and data.structure_id in selected_structure_ids]
        data_list = filtered_data_list
        if not silent: print(f"Total data files (including all directions) for selected structures: {len(data_list)} files")

    if not data_list:
        return {'test_loss': float('nan'), 'test_r2': float('nan'), 'train_loss': float('nan'), 'val_loss': float('nan')}

    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    # Validation data should be split from train data
    if len(train_data) > 1:
        train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42) # 0.25 * 0.8 = 0.2
    else:
        val_data = [] # 데이터가 너무 적을 경우 예외 처리

    if not silent:
        print(f"Total samples: {len(data_list)}")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")

    # --- 3. 데이터 로더 ---
    batch_size = 32
    # Drop last if batch size 1 to avoid BatchNorm error
    drop_last = len(train_data) > batch_size 
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False) if val_data else None
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # --- Standardization ---
    all_y = torch.cat([data.y for data in data_list], dim=0)
    y_mean = all_y.mean().to(device)
    y_std = all_y.std().to(device)
    
    # --- 4. 모델 초기화 ---
    first_data = data_list[0]
    node_dim = first_data.x.shape[1]
    edge_dim = first_data.edge_attr.shape[1] if first_data.edge_attr is not None else 0
    global_dim = first_data.u.shape[1] if first_data.u is not None else 0
    output_dim = first_data.y.shape[1] if first_data.y is not None else 100 

    if not silent:
        print(f"Node Dim: {node_dim}, Edge Dim: {edge_dim}, Global Dim: {global_dim}")

    model = PushoverGNN(
        node_dim=node_dim,
        edge_dim=edge_dim,
        global_dim=global_dim,
        hidden_dim=64,
        output_dim=output_dim,
        num_layers=3, 
        heads=2,
        dropout=0.1
    ).to(device)

    # --- 5. 손실 함수 및 최적화 ---
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # --- 6. 학습 루프 ---
    best_val_loss = float('inf')
    avg_train_loss = 0.0
    
    # 모델 저장 경로는 실험 중 덮어쓰기 방지를 위해 별도 처리하거나 임시 경로 사용 권장
    # 여기서는 간단히 덮어씀 (실험 스크립트에서 관리 필요 시 수정)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        steps = 0
        for batch_data in train_loader:
            batch_data = batch_data.to(device)
            optimizer.zero_grad()
            out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, batch_data.u)
            y_target = (batch_data.y.squeeze(1) - y_mean) / y_std
            loss = criterion(out, y_target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            steps += 1
        
        avg_train_loss = train_loss / steps if steps > 0 else 0

        # --- 검증 ---
        avg_val_loss = 0
        if val_loader:
            model.eval()
            val_loss = 0
            val_steps = 0
            with torch.no_grad():
                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, batch_data.u)
                    y_target = (batch_data.y.squeeze(1) - y_mean) / y_std
                    loss = criterion(out, y_target)
                    val_loss += loss.item()
                    val_steps += 1
            avg_val_loss = val_loss / val_steps if val_steps > 0 else 0
            
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # 모델 저장
                torch.save(model.state_dict(), model_path)
                # Scaler (y_mean, y_std) 저장
                torch.save({'mean': y_mean, 'std': y_std}, scaler_path)
                # if not silent: print(f"  --> Model and Scaler saved. Best Val Loss: {best_val_loss:.4f}")
        
        if not silent and (epoch + 1) % 20 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs}, Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}, LR: {current_lr:.1e}")

    # --- 7. 테스트 및 R2 계산 ---
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, weights_only=True)) # Best model 로드
    else:
        print("Warning: No best model saved. Using current weights.")

    model.eval()
    test_loss = 0
    test_steps = 0
    
    # R2 Score 계산을 위한 변수
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            out = model(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, batch_data.u)
            
            # Loss 계산 (Normalized scale)
            y_target_norm = (batch_data.y.squeeze(1) - y_mean) / y_std
            loss = criterion(out, y_target_norm)
            test_loss += loss.item()
            test_steps += 1
            
            # R2 계산을 위해 원래 스케일로 복원
            y_pred_original = out * y_std + y_mean
            y_true_original = batch_data.y.squeeze(1)
            
            y_true_all.append(y_true_original.cpu())
            y_pred_all.append(y_pred_original.cpu())
    
    avg_test_loss = test_loss / test_steps if test_steps > 0 else 0
    
    # R2 Score
    if y_true_all:
        y_true_all = torch.cat(y_true_all, dim=0)
        y_pred_all = torch.cat(y_pred_all, dim=0)
        
        ss_res_flat = torch.sum((y_true_all - y_pred_all) ** 2)
        ss_tot_flat = torch.sum((y_true_all - torch.mean(y_true_all)) ** 2)
        
        r2_score = 1 - ss_res_flat / (ss_tot_flat + 1e-8)
        r2_score = r2_score.item()
    else:
        r2_score = float('nan')

    if not silent:
        print(f"Test Loss: {avg_test_loss:.6f}")
        print(f"Test R2 Score: {r2_score:.4f}")

    return {
        'dataset': dataset_dir_name,
        'test_r2': r2_score,
        'test_loss': avg_test_loss,
        'train_loss': avg_train_loss,
        'val_loss': best_val_loss
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Pushover GNN')
    parser.add_argument('--dataset_dir', type=str, default='processed', help='Directory name under data/')
    parser.add_argument('--model_name', type=str, default='best_pushover_gnn_model.pt', help='Filename for saving the model')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--sample_count', type=int, default=None, help='Number of samples to use')
    args = parser.parse_args()

    train_model(dataset_dir_name=args.dataset_dir, model_name=args.model_name, epochs=args.epochs, sample_count=args.sample_count)
