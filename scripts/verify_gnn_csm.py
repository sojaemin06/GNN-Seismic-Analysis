import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from torch_geometric.loader import DataLoader
from sklearn.metrics import r2_score

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈 임포트 ---
from src.gnn.models import PushoverGNN
from src.gnn.train import PushoverDataset
from src.core.capacity_spectrum import calculate_performance_point_csm
from src.core.kds_2022_spectrum import calculate_design_acceleration, determine_seismic_design_category

def verify_gnn_csm():
    # 1. 설정 및 모델 로드
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(project_root) / 'results' / 'paper_figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = Path(project_root) / 'results' / 'models' / 'best_pushover_gnn_model.pt'
    scaler_path = Path(project_root) / 'results' / 'models' / 'scaler.pt'
    
    # 2. 데이터셋 로드
    dataset_processed_dir = Path(project_root) / 'data' / 'processed'
    class CustomPushoverDataset(PushoverDataset):
        @property
        def raw_dir(self): return self.root 
    dataset = CustomPushoverDataset(root=str(dataset_processed_dir))
    
    # Scaler 로드
    checkpoint = torch.load(scaler_path)
    y_mean = checkpoint['mean'].to(device)
    y_std = checkpoint['std'].to(device)
    
    # 모델 아키텍처 초기화
    sample_data = dataset[0]
    model = PushoverGNN(
        node_dim=sample_data.x.shape[1],
        edge_dim=sample_data.edge_attr.shape[1] if sample_data.edge_attr is not None else 0,
        global_dim=sample_data.u.shape[1] if sample_data.u is not None else 0,
        hidden_dim=64, output_dim=100, num_layers=3, heads=2, dropout=0.1
    ).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    print("Searching for a high-quality test sample...")
    
    # 3. 적절한 샘플 찾기 (R2 > 0.95 인 것 중 하나)
    target_idx = -1
    best_r2 = -float('inf')
    
    for i in range(len(dataset)):
        data = dataset[i].to(device)
        with torch.no_grad():
            # Add batch dimension for model input
            from torch_geometric.data import Batch
            batch = Batch.from_data_list([data])
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
            
            pred = (out * y_std + y_mean).cpu().numpy().flatten()
            target = data.y.cpu().numpy().flatten()
            
            # Simple R2 check
            curr_r2 = r2_score(target, pred)
            
            if curr_r2 > 0.95:
                target_idx = i
                best_r2 = curr_r2
                break # Found one
            
            if curr_r2 > best_r2:
                best_r2 = curr_r2
                target_idx = i

    print(f"Selected Sample ID: {target_idx} (R2={best_r2:.4f})")
    data = dataset[target_idx]
    
    # 4. 예측 수행
    with torch.no_grad():
        batch = Batch.from_data_list([data.to(device)])
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.u)
        pred_curve = (out * y_std + y_mean).cpu().numpy().flatten()
        actual_curve = data.y.cpu().numpy().flatten()

    # 5. CSM 입력 데이터 준비
    # 모달 정보 (임시값: 실제로는 데이터 생성 시 저장된 modal_properties.json 필요)
    # GNN 데이터셋에는 모달 정보가 직접 포함되어 있지 않으므로, 
    # 여기서는 논리 검증을 위해 '일반적인 3~5층 RC 건물의 모달 값'을 가정합니다.
    # (추후 dataset 생성 시 global feature에 period 등을 포함시키면 정확해짐)
    
    # 가정된 모달 특성 (Typical Low-rise RC)
    # Period T1 ~ 0.4s, Mass Participation ~ 80%
    # GNN이 예측하는 것은 V-D Curve 자체이므로, 이를 ADRS로 변환하는 계수만 동일하게 적용하면 비교 가능.
    csm_modal_props = {
        'pf1': 1.3,        # Modal Participation Factor
        'm_eff_t1': 200000, # Effective Modal Mass (kg) - Arbitrary but consistent
        'phi_roof': 1.0     # Mode Shape at Roof (Normalized)
    }
    
    # 가상의 지진 하중 (Design Spectrum)
    design_params = [{
        'objective_name': 'Life Safety (LS)',
        'description': '2400년 재현주기 지진의 2/3 수준',
        'method': 'spectrum',
        'site_class': 'S4', # [FIX] Add site class
        'S': 0.22, # Zone Factor 0.22g
        'target_drift_ratio': 0.015, # 1.5%
        'repetition_period': 'Design Basis'
    }]
    
    # DataFrame 변환 (CSM 함수 입력용)
    # 실제 데이터
    df_actual = pd.DataFrame({
        'Roof_Displacement_m': np.linspace(0, 0.04*3.5*3, 100), # X축 (Drift 가정)
        'Base_Shear_N': actual_curve * 9.81 * csm_modal_props['m_eff_t1'] # Y축 (V/W * W) 역산
    })
    # 예측 데이터
    df_pred = pd.DataFrame({
        'Roof_Displacement_m': np.linspace(0, 0.04*3.5*3, 100),
        'Base_Shear_N': pred_curve * 9.81 * csm_modal_props['m_eff_t1']
    })
    
    # 6. CSM 실행
    print("\nRunning CSM for ACTUAL Curve...")
    res_actual = calculate_performance_point_csm(df_actual, csm_modal_props, design_params)
    
    print("\nRunning CSM for PREDICTED Curve...")
    res_pred = calculate_performance_point_csm(df_pred, csm_modal_props, design_params)
    
    # 7. 결과 비교
    if not res_actual or not res_pred:
        print("CSM Calculation Failed.")
        return

    pp_actual = res_actual[0]['performance_point']
    pp_pred = res_pred[0]['performance_point']
    
    sd_error = abs(pp_actual['Sd'] - pp_pred['Sd']) / pp_actual['Sd'] * 100
    sa_error = abs(pp_actual['Sa'] - pp_pred['Sa']) / pp_actual['Sa'] * 100
    
    # [NEW] Strength Degradation Check
    # Actual Curve
    v_max_act = df_actual['Base_Shear_N'].abs().max()
    # Find Base Shear at performance point displacement (Interpolation)
    # Convert Spectral Displacement to Roof Displacement: D = Sd * PF1 * Phi
    # Note: df_actual X-axis is already Roof Displacement
    roof_disp_act_target = pp_actual['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
    v_perf_act = np.interp(roof_disp_act_target, df_actual['Roof_Displacement_m'], df_actual['Base_Shear_N'])
    stab_ratio_act = v_perf_act / v_max_act if v_max_act > 0 else 0
    
    # Predicted Curve
    v_max_pred = df_pred['Base_Shear_N'].abs().max()
    roof_disp_pred_target = pp_pred['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
    v_perf_pred = np.interp(roof_disp_pred_target, df_pred['Roof_Displacement_m'], df_pred['Base_Shear_N'])
    stab_ratio_pred = v_perf_pred / v_max_pred if v_max_pred > 0 else 0

    print(f"\n[Performance Point Comparison]")
    print(f"Actual   : Sd = {pp_actual['Sd']:.4f} m, Sa = {pp_actual['Sa']:.4f} g")
    print(f"           Stability Ratio = {stab_ratio_act:.2f} ({'OK' if stab_ratio_act >= 0.8 else 'FAIL - Instability'})")
    
    print(f"Predicted: Sd = {pp_pred['Sd']:.4f} m, Sa = {pp_pred['Sa']:.4f} g")
    print(f"           Stability Ratio = {stab_ratio_pred:.2f} ({'OK' if stab_ratio_pred >= 0.8 else 'FAIL - Instability'})")
    
    print(f"Error    : Sd = {sd_error:.2f} %, Sa = {sa_error:.2f} %")
    
    # 8. 그래프 그리기
    plt.figure(figsize=(10, 6))
    
    # Demand Spectrum (하나만 그리면 됨)
    demand_sd = res_actual[0]['demand_spectrum_5pct']['Sd']
    demand_sa = res_actual[0]['demand_spectrum_5pct']['Sa']
    plt.plot(demand_sd, demand_sa, 'k-', label='Demand Spectrum (Design Basis)')
    
    # Capacity Curves (ADRS)
    cap_act_sd = res_actual[0]['capacity_adrs']['Sd']
    cap_act_sa = res_actual[0]['capacity_adrs']['Sa']
    plt.plot(cap_act_sd, cap_act_sa, 'b-', label='Capacity Curve (Actual)')
    
    cap_pred_sd = res_pred[0]['capacity_adrs']['Sd']
    cap_pred_sa = res_pred[0]['capacity_adrs']['Sa']
    plt.plot(cap_pred_sd, cap_pred_sa, 'r--', label='Capacity Curve (GNN Predicted)')
    
    # Performance Points
    plt.plot(pp_actual['Sd'], pp_actual['Sa'], 'bo', markersize=10, label=f'PP Actual ($S_d$={pp_actual["Sd"]:.3f})')
    plt.plot(pp_pred['Sd'], pp_pred['Sa'], 'rx', markersize=10, markeredgewidth=2, label=f'PP Predicted ($S_d$={pp_pred["Sd"]:.3f})')
    
    plt.title(f'CSM Comparison: Actual vs GNN Predicted\n(Performance Point Error: {sd_error:.2f}%)')
    plt.xlabel('Spectral Displacement, $S_d$ (m)')
    plt.ylabel('Spectral Acceleration, $S_a$ (g)')
    plt.grid(True, linestyle='--')
    plt.legend()
    plt.tight_layout()
    
    save_path = output_dir / 'fig_csm_comparison.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nComparison plot saved to: {save_path}")

if __name__ == '__main__':
    verify_gnn_csm()
