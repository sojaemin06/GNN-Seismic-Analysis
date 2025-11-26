# -*- coding: utf-8 -*-
import sys
import os
import random
import matplotlib.pyplot as plt
from pathlib import Path

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈 임포트 ---
from scripts.run_single_analysis import get_single_run_parameters
from src.core.model_builder import build_model
from src.core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from src.core.post_processor import process_pushover_results
from src.core.capacity_spectrum import pushover_to_adrs
import openseespy.opensees as ops

def plot_capacity_spectrum_for_specific_design(direction='X', seed=42):
    """
    특정 설계안(seed로 고정)에 대한 역량 스펙트럼을 생성하고 그래프로 저장합니다.
    """
    print(f"\n--- [{direction}-Direction] Capacity Spectrum Generation ---")
    
    # 1. 재현 가능한 설계안 생성을 위해 random seed 고정
    random.seed(seed)
    print(f"Using random seed: {seed} for a reproducible design.")

    # 2. 단일 실행용 파라미터 가져오기
    params = get_single_run_parameters()
    params['skip_post_processing'] = True # 기존 시각화는 건너뜁니다.
    
    # 출력 디렉토리 이름에 방향 추가
    original_name = params['analysis_name']
    original_dir = params['output_dir']
    params['analysis_name'] = f"{original_name}_{direction}"
    params['output_dir'] = original_dir.parent / f"{original_dir.name}_{direction}"

    # 3. 해석 실행 (run_performance_analysis.py 로직 재사용)
    params['output_dir'].mkdir(parents=True, exist_ok=True)
    model_nodes_info = build_model(params)
    
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. 해석을 중단합니다."); ops.wipe(); return

    ok, modal_props = run_eigen_analysis(params, model_nodes_info)
    if not ok:
        print("\n고유치 해석 실패. 해석을 중단합니다."); ops.wipe(); return
    
    ok, dominant_mode = run_pushover_analysis(params, model_nodes_info, modal_props, direction=direction)
    if not ok:
        print("\n푸쉬오버 해석 실행 중 오류 발생."); ops.wipe(); return
    
    df_curve, _, _, _ = process_pushover_results(params, model_nodes_info, dominant_mode, direction=direction)
    ops.wipe()
    
    if df_curve is None or df_curve.empty:
        print("푸쉬오버 곡선 생성 실패."); return
    
    print("\n1. Pushover analysis completed.")

    # 4. 역량 스펙트럼(ADRS)으로 변환
    csm_modal_props = {
        'pf1': dominant_mode[f'gamma_{direction.lower()}'],
        'm_eff_t1': dominant_mode[f'M_star_{direction.lower()}']
    }
    capacity_adrs = pushover_to_adrs(df_curve, csm_modal_props['pf1'], csm_modal_props['m_eff_t1'])
    
    if capacity_adrs.empty:
        print("Failed to convert pushover curve to ADRS format.")
        return
        
    print("2. Converted Pushover Curve to Capacity Spectrum (ADRS).")

    # 5. 그래프 생성 및 저장
    plt.figure(figsize=(10, 8))
    plt.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], marker='o', linestyle='-', markersize=3)
    plt.title(f'Capacity Spectrum (ADRS) - Direction {direction} (Seed: {seed})')
    plt.xlabel('Spectral Displacement, Sd (m)')
    plt.ylabel('Spectral Acceleration, Sa (g)')
    plt.grid(True)
    
    # 결과 저장 경로 설정
    plot_path = params['output_dir'] / f"capacity_spectrum_{direction}.png"
    plt.savefig(plot_path)
    
    print(f"\n3. Capacity spectrum plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    # X 방향에 대해 특정 설계안의 역량 스펙트럼 그래프 생성
    plot_capacity_spectrum_for_specific_design(direction='X', seed=42)
    # Z 방향에 대해 동일한 설계안의 역량 스펙트럼 그래프 생성
    plot_capacity_spectrum_for_specific_design(direction='Z', seed=42)
