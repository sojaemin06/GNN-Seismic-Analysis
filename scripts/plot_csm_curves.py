# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈 임포트 ---
from src.core.capacity_spectrum import pushover_to_adrs, generate_kds2022_demand_spectrum

def plot_csm_curves(results_dir: Path):
    """
    지정된 결과 디렉토리의 파일을 읽어 역량 스펙트럼과 요구 스펙트럼을 함께 도시합니다.
    """
    print(f"\n--- Plotting CSM Curves for: {results_dir.name} ---")

    # 1. 파일 경로 찾기 및 데이터 로드
    direction = 'X' if '_X' in results_dir.name else 'Z'
    pushover_csv_path = next(results_dir.glob(f'*_pushover_curve_{direction}.csv'), None)
    modal_json_path = results_dir / 'modal_properties.json'

    if not pushover_csv_path or not modal_json_path.exists():
        print(f"Error: Required data files not found in {results_dir}"); return

    try:
        df_curve = pd.read_csv(pushover_csv_path)
        with open(modal_json_path, 'r') as f:
            modal_data = json.load(f)
        dominant_mode = modal_data['dominant_mode']
        
        # 모드 형상 값 추출 (지붕 레벨 = 마지막 요소)
        phi_roof = dominant_mode[f'phi_{direction.lower()}'][-1]
        
        modal_props = {
            'pf1': dominant_mode[f'gamma_{direction.lower()}'],
            'm_eff_t1': dominant_mode[f'M_star_{direction.lower()}'],
            'phi_roof': phi_roof
        }
        print("Successfully loaded pushover curve and modal properties.")
    except Exception as e:
        print(f"Error reading data files: {e}"); return

    # 2. 역량 스펙트럼 생성
    capacity_adrs = pushover_to_adrs(df_curve, modal_props['pf1'], modal_props['m_eff_t1'], modal_props['phi_roof'])
    if capacity_adrs.empty:
        print("Failed to generate capacity spectrum."); return
    print("Generated Capacity Spectrum (ADRS).")

    # 3. 요구 스펙트럼 생성 (5% 감쇠)
    design_params = {'S': 0.16, 'site_class': 'S4'}
    Sd_demand_5pct, Sa_demand_5pct = generate_kds2022_demand_spectrum(**design_params)
    print("Generated 5% Damped Demand Spectrum.")

    # 4. 두 스펙트럼을 함께 그래프로 그리기
    plt.figure(figsize=(12, 9))
    plt.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], 'b-', marker='.', label='Capacity Spectrum (Pushover)')
    plt.plot(Sd_demand_5pct, Sa_demand_5pct, 'r--', label='5% Damped Demand Spectrum (KDS 2022)')
    
    plt.title(f'Capacity vs. Demand Spectrum - {direction} direction')
    plt.xlabel('Spectral Displacement, Sd (m)')
    plt.ylabel('Spectral Acceleration, Sa (g)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Zoom in on the area of interest if Sd is small
    if not capacity_adrs.empty:
        plt.xlim(0, max(capacity_adrs['Sd'].max(), Sd_demand_5pct.max()) * 1.1)
        plt.ylim(0, max(capacity_adrs['Sa'].max(), Sa_demand_5pct.max()) * 1.1)

    # 5. 그래프 저장
    plot_path = results_dir / f"CSM_curves_plot_{direction}.png"
    plt.savefig(plot_path)
    print(f"\nCSM curves plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    default_dir_x = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X'
    default_dir_z = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z'
    
    if default_dir_x.exists():
        plot_csm_curves(default_dir_x)
    else:
        print(f"Default directory not found: {default_dir_x}")
        
    if default_dir_z.exists():
        plot_csm_curves(default_dir_z)
    else:
        print(f"Default directory not found: {default_dir_z}")
