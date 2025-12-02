# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
import argparse
from pathlib import Path

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈 임포트 ---
from src.core.capacity_spectrum import calculate_performance_point_csm
from src.core.kds_2022_spectrum import calculate_design_acceleration, determine_seismic_design_category
from src.core.kds_performance_criteria import get_performance_objectives
from src.visualization.animate_csm import animate_csm_process

def load_config(config_path):
    """설정 파일을 로드합니다."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def run_evaluation(results_dir: Path, config: dict):
    """
    설정(config)에 따라 특정 결과 디렉토리의 구조물에 대한 내진성능평가를 수행합니다.
    """
    print(f"\n{'='*60}")
    print(f"Processing Directory: {results_dir.name}")
    print(f"{'='*60}")

    # 1. 데이터 파일 경로 찾기
    direction = 'X' if '_X' in results_dir.name else 'Z'
    pushover_csv_path = next(results_dir.glob(f'*_pushover_curve_{direction}.csv'), None)
    modal_json_path = results_dir / 'modal_properties.json'

    if not pushover_csv_path or not pushover_csv_path.exists():
        print(f"Error: Pushover curve CSV not found in {results_dir}")
        return
    if not modal_json_path.exists():
        print(f"Error: modal_properties.json not found. Run analysis first.")
        return

    # 2. 데이터 로드
    try:
        df_curve = pd.read_csv(pushover_csv_path)
        with open(modal_json_path, 'r') as f:
            modal_data = json.load(f)
        
        dominant_mode = modal_data['dominant_mode']
        csm_modal_props = {
            'pf1': dominant_mode[f'gamma_{direction.lower()}'],
            'm_eff_t1': dominant_mode[f'M_star_{direction.lower()}'],
            'phi_roof': dominant_mode[f'phi_{direction.lower()}'][-1]
        }
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error reading data: {e}")
        return

    # 3. 설정 파일 내용을 기반으로 파라미터 리스트 구성 및 내진설계범주 결정
    site_params = config['site_parameters']
    Z_factor = site_params['Z'] # 지진구역계수
    site_class = site_params['site_class']
    importance_class = site_params.get('importance_class', 'I') # 기본값 I

    # 2400년 재현주기 기준 S값 (MCE) 산정 (I=2.0)
    S_MCE = Z_factor * 2.0
    
    # 기본설계지진 S값 (DBE) 산정 (MCE * 2/3)
    S_DBE = S_MCE * (2.0 / 3.0)

    # 내진설계범주 결정
    SDS, SD1 = calculate_design_acceleration(S_DBE, site_class)
    design_category = determine_seismic_design_category(SDS, SD1, importance_class)

    print(f"\n[Site & Seismic Design Parameters]")
    print(f" - Site Class: {site_class}")
    print(f" - Seismic Zone Factor (Z): {Z_factor}g")
    print(f" - S_MCE (2400yr): {S_MCE:.4f}g")
    print(f" - S_DBE (Basic Design): {S_DBE:.4f}g")
    print(f" - Importance Class: {importance_class}")
    print(f" - Calculated SDS: {SDS:.4f}g, SD1: {SD1:.4f}g")
    print(f" - Seismic Design Category (SDC): {design_category}")

    # --- 자동 성능 목표 설정 ---
    try:
        performance_objectives = get_performance_objectives(importance_class)
        print(f"\n[Performance Objectives for Class '{importance_class}']")
        for obj in performance_objectives:
            print(f" - {obj['description']} (Drift Limit: {obj['target_drift_ratio_limit']*100}%)")
    except ValueError as e:
        print(f"Error setting performance objectives: {e}")
        return

    design_params_list = []
    for obj in performance_objectives:
        method = obj['method']
        params = {
            'site_class': site_class,
            'objective_name': obj['name'],
            'target_drift_ratio': obj['target_drift_ratio_limit'],
            'method': method
        }
        
        if method == 'scaling':
            params['S_DBE'] = S_DBE
            params['I_E'] = obj['factor'] 
        else: 
            params['S'] = Z_factor * obj['factor']
            
        design_params_list.append(params)

    # 4. CSM 계산 실행
    results = calculate_performance_point_csm(df_curve, csm_modal_props, design_params_list)

    if not results:
        print("Calculation failed.")
        return

    # 5. 결과 리포트 및 애니메이션 생성
    total_height = 3.5 * 4 # (참고: 실제 높이는 모델 데이터에서 가져오는 것이 좋으나 현재는 고정값 사용) 
    
    print(f"\n{'[Evaluation Results]':^70}")
    print(f"{'-'*70}")
    print(f"{'Objective':<30} | {'Sa(g)':<7} | {'Sd(m)':<8} | {'Drift(%)':<8} | {'Limit(%)':<8} | {'Result'}")
    print(f"{'-'*70}")

    for res in results:
        obj_name = res['objective_name']
        pp = res['performance_point']
        
        # 변위 계산
        roof_disp_actual = pp['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
        drift_ratio = (abs(roof_disp_actual) / total_height) * 100 # % 단위
        
        # 기준 찾기
        target_limit = next(p['target_drift_ratio'] for p in design_params_list if p['objective_name'] == obj_name) * 100
        
        status = "PASS" if drift_ratio <= target_limit else "FAIL"
        
        print(f"{obj_name:<30} | {pp['Sa']:<7.4f} | {pp['Sd']:<8.4f} | {drift_ratio:<8.3f} | {target_limit:<8.3f} | {status}")

        # 애니메이션 생성
        # 파일명에 성능목표 이름 포함 (공백은 언더스코어로 변경)
        safe_obj_name = obj_name.replace(" ", "_").replace("(", "").replace(")", "")
        animation_filename = f"CSM_{safe_obj_name}_{direction}.gif"
        
        # 기존 animate_csm_process는 파일명을 직접 생성하므로, 
        # 여기서는 결과 딕셔너리에 파일명 힌트를 추가하거나 함수를 수정해야 함.
        # 일단 현재 animate_csm_process는 direction만 받아서 파일명을 자동 생성함.
        # 다중 목표를 위해 animate_csm.py를 수정하거나, 임시로 호출함.
        # 여기서는 animate_csm_process를 수정하여 'filename_suffix'를 받을 수 있게 하고 호출함.
        
        animate_csm_process(res, results_dir, direction, filename_suffix=safe_obj_name)

    print(f"{'-'*70}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Seismic Performance Evaluation based on Config.")
    parser.add_argument('--config', type=str, default='scripts/seismic_design_config.json', help='Path to configuration JSON file')
    parser.add_argument('--dir', type=str, help='Target results directory (optional)')
    
    args = parser.parse_args()
    
    config_path = Path(project_root) / args.config
    config_data = load_config(config_path)
    
    if args.dir:
        target_dir = Path(args.dir)
        if target_dir.exists():
            run_evaluation(target_dir, config_data)
        else:
            print(f"Directory not found: {args.dir}")
    else:
        default_dirs = [
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X',
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z'
        ]
        for d in default_dirs:
            if d.exists():
                run_evaluation(d, config_data)
            else:
                print(f"Default directory not found: {d}")