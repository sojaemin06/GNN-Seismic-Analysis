# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
import argparse
from pathlib import Path
import numpy as np

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

def convert_numpy_to_native(obj):
    """Numpy 데이터 타입을 Python 네이티브 타입으로 변환 (JSON 직렬화용)."""
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(elem) for elem in obj]
    return obj

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
            print(f" - {obj['description']} (Drift Limit: {obj['target_drift_ratio_limit']*100:.2f}%)")
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
            'repetition_period': obj['repetition_period'], # 추가
            'description': obj['description'], # 추가
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
    
    # --- CSM 평가 요약 데이터 추출 및 저장 ---
    summary_results = []
    total_height = 3.5 * 4 # 임시 (추후 모델 데이터에서 가져오도록 수정)
    
    for res in results:
        obj_name = res['objective_name']
        pp = res['performance_point']
        design_params = next(dp for dp in design_params_list if dp['objective_name'] == obj_name) # 해당 목표의 원본 파라미터 찾기
        
        # 변위 계산
        roof_disp_actual = pp['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
        drift_ratio = (abs(roof_disp_actual) / total_height) * 100 # % 단위
        
        status = "PASS" if drift_ratio <= (design_params['target_drift_ratio'] * 100) else "FAIL" # 타겟 변위는 %로 저장되어있으므로 100곱함
        
        summary_results.append({
            "objective_name": obj_name,
            "repetition_period": design_params['repetition_period'],
            "direction": direction,
            "perf_point_Sd_m": float(f"{pp['Sd']:.4f}"),
            "perf_point_Sa_g": float(f"{pp['Sa']:.4f}"),
            "effective_period_s": float(f"{pp['T_eff']:.3f}"),
            "effective_damping_pct": float(f"{pp['beta_eff']*100:.1f}"),
            "calculated_drift_pct": float(f"{drift_ratio:.3f}"),
            "allowed_drift_pct": float(f"{design_params['target_drift_ratio']*100:.3f}"),
            "status": status,
            "description": design_params['description'] # 성능 목표 설명 추가
        })

    # Summary 결과를 JSON 파일로 저장
    summary_json_path = results_dir / f"csm_evaluation_summary_{direction}.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_native(summary_results), f, indent=4, ensure_ascii=False)
    print(f"CSM evaluation summary saved to: {summary_json_path}")


    # 5. 결과 리포트 및 애니메이션 생성
    
    print(f"\n[{'[Evaluation Results]':^70}]")
    print(f"{'-'*70}")
    print(f"{'Objective':<30} | {'Sa(g)':<7} | {'Sd(m)':<8} | {'Drift(%)':<8} | {'Limit(%)':<8} | {'Result':<8}")
    print(f"{'-'*70}")

    for i, res in enumerate(results): # res는 calculate_performance_point_csm의 반환값
        obj_name = res['objective_name']
        pp = res['performance_point']
        
        # summary_results에서 해당 목표의 결과 가져오기 (이미 계산됨)
        summary = summary_results[i] # 순서가 같다고 가정
        
        print(f"{obj_name:<30} | {summary['perf_point_Sa_g']:<7.4f} | {summary['perf_point_Sd_m']:<8.4f} | {summary['calculated_drift_pct']:<8.3f} | {summary['allowed_drift_pct']:<8.3f} | {summary['status']}")

        # 애니메이션 생성
        safe_obj_name = obj_name.replace(" ", "_").replace("(", "").replace(")", "")
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
