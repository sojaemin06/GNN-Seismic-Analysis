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
    print(f"{ '='*60}")

    # 디렉토리 이름에서 방향과 부호 추론 (예: Run_Single_..._X_pos)
    dir_name = results_dir.name
    if '_X_pos' in dir_name:
        direction, sign, sign_str = 'X', '+', 'pos'
    elif '_X_neg' in dir_name:
        direction, sign, sign_str = 'X', '-', 'neg'
    elif '_Z_pos' in dir_name:
        direction, sign, sign_str = 'Z', '+', 'pos'
    elif '_Z_neg' in dir_name:
        direction, sign, sign_str = 'Z', '-', 'neg'
    else:
        # 이전 버전 호환성 (X, Z만 있는 경우)
        if '_X' in dir_name: direction, sign, sign_str = 'X', '+', 'pos'
        elif '_Z' in dir_name: direction, sign, sign_str = 'Z', '+', 'pos'
        else:
            print(f"Skipping directory {dir_name}: Unknown direction/sign format")
            return

    # 1. 데이터 파일 경로 찾기
    # run_single_analysis.py에서 파일명에 direction만 붙였는지, _pos/_neg까지 붙였는지 확인 필요
    # 현재 run_single_analysis.py는 params['analysis_name']에 _X_pos 등을 붙이므로 파일명에도 반영됨
    
    # 파일명 패턴: *pushover_curve_{direction}.csv 또는 *pushover_curve_{direction}_{sign_str}.csv
    # run_single_analysis.py의 process_pushover_results에서 저장할 때 
    # f"{analysis_name}_pushover_curve_{direction}.csv" 로 저장함.
    # analysis_name 자체가 "Run_Single_..._X_pos" 이므로
    # 최종 파일명은 "Run_Single_..._X_pos_pushover_curve_X.csv" 형태가 됨.
    
    pushover_csv_path = next(results_dir.glob(f'*pushover_curve_{direction}.csv'), None)
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
        print(f"Data loaded successfully. (Direction: {direction}, Sign: {sign})")
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
            'repetition_period': obj['repetition_period'],
            'description': obj['description'],
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
    
    # [NEW] 회전각 이력 데이터 로드 (Step별 힌지 상태 확인용)
    # 주의: 파일명이 run_single_analysis에서 생성된 규칙을 따라야 함.
    # *all_col_plastic_rotation_{direction}.out
    col_rot_path = next(results_dir.glob(f'*all_col_plastic_rotation_{direction}.out'), None)
    beam_rot_path = next(results_dir.glob(f'*all_beam_plastic_rotation_{direction}.out'), None)
    
    col_rot_data, beam_rot_data = None, None
    if col_rot_path and col_rot_path.exists():
        try:
            col_rot_data = np.loadtxt(col_rot_path)
            if col_rot_data.ndim == 1: col_rot_data = col_rot_data.reshape(1, -1)
        except: pass
    if beam_rot_path and beam_rot_path.exists():
        try:
            beam_rot_data = np.loadtxt(beam_rot_path)
            if beam_rot_data.ndim == 1: beam_rot_data = beam_rot_data.reshape(1, -1)
        except: pass

    # 회전각 한계 (animate_results.py와 동일하게 적용)
    # TODO: KDS 41 17 00 표 5.4.6 ~ 5.4.8을 참조하여 부재별 동적 허용기준 적용 고려 필요
    ROT_CP = 0.04 # Collapse Prevention Limit (rad)
    
    # --- CSM 평가 요약 데이터 추출 및 저장 ---
    summary_results = []
    total_height = 3.5 * 4 # 임시 (추후 모델 데이터에서 가져오도록 수정)
    
    # pushover_curve의 Roof_Displacement_m과 Step 매핑을 위해
    # df_curve는 이미 로드됨 ('Roof_Displacement_m', 'Base_Shear_N', 'Pseudo_Time')
    # .out 파일의 첫 번째 컬럼은 Time (Pseudo Time)
    
    max_base_shear = df_curve['Base_Shear_N'].abs().max()

    for res in results:
        obj_name = res['objective_name']
        pp = res['performance_point']
        design_params = next(dp for dp in design_params_list if dp['objective_name'] == obj_name)
        
        # 변위 계산
        roof_disp_actual = pp['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
        drift_ratio = (abs(roof_disp_actual) / total_height) * 100 # % 단위
        
        status = "PASS" if drift_ratio <= (design_params['target_drift_ratio'] * 100) else "FAIL"
        
        # [NEW] 1. 중력하중 저항능력 평가 (Stability Check)
        # 성능점에서의 전단력이 최대 강도의 80% 미만으로 떨어졌는지 확인
        # (간접적인 중력 저항 능력 상실 또는 불안정성 척도)
        
        # 푸쉬오버 곡선에서 최대 전단력 발생 시의 변위 찾기
        abs_roof_disp = df_curve['Roof_Displacement_m'].abs()
        abs_base_shear = df_curve['Base_Shear_N'].abs()
        
        peak_idx = abs_base_shear.idxmax()
        disp_at_peak = abs_roof_disp.loc[peak_idx]
        
        # 성능점 지붕 변위에서의 베이스 전단력을 푸쉬오버 곡선에서 보간하여 가져옴
        current_base_shear = np.interp(abs(roof_disp_actual), abs_roof_disp, abs_base_shear)
        
        # 실제 전단력 비율 계산
        actual_shear_ratio = current_base_shear / max_base_shear if max_base_shear > 0 else 0
        
        # Case A: 성능점이 최대 강도 도달 전 (pre-peak)
        if abs(roof_disp_actual) < disp_at_peak:
            # 최대 강도 도달 전에는 내력 저하가 없으므로 안정성 비율을 1.0으로 간주
            stability_ratio = 1.0
            stability_status = "OK" 
            status = "PASS" if drift_ratio <= (design_params['target_drift_ratio'] * 100) else "FAIL" 
        # Case B: 성능점이 최대 강도 도달 후 (post-peak)
        else:
            stability_ratio = actual_shear_ratio
            if stability_ratio < 0.8:
                 stability_status = "FAIL (Instability)"
                 status = "FAIL (Instability)"
            else:
                 stability_status = "OK"
                 status = "PASS" if drift_ratio <= (design_params['target_drift_ratio'] * 100) else "FAIL"
        # else는 필요 없음. Instability가 아니면 다음 붕괴 부재 체크로 넘어감.

        # [NEW] 2. 붕괴 부재 판별 (Collapsed Members)
        collapsed_count = 0
        collapsed_members = [] 
        
        # 성능점 시점(Time) 찾기
        target_time = np.interp(abs(roof_disp_actual), df_curve['Roof_Displacement_m'].abs(), df_curve['Pseudo_Time'])
        
        # 해당 Time에 가장 가까운 스텝 인덱스 찾기
        step_idx = 0 
        if col_rot_data is not None and len(col_rot_data) > 0: # 데이터가 있고 비어있지 않은지 확인
            times = col_rot_data[:, 0]
            step_idx = (np.abs(times - target_time)).argmin()
            
            # 해당 스텝의 회전각 확인 (col_rot_data는 1열이 time, 나머지가 data)
            # data는 각 IP당 eps, kz, ky 3개. 5개 IP 가정 (총 15개 데이터)
            # col_rot_data의 형태: [time, ele1_ip1_eps, ele1_ip1_kz, ele1_ip1_ky, ..., eleN_ip5_ky]
            
            if col_rot_data.shape[1] > 1: # time 컬럼 외 데이터가 있는지 확인
                all_vals = col_rot_data[step_idx, 1:]
                # 3개씩 묶음 (eps, kz, ky)
                # 각 요소의 적분점 수 = 5개 (model_builder의 num_int_pts)
                # 따라서 각 요소당 데이터는 5 * 3 = 15개
                # 전체 요소 수 = (all_vals의 길이) / 15
                num_elements = int(len(all_vals) / (5*3)) # 각 요소에 5개의 적분점과 3가지 데이터 (eps, kz, ky)
                
                # 각 요소별로 가장 큰 회전각을 찾아서 확인
                for i in range(num_elements):
                    element_data = all_vals[i * 15 : (i+1) * 15] # 15개 데이터 (5개 IP * 3)
                    max_theta_element = 0.0
                    for ip_idx in range(5): # 5개 적분점
                        kz = element_data[ip_idx * 3 + 1] # kz
                        ky = element_data[ip_idx * 3 + 2] # ky
                        theta_ip = (abs(kz) + abs(ky)) * 0.5 # Lp=0.5m 가정
                        if theta_ip > max_theta_element:
                            max_theta_element = theta_ip
                    
                    if max_theta_element > ROT_CP:
                        collapsed_count += 1
        
        # 빔에 대해서도 동일 수행
        if beam_rot_data is not None and len(beam_rot_data) > 0: # 데이터가 있고 비어있지 않은지 확인
             times_b = beam_rot_data[:, 0]
             # step_idx가 col_rot_data가 없을 때 0으로 초기화되므로,
             # beam_rot_data의 times_b 길이보다 크지 않도록 다시 체크
             if step_idx < len(times_b) and beam_rot_data.shape[1] > 1: # 인덱스 유효성 체크 및 데이터 확인
                 all_vals_b = beam_rot_data[step_idx, 1:]
                 num_elements_b = int(len(all_vals_b) / (5*3))
                 
                 for i in range(num_elements_b):
                    element_data_b = all_vals_b[i * 15 : (i+1) * 15]
                    max_theta_element_b = 0.0
                    for ip_idx in range(5):
                        kz_b = element_data_b[ip_idx * 3 + 1]
                        ky_b = element_data_b[ip_idx * 3 + 2]
                        theta_ip_b = (abs(kz_b) + abs(ky_b)) * 0.5
                        if theta_ip_b > max_theta_element_b:
                            max_theta_element_b = theta_ip_b
                    
                    if max_theta_element_b > ROT_CP:
                        collapsed_count += 1
        
        # [UPDATE] 붕괴 부재 발생 시 상태 업데이트
        # 안정성 실패(Instability)가 가장 심각하므로 그게 아닐 때만 체크
        if status != "FAIL (Instability)" and collapsed_count > 0: # 안정성 실패가 아니라면 붕괴 부재 여부로 최종 판정
            status = f"FAIL ({collapsed_count} Members Collapsed)"

        summary_results.append({
            "objective_name": obj_name,
            "repetition_period": design_params['repetition_period'],
            "direction": direction,
            "sign": sign, # 부호 정보 추가
            "perf_point_Sd_m": float(f"{pp['Sd']:.4f}"),
            "perf_point_Sa_g": float(f"{pp['Sa']:.4f}"),
            "effective_period_s": float(f"{pp['T_eff']:.3f}"),
            "effective_damping_pct": float(f"{pp['beta_eff']*100:.1f}"),
            "calculated_drift_pct": float(f"{drift_ratio:.3f}"),
            "allowed_drift_pct": float(f"{design_params['target_drift_ratio']*100:.3f}"),
            "status": status,
            "stability_ratio": float(f"{stability_ratio:.2f}"),
            "collapsed_member_count": collapsed_count,
            "description": design_params['description']
        })

    # Summary 결과를 JSON 파일로 저장 (부호 포함 파일명)
    summary_json_path = results_dir / f"csm_evaluation_summary_{direction}_{sign_str}.json"
    with open(summary_json_path, 'w', encoding='utf-8') as f:
        json.dump(convert_numpy_to_native(summary_results), f, indent=4, ensure_ascii=False)
    print(f"CSM evaluation summary saved to: {summary_json_path}")


    # 5. 결과 리포트 및 애니메이션 생성
    
    print(f"\n[{'[Evaluation Results]':^70}]")
    print(f"{'-'*70}")
    print(f"{ 'Objective':<30} | {'Sa(g)':<7} | {'Sd(m)':<8} | {'Drift(%)':<8} | {'Limit(%)':<8} | {'Result':<8}")
    print(f"{'-'*70}")

    for i, res in enumerate(results): 
        obj_name = res['objective_name']
        pp = res['performance_point']
        
        summary = summary_results[i]
        
        print(f"{obj_name:<30} | {summary['perf_point_Sa_g']:<7.4f} | {summary['perf_point_Sd_m']:<8.4f} | {summary['calculated_drift_pct']:<8.3f} | {summary['allowed_drift_pct']:<8.3f} | {summary['status']}")

        # 애니메이션 생성 (파일명에 부호 포함)
        safe_obj_name = obj_name.replace(" ", "_").replace("(", "").replace(")", "")
        # animate_csm_process는 filename_suffix만 받으므로, 여기서 구분자를 잘 넣어줘야 함
        suffix = f"{sign_str}_{safe_obj_name}"
        animate_csm_process(res, results_dir, direction, filename_suffix=suffix)

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
        # [수정] 기본 디렉토리 목록을 4가지 케이스로 확장
        default_dirs = [
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X_pos',
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X_neg',
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z_pos',
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z_neg'
        ]
        
        # 호환성을 위해 기존 폴더가 있다면 추가 (선택 사항)
        legacy_dirs = [
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X',
            Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z'
        ]
        
        all_dirs = default_dirs + legacy_dirs
        
        for d in all_dirs:
            if d.exists():
                run_evaluation(d, config_data)
            # else:
            #     print(f"Default directory not found: {d}")
