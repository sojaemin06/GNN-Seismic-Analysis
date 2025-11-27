# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
from pathlib import Path
import argparse

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈 임포트 ---
from src.core.capacity_spectrum import calculate_performance_point_csm
from src.visualization.animate_csm import animate_csm_process
from src.core.post_processor import find_first_material_failure

def calculate_from_files(results_dir: Path):
    """
    지정된 결과 디렉토리의 파일들을 읽어 성능점을 계산합니다.
    - ..._pushover_curve_[X/Z].csv
    - modal_properties.json
    """
    print(f"\n--- Calculating Performance Point from files in: {results_dir.name} ---")

    # 1. 파일 경로 찾기
    direction = 'X' if '_X' in results_dir.name else 'Z'
    
    pushover_csv_path = next(results_dir.glob(f'*_pushover_curve_{direction}.csv'), None)
    modal_json_path = results_dir / 'modal_properties.json'

    if not pushover_csv_path or not pushover_csv_path.exists():
        print(f"Error: Pushover curve CSV not found in {results_dir}")
        return
    if not modal_json_path.exists():
        print(f"Error: modal_properties.json not found in {results_dir}")
        print("Please run 'run_single_analysis.py' once to generate the required file.")
        return

    # 2. 데이터 파일 읽기
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
        print("Successfully loaded pushover curve and modal properties.")
    except Exception as e:
        print(f"Error reading data files: {e}")
        return

    # 3. 설계 응답 스펙트럼 파라미터 정의 (KDS 2022 기준, 여러 성능 목표)
    S_MCE_value = 0.135 # 충남 지역의 2400년 재현주기 유효지반가속도 (맵에서 추출)
    site_class_value = 'S2' # 얕고 단단한 지반
    
    design_params_list = [
        {
            'S_MCE': S_MCE_value,
            'site_class': site_class_value,
            'S_factor': 1.0,
            'objective_name': '2400-year Collapse Prevention (CP)',
            'target_drift_ratio': 0.03 # 붕괴방지 목표 층간변위비 (3%)
        },
        {
            'S_MCE': S_MCE_value,
            'site_class': site_class_value,
            'S_factor': 0.8, # 1400년 재현주기 지진력은 2400년의 0.8배
            'objective_name': '1400-year Life Safety (LS)',
            'target_drift_ratio': 0.02 # 인명보호 목표 층간변위비 (2%)
        }
    ]

    print("\nUsing Design Spectrum Parameters (KDS 2022) for multiple objectives:")
    print(f"   - Effective Ground Acceleration (S_MCE): {S_MCE_value}g")
    print(f"   - Site Class: {site_class_value}")

    # 4. 역량스펙트럼법(CSM)으로 성능점 계산 (여러 성능 목표에 대해)
    all_csm_results = calculate_performance_point_csm(df_curve, csm_modal_props, design_params_list)

    if not all_csm_results:
        print("Failed to calculate performance points.")
        return

    # 5. 각 성능 목표에 대한 결과 출력 및 애니메이션 생성
    # 건물 전체 높이 (층간변위비 계산용)
    total_height = 3.5 * 4 # 4층 * 3.5m/층 (run_single_analysis.py에서 설정된 값)

    for csm_results in all_csm_results:
        objective_name = csm_results['objective_name']
        pp = csm_results['performance_point']
        
        print(f"\n--- Performance Point Results for: {objective_name} ---")
        print(f"  - Spectral Displacement (Sd): {pp['Sd']:.4f} m")
        print(f"  - Spectral Acceleration (Sa): {pp['Sa']:.4f} g")
        print(f"  - Effective Period (T_eff): {pp['T_eff']:.3f} s")
        print(f"  - Effective Damping (beta_eff): {pp['beta_eff']:.3f} ({pp['beta_eff']*100:.1f}%)")
        print(f"  - SDS: {csm_results['SDS']:.4f}g, SD1: {csm_results['SD1']:.4f}g")
        
        # 성능 검토
        # 1. 층간변위 검토
        # Sd를 실제 지붕 변위로 변환
        roof_disp_actual = pp['Sd'] * csm_modal_props['pf1'] * csm_modal_props['phi_roof']
        # 평균 층간변위비 (근사적으로 전체 변위 / 전체 높이)
        avg_drift_ratio = abs(roof_disp_actual) / total_height
        target_drift_ratio = next(p['target_drift_ratio'] for p in design_params_list if p['objective_name'] == objective_name)
        
        drift_check = "OK" if avg_drift_ratio <= target_drift_ratio else "FAIL"
        print(f"  - Roof Displacement (Actual): {roof_disp_actual:.4f} m")
        print(f"  - Approx. Drift Ratio: {avg_drift_ratio:.4f} (Limit: {target_drift_ratio}) -> {drift_check}")

        # 2. 붕괴 부재 검토 (find_first_material_failure 활용)
        # 성능점의 시간(Pseudo_Time)을 찾아서 그 시점 이전에 파괴가 있었는지 확인해야 함
        # 여기서는 간단히 성능점 변위에서의 파괴 여부를 확인
        # (실제로는 find_first_material_failure 함수를 호출하여 전체 이력을 확인해야 함)
        # 모듈화된 함수 사용을 위해 model_info 등 필요한 정보를 가져와야 하는데, 
        # 현재 스크립트에서는 결과 파일만 읽으므로 제한적임. 
        # 따라서 여기서는 결과 파일에 포함된 변형률 데이터를 읽어 간접적으로 확인하는 로직을 추가할 수 있으나,
        # 파일 구조상 복잡하므로, 일단 변위 기반 검토만 수행하고 붕괴 부재 없음은 가정하거나 
        # 별도 상세 해석이 필요함을 명시함.
        print("  - Member Failure Check: Requires detailed fiber strain data analysis (not loaded here).")
        
        print("---------------------------------")
        
        # 6. CSM 과정 애니메이션 생성
        # 애니메이션 파일명 구분
        suffix = "CP" if "Collapse" in objective_name else "LS"
        csm_results['suffix'] = suffix # 애니메이션 함수에서 파일명에 사용
        # animate_csm_process(csm_results, results_dir, direction) # Call disabled to avoid index error for now


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate Performance Point from existing analysis results.")
    parser.add_argument(
        '--dir',
        type=str,
        help="Path to the results directory for a specific analysis run.",
        default=None
    )
    args = parser.parse_args()

    if args.dir:
        results_directory = Path(args.dir)
        if results_directory.exists() and results_directory.is_dir():
            calculate_from_files(results_directory)
        else:
            print(f"Error: Provided directory does not exist or is not a directory: {args.dir}")
    else:
        # 기본 예시 디렉토리 사용
        print("No directory provided. Using default example directories...")
        default_dir_x = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X'
        default_dir_z = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z'
        
        if default_dir_x.exists():
            calculate_from_files(default_dir_x)
        else:
            print(f"\nDefault directory not found: {default_dir_x}")
            
        if default_dir_z.exists():
            calculate_from_files(default_dir_z)
        else:
            print(f"\nDefault directory not found: {default_dir_z}")