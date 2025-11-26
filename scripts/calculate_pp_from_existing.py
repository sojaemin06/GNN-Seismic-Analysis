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

    # 3. 설계 응답 스펙트럼 파라미터 정의 (KDS 2022 기준)
    design_params = {
        'S': 0.16,  # 유효지반가속도 (예: 서울)
        'site_class': 'S4',
    }
    print("\nUsing Design Spectrum Parameters (KDS 2022):")
    print(f"   - Effective Ground Acceleration (S): {design_params['S']}g")
    print(f"   - Site Class: {design_params['site_class']}")

    # 4. 역량스펙트럼법(CSM)으로 성능점 계산
    csm_results = calculate_performance_point_csm(df_curve, csm_modal_props, design_params)

    if not csm_results:
        print("Failed to calculate performance point.")
        return

    # 5. 결과 출력
    pp = csm_results['performance_point']
    print("\n--- Performance Point Results ---")
    print(f"  - Spectral Displacement (Sd): {pp['Sd']:.4f} m")
    print(f"  - Spectral Acceleration (Sa): {pp['Sa']:.4f} g")
    print(f"  - Effective Period (T_eff): {pp['T_eff']:.3f} s")
    print(f"  - Effective Damping (beta_eff): {pp['beta_eff']:.3f} ({pp['beta_eff']*100:.1f}%)")
    print("---------------------------------")
    
    # 6. CSM 과정 애니메이션 생성
    animate_csm_process(csm_results, results_dir, direction)


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
