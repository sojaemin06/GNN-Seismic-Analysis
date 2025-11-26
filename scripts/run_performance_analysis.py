# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
from pathlib import Path

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 기존 모듈 임포트 ---
# run_single_analysis의 함수들을 재사용합니다.
from scripts.run_single_analysis import get_single_run_parameters, main as run_single_main
from src.core.capacity_spectrum import calculate_performance_point_csm, create_design_spectrum_kbc2016

def run_performance_analysis(direction='X'):
    """
    단일 구조물에 대한 성능점 평가 전체 프로세스를 실행합니다.
    1. 단일 해석을 실행하여 푸쉬오버 곡선 및 모드 특성을 얻습니다.
    2. 특정 지역/조건에 대한 설계 응답 스펙트럼을 생성합니다.
    3. 역량스펙트럼법(CSM)을 사용하여 성능점을 계산합니다.
    4. 결과를 출력합니다.
    """
    print(f"\n--- [{direction}-Direction] Performance Point Analysis ---")

    # 1. 단일 실행용 파라미터 가져오기 및 해석 실행
    params = get_single_run_parameters()
    
    # 출력 디렉토리 이름에 방향 추가
    original_name = params['analysis_name']
    original_dir = params['output_dir']
    params['analysis_name'] = f"{original_name}_{direction}"
    params['output_dir'] = original_dir.parent / f"{original_dir.name}_{direction}"

    # --- 중요: 후처리(시각화 등)는 건너뛰고 결과만 받음 ---
    params['skip_post_processing'] = True
    
    # run_single_analysis.py의 main 함수를 호출하여 해석 결과 확보
    # main 함수는 (perf_points, model_nodes_info, df_curve, direction)를 반환
    # 이 중 model_nodes_info와 df_curve가 필요.
    # 하지만, modal_properties를 직접 얻으려면 main 함수를 분해해야 함.
    # 여기서는 간단히 하기 위해 run_single_analysis의 로직 일부를 다시 가져옴.
    
    # --- run_single_analysis.py의 main 함수 로직 재구성 ---
    from src.core.model_builder import build_model
    from src.core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
    from src.core.post_processor import process_pushover_results
    import openseespy.opensees as ops

    params['output_dir'].mkdir(parents=True, exist_ok=True)
    model_nodes_info = build_model(params)
    
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. 해석을 중단합니다."); ops.wipe(); return

    ok, modal_props = run_eigen_analysis(params, model_nodes_info)
    if not ok:
        print("\n고유치 해석 실패. 해석을 중단합니다."); ops.wipe(); return
    
    # 푸쉬오버 해석 실행
    ok, dominant_mode = run_pushover_analysis(params, model_nodes_info, modal_props, direction=direction)
    if not ok:
        print("\n푸쉬오버 해석 실행 중 오류 발생."); ops.wipe(); return
    
    # 푸쉬오버 결과 처리
    df_curve, _, _, _ = process_pushover_results(params, model_nodes_info, dominant_mode, direction=direction)
    ops.wipe()
    
    if df_curve is None or df_curve.empty:
        print("푸쉬오버 곡선 생성 실패."); return
    
    print("\n1. Pushover analysis completed.")
    print(f"   - Dominant Period (T1): {modal_props['T1']:.3f} s")
    print(f"   - Effective Modal Mass (M_eff): {modal_props['m_eff_t1'] / 1e3:.1f} tons")
    print(f"   - Participation Factor (PF1): {modal_props['pf1']:.3f}")

    # 2. 설계 응답 스펙트럼 정의 (예: 서울, S4 지반)
    # KBC 2016 기준, 지역계수(S) 0.11, 위험도계수(I) 1.0 -> 유효지반가속도 0.11g
    # 지반종류 S4에 대해 Sa=0.242, Sv=0.154 (근사값)
    design_spectrum_params = {
        'soil_type': 'S4',
        'Sa': 0.242, # 단주기 설계 스펙트럼 가속도
        'Sv': 0.154, # 1초 주기 설계 스펙트럼 가속도
    }
    print("\n2. Design spectrum (KBC 2016) defined:")
    print(f"   - Soil Type: {design_spectrum_params['soil_type']}")
    print(f"   - Sa (for spectrum shape): {design_spectrum_params['Sa']}g")
    print(f"   - Sv (for spectrum shape): {design_spectrum_params['Sv']}g")

    # 3. 역량스펙트럼법(CSM)으로 성능점 계산
    csm_results = calculate_performance_point_csm(df_curve, modal_props, design_spectrum_params)
    
    if not csm_results:
        print("성능점 계산 실패."); return
        
    print("\n3. Performance point calculation (CSM) completed.")

    # 4. 결과 출력
    pp = csm_results['performance_point']
    print("\n--- Performance Point Results ---")
    print(f"  - Spectral Displacement (Sd): {pp['Sd']:.4f} m")
    print(f"  - Spectral Acceleration (Sa): {pp['Sa']:.4f} g")
    print(f"  - Effective Period (T_eff): {pp['T_eff']:.3f} s")
    print(f"  - Effective Damping (beta_eff): {pp['beta_eff']:.3f} ({pp['beta_eff']*100:.1f}%)")
    print("---------------------------------")
    
    # To be used in the next step
    # from src.visualization.animate_csm import animate_csm_process
    # animate_csm_process(csm_results, params, direction)

if __name__ == '__main__':
    # X, Z 방향에 대해 순차적으로 해석 실행
    run_performance_analysis(direction='X')
    run_performance_analysis(direction='Z')
