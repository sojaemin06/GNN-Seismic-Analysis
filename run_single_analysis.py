import openseespy.opensees as ops
import math
import numpy as np
import pandas as pd
import sys
import os 
from pathlib import Path 

from core.model_builder import build_model
from core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from core.post_processor import process_pushover_results, calculate_performance_points
from core.verification import verify_nsp_applicability

def main(params, directions=['X', 'Z']):
    """
    [수정] 전체 해석 파이프라인을 실행하며, 여러 방향에 대한 해석을 지원합니다.
    """
    # 1. 모델 구축
    model_nodes_info = build_model(params)
    if not isinstance(model_nodes_info, dict):
        print(f"Error: Model building failed for {params['analysis_name']}.")
        ops.wipe()
        return None, None

    # 2. 중력 해석
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. Pushover 해석을 중단합니다.")
        ops.wipe()
        return model_nodes_info, None

    # 3. 고유치 해석 및 검증
    ok, mpr_x, mpr_z = run_eigen_analysis(params, model_nodes_info)
    if not ok or mpr_x < 0.9 or mpr_z < 0.9:
        print(f"\n고유치 해석 실패 또는 질량 참여율 부족. (X: {mpr_x*100:.1f}%, Z: {mpr_z*100:.1f}%).")
        ops.wipe()
        return model_nodes_info, None

    is_nsp_valid_x, is_nsp_valid_y = verify_nsp_applicability(params, model_nodes_info)
    if not (is_nsp_valid_x and is_nsp_valid_y):
        print("\n[Warning] 1차 모드 지배 조건을 만족하지 않음.")
        ops.wipe()
        return model_nodes_info, None

    # 4. 각 방향에 대한 푸쉬오버 해석 루프
    results = {}
    for direction in directions:
        # 푸쉬오버 해석은 각 방향에 대해 독립적인 상태에서 시작해야 함
        # 중력 해석 상태를 복원하기 위해 constant-gravity load pattern을 다시 적용
        ops.loadConst('-time', 0.0)

        if not run_pushover_analysis(params, model_nodes_info, direction=direction):
            print(f"\n푸쉬오버 해석 실패 (Direction: {direction}).")
            # 한 방향이라도 실패하면 전체를 실패로 간주
            ops.wipe()
            return model_nodes_info, None

        df_curve, _, _, _ = process_pushover_results(params, model_nodes_info, direction=direction)
        
        if df_curve is None or df_curve.empty or len(df_curve) < 2:
            print(f"\n결과 처리 실패 (Direction: {direction}).")
            ops.wipe()
            return model_nodes_info, None

        perf_points = calculate_performance_points(df_curve)
        results[direction] = (perf_points, df_curve)

    ops.wipe()
    
    return model_nodes_info, results


if __name__ == '__main__':
    
    analysis_params = {
        'analysis_name': 'Biaxial_Test', 
        'output_dir': Path('debug_output/biaxial_test'), 
        'target_drift': 0.03, 'num_steps': 500, 'num_modes': 20, 'num_int_pts': 5,
        'num_bays_x': 2, 'num_bays_z': 2, 'num_stories': 3,     
        'bay_width_x': 5.0, 'bay_width_z': 5.0, 'story_height': 3.5,
        'build_core': False, 'skip_post_processing': True,
    }
        
    common_params = {
        'fc': -30e6, 'Fy': 500e6, 'E_steel': 200e9, 'g': 9.81,
        'dead_load_pa': 5000.0, 'live_load_pa': 0.0
    }
    
    # 단일 속성으로 테스트
    col_props = {'dims': (0.8, 0.8), 'fc': -30e6, 'Fy': 500e6, 'cover': 0.04, 'rebar_Area': 0.00049, 'num_bars_x': 6, 'num_bars_z': 6}
    beam_props = {'dims': (0.4, 0.6), 'fc': -30e6, 'Fy': 500e6, 'cover': 0.04, 'rebar_Area': 0.00049, 'num_bars_top': 6, 'num_bars_bot': 6}
    
    # Tapering을 위해 리스트로 전달
    parameters = {**analysis_params, **common_params, 'col_props_list': [col_props]*3, 'beam_props': beam_props}
    
    print("\n--- Running Biaxial analysis for stability test ---")
    model_info, results = main(parameters, directions=['X', 'Z'])
    
    if results:
        print("\nBiaxial analysis successful.")
        print("X-dir peak shear:", results['X'][0]['peak_shear'])
        print("Z-dir peak shear:", results['Z'][0]['peak_shear'])
    else:
        print("\nBiaxial analysis failed.")
