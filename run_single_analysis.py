import openseespy.opensees as ops
import math
import numpy as np
import pandas as pd
import sys
import os 
from pathlib import Path 

# --- 모듈화된 함수 임포트 ---
from core.model_builder import build_model
from core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from core.post_processor import process_pushover_results, calculate_performance_points

from visualization.plot_matplotlib import plot_model_matplotlib
from visualization.plot_opsvis import plot_with_opsvis
from visualization.plot_hinges import plot_plastic_damage_distribution
from visualization.animate_results import animate_and_plot_pushover, animate_and_plot_M_phi
# --- ---
# 가나다라
# ### 11. 메인 실행 함수 ###
def main(params):
    """
    전체 해석 파이프라인을 실행합니다.
    [데이터셋 생성용 수정] 
    1. 'skip_post_processing' 파라미터 추가
    2. 'perf_points'를 반환
    """
    # 0. 출력 디렉토리 생성
    try:
        params['output_dir'].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {params['output_dir']}: {e}")
        return None # 디렉토리 생성 실패 시 None 반환

    # 1. 모델 구축 (모든 기하정보가 포함된 model_nodes_info 반환)
    model_nodes_info = build_model(params)
    
    skip_plots = params.get('skip_post_processing', False)

    # 2. Matplotlib 3D/2D 플롯 (전단벽 포함)
    if not skip_plots:
        plot_model_matplotlib(params, model_nodes_info)
    
    # 2.5. opsvis 시각화 (Wireframe + Fiber Sections)
    if not skip_plots:
        plot_with_opsvis(params)
    
    # 3. 중력 해석
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. Pushover 해석을 중단합니다.")
        ops.wipe() # 실패 시 wipe 추가
        return None # None 반환

    # 4. 고유치 해석
    if not run_eigen_analysis(params):
        print("\n고유치 해석 실패. Pushover 해석을 중단합니다.")
        ops.wipe()
        return None # None 반환

    # 5. 푸쉬오버 해석 실행
    if not run_pushover_analysis(params, model_nodes_info):
        print("\n푸쉬오버 해석 실행 중 오류 발생.")
        ops.wipe()
        pass 

    # 6. 푸쉬오버 결과 처리 (4개 값 반환)
    df_curve, df_disp, final_states_dfs, df_m_phi = process_pushover_results(params, model_nodes_info)
    
    if df_curve is None or df_disp is None or df_curve.empty or len(df_curve) < 2:
        print("\n결과 파일 처리 실패 또는 데이터 부족. 후처리를 건너뜁니다.")
        ops.wipe()
        return None # None 반환

    # 7. 성능점 계산
    perf_points = calculate_performance_points(df_curve)
    
    # 8. 소성/손상 분포도 플로팅
    if not skip_plots:
        plot_plastic_damage_distribution(params, model_nodes_info, final_states_dfs)

    # 9. 애니메이션 및 플롯 생성
    if not skip_plots:
        animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs)
    
    # 10. M-Phi 애니메이션 플롯 생성
    if not skip_plots and df_m_phi is not None:
        animate_and_plot_M_phi(df_m_phi, params)
    
    print("\n단일 해석 및 후처리 완료.")
    
    ops.wipe() # 마지막에 wipe 추가
    
    return perf_points # 계산된 성능점 반환


# ### 12. 메인 실행부 (Driver) ###
if __name__ == '__main__':
    
    # --- 1. 빠른 테스트 / 상세 해석 설정 ---
    FAST_TEST_CONFIG = True 
    BUILD_CORE_SWITCH = False 
    
    if FAST_TEST_CONFIG:
        print("--- [!] 빠른 테스트 모드 (Fast Test Mode) 활성화 ---")
        analysis_params = {
            'analysis_name': 'Run_Fast_Test_3x3_Core', 
            'output_dir': Path('results/Run_Fast_Test_3x3_Core'), 
            'target_drift': 0.04, 
            'num_steps': 500,     
            'num_modes': 3,       
            'num_int_pts': 2,     
            'plot_z_line_index': 1, 
            
            # Geometry
            'num_bays_x': 3,      
            'num_bays_z': 3,      
            'num_stories': 6,     
            'bay_width_x': 6.0, 
            'bay_width_z': 6.0, 
            'story_height': 3.0,
            
            'build_core': BUILD_CORE_SWITCH,
            
            'core_z_start_bay_idx': 1, 
            'core_x_start_bay_idx': 1, 
            'num_core_bays_z': 1,      
            'num_core_bays_x': 1,
            
            # [수정] 데이터셋 생성이 아닌 단일 실행이므로 플로팅 활성화
            'skip_post_processing': False,
        }
    else:
        # ... (상세 해석용 설정) ...
        print("--- [!] 상세 해석 모드 (Full Analysis Mode) 활성화 ---")
        analysis_params = {
            # ... (기존과 동일) ...
            'skip_post_processing': False,
        }
        
    if not BUILD_CORE_SWITCH:
        analysis_params['analysis_name'] = analysis_params['analysis_name'].replace('_Core', '_NoCore')
        analysis_params['output_dir'] = Path(str(analysis_params['output_dir']).replace('_Core', '_NoCore'))


    # --- 2. 재료/단면/하중 공통 파라미터 ---
    common_params = {
        'fc': -30e6,      # Pa
        'Fy': 400e6,      # Pa
        'E_steel': 200e9, # Pa
        'col_dims': (0.4, 0.4),
        'beam_dims': (0.3, 0.5),
        'cover': 0.04,
        'rebar_Area': 0.00049,
        'wall_thickness': 0.20,
        'wall_reinf_ratio': 0.003,
        'g': 9.81,
        'dead_load_pa': 5000.0, 
        'live_load_pa': 2000.0
    }

    # --- 3. 파라미터 결합 및 메인 함수 실행 ---
    parameters = {**analysis_params, **common_params}
    
    # main 함수를 직접 호출
    main(parameters)