import openseespy.opensees as ops
import math
import numpy as np
import pandas as pd
import sys
import os 
from pathlib import Path 

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈화된 함수 임포트 ---
from src.core.model_builder import build_model
from src.core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from src.core.post_processor import process_pushover_results, calculate_performance_points, find_first_material_failure
from src.core.verification import verify_nsp_applicability

from src.visualization.plot_matplotlib import plot_model_matplotlib
from src.visualization.plot_opsvis import plot_with_opsvis
from src.visualization.plot_hinges import plot_plastic_damage_distribution
from src.visualization.animate_results import animate_and_plot_pushover, animate_and_plot_M_phi
from src.visualization.plot_materials import plot_material_stress_strain
# --- ---

# ### 11. 메인 실행 함수 ###
def main(params, direction='X'):
    """
    전체 해석 파이프라인을 실행합니다.
    [데이터셋 생성용 수정] 
    1. 'skip_post_processing' 파라미터 추가
    2. 'perf_points'를 반환
    [신규] 'direction' 파라미터를 추가하여 해석 방향을 지정합니다.
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
        plot_model_matplotlib(params, model_nodes_info, direction)
    
    # 2.5. opsvis 시각화 (Wireframe + Fiber Sections)
    if not skip_plots:
        plot_with_opsvis(params)
        plot_material_stress_strain(params) # [신규] 재료 모델 그래프 출력
    
    # 3. 중력 해석
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. Pushover 해석을 중단합니다.")
        ops.wipe() # 실패 시 wipe 추가
        return None # None 반환

    # 4. 고유치 해석
    ok, mpr_x, mpr_z = run_eigen_analysis(params, model_nodes_info)
    if not ok:
        print("\n고유치 해석 실패. Pushover 해석을 중단합니다.")
        ops.wipe()
        return None, None, None, None # None 반환
    
    # [신규] 질량 참여율 90% 검증
    if mpr_x < 0.9 or mpr_z < 0.9:
        print(f"\n[Warning] 질량 참여율이 90% 미만입니다 (X: {mpr_x*100:.1f}%, Z: {mpr_z*100:.1f}%). 데이터셋에서 제외합니다.")
        ops.wipe()
        return None, None, None, None

    # [신규] 4.5. 비선형 정적해석(Pushover) 적용 타당성 검증
    is_nsp_valid_x, is_nsp_valid_y = verify_nsp_applicability(params, model_nodes_info)
    if not (is_nsp_valid_x and is_nsp_valid_y):
        print("\n[Warning] 현재 모델은 1차 모드 지배 조건을 만족하지 않습니다. 데이터셋에서 제외합니다.")
        ops.wipe()
        return None, None, None, None # 필요시 주석 해제

    # 5. 푸쉬오버 해석 실행
    if not run_pushover_analysis(params, model_nodes_info, direction=direction):
        print("\n푸쉬오버 해석 실행 중 오류 발생.")
        ops.wipe()
        # pass # 해석 실패 시 조용히 넘어가는 대신, None을 반환하여 실행 중단
        return None # None 반환

    # 6. 푸쉬오버 결과 처리 (4개 값 반환)
    df_curve, df_disp, final_states_dfs, df_m_phi = process_pushover_results(params, model_nodes_info, direction=direction)
    
    if df_curve is None or df_disp is None or df_curve.empty or len(df_curve) < 2:
        print("\n결과 파일 처리 실패 또는 데이터 부족. 후처리를 건너뜁니다.")
        ops.wipe()
        return None # None 반환

    # 7. 성능점 계산
    perf_points = calculate_performance_points(df_curve)
    
    # [신규] 8. 재료 파괴 검사
    failure_report = find_first_material_failure(model_nodes_info, params, df_curve, direction)
    if failure_report:
        print("\n--- [!] 재료 변형률 한계 초과로 인한 부재 파괴가 감지되었습니다. ---")
        print(f"  - Element: {failure_report['element_tag']} ({failure_report['element_type']})")
        print(f"  - Failure: {failure_report['failure_type']}")
        print(f"  - Time: {failure_report['time']:.4f}s, Roof Disp: {failure_report['roof_disp']:.4f}m")
        print(f"  - Strain: {failure_report['strain']:.6f}, Limit: {failure_report['limit']:.6f}")
        print("--------------------------------------------------------------------")
    
    # 9. 소성/손상 분포도 플로팅
    if not skip_plots:
        plot_plastic_damage_distribution(params, model_nodes_info, final_states_dfs, direction)

    # 10. 애니메이션 및 플롯 생성
    if not skip_plots:
        animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs, failure_report, direction)
    if not skip_plots and df_m_phi is not None:
        animate_and_plot_M_phi(df_m_phi, params)
    
    print("\n단일 해석 및 후처리 완료.")
    
    ops.wipe() # 마지막에 wipe 추가
    
    return perf_points, model_nodes_info, df_curve, direction # 계산된 성능점, 모델 정보, 푸쉬오버 곡선, 방향 반환


import random
import json

# generate_dataset.py의 헬퍼 함수 임포트
try:
    from generate_dataset import get_fc_expected_strength_factor, get_fy_expected_strength_factor
except ImportError:
    print("Warning: Could not import helper functions from generate_dataset.py. Using fallback values.")
    def get_fc_expected_strength_factor(fc): return 1.1
    def get_fy_expected_strength_factor(fy): return 1.1

def get_single_run_parameters():
    """
    [수정됨] dataset_config.json에서 임의의 설계안 하나를 생성하여 반환합니다.
    """
    print("--- [!] 단일 RC 모멘트 골조 해석 모드 활성화 (설계안 샘플링) ---")
    
    config_path = Path(__file__).parent / 'dataset_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    geo_params = config['building_geometry']
    member_params = config['member_properties']
    material_params = config['material_properties']

    num_stories = random.choice(geo_params['num_stories_range'])
    num_bays_x = random.choice(geo_params['num_bays_x_range'])
    num_bays_z = random.choice(geo_params['num_bays_z_range'])

    fc_nominal_mpa = random.choice(material_params['nominal_strengths']['fc_MPa_range'])
    fy_nominal_mpa = random.choice(material_params['nominal_strengths']['Fy_MPa_range'])
    
    fc_factor = get_fc_expected_strength_factor(fc_nominal_mpa)
    fy_factor = get_fy_expected_strength_factor(fy_nominal_mpa)

    col_props_by_group = {}
    beam_props_by_group = {}
    last_ext_col_dim, last_int_col_dim = (0, 0), (0, 0)
    num_story_groups = math.ceil(num_stories / 2)

    for group_idx in reversed(range(num_story_groups)):
        ext_col_choices = [tuple(d) for d in member_params['col_section_tiers_m']['exterior'] if d[0] >= last_ext_col_dim[0]]
        int_col_choices_all = [tuple(d) for d in member_params['col_section_tiers_m']['interior'] if d[0] >= last_int_col_dim[0]]
        
        current_ext_col_dim = random.choice(ext_col_choices if ext_col_choices else member_params['col_section_tiers_m']['exterior'])
        int_col_choices_filtered = [d for d in int_col_choices_all if d[0] >= current_ext_col_dim[0]]
        current_int_col_dim = random.choice(int_col_choices_filtered if int_col_choices_filtered else int_col_choices_all)

        col_props_by_group[group_idx] = {'exterior': current_ext_col_dim, 'interior': current_int_col_dim}
        last_ext_col_dim, last_int_col_dim = current_ext_col_dim, current_int_col_dim

        beam_props_by_group[group_idx] = {
            'exterior': tuple(random.choice(member_params['beam_section_tiers_m']['exterior'])),
            'interior': tuple(random.choice(member_params['beam_section_tiers_m']['interior']))
        }
    
    params = {
        'analysis_name': 'Run_Single_RC_Moment_Frame_Sampled',
        'output_dir': Path('results/Run_Single_RC_Moment_Frame_Sampled'),
        'target_drift': 0.04, 'num_steps': 1000, 'num_modes': 20,
        'num_int_pts': 5,
        'plot_z_line_index': 0,
        'plot_x_line_index': 0,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': random.choice(geo_params['bay_width_x_m_range']),
        'bay_width_z': random.choice(geo_params['bay_width_z_m_range']),
        'story_height': 3.5,
        'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
        'skip_post_processing': False,
        'g': 9.81,
        'dead_load_pa': 5000,
        'live_load_pa': 2500,
        'cover': 0.04,
        'rebar_Area': 0.000507,
        'num_bars_z': 5, 'num_bars_x': 5,
        'num_bars_top': 3, 'num_bars_bot': 3,
        'E_steel': 200e9,
        'f_ck_nominal': -fc_nominal_mpa * 1e6,
        'fc': -fc_nominal_mpa * 1e6 * fc_factor,
        'Fy_nominal': fy_nominal_mpa * 1e6,
        'Fy': fy_nominal_mpa * 1e6 * fy_factor,
        'col_props_by_group': col_props_by_group,
        'beam_props_by_group': beam_props_by_group,
    }
    
    return {**params, **config['nonlinear_materials']}


# ### 12. 메인 실행부 (Driver) ###
if __name__ == '__main__':
    
    # 1. 단일 실행용 파라미터 가져오기
    parameters = get_single_run_parameters()
    
    # 2. X, Z 방향에 대해 순차적으로 해석 실행
    for direction in ['X', 'Z']:
        print(f"\n--- Running {direction}-direction analysis ---")
        # 각 방향 해석을 위해 파라미터 복사 (ops.wipe()가 이전 상태를 지우므로)
        params_for_run = parameters.copy()
        
        # 출력 디렉토리 이름에 방향 추가
        original_name = params_for_run['analysis_name']
        original_dir = params_for_run['output_dir']
        
        params_for_run['analysis_name'] = f"{original_name}_{direction}"
        params_for_run['output_dir'] = original_dir.parent / f"{original_dir.name}_{direction}"

        main(params_for_run, direction=direction)