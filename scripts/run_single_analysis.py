import openseespy.opensees as ops
import math
import numpy as np
import pandas as pd
import sys
import os 
from pathlib import Path 
import random
import json

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈화된 함수 임포트 ---
from src.core.model_builder import build_model
from src.core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from src.core.post_processor import process_pushover_results, calculate_performance_points
from src.core.verification import verify_nsp_applicability
from src.visualization.plot_matplotlib import plot_model_matplotlib
# from src.visualization.plot_opsvis import plot_with_opsvis # [REMOVED]
from src.visualization.plot_hinges import plot_plastic_damage_distribution
from src.visualization.animate_results import animate_and_plot_pushover
from src.visualization.plot_materials import plot_material_stress_strain
from src.visualization.plot_section import plot_section_matplotlib # [NEW]

# --- 헬퍼 함수 임포트 ---
try:
    from generate_dataset import get_fc_expected_strength_factor, get_fy_expected_strength_factor
except ImportError:
    print("Warning: Could not import helper functions from generate_dataset.py. Using fallback values.")
    def get_fc_expected_strength_factor(fc): return 1.1
    def get_fy_expected_strength_factor(fy): return 1.1

# --- 함수 정의 ---

def get_single_run_parameters():
    """[수정됨] dataset_config.json에서 임의의 설계안 하나를 생성하여 반환합니다."""
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

    # [NEW] 철근 직경 선택 (기둥/보 분리)
    col_rebar_pool = member_params.get('rebar_col_list', [{'name': 'D25', 'area': 0.000507}])
    beam_rebar_pool = member_params.get('rebar_beam_list', [{'name': 'D22', 'area': 0.000387}])
    
    selected_col_rebar = random.choice(col_rebar_pool)
    selected_beam_rebar = random.choice(beam_rebar_pool)
    
    # [NEW] 목표 철근비 선택 (Random Range)
    # 기둥: 1.0% ~ 2.5%
    target_rho_col = random.uniform(0.010, 0.025)
    # 보: 0.6% ~ 1.5%
    target_rho_beam = random.uniform(0.006, 0.015)
    
    print(f"Selected Rebar -> Col: {selected_col_rebar['name']} (Rho: {target_rho_col*100:.1f}%), Beam: {selected_beam_rebar['name']} (Rho: {target_rho_beam*100:.1f}%)")

    col_props_by_group = {}
    beam_props_by_group = {}
    last_ext_col_dim, last_int_col_dim = (0, 0), (0, 0)
    num_story_groups = math.ceil(num_stories / 2)

    for group_idx in reversed(range(num_story_groups)):
        # --- Column Section ---
        ext_col_choices = [tuple(d) for d in member_params['col_section_tiers_m']['exterior'] if d[0] >= last_ext_col_dim[0]]
        int_col_choices_all = [tuple(d) for d in member_params['col_section_tiers_m']['interior'] if d[0] >= last_int_col_dim[0]]
        
        current_ext_col_dim = random.choice(ext_col_choices if ext_col_choices else member_params['col_section_tiers_m']['exterior'])
        int_col_choices_filtered = [d for d in int_col_choices_all if d[0] >= current_ext_col_dim[0]]
        current_int_col_dim = random.choice(int_col_choices_filtered if int_col_choices_filtered else int_col_choices_all)
        
        # [Logic] Calculate Number of Bars for Columns
        def calc_col_bars(dims, rho, bar_area):
            Ag = dims[0] * dims[1]
            As_req = Ag * rho
            num_total = max(4, round(As_req / bar_area))
            if num_total % 2 != 0: num_total += 1 # 짝수 보정
            # 4면 배근 가정: num_bars_x + num_bars_z = num_total/2 + 2 ??
            # 단순화: 각 면에 고르게 배치. 
            # num_bars_z means bars along z-axis (top/bottom face). Actually bars along width.
            # Let's assume equal spacing on all sides.
            n_side = max(2, int(num_total / 4) + 1)
            # Total = (n_side * 2) + ((n_side-2) * 2) = 2*n_side + 2*n_side - 4 = 4*n_side - 4
            return n_side, n_side # num_bars_z, num_bars_x (number of bars along the face)

        ext_nz, ext_nx = calc_col_bars(current_ext_col_dim, target_rho_col, selected_col_rebar['area'])
        int_nz, int_nx = calc_col_bars(current_int_col_dim, target_rho_col, selected_col_rebar['area'])

        col_props_by_group[group_idx] = {
            'exterior': {
                'dims': current_ext_col_dim, 
                'rebar': {'area': selected_col_rebar['area'], 'nz': ext_nz, 'nx': ext_nx}
            },
            'interior': {
                'dims': current_int_col_dim, 
                'rebar': {'area': selected_col_rebar['area'], 'nz': int_nz, 'nx': int_nx}
            }
        }
        last_ext_col_dim, last_int_col_dim = current_ext_col_dim, current_int_col_dim

        # --- Beam Section ---
        ext_beam_dim = tuple(random.choice(member_params['beam_section_tiers_m']['exterior']))
        int_beam_dim = tuple(random.choice(member_params['beam_section_tiers_m']['interior']))
        
        # [Logic] Calculate Number of Bars for Beams (Top/Bot)
        def calc_beam_bars(dims, rho, bar_area):
            Ag = dims[0] * dims[1]
            As_req = Ag * rho
            # 보 철근은 상/하부 배근. As_req를 상하부 합계로 볼 것인가?
            # 보통 rho는 인장 철근비 기준. 상부/하부 각각 rho 적용? -> 너무 많음.
            # As_total = As_req. Top/Bot 각각 절반씩?
            # 실무: As_top ≈ 0.6~0.7 * As_total, As_bot ≈ 0.3~0.5 * As_total (단부는 상부 인장)
            # 단순화: 상부/하부 동일하게 As_req / 2 씩 배근 (대칭 배근 가정)
            num_one_side = max(2, round((As_req / 2) / bar_area))
            return num_one_side, num_one_side # top, bot

        ext_top, ext_bot = calc_beam_bars(ext_beam_dim, target_rho_beam, selected_beam_rebar['area'])
        int_top, int_bot = calc_beam_bars(int_beam_dim, target_rho_beam, selected_beam_rebar['area'])

        beam_props_by_group[group_idx] = {
            'exterior': {
                'dims': ext_beam_dim,
                'rebar': {'area': selected_beam_rebar['area'], 'top': ext_top, 'bot': ext_bot}
            },
            'interior': {
                'dims': int_beam_dim,
                'rebar': {'area': selected_beam_rebar['area'], 'top': int_top, 'bot': int_bot}
            }
        }
    
    params = {
        'analysis_name': 'Run_Single_RC_Moment_Frame_Sampled',
        'output_dir': Path('results/Run_Single_RC_Moment_Frame_Sampled'),
        'target_drift': 0.04, 'num_steps': 1000, 'num_modes': 20,
        'num_int_pts': 5, 'plot_z_line_index': 0, 'plot_x_line_index': 0,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': random.choice(geo_params['bay_width_x_m_range']),
        'bay_width_z': random.choice(geo_params['bay_width_z_m_range']),
        'story_height': 3.5,
        'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
        'skip_post_processing': False, 'g': 9.81, 'dead_load_pa': 5000, 'live_load_pa': 2500,
        'cover': 0.04, 
        # [Deleted] Global fixed rebar params
        'E_steel': 200e9,
        'f_ck_nominal': -fc_nominal_mpa * 1e6, 'fc': -fc_nominal_mpa * 1e6 * fc_factor,
        'Fy_nominal': fy_nominal_mpa * 1e6, 'Fy': fy_nominal_mpa * 1e6 * fy_factor,
        'col_props_by_group': col_props_by_group,
        'beam_props_by_group': beam_props_by_group,
    }
    
    return {**params, **config['nonlinear_materials']}

def main(params, direction='X', sign='+'):
    """[수정] 리팩토링된 함수들을 사용하여 전체 해석 파이프라인을 실행합니다.
    sign: '+' 또는 '-' (가력 방향 부호)
    """
    try:
        params['output_dir'].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {params['output_dir']}: {e}")
        return None

    # 방향 부호에 따른 target_drift 조정
    original_target_drift = params['target_drift']
    if sign == '-':
        params['target_drift'] = -abs(original_target_drift)
    else:
        params['target_drift'] = abs(original_target_drift)

    print(f"Target Drift set to: {params['target_drift']} (Sign: {sign})")

    model_nodes_info = build_model(params)
    skip_plots = params.get('skip_post_processing', False)

    if not skip_plots:
        # 플롯은 한 번만 그리면 됨 (양방향 모두 구조는 같음)
        # 하지만 파일명 중복 방지를 위해 여기서도 그림
        plot_model_matplotlib(params, model_nodes_info, direction)
        # plot_with_opsvis(params) # [REMOVED] Deprecated in favor of custom matplotlib plotter
        plot_section_matplotlib(params, params['output_dir']) # [NEW] Clean Section Plots
        plot_material_stress_strain(params)
    
    if not run_gravity_analysis(params):
        print("\n중력 해석 실패. 해석을 중단합니다."); ops.wipe(); return None

    ok, modal_props = run_eigen_analysis(params, model_nodes_info)
    if not ok:
        print("\n고유치 해석 실패. 해석을 중단합니다."); ops.wipe(); return None

    is_nsp_valid_x, is_nsp_valid_z = verify_nsp_applicability(params, model_nodes_info, modal_props)
    
    # 푸쉬오버 해석 실행 (sign 정보는 params['target_drift']에 이미 반영됨)
    # run_pushover_analysis 내부에서 target_drift 부호를 사용함
    ok, dominant_mode = run_pushover_analysis(params, model_nodes_info, modal_props, direction=direction)
    if not ok:
        print("\n푸쉬오버 해석 실행 중 오류 발생."); ops.wipe(); return None
    
    # 결과 처리 및 파일명에 sign 반영을 위해 process_pushover_results 호출 시 주의 필요
    # 현재 process_pushover_results는 파일명을 params['analysis_name']에서 가져옴
    # 따라서 params['analysis_name']을 미리 변경해두어야 함 (main 호출 전에 처리됨)
    
    df_curve, df_disp, final_states_dfs, df_m_phi = process_pushover_results(params, model_nodes_info, dominant_mode, direction=direction)
    
    # Save modal properties to a json file
    modal_data_to_save = {
        'modal_properties': modal_props,
        'dominant_mode': dominant_mode,
        'nsp_validity': {'X': is_nsp_valid_x, 'Z': is_nsp_valid_z}
    }
    modal_json_path = params['output_dir'] / 'modal_properties.json'
    with open(modal_json_path, 'w') as f:
        def convert_numpy_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy_to_list(i) for i in obj]
            return obj
        
        json.dump(convert_numpy_to_list(modal_data_to_save), f, indent=4)
    print(f"Modal properties saved to: {modal_json_path}")
    
    if df_curve is None or df_disp is None or df_curve.empty or len(df_curve) < 2:
        print("\n결과 파일 처리 실패 또는 데이터 부족. 후처리를 건너뜁니다."); ops.wipe(); return None

    perf_points = calculate_performance_points(df_curve)
    
    if not skip_plots:
        # 애니메이션 생성 시에도 sign 정보가 반영된 결과 파일을 사용하게 됨
        animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs, None, direction)
    
    print("\n단일 해석 및 후처리 완료."); ops.wipe()
    return perf_points, model_nodes_info, df_curve, direction

# ### 메인 실행부 (Driver) ###
if __name__ == '__main__':
    
    # 1. 단일 실행용 파라미터 가져오기
    parameters = get_single_run_parameters()
    
    # 2. X(+,-), Z(+,-) 방향에 대해 순차적으로 해석 실행
    directions = ['X', 'Z']
    signs = ['+', '-']
    
    for direction in directions:
        for sign in signs:
            print(f"\n--- Running {direction}-direction ({sign}) analysis ---")
            
            params_for_run = parameters.copy()
            
            # 출력 디렉토리 및 해석 이름에 방향과 부호 추가
            # 예: Run_Single_..._X_pos, Run_Single_..._X_neg
            original_name = params_for_run['analysis_name']
            original_dir = params_for_run['output_dir']
            
            sign_str = "pos" if sign == '+' else "neg"
            
            params_for_run['analysis_name'] = f"{original_name}_{direction}_{sign_str}"
            # 디렉토리명도 구분 (선택사항, 파일명만 구분해도 되지만 관리가 편함)
            params_for_run['output_dir'] = original_dir.parent / f"{original_dir.name}_{direction}_{sign_str}"

            main(params_for_run, direction=direction, sign=sign)
