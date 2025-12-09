import math
import sys
import os
import json
import time # [NEW]
import pandas as pd
from pathlib import Path 
import random
import torch
from tqdm import tqdm
import openseespy.opensees as ops
import traceback

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- 모듈화된 함수 임포트 ---
from src.core.model_builder import build_model
from src.core.analysis_runner import run_gravity_analysis, run_eigen_analysis, run_pushover_analysis
from src.core.post_processor import process_pushover_results
from src.data.graph_exporter import extract_graph_data, process_pushover_curve

# --- 헬퍼 함수 임포트 (generate_dataset.py 자체에 포함) ---
# 여기서 get_fc_expected_strength_factor, get_fy_expected_strength_factor 정의
def get_fc_expected_strength_factor(fc):
    if fc <= 20: return 1.25
    elif fc <= 30: return 1.20
    elif fc <= 40: return 1.15
    else: return 1.10

def get_fy_expected_strength_factor(fy):
    if fy <= 300: return 1.10
    elif fy <= 400: return 1.05
    else: return 1.03

def generate_random_design_parameters(config_path):
    """
    dataset_config.json에서 임의의 설계안 하나를 생성하여 반환합니다.
    run_single_analysis.py의 get_single_run_parameters와 유사.
    """
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
    target_rho_col = random.uniform(0.010, 0.025)
    target_rho_beam = random.uniform(0.006, 0.015)
    
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
            n_side = max(2, int(num_total / 4) + 1)
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
        'analysis_name': 'Generated_RC_Moment_Frame',
        'output_dir': Path('results_temp/Generated_RC_Moment_Frame'), # 임시 결과 저장 디렉토리
        'target_drift': 0.04, 
        'num_steps': config['analysis_parameters']['num_steps'], # [MODIFIED] Use value from config
        'num_modes': 20,
        'num_int_pts': 5, 'plot_z_line_index': 0, 'plot_x_line_index': 0,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': random.choice(geo_params['bay_width_x_m_range']),
        'bay_width_z': random.choice(geo_params['bay_width_z_m_range']),
        'story_height': 3.5,
        'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
        'skip_post_processing': True, # 데이터 생성 시 플롯 건너뛰기
        'g': 9.81, 'dead_load_pa': 5000, 'live_load_pa': 2500,
        'cover': 0.04, 
        'E_steel': 200e9,
        'f_ck_nominal': -fc_nominal_mpa * 1e6, 'fc': -fc_nominal_mpa * 1e6 * fc_factor,
        'Fy_nominal': fy_nominal_mpa * 1e6, 'Fy': fy_nominal_mpa * 1e6 * fy_factor,
        'col_props_by_group': col_props_by_group,
        'beam_props_by_group': beam_props_by_group,
    }
    
    return {**params, **config['nonlinear_materials']}


def run_and_export_graph_data(sample_id, dataset_config_path, output_data_dir):
    """
    단일 해석을 수행하고 결과를 GNN 그래프 데이터로 변환하여 저장합니다.
    """
    start_time = time.time() # [NEW] Start timer
    params = generate_random_design_parameters(dataset_config_path)
    
    # 임시 결과 저장 디렉토리를 각 샘플별로 고유하게 설정
    temp_results_dir_name = f"temp_results_{os.getpid()}_{sample_id}"
    params['output_dir'] = Path(os.getcwd()) / temp_results_dir_name # 프로젝트 루트 아래
    
    try:
        params['output_dir'].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating temporary directory {params['output_dir']}: {e}")
        return False, f"Dir_Error: {e}", time.time() - start_time

    graph_data_list = []
    directions = ['X', 'Z']
    signs = ['pos', 'neg'] # target_drift 부호는 analysis_runner에서 처리

    for direction in directions:
        for sign_str in signs:
            # 매 실행마다 OpenSees 모델을 초기화해야 하므로, build_model 호출 전에 wipe
            ops.wipe() # Assuming ops is imported and available
            
            # 각 방향/부호별로 파라미터 복사 및 이름/디렉토리 설정
            params_for_run = params.copy()
            params_for_run['analysis_name'] = f"{params['analysis_name']}_{sample_id}_{direction}_{sign_str}"
            params_for_run['output_dir'] = params['output_dir'] / f"{direction}_{sign_str}"
            
            # 임시 결과 디렉토리 생성
            try:
                params_for_run['output_dir'].mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"SubDir_Error: {e}", time.time() - start_time

            # target_drift 설정
            if sign_str == 'neg':
                params_for_run['target_drift'] = -abs(params_for_run['target_drift'])
            else:
                params_for_run['target_drift'] = abs(params_for_run['target_drift'])

            # Model Build
            model_nodes_info = build_model(params_for_run)
            
            # [NEW] Calculate Mass Properties (for Node Features)
            node_mass_map = {}
            for node_tag in model_nodes_info['all_node_coords']:
                mass_vals = ops.nodeMass(node_tag)
                if mass_vals:
                    node_mass_map[node_tag] = mass_vals[0]
                else:
                    node_mass_map[node_tag] = 0.0
            
            params_for_run['node_mass_map'] = node_mass_map
            
            # Gravity Analysis (Get accurate weight from reactions)
            ok_gravity, total_reaction_y = run_gravity_analysis(params_for_run, model_nodes_info)
            if not ok_gravity:
                return False, f"Gravity_Fail_{direction}_{sign_str}", time.time() - start_time
            
            params_for_run['total_weight'] = abs(total_reaction_y) # Use Reaction Y as Total Weight
            
            # Eigen Analysis
            ok_eigen, modal_props = run_eigen_analysis(params_for_run, model_nodes_info, silent=True)
            if not ok_eigen:
                return False, f"Eigen_Fail_{direction}_{sign_str}", time.time() - start_time

            # Pushover Analysis
            ok_pushover, dominant_mode = run_pushover_analysis(params_for_run, model_nodes_info, modal_props, direction=direction)
            if not ok_pushover:
                return False, f"Pushover_Fail_{direction}_{sign_str}", time.time() - start_time
            
            # Post-processing (Results CSV to DataFrame)
            df_curve, _, _, _ = process_pushover_results(params_for_run, model_nodes_info, dominant_mode, direction=direction, skip_plots=True)
            
            if df_curve is None or df_curve.empty or len(df_curve) < 2:
                return False, f"PostProcess_Fail_{direction}_{sign_str}", time.time() - start_time

            # Graph Data Extraction
            graph_data = extract_graph_data(model_nodes_info, params_for_run, direction) # params_for_run 전달
            
            if graph_data is None: # None이 반환되면 실패 처리
                return False, f"GraphData_Extraction_Fail_{direction}_{sign_str}", time.time() - start_time
            
            # Target (Pushover Curve) Processing
            max_roof_disp = abs(params_for_run['target_drift'] * params_for_run['story_height'] * params_for_run['num_stories'])
            processed_curve = process_pushover_curve(df_curve, max_roof_disp, total_weight=params_for_run.get('total_weight'))
            
            if processed_curve is None:
                return False, f"CurveProcess_Fail_{direction}_{sign_str}", time.time() - start_time
            
            graph_data.y = processed_curve.unsqueeze(0) # [1, 100] 형태로 저장

            # Save Graph Data
            output_file_path = output_data_dir / f"data_{sample_id}_{direction}_{sign_str}.pt"
            torch.save(graph_data, output_file_path)
            graph_data_list.append(graph_data)
            
    return True, "Success", time.time() - start_time


def main_generate_dataset(num_samples: int = 100):
    dataset_config_path = Path(__file__).parent / 'dataset_config.json'
    processed_data_dir = Path(project_root) / 'data' / 'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = processed_data_dir / 'dataset_generation_log.txt'
    error_log_file_path = processed_data_dir / 'dataset_generation_errors.txt'

    success_count = 0
    failure_count = 0
    skipped_count = 0 # [NEW] Skipped count

    # [NEW] Find the starting sample ID to avoid overwriting
    existing_files = list(processed_data_dir.glob('data_*.pt'))
    existing_sample_ids = []
    for f in existing_files:
        try:
            # Extract sample_id from filename like "data_X_X_pos.pt"
            parts = f.stem.split('_') # e.g., ['data', '0', 'X', 'pos']
            if len(parts) >= 2 and parts[1].isdigit():
                existing_sample_ids.append(int(parts[1]))
        except ValueError:
            continue
    
    start_sample_id = 0
    if existing_sample_ids:
        start_sample_id = max(existing_sample_ids) + 1
    
    print(f"Starting data generation from Sample ID: {start_sample_id}")
    
    session_start_time = time.time() # [NEW] Session timer

    with open(log_file_path, 'a', encoding='utf-8') as log_f, \
         open(error_log_file_path, 'a', encoding='utf-8') as error_f: # Append mode for logs
        
        log_f.write(f"\n--- Dataset Generation Session Started: {pd.Timestamp.now()} ---\n")
        log_f.write(f"Attempting to generate {num_samples} NEW samples (starting from ID {start_sample_id}).\n\n")
        
        for i in tqdm(range(num_samples), desc="Generating Dataset"):
            current_sample_id = start_sample_id + i
            
            # [NEW] Check if sample already exists (all 4 directions)
            all_directions_exist = True
            for direction in ['X', 'Z']:
                for sign_str in ['pos', 'neg']:
                    output_file_path = processed_data_dir / f"data_{current_sample_id}_{direction}_{sign_str}.pt"
                    if not output_file_path.exists():
                        all_directions_exist = False
                        break
                if not all_directions_exist:
                    break
            
            if all_directions_exist:
                skipped_count += 1
                log_f.write(f"Sample {current_sample_id}: SKIPPED (already exists)\n")
                continue # Skip to next sample
            
            try:
                success, message, elapsed = run_and_export_graph_data(current_sample_id, dataset_config_path, processed_data_dir)
                if success:
                    success_count += 1
                    log_f.write(f"Sample {current_sample_id}: SUCCESS - {message} (Time: {elapsed:.2f}s)\n")
                else:
                    failure_count += 1
                    error_f.write(f"Sample {current_sample_id}: FAILED - {message} (Time: {elapsed:.2f}s)\n")
                    log_f.write(f"Sample {current_sample_id}: FAILED - {message} (Time: {elapsed:.2f}s)\n")
            except Exception:
                failure_count += 1
                error_detail = traceback.format_exc()
                error_f.write(f"Sample {current_sample_id}: UNCAUGHT EXCEPTION:\n{error_detail}\n")
                log_f.write(f"Sample {current_sample_id}: UNCAUGHT EXCEPTION - See error log for details.\n")
            finally:
                # 임시 결과 디렉토리 삭제
                temp_dir_name = f"temp_results_{os.getpid()}_{current_sample_id}"
                temp_dir_path = Path(os.getcwd()) / temp_dir_name
                if temp_dir_path.exists():
                    import shutil
                    shutil.rmtree(temp_dir_path)
        
        session_end_time = time.time()
        total_duration = session_end_time - session_start_time
        
        log_f.write(f"\n--- Dataset Generation Session Finished: {pd.Timestamp.now()} ---\n")
        log_f.write(f"Total Attempts (New Samples): {num_samples}\n")
        log_f.write(f"Successful New Samples: {success_count}\n")
        log_f.write(f"Failed New Samples: {failure_count}\n")
        log_f.write(f"Skipped Existing Samples: {skipped_count}\n")
        log_f.write(f"Total Session Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)\n")
        
    print(f"\nDataset generation completed. Successful New: {success_count}, Failed: {failure_count}, Skipped: {skipped_count}")
    print(f"Total Duration: {time.time() - session_start_time:.2f}s")
    print(f"Logs: {log_file_path}")
    print(f"Errors: {error_log_file_path}")

if __name__ == '__main__':
    # 기본 100개 샘플 생성. 필요시 argparse로 개수 조절 가능
    main_generate_dataset(num_samples=99) # 테스트를 위해 1개로 설정
