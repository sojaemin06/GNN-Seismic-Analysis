import math
import sys
import os
import json
import time 
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
from src.core.verification import verify_nsp_applicability 

# --- 헬퍼 함수 임포트 ---
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

    col_rebar_pool = member_params.get('rebar_col_list', [{'name': 'D25', 'area': 0.000507}])
    beam_rebar_pool = member_params.get('rebar_beam_list', [{'name': 'D22', 'area': 0.000387}])
    
    selected_col_rebar = random.choice(col_rebar_pool)
    selected_beam_rebar = random.choice(beam_rebar_pool)
    
    target_rho_col = random.uniform(0.010, 0.025)
    target_rho_beam = random.uniform(0.006, 0.015)
    
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
        
        def calc_col_bars(dims, rho, bar_area):
            Ag = dims[0] * dims[1]
            As_req = Ag * rho
            num_total = max(4, round(As_req / bar_area))
            if num_total % 2 != 0: num_total += 1 
            n_side = max(2, int(num_total / 4) + 1)
            return n_side, n_side 

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

        ext_beam_dim = tuple(random.choice(member_params['beam_section_tiers_m']['exterior']))
        int_beam_dim = tuple(random.choice(member_params['beam_section_tiers_m']['interior']))
        
        def calc_beam_bars(dims, rho, bar_area):
            Ag = dims[0] * dims[1]
            As_req = Ag * rho
            num_one_side = max(2, round((As_req / 2) / bar_area))
            return num_one_side, num_one_side 

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
        'output_dir': Path('results_temp/Generated_RC_Moment_Frame'),
        'target_drift': 0.04, 
        'num_steps': config['analysis_parameters']['num_steps'], 
        'num_modes': 20,
        'num_int_pts': 5, 'plot_z_line_index': 0, 'plot_x_line_index': 0,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': random.choice(geo_params['bay_width_x_m_range']),
        'bay_width_z': random.choice(geo_params['bay_width_z_m_range']),
        'story_height': 3.5,
        'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
        'skip_post_processing': True, 
        'g': 9.81, 'dead_load_pa': 5000, 'live_load_pa': 2500,
        'cover': 0.04, 
        'E_steel': 200e9,
        'f_ck_nominal': -fc_nominal_mpa * 1e6, 'fc': -fc_nominal_mpa * 1e6 * fc_factor,
        'Fy_nominal': fy_nominal_mpa * 1e6, 'Fy': fy_nominal_mpa * 1e6 * fy_factor,
        'col_props_by_group': col_props_by_group,
        'beam_props_by_group': beam_props_by_group,
    }
    
    return {**params, **config['nonlinear_materials']}


def run_and_export_graph_data(run_id, save_id, dataset_config_path, output_data_dir):
    start_time = time.time() 
    params = generate_random_design_parameters(dataset_config_path)
    
    temp_results_dir_name = f"temp_results_{os.getpid()}_{run_id}"
    params['output_dir'] = Path(os.getcwd()) / temp_results_dir_name 
    
    try:
        params['output_dir'].mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Error creating temporary directory {params['output_dir']}: {e}")
        return False, f"Dir_Error: {e}", time.time() - start_time

    data_to_save = [] 
    directions = ['X', 'Z']
    signs = ['pos', 'neg'] 

    for direction in directions:
        for sign_str in signs:
            ops.wipe() 
            
            params_for_run = params.copy()
            params_for_run['analysis_name'] = f"{params['analysis_name']}_{run_id}_{direction}_{sign_str}"
            params_for_run['output_dir'] = params['output_dir'] / f"{direction}_{sign_str}"
            
            try:
                params_for_run['output_dir'].mkdir(parents=True, exist_ok=True)
            except Exception as e:
                return False, f"SubDir_Error: {e}", time.time() - start_time

            if sign_str == 'neg':
                params_for_run['target_drift'] = -abs(params_for_run['target_drift'])
            else:
                params_for_run['target_drift'] = abs(params_for_run['target_drift'])

            model_nodes_info = build_model(params_for_run)
            
            node_mass_map = {}
            for node_tag in model_nodes_info['all_node_coords']:
                mass_vals = ops.nodeMass(node_tag)
                if mass_vals:
                    node_mass_map[node_tag] = mass_vals[0]
                else:
                    node_mass_map[node_tag] = 0.0
            
            params_for_run['node_mass_map'] = node_mass_map
            
            ok_gravity, total_reaction_y = run_gravity_analysis(params_for_run, model_nodes_info)
            if not ok_gravity:
                return False, f"Gravity_Fail_{direction}_{sign_str}", time.time() - start_time
            
            params_for_run['total_weight'] = abs(total_reaction_y) 
            
            ok_eigen, modal_props = run_eigen_analysis(params_for_run, model_nodes_info, silent=True)
            if not ok_eigen:
                return False, f"Eigen_Fail_{direction}_{sign_str}", time.time() - start_time

            is_nsp_valid_x, is_nsp_valid_z = verify_nsp_applicability(params_for_run, model_nodes_info, modal_props, silent=True) 
            
            is_current_direction_valid = False
            if direction == 'X':
                is_current_direction_valid = is_nsp_valid_x
            elif direction == 'Z':
                is_current_direction_valid = is_nsp_valid_z
            
            if not is_current_direction_valid:
                return False, f"BadData_130Rule_Fail_{direction}_{sign_str}", time.time() - start_time
            
            ok_pushover, dominant_mode = run_pushover_analysis(params_for_run, model_nodes_info, modal_props, direction=direction)
            if not ok_pushover:
                return False, f"Pushover_Fail_{direction}_{sign_str}", time.time() - start_time
            
            df_curve, _, _, _ = process_pushover_results(params_for_run, model_nodes_info, dominant_mode, direction=direction, skip_plots=True)
            
            if df_curve is None or df_curve.empty or len(df_curve) < 2:
                return False, f"PostProcess_Fail_{direction}_{sign_str}", time.time() - start_time

            direction_modal_props = {}
            if dominant_mode:
                dir_lower = direction.lower()
                direction_modal_props = {
                    'T1': dominant_mode.get('period', 0.0),
                    'PF1': dominant_mode.get(f'gamma_{dir_lower}', 0.0),
                    'MassRatio': dominant_mode.get(f'mpr_{dir_lower}', 0.0),
                    'PhiRoof': dominant_mode.get(f'phi_{dir_lower}', [0.0])[-1] 
                }
            
            graph_data = extract_graph_data(model_nodes_info, params_for_run, direction, modal_props=direction_modal_props) 
            
            if graph_data is None: 
                return False, f"GraphData_Extraction_Fail_{direction}_{sign_str}", time.time() - start_time
            
            max_roof_disp = abs(params_for_run['target_drift'] * params_for_run['story_height'] * params_for_run['num_stories'])
            processed_curve = process_pushover_curve(df_curve, max_roof_disp, total_weight=params_for_run.get('total_weight'))
            
            if processed_curve is None:
                return False, f"CurveProcess_Fail_{direction}_{sign_str}", time.time() - start_time
            
            if torch.std(processed_curve) < 1e-4:
                return False, f"BadData_FlatCurve_{direction}_{sign_str}", time.time() - start_time
            
            if torch.max(torch.abs(processed_curve)) < 0.001:
                 return False, f"BadData_LowStrength_{direction}_{sign_str}", time.time() - start_time

            graph_data.y = processed_curve.unsqueeze(0) 

            output_file_path = output_data_dir / f"data_{save_id}_{direction}_{sign_str}.pt"
            data_to_save.append((graph_data, output_file_path))
    
    for data, path in data_to_save:
        torch.save(data, path)
            
    return True, "Success", time.time() - start_time


def main_generate_dataset(num_samples: int = 500):
    dataset_config_path = Path(__file__).parent / 'dataset_config.json'
    processed_data_dir = Path(project_root) / 'data' / 'processed'
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    log_file_path = processed_data_dir / 'dataset_generation_log.txt'
    error_log_file_path = processed_data_dir / 'dataset_generation_errors.txt'

    success_count = 0
    failure_count = 0
    
    existing_files = list(processed_data_dir.glob('data_*.pt'))
    existing_sample_ids = []
    for f in existing_files:
        try:
            parts = f.stem.split('_') 
            if len(parts) >= 2 and parts[1].isdigit():
                existing_sample_ids.append(int(parts[1]))
        except ValueError:
            continue
    
    start_sample_id = 0
    if existing_sample_ids:
        start_sample_id = max(existing_sample_ids) + 1
    
    print(f"Starting data generation from Sample ID: {start_sample_id}")
    print(f"Target: Generate {num_samples} NEW successful samples.")
    
    session_start_time = time.time() 

    current_save_id = start_sample_id
    total_attempts = 0 

    with open(log_file_path, 'a', encoding='utf-8') as log_f, \
         open(error_log_file_path, 'a', encoding='utf-8') as error_f: 
        
        log_f.write(f"\n--- Dataset Generation Session Started: {pd.Timestamp.now()} ---\n")
        log_f.write(f"Target: {num_samples} successful samples (Starting ID: {start_sample_id}).\n\n")
        
        with tqdm(total=num_samples, desc="Generating Dataset") as pbar:
            while success_count < num_samples:
                run_id = start_sample_id + total_attempts 
                
                try:
                    success, message, elapsed = run_and_export_graph_data(run_id, current_save_id, dataset_config_path, processed_data_dir)
                    
                    if success:
                        success_count += 1
                        log_f.write(f"Sample {current_save_id} (Run {run_id}): SUCCESS - {message} (Time: {elapsed:.2f}s)\n")
                        current_save_id += 1
                        pbar.update(1)
                    else:
                        failure_count += 1
                        error_f.write(f"Run {run_id} (Target {current_save_id}): FAILED - {message} (Time: {elapsed:.2f}s)\n")
                        
                except Exception:
                    failure_count += 1
                    error_detail = traceback.format_exc()
                    error_f.write(f"Run {run_id} (Target {current_save_id}): UNCAUGHT EXCEPTION:\n{error_detail}\n")
                    log_f.write(f"Run {run_id}: UNCAUGHT EXCEPTION - See error log.\n")
                
                finally:
                    temp_dir_name = f"temp_results_{os.getpid()}_{run_id}"
                    temp_dir_path = Path(os.getcwd()) / temp_dir_name
                    if temp_dir_path.exists():
                        import shutil
                        try:
                            shutil.rmtree(temp_dir_path)
                        except Exception as e:
                            print(f"Warning: Failed to delete temp dir {temp_dir_path}: {e}")
                
                total_attempts += 1
        
        session_end_time = time.time()
        total_duration = session_end_time - session_start_time
        
        log_f.write(f"\n--- Dataset Generation Session Finished: {pd.Timestamp.now()} ---\n")
        log_f.write(f"Total Attempts: {total_attempts}\n")
        log_f.write(f"Successful New Samples: {success_count}\n")
        log_f.write(f"Failed Attempts: {failure_count}\n")
        log_f.write(f"Total Session Duration: {total_duration:.2f}s ({total_duration/60:.2f} min)\n")
        
    print(f"\nDataset generation completed.")
    print(f"Generated {success_count} samples (ID {start_sample_id} to {current_save_id - 1}).")
    print(f"Total Attempts: {total_attempts}, Failures: {failure_count}")
    print(f"Total Duration: {time.time() - session_start_time:.2f}s")
    print(f"Logs: {log_file_path}")
    print(f"Errors: {error_log_file_path}")

if __name__ == '__main__':
    main_generate_dataset(num_samples=500) 
