import itertools
import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
import math
import os
import traceback
import json
import shutil

# run_single_analysis.py의 main 함수를 직접 호출하기 위해 import
from run_single_analysis import main as run_single_analysis_main

# --- [수정] 그래프 데이터 추출 함수 임포트 ---
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from src.data.graph_exporter import extract_graph_data, process_pushover_curve

def get_unique_building_indices(dataset_info):
    """Helper function to count unique buildings from the dataset_info list."""
    if not dataset_info:
        return 0
    return len(set(int(item['analysis_name'].split('_')[1]) for item in dataset_info))


# --- [수정] 멀티프로세싱을 위한 단일 작업 함수 ---
def process_single_combination(args):
    """
    [수정됨] 단일 파라미터 조합에 대해, 그룹화 및 계층화된 샘플링을 적용하여
    X, Z 양방향으로 해석을 실행하고, 실패 시 생성된 모든 파일을 삭제하여 원자성을 보장합니다.
    """
    i, combo, output_root_dir, config = args
    num_stories, num_bays_x, num_bays_z = combo # build_core 제거
    build_core = False # RC 모멘트 골조만 대상으로 하므로 항상 False

    geo_params = config['building_geometry']
    member_params = config['member_properties']
    material_params = config['material_properties']
    
    building_results_for_summary = [] # 현재 건물(combo)의 X, Z 방향 결과를 임시 저장
    building_dirs = [] # [추가] 현재 건물의 모든 출력 폴더 경로를 추적

    # --- 1. 건물 전체에 적용될 균일 파라미터 샘플링 ---
    bay_width_x = random.choice(geo_params['bay_width_x_m_range'])
    bay_width_z = random.choice(geo_params['bay_width_z_m_range'])
    
    fc_nominal = random.choice(material_params['nominal_strengths']['fc_MPa_range'])
    Fy_nominal = random.choice(material_params['nominal_strengths']['Fy_MPa_range'])

    fc_expected = fc_nominal * material_params['expected_strength_factors']['fc_factor']
    Fy_expected = Fy_nominal * material_params['expected_strength_factors']['Fy_factor']

    uniform_material_props = {
        'fc': -fc_expected * 1e6,
        'Fy': Fy_expected * 1e6,
        'cover': round(random.uniform(0.04, 0.06), 3),
        'rebar_Area': random.choice(member_params['rebar_As_m2_list'])['area'],
        'num_bars_x': random.choice(member_params['num_bars_main_range']),
        'num_bars_z': random.choice(member_params['num_bars_main_range']),
        'num_bars_top': random.choice(member_params['num_bars_main_range']),
        'num_bars_bot': random.choice(member_params['num_bars_main_range']),
        'E_steel': 200e9, 'g': 9.81, 'dead_load_pa': 5000.0, 'live_load_pa': 0.0,
    }

    # --- 2. 2개층 단위 그룹별 부재 단면 샘플링 ---
    num_story_groups = math.ceil(num_stories / 2)
    col_props_by_group = {}
    beam_props_by_group = {}
    
    last_ext_col_dim = (0, 0)
    last_int_col_dim = (0, 0)

    for group_idx in reversed(range(num_story_groups)):
        ext_col_choices = [tuple(d) for d in member_params['col_section_tiers_m']['exterior'] if d[0] >= last_ext_col_dim[0]]
        int_col_choices = [tuple(d) for d in member_params['col_section_tiers_m']['interior'] if d[0] >= last_int_col_dim[0]]
        
        ext_col_dim = random.choice(ext_col_choices)
        int_col_choices_filtered = [d for d in int_col_choices if d[0] >= ext_col_dim[0]]
        int_col_dim = random.choice(int_col_choices_filtered)

        col_props_by_group[group_idx] = {'exterior': ext_col_dim, 'interior': int_col_dim}
        last_ext_col_dim, last_int_col_dim = ext_col_dim, int_col_dim

        beam_props_by_group[group_idx] = {
            'exterior': tuple(random.choice(member_params['beam_section_tiers_m']['exterior'])),
            'interior': tuple(random.choice(member_params['beam_section_tiers_m']['interior']))
        }

    # --- 3. 해석에 필요한 파라미터 구조화 ---
    member_props_for_analysis = {**uniform_material_props, 'col_props_by_group': col_props_by_group, 'beam_props_by_group': beam_props_by_group}

    # --- 4. 양방향 해석 실행 ---
    analysis_failed_for_building = False
    for direction in ['X', 'Z']:
        current_analysis_name = f"Building_{i:04d}_S{num_stories}_BX{num_bays_x}_BZ{num_bays_z}_{direction}"
        current_output_dir = Path(output_root_dir) / current_analysis_name
        building_dirs.append(current_output_dir) # [추가] 생성될 폴더 경로 저장

        analysis_params = {
            'analysis_name': current_analysis_name, 'output_dir': current_output_dir, 'target_drift': 0.04, 
            'num_steps': 1000, 'num_modes': 20, 'num_int_pts': 5, 'plot_z_line_index': 1,
            'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
            'bay_width_x': bay_width_x, 'bay_width_z': bay_width_z, 'story_height': 3.5,
            'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
            'skip_post_processing': True,
        }
        
        parameters = {**analysis_params, **member_props_for_analysis, **config['nonlinear_materials']}

        try:
            print(f"[{current_analysis_name}] 1. Running single analysis...")
            perf_points, model_nodes_info, df_curve, actual_direction = run_single_analysis_main(parameters, direction=direction)
            print(f"[{current_analysis_name}] 2. Analysis finished.")

            if model_nodes_info is None or df_curve is None or df_curve.empty:
                print(f"Skipping {current_analysis_name} due to analysis failure or insufficient data.")
                analysis_failed_for_building = True
                break 

            target_disp = parameters['target_drift'] * parameters['story_height'] * parameters['num_stories']
            max_roof_disp_actual = df_curve['Roof_Displacement_m'].max()
            if max_roof_disp_actual < target_disp * 0.95:
                print(f"Skipping {current_analysis_name} due to premature analysis termination (reached {max_roof_disp_actual:.3f}m / target {target_disp:.3f}m).")
                analysis_failed_for_building = True
                break 
                
            print(f"[{current_analysis_name}] 3. Extracting graph data...")
            graph_data = extract_graph_data(model_nodes_info, parameters, direction=actual_direction)
            print(f"[{current_analysis_name}] 4. Graph data extracted.")
            
            print(f"[{current_analysis_name}] 5. Processing pushover curve...")
            target_y = process_pushover_curve(df_curve, max_roof_disp=max_roof_disp_actual)
            print(f"[{current_analysis_name}] 6. Pushover curve processed.")

            if target_y is None:
                print(f"Skipping {current_analysis_name} due to pushover curve processing failure.")
                analysis_failed_for_building = True
                break 
                
            graph_data.y = target_y

            data_file_path = current_output_dir / f"{current_analysis_name}_graph_data.pt"
            
            print(f"[{current_analysis_name}] 7. Saving graph data to {data_file_path}...")
            torch.save(graph_data, data_file_path)
            print(f"[{current_analysis_name}] 8. Graph data saved.")

            summary_data = {
                'analysis_name': current_analysis_name, 'num_stories': num_stories, 'num_bays_x': num_bays_x,
                'num_bays_z': num_bays_z, 'data_file': str(data_file_path),
                'max_roof_disp_actual': max_roof_disp_actual,
                'peak_shear_kN': perf_points.get('peak_shear', 0) / 1000 if perf_points else 0,
                'direction': actual_direction
            }
            summary_data.update({f'group_{g}_col_ext': str(d['exterior']) for g, d in col_props_by_group.items()})
            summary_data.update({f'group_{g}_col_int': str(d['interior']) for g, d in col_props_by_group.items()})
            summary_data.update({f'group_{g}_beam_ext': str(d['exterior']) for g, d in beam_props_by_group.items()})
            summary_data.update({f'group_{g}_beam_int': str(d['interior']) for g, d in beam_props_by_group.items()})
            
            building_results_for_summary.append(summary_data)
            
        except Exception as e:
            print(f"Error processing {current_analysis_name}: {e}")
            print(f"Exception Type: {type(e)}")
            print(traceback.format_exc())
            analysis_failed_for_building = True
            break

    # --- 5. [수정] 실패 시 생성된 모든 폴더 삭제 ---
    if analysis_failed_for_building:
        print(f"  -> Analysis failed for Building_{i:04d}. Deleting intermediate directories.")
        for d in building_dirs:
            if d.exists():
                shutil.rmtree(d)
        return None # 실패 시 아무것도 반환하지 않음

    return building_results_for_summary # 양방향 모두 성공 시에만 결과 반환

# --- [수정] 메인 데이터셋 생성 함수 (멀티프로세싱 적용) ---
def generate_dataset(output_root_dir='dataset', target_samples=10, num_workers=None):
    """
    [수정됨] GNN 학습을 위한 데이터셋을 병렬로 생성합니다. 실패를 고려하여,
    목표한 개수의 성공 데이터가 생성될 때까지 해석을 반복합니다.
    """
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)

    # --- 1. 파라미터 범위 정의 및 모든 조합 생성 ---
    config_path = Path(__file__).parent / 'dataset_config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    geo_params = config['building_geometry']
    all_param_combinations = list(itertools.product(
        geo_params['num_stories_range'], 
        geo_params['num_bays_x_range'], 
        geo_params['num_bays_z_range']
    ))
    
    # --- 2. [수정] 목표 개수에 도달할 때까지 멀티프로세싱 반복 실행 ---
    dataset_info = []
    tried_combinations = set()
    pbar = tqdm(total=target_samples, desc="Succeeded Buildings")
    
    task_id_counter = 0
    
    while get_unique_building_indices(dataset_info) < target_samples:
        current_successful_count = get_unique_building_indices(dataset_info)
        
        # 다음 배치 크기 결정
        needed = target_samples - current_successful_count
        batch_size = min(needed * 2, num_workers * 2) # 추정 성공률 50% 가정, 배치 크기 조절

        # 시도하지 않은 새로운 조합 샘플링
        available_combinations = [c for c in all_param_combinations if tuple(c) not in tried_combinations]
        if not available_combinations:
            print("\nWarning: All possible parameter combinations have been tried, but target not reached.")
            break
            
        sample_size = min(batch_size, len(available_combinations))
        new_combos_to_try = random.sample(available_combinations, sample_size)
        
        tasks = []
        for combo in new_combos_to_try:
            tried_combinations.add(tuple(combo))
            tasks.append((task_id_counter, combo, output_root_dir, config))
            task_id_counter += 1
            
        print(f"\n--- Starting a new batch of {len(tasks)} samples to reach target {target_samples} (currently {current_successful_count}) ---")

        with multiprocessing.Pool(processes=num_workers) as pool:
            for results_for_building in pool.imap_unordered(process_single_combination, tasks):
                if results_for_building:
                    dataset_info.extend(results_for_building)
                    pbar.update(1) # 건물 하나 성공 시 pbar 1 증가

    pbar.close()

    # --- 3. 결과 후처리: 순차적 번호 부여 및 요약 저장 ---
    if not dataset_info:
        print("\nWarning: No data was successfully generated.")
        return

    # 성공한 빌딩의 원본 인덱스 추출 및 정렬
    original_indices = sorted(list(set([
        int(item['analysis_name'].split('_')[1]) for item in dataset_info
    ])))

    # 원본 인덱스 -> 새 순차 인덱스 매핑 생성
    old_to_new_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(original_indices)}
    
    print(f"\n--- Renumbering {len(original_indices)} successful buildings sequentially... ---")

    final_dataset_info = []
    for item in dataset_info:
        original_name = item['analysis_name']
        parts = original_name.split('_')
        old_idx = int(parts[1])
        geo_part = "_".join(parts[2:-1])
        direction = parts[-1]
        new_idx = old_to_new_index_map[old_idx]
        new_dir_name = f"Building_{new_idx:04d}_{geo_part}_{direction}"
        
        old_dir = Path(item['data_file']).parent
        new_dir = old_dir.parent / new_dir_name
        
        old_pt_filename = Path(item['data_file']).name
        new_pt_filename = f"{new_dir_name}_graph_data.pt"

        if old_dir.exists() and not old_dir == new_dir:
            os.rename(old_dir, new_dir)
            pt_file_to_rename = new_dir / old_pt_filename
            if pt_file_to_rename.exists():
                os.rename(pt_file_to_rename, new_dir / new_pt_filename)
        
        updated_item = item.copy()
        updated_item['analysis_name'] = new_dir_name
        updated_item['data_file'] = str(new_dir / new_pt_filename)
        final_dataset_info.append(updated_item)

    final_dataset_info.sort(key=lambda item: item['analysis_name'])
    df_dataset_info = pd.DataFrame(final_dataset_info)
    summary_path = output_root_dir / 'dataset_summary.csv'
    df_dataset_info.to_csv(summary_path, index=False, encoding='utf-8-sig')
    
    print(f"\n--- Dataset generation complete ---")
    print(f"Successfully generated {len(original_indices)} buildings ({len(df_dataset_info)} models).")
    print(f"Total combinations tried: {len(tried_combinations)}.")
    print(f"Final summary saved to {summary_path}")

if __name__ == '__main__':
    project_root_path = Path(__file__).parent.parent
    output_dir_abs = project_root_path / 'data' / 'raw_dataset'

    # [수정] 목표 성공 샘플 개수를 100개로 설정하여 실행
    generate_dataset(
        output_root_dir=output_dir_abs, 
        target_samples=100,
        num_workers=2 # CPU 코어 사용을 2개로 제한
    )
