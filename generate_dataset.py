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

# run_single_analysis.py의 main 함수를 직접 호출하기 위해 import
from run_single_analysis import main as run_single_analysis_main

# --- GNN 데이터 추출 및 가공을 위한 헬퍼 함수 (향후 core/graph_exporter.py로 분리 예정) ---
def extract_graph_data(model_nodes_info, col_props, beam_props):
    """
    model_nodes_info에서 PyG (PyTorch Geometric) 그래프 데이터를 추출합니다.
    [수정] 엣지 특성에 부재 종류, 단면 크기, 재료 강도, 철근 정보 등을 모두 포함시킵니다.
    """
    # 1. 노드 특성 (노드 좌표)
    node_coords = np.array(list(model_nodes_info['all_node_coords'].values()))
    x = torch.tensor(node_coords, dtype=torch.float)

    # 2. 엣지 인덱스 및 엣지 특성 생성
    edge_list = []
    edge_attributes = []
    
    column_tags = set(model_nodes_info['all_column_tags'])
    beam_tags = set(model_nodes_info['all_beam_tags'])

    # 엣지 특성 순서 정의 (GNN 모델 입력 순서와 일치해야 함)
    # [is_column, is_beam, width, depth, fc, Fy, cover, As, num_bars_1, num_bars_2]
    col_attr_list = [
        1.0, 0.0, 
        col_props['dims'][0], col_props['dims'][1], 
        abs(col_props['fc'] / 1e6), # MPa 단위로 정규화
        col_props['Fy'] / 1e6,      # MPa 단위로 정규화
        col_props['cover'], 
        col_props['rebar_Area'], 
        col_props['num_bars_x'], col_props['num_bars_z']
    ]
    beam_attr_list = [
        0.0, 1.0, 
        beam_props['dims'][0], beam_props['dims'][1], 
        abs(beam_props['fc'] / 1e6), # MPa 단위로 정규화
        beam_props['Fy'] / 1e6,      # MPa 단위로 정규화
        beam_props['cover'], 
        beam_props['rebar_Area'], 
        beam_props['num_bars_top'], beam_props['num_bars_bot']
    ]

    for ele_tag, (node_i, node_j) in model_nodes_info['all_line_elements'].items():
        edge_list.append([node_i - 1, node_j - 1])
        edge_list.append([node_j - 1, node_i - 1])

        if ele_tag in column_tags:
            attr = col_attr_list
        elif ele_tag in beam_tags:
            attr = beam_attr_list
        else:
            attr = [0.0] * 10

        edge_attributes.append(attr)
        edge_attributes.append(attr)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    from torch_geometric.data import Data
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.randn(100))
    
    return data

def process_pushover_curve(df_curve, max_roof_disp, num_points=100):
    """
    푸쉬오버 곡선을 표준화된 길이의 벡터로 가공합니다.
    """
    if df_curve is None or df_curve.empty or len(df_curve) < 2:
        return None

    standard_displacements = np.linspace(0, max_roof_disp, num_points)
    original_displacements = df_curve['Roof_Displacement_m'].values
    original_shears = df_curve['Base_Shear_N'].values
    standardized_shears = np.interp(standard_displacements, original_displacements, original_shears)
    
    return torch.tensor(standardized_shears, dtype=torch.float)

# --- [수정] 멀티프로세싱을 위한 단일 작업 함수 ---
def process_single_combination(args):
    """
    단일 파라미터 조합에 대해 해석을 실행하고 결과를 반환합니다.
    """
    i, combo, output_root_dir, section_params_ranges = args
    num_stories, num_bays_x, num_bays_z, build_core = combo
    
    # --- [신규] 단면 및 재료 파라미터 랜덤 샘플링 ---
    col_props = {
        'dims': random.choice(section_params_ranges['col_dims_range']),
        'fc': -random.choice(section_params_ranges['fc_range']) * 1e6,
        'Fy': random.choice(section_params_ranges['Fy_range']) * 1e6,
        'cover': round(random.uniform(*section_params_ranges['cover_range']), 3),
        'rebar_Area': random.choice(section_params_ranges['rebar_Area_range']),
        'num_bars_x': random.choice(section_params_ranges['col_num_bars_range']),
        'num_bars_z': random.choice(section_params_ranges['col_num_bars_range']),
    }
    beam_props = {
        'dims': random.choice(section_params_ranges['beam_dims_range']),
        'fc': col_props['fc'], # 보와 기둥의 콘크리트 강도는 동일하다고 가정
        'Fy': col_props['Fy'], # 철근 강도도 동일하다고 가정
        'cover': round(random.uniform(*section_params_ranges['cover_range']), 3),
        'rebar_Area': random.choice(section_params_ranges['rebar_Area_range']),
        'num_bars_top': random.choice(section_params_ranges['beam_num_bars_range']),
        'num_bars_bot': random.choice(section_params_ranges['beam_num_bars_range']),
    }
    
    core_config = {}
    if build_core:
        core_config_options = [
            {'core_z_start_bay_idx': 0, 'core_x_start_bay_idx': 0, 'num_core_bays_z': 1, 'num_core_bays_x': 1},
            {'core_z_start_bay_idx': 1, 'core_x_start_bay_idx': 1, 'num_core_bays_z': 1, 'num_core_bays_x': 1},
        ]
        valid_core_configs = [
            cfg for cfg in core_config_options
            if (cfg['core_x_start_bay_idx'] + cfg['num_core_bays_x'] <= num_bays_x and
                cfg['core_z_start_bay_idx'] + cfg['num_core_bays_z'] <= num_bays_z)
        ]
        if not valid_core_configs:
            build_core = False
        else:
            core_config = random.choice(valid_core_configs)
    
    analysis_name = f"Building_{i:04d}_S{num_stories}_BX{num_bays_x}_BZ{num_bays_z}_C{build_core}"
    output_dir = Path(output_root_dir) / analysis_name

    analysis_params = {
        'analysis_name': analysis_name, 'output_dir': output_dir, 'target_drift': 0.04, 
        'num_steps': 1000, 'num_modes': 20, 'num_int_pts': 5, 'plot_z_line_index': 1,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': 6.0, 'bay_width_z': 6.0, 'story_height': 3.5,
        'build_core': build_core, **core_config,
        'seismic_zone_factor': 0.11, 'hazard_factor': 1.0, 'soil_type': 'S4',
        'skip_post_processing': True,
    }
    
    # [수정] 해석에 필요한 모든 파라미터를 common_params에 통합
    common_params = {
        'E_steel': 200e9, 'wall_thickness': 0.20, 'wall_reinf_ratio': 0.003,
        'g': 9.81, 'dead_load_pa': 5000.0, 'live_load_pa': 0.0,
        'col_dims': col_props['dims'], 'beam_dims': beam_props['dims'],
        'fc': col_props['fc'], 'Fy': col_props['Fy'], 'cover': col_props['cover'], 'rebar_Area': col_props['rebar_Area'],
        'num_bars_x': col_props['num_bars_x'], 'num_bars_z': col_props['num_bars_z'],
        'num_bars_top': beam_props['num_bars_top'], 'num_bars_bot': beam_props['num_bars_bot'],
    }

    parameters = {**analysis_params, **common_params}
    
    try:
        perf_points, model_nodes_info, df_curve = run_single_analysis_main(parameters)

        if model_nodes_info is None or df_curve is None:
            return None

        # [수정] 상세 속성 정보를 extract_graph_data에 전달
        graph_data = extract_graph_data(model_nodes_info, col_props, beam_props)
        
        max_roof_disp_actual = df_curve['Roof_Displacement_m'].max()
        target_y = process_pushover_curve(df_curve, max_roof_disp=max_roof_disp_actual)

        if target_y is None:
            return None
            
        graph_data.y = target_y

        data_file_path = output_dir / f"{analysis_name}_graph_data.pt"
        torch.save(graph_data, data_file_path)

        # [수정] 요약 정보에 상세 속성 추가
        summary_data = {
            'analysis_name': analysis_name, 'num_stories': num_stories, 'num_bays_x': num_bays_x,
            'num_bays_z': num_bays_z, 'build_core': build_core, 'data_file': str(data_file_path),
            'max_roof_disp_actual': max_roof_disp_actual,
            'peak_shear_kN': perf_points.get('peak_shear', 0) / 1000 if perf_points else 0
        }
        # col_props와 beam_props의 각 항목을 개별 열로 추가
        for k, v in col_props.items(): summary_data[f'col_{k}'] = str(v)
        for k, v in beam_props.items(): summary_data[f'beam_{k}'] = str(v)
        
        return summary_data
        
    except Exception as e:
        print(f"Error processing {analysis_name}: {e}")
        return None

# --- [수정] 메인 데이터셋 생성 함수 (멀티프로세싱 적용) ---
def generate_dataset(output_root_dir='dataset', num_samples=10, num_workers=None):
    """
    GNN 학습을 위한 데이터셋을 병렬로 생성합니다.
    """
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. 파라미터 범위 정의 ---
    # [수정] 저층 건물에 적합한 파라미터 범위로 수정
    num_stories_range = [3, 4, 5]
    num_bays_x_range = [2, 3, 4]
    num_bays_z_range = [2, 3, 4]
    build_core_options = [False]
    
    section_params_ranges = {
        'col_dims_range': [(round(w, 2), round(w, 2)) for w in np.arange(0.4, 0.61, 0.1)], # 400x400 ~ 600x600
        'beam_dims_range': [(0.3, 0.5), (0.3, 0.6), (0.4, 0.5), (0.4, 0.6)],
        'fc_range': [21, 24, 27],  # MPa (Discrete values)
        'Fy_range': [400, 500], # MPa (Discrete values)
        'cover_range': (0.04, 0.06), # m
        'rebar_Area_range': [0.000284, 0.000387, 0.000491], # D19, D22, D25
        'col_num_bars_range': [3, 4, 5],
        'beam_num_bars_range': [3, 4, 5],
    }
    
    param_combinations = list(itertools.product(
        num_stories_range, num_bays_x_range, num_bays_z_range, build_core_options
    ))
    
    if len(param_combinations) > num_samples:
        param_combinations = random.sample(param_combinations, num_samples)

    # --- 2. 멀티프로세싱 실행 ---
    tasks = [(i, combo, output_root_dir, section_params_ranges) for i, combo in enumerate(param_combinations)]
    
    if num_workers is None:
        num_workers = min(multiprocessing.cpu_count(), 8)
        
    print(f"--- Starting dataset generation for {len(tasks)} samples using {num_workers} workers ---")
    
    dataset_info = []
    # 멀티프로세싱 Pool 사용
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Generating Dataset") as pbar:
            for result in pool.imap_unordered(process_single_combination, tasks):
                if result:
                    dataset_info.append(result)
                pbar.update()

    # --- 3. 결과 요약 및 저장 ---
    if not dataset_info:
        print("\nWarning: No data was successfully generated.")
        return

    df_dataset_info = pd.DataFrame(dataset_info)
    summary_path = output_root_dir / 'dataset_summary.csv'
    df_dataset_info.to_csv(summary_path, index=False)
    
    print(f"\n--- Dataset generation complete ---")
    print(f"{len(dataset_info)} / {len(tasks)} samples successfully generated.")
    print(f"Summary saved to {summary_path}")

if __name__ == '__main__':
    # 스크립트의 현재 작업 디렉토리 문제 해결을 위해 절대 경로 사용
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.join(project_root, 'dataset_output_parallel')

    # 데이터셋 생성 실행 (샘플 수 증가)
    generate_dataset(
        output_root_dir=output_dir_abs, 
        num_samples=100, # 훈련을 위해 샘플 수를 100개로 늘림
        num_workers=None # 시스템의 CPU 코어 수에 맞춰 자동으로 워커 수 설정
    )
