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

from run_single_analysis import main as run_single_analysis_main

def extract_graph_data(model_nodes_info, col_props_list, beam_props):
    node_coords_map = model_nodes_info['all_node_coords']
    # 노드 태그를 정수형으로 변환하고 정렬하여 일관된 순서 보장
    node_tags_sorted = sorted(list(map(int, node_coords_map.keys())))
    node_map = {tag: i for i, tag in enumerate(node_tags_sorted)}
    
    coords_array = np.array([node_coords_map[tag] for tag in node_tags_sorted])
    x = torch.tensor(coords_array, dtype=torch.float)

    edge_list = []
    edge_attributes = []
    
    column_tags = set(model_nodes_info['all_column_tags'])
    story_height = model_nodes_info.get('story_height', 3.5)

    def get_col_props_for_node_y(y_coord):
        story_idx = int(round(y_coord / story_height))
        props_idx = min(story_idx, len(col_props_list) - 1)
        return col_props_list[props_idx]

    beam_attr = [
        0.0, 1.0, 
        beam_props['dims'][0], beam_props['dims'][1], 
        abs(beam_props['fc'] / 1e6), beam_props['Fy'] / 1e6,
        beam_props['cover'], beam_props['rebar_Area'], 
        beam_props['num_bars_top'], beam_props['num_bars_bot']
    ]

    for ele_tag, (node_i_tag, node_j_tag) in model_nodes_info['all_line_elements'].items():
        edge_list.append([node_map[node_i_tag], node_map[node_j_tag]])
        edge_list.append([node_map[node_j_tag], node_map[node_i_tag]])

        if ele_tag in column_tags:
            lower_node_y = min(node_coords_map[node_i_tag][1], node_coords_map[node_j_tag][1])
            col_props = get_col_props_for_node_y(lower_node_y)
            attr = [
                1.0, 0.0, 
                col_props['dims'][0], col_props['dims'][1], 
                abs(col_props['fc'] / 1e6), col_props['Fy'] / 1e6,
                col_props['cover'], col_props['rebar_Area'], 
                col_props['num_bars_x'], col_props['num_bars_z']
            ]
        else: # Beam
            attr = beam_attr

        edge_attributes.extend([attr, attr])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    from torch_geometric.data import Data
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.randn(200))

def process_pushover_curve(df_curve, max_roof_disp, num_points=100):
    if df_curve is None or df_curve.empty or len(df_curve) < 2: return None
    standard_displacements = np.linspace(0, max_roof_disp, num_points)
    original_displacements = df_curve['Roof_Displacement_m'].values
    original_shears = df_curve['Base_Shear_N'].values
    standardized_shears = np.interp(standard_displacements, original_displacements, original_shears)
    return torch.tensor(standardized_shears, dtype=torch.float)

def calculate_moment_capacity(props, is_column=False):
    phi = 0.9
    fc_prime = abs(props['fc']) / 1e6
    Fy = props['Fy'] / 1e6
    b, h = props['dims']
    d = h - props['cover']
    
    if is_column:
        As_x = props['num_bars_x'] * props['rebar_Area']
        a_x = (As_x * Fy) / (0.85 * fc_prime * b) if fc_prime > 0 and b > 0 else 0
        Mn_x = As_x * Fy * (d - a_x / 2) if a_x < d else 0
        d_z = b - props['cover']
        As_z = props['num_bars_z'] * props['rebar_Area']
        a_z = (As_z * Fy) / (0.85 * fc_prime * h) if fc_prime > 0 and h > 0 else 0
        Mn_z = As_z * Fy * (d_z - a_z / 2) if a_z < d_z else 0
        return phi * max(Mn_x, Mn_z) * 1e3
    else:
        As = props['num_bars_top'] * props['rebar_Area']
        a = (As * Fy) / (0.85 * fc_prime * b) if fc_prime > 0 and b > 0 else 0
        Mn = As * Fy * (d - a / 2) if a < d else 0
        return phi * Mn * 1e3

def process_single_combination(args):
    i, combo, output_root_dir, section_params_ranges = args
    num_stories, num_bays_x, num_bays_z, build_core = combo
    
    MAX_RETRIES = 10
    for _ in range(MAX_RETRIES):
        col_props_list = []
        for s in range(num_stories):
            dim_range = [0.7, 0.8] if s < 2 else ([0.6, 0.7] if s < 4 else [0.5, 0.6])
            col_dim_val = random.choice(dim_range)
            num_bars = random.choice([5, 6]) if col_dim_val >= 0.7 else random.choice([4, 5])
            col_props_list.append({
                'dims': (round(col_dim_val, 2), round(col_dim_val, 2)),
                'fc': -random.choice(section_params_ranges['fc_range']) * 1e6,
                'Fy': random.choice(section_params_ranges['Fy_range']) * 1e6,
                'cover': round(random.uniform(*section_params_ranges['cover_range']), 3),
                'rebar_Area': random.choice(section_params_ranges['rebar_Area_range']),
                'num_bars_x': num_bars, 'num_bars_z': num_bars,
            })
        beam_props = {
            'dims': random.choice(section_params_ranges['beam_dims_range']),
            'fc': col_props_list[0]['fc'], 'Fy': col_props_list[0]['Fy'],
            'cover': round(random.uniform(*section_params_ranges['cover_range']), 3),
            'rebar_Area': random.choice(section_params_ranges['rebar_Area_range']),
            'num_bars_top': random.choice(section_params_ranges['beam_num_bars_range']),
            'num_bars_bot': random.choice(section_params_ranges['beam_num_bars_range']),
        }
        Mn_col = calculate_moment_capacity(col_props_list[0], is_column=True)
        Mn_beam = calculate_moment_capacity(beam_props)
        if (2 * Mn_col) > (1.2 * 2 * Mn_beam): break
    else: return None

    analysis_name = f"Building_{i:04d}_S{num_stories}_BX{num_bays_x}_BZ{num_bays_z}_C{build_core}"
    output_dir = Path(output_root_dir) / analysis_name
    output_dir.mkdir(parents=True, exist_ok=True)

    parameters = {
        'analysis_name': analysis_name, 'output_dir': output_dir, 'target_drift': 0.04, 
        'num_steps': 1000, 'num_modes': 20, 'num_int_pts': 5,
        'num_bays_x': num_bays_x, 'num_bays_z': num_bays_z, 'num_stories': num_stories,
        'bay_width_x': 6.0, 'bay_width_z': 6.0, 'story_height': 3.5,
        'build_core': build_core, 'skip_post_processing': True,
        'E_steel': 200e9, 'g': 9.81, 'dead_load_pa': 5000.0, 'live_load_pa': 0.0,
        'fc': col_props_list[0]['fc'], 'Fy': col_props_list[0]['Fy'],
        'col_props_list': col_props_list, 'beam_props': beam_props,
    }
    
    try:
        model_nodes_info, results = run_single_analysis_main(parameters, directions=['X', 'Z'])
        if model_nodes_info is None or results is None: return None

        (perf_points_x, df_curve_x) = results['X']
        (perf_points_z, df_curve_z) = results['Z']

        graph_data = extract_graph_data(model_nodes_info, col_props_list, beam_props)
        max_roof_disp_x = df_curve_x['Roof_Displacement_m'].max()
        target_y_x = process_pushover_curve(df_curve_x, max_roof_disp=max_roof_disp_x)
        max_roof_disp_z = df_curve_z['Roof_Displacement_m'].max()
        target_y_z = process_pushover_curve(df_curve_z, max_roof_disp=max_roof_disp_z)
        if target_y_x is None or target_y_z is None: return None
        graph_data.y = torch.cat((target_y_x, target_y_z), dim=0)
        data_file_path = output_dir / f"{analysis_name}_graph_data.pt"
        torch.save(graph_data, data_file_path)

        return {
            'analysis_name': analysis_name, 'num_stories': num_stories,
            'max_roof_disp_x': max_roof_disp_x,
            'peak_shear_kN_x': perf_points_x.get('peak_shear', 0) / 1000 if perf_points_x else 0,
            'max_roof_disp_z': max_roof_disp_z,
            'peak_shear_kN_z': perf_points_z.get('peak_shear', 0) / 1000 if perf_points_z else 0
        }
    except Exception:
        return None

def generate_dataset(output_root_dir='dataset', num_samples=1, num_workers=None):
    output_root_dir = Path(output_root_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)
    num_stories_range = [3, 4, 5]; num_bays_x_range = [2, 3, 4]; num_bays_z_range = [2, 3, 4]
    section_params_ranges = {
        'beam_dims_range': [(0.35, 0.55), (0.4, 0.6), (0.4, 0.65)],
        'fc_range': [24, 27, 30], 'Fy_range': [400, 500], 'cover_range': (0.04, 0.06),
        'rebar_Area_range': [0.000284, 0.000387, 0.000491], 'beam_num_bars_range': [4, 5, 6],
    }
    all_combinations = list(itertools.product(num_stories_range, num_bays_x_range, num_bays_z_range, [False]))
    tasks = [(i * num_samples + j, combo, output_root_dir, section_params_ranges)
             for i, combo in enumerate(all_combinations) for j in range(num_samples)]
    if num_workers is None: num_workers = min(multiprocessing.cpu_count(), 8)
    print(f"--- Starting dataset generation for {len(tasks)} samples using {num_workers} workers ---")
    dataset_info = []
    with multiprocessing.Pool(processes=num_workers) as pool:
        with tqdm(total=len(tasks), desc="Generating Dataset") as pbar:
            for result in pool.imap_unordered(process_single_combination, tasks):
                if result: dataset_info.append(result)
                pbar.update()
    if not dataset_info:
        print("\nWarning: No data was successfully generated.")
        return
    df = pd.DataFrame(dataset_info)
    df.to_csv(output_root_dir / 'dataset_summary.csv', index=False)
    print(f"\n--- Dataset generation complete ---")
    print(f"{len(dataset_info)} / {len(tasks)} samples successfully generated.")
    print(f"Summary saved to {output_root_dir / 'dataset_summary.csv'}")

if __name__ == '__main__':
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir_abs = os.path.join(project_root, 'dataset_output_parallel')
    generate_dataset(output_root_dir=output_dir_abs, num_samples=1, num_workers=None)