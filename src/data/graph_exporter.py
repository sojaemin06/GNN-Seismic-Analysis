import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# --- GNN 데이터 추출 및 가공을 위한 헬퍼 함수 ---
def extract_graph_data(model_nodes_info, col_props, beam_props, direction):
    """
    model_nodes_info에서 PyG (PyTorch Geometric) 그래프 데이터를 추출합니다.
    [수정] 엣지 특성에 부재 종류, 단면 크기, 재료 강도, 철근 정보 등을 모두 포함시킵니다.
    [수정] 노드 특성에 좌표와 함께 경계조건(기초) 정보를 추가합니다.
    [수정] 그래프 레벨에 해석 방향을 나타내는 전역 특성을 추가합니다.
    """
    # 1. 노드 특성 (Node Features)
    node_coords = np.array(list(model_nodes_info['all_node_coords'].values()))
    is_base_node = np.isclose(node_coords[:, 1], 0).astype(float).reshape(-1, 1)
    node_features = np.hstack([node_coords, is_base_node])
    x = torch.tensor(node_features, dtype=torch.float)

    # 2. 엣지 인덱스 및 엣지 특성 생성
    edge_list = []
    edge_attributes = []
    column_tags = set(model_nodes_info['all_column_tags'])
    beam_tags = set(model_nodes_info['all_beam_tags'])
    col_attr_list = [
        1.0, 0.0, col_props['dims'][0], col_props['dims'][1], 
        abs(col_props['fc'] / 1e6), col_props['Fy'] / 1e6, col_props['cover'], 
        col_props['rebar_Area'], col_props['num_bars_x'], col_props['num_bars_z']
    ]
    beam_attr_list = [
        0.0, 1.0, beam_props['dims'][0], beam_props['dims'][1], 
        abs(beam_props['fc'] / 1e6), beam_props['Fy'] / 1e6, beam_props['cover'], 
        beam_props['rebar_Area'], beam_props['num_bars_top'], beam_props['num_bars_bot']
    ]
    for ele_tag, (node_i, node_j) in model_nodes_info['all_line_elements'].items():
        edge_list.extend([[node_i - 1, node_j - 1], [node_j - 1, node_i - 1]])
        attr = col_attr_list if ele_tag in column_tags else beam_attr_list if ele_tag in beam_tags else [0.0] * 10
        edge_attributes.extend([attr, attr])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # 3. 전역 특성 (Global Feature) - 해석 방향
    direction_vector = torch.tensor([[1.0, 0.0]] if direction == 'X' else [[0.0, 1.0]], dtype=torch.float)

    # 4. Data 객체 생성
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, u=direction_vector, y=torch.randn(100)) # y는 placeholder
    
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
    
    # Ensure original_displacements is monotonically increasing for np.interp
    if not np.all(np.diff(original_displacements) >= 0):
        # If not, sort both arrays based on displacements
        sort_indices = np.argsort(original_displacements)
        original_displacements = original_displacements[sort_indices]
        original_shears = original_shears[sort_indices]

    standardized_shears = np.interp(standard_displacements, original_displacements, original_shears)
    
    return torch.tensor(standardized_shears, dtype=torch.float)