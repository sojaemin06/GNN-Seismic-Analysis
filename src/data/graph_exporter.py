import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

# --- GNN 데이터 추출 및 가공을 위한 헬퍼 함수 ---
def extract_graph_data(model_nodes_info, member_props, direction):
    """
    [수정됨] 그룹화된 member_props를 사용하여 PyG 그래프 데이터를 추출합니다.
    """
    # 1. 노드 특성 (Node Features)
    node_coords_dict = model_nodes_info['all_node_coords']
    node_coords = np.array(list(node_coords_dict.values()))
    is_base_node = np.isclose(node_coords[:, 1], 0).astype(float).reshape(-1, 1)
    node_features = np.hstack([node_coords, is_base_node])
    x = torch.tensor(node_features, dtype=torch.float)

    # --- 헬퍼: 부재 위치 및 그룹 식별 ---
    story_height = model_nodes_info['story_height']
    x_coords = sorted(list(set(node_coords[:, 0])))
    z_coords = sorted(list(set(node_coords[:, 2])))
    min_x, max_x = x_coords[0], x_coords[-1]
    min_z, max_z = z_coords[0], z_coords[-1]

    def get_element_info(node_i_coord, node_j_coord):
        # 층 및 그룹 식별 (두 노드의 y좌표 평균 사용)
        y_coord = (node_i_coord[1] + node_j_coord[1]) / 2
        story = int(round(y_coord / story_height))
        story_group_idx = (story - 1) // 2
        
        # 위치 식별 (x, z 좌표 사용)
        x_coord = (node_i_coord[0] + node_j_coord[0]) / 2
        z_coord = (node_i_coord[2] + node_j_coord[2]) / 2
        
        is_exterior = np.isclose(x_coord, min_x) or np.isclose(x_coord, max_x) or \
                      np.isclose(z_coord, min_z) or np.isclose(z_coord, max_z)
        
        return story_group_idx, 'exterior' if is_exterior else 'interior'

    # 2. 엣지 인덱스 및 엣지 특성 생성
    edge_list = []
    edge_attributes = []
    column_tags = set(model_nodes_info['all_column_tags'])
    beam_tags = set(model_nodes_info['all_beam_tags'])

    for ele_tag, (node_i, node_j) in model_nodes_info['all_line_elements'].items():
        node_i_coord = node_coords_dict[node_i]
        node_j_coord = node_coords_dict[node_j]
        
        story_group, location_type = get_element_info(node_i_coord, node_j_coord)
        
        # story_group이 음수이면 지상층이 아닌 부재로 간주하고 건너뜁니다.
        if story_group < 0:
            continue
            
        edge_list.extend([[node_i - 1, node_j - 1], [node_j - 1, node_i - 1]])
        
        attr = [0.0] * 10 # 기본값
        if ele_tag in column_tags:
            dims = member_props['col_props_by_group'][story_group][location_type]
            attr = [
                1.0, 0.0, dims[0], dims[1],
                abs(member_props['fc'] / 1e6), member_props['Fy'] / 1e6, member_props['cover'],
                member_props['rebar_Area'], member_props['num_bars_x'], member_props['num_bars_z']
            ]
        elif ele_tag in beam_tags:
            dims = member_props['beam_props_by_group'][story_group][location_type]
            attr = [
                0.0, 1.0, dims[0], dims[1],
                abs(member_props['fc'] / 1e6), member_props['Fy'] / 1e6, member_props['cover'],
                member_props['rebar_Area'], member_props['num_bars_top'], member_props['num_bars_bot']
            ]
            
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