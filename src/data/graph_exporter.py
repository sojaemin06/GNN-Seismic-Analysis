import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def extract_graph_data(model_nodes_info, params, direction):
    """
    [Updated] 4-Direction GNN Spec 반영.
    - Node: 5 features (x, y, z, is_base, mass_norm)
    - Edge: 12 features (geo, mat_norm, rebar_detail, engineered_props)
    - Global: 4 features (One-hot direction)
    """
    
    # ---------------------------------------------------------
    # 1. Node Features (dim=5)
    # ---------------------------------------------------------
    node_coords_dict = model_nodes_info['all_node_coords']
    node_mass_map = params.get('node_mass_map', {})
    
    node_ids = sorted(node_coords_dict.keys()) # Ensure deterministic order
    node_id_map = {old_id: new_id for new_id, old_id in enumerate(node_ids)}
    
    node_features_list = []
    
    # Normalization Constants
    MASS_SCALE = 100000.0 # 100 ton

    for node_id in node_ids:
        x, y, z = node_coords_dict[node_id]
        is_base = 1.0 if np.isclose(y, 0) else 0.0
        mass = node_mass_map.get(node_id, 0.0)
        
        # [Feature 0-4]: x, y, z, is_base, mass_norm
        node_features_list.append([x, y, z, is_base, mass / MASS_SCALE])

    x_tensor = torch.tensor(node_features_list, dtype=torch.float)

    # ---------------------------------------------------------
    # Helper: Element Group Identification
    # ---------------------------------------------------------
    story_height = params['story_height']
    node_coords_arr = np.array([node_coords_dict[nid] for nid in node_ids])
    x_set = sorted(list(set(node_coords_arr[:, 0])))
    z_set = sorted(list(set(node_coords_arr[:, 2])))
    min_x, max_x = x_set[0], x_set[-1]
    min_z, max_z = z_set[0], z_set[-1]
    
    def get_element_info(node_i, node_j):
        coord_i = node_coords_dict[node_i]
        coord_j = node_coords_dict[node_j]
        
        # Story Group
        y_avg = (coord_i[1] + coord_j[1]) / 2.0
        story_idx = int(round(y_avg / story_height - 0.5))
        if story_idx < 0: return -1, 'unknown'
        story_group = story_idx // 2
        
        # Location (Exterior if on boundary)
        mid_x = (coord_i[0] + coord_j[0]) / 2.0
        mid_z = (coord_i[2] + coord_j[2]) / 2.0
        tol = 1e-6
        is_ext = (np.isclose(mid_x, min_x, atol=tol) or np.isclose(mid_x, max_x, atol=tol) or
                  np.isclose(mid_z, min_z, atol=tol) or np.isclose(mid_z, max_z, atol=tol))
        return story_group, 'exterior' if is_ext else 'interior'

    # ---------------------------------------------------------
    # 2. Edge Features (dim=12)
    # ---------------------------------------------------------
    edge_list = []
    edge_attr_list = []
    
    # Normalization Constants
    FC_SCALE = 50e6 # 50 MPa
    FY_SCALE = 600e6 # 600 MPa
    AREA_SCALE = 1000.0 # m^2 -> scaled
    I_SCALE = 1e4 # m^4 -> * 10000

    for ele_tag, (node_i, node_j) in model_nodes_info['all_line_elements'].items():
        story_group, loc_type = get_element_info(node_i, node_j)
        if story_group == -1: continue
        
        u, v = node_id_map[node_i], node_id_map[node_j]
        
        # Initialize Feature Vector
        # [is_col, is_beam, w, d, fc_n, fy_n, cov, area_n, n1, n2, I_n, rho]
        feat = [0.0] * 12
        is_valid = False
        
        # Retrieve Properties
        props = None
        if ele_tag in model_nodes_info['all_column_tags']:
            if story_group in params['col_props_by_group']:
                props = params['col_props_by_group'][story_group].get(loc_type)
                feat[0] = 1.0 # is_col
        elif ele_tag in model_nodes_info['all_beam_tags']:
            if story_group in params['beam_props_by_group']:
                props = params['beam_props_by_group'][story_group].get(loc_type)
                feat[1] = 1.0 # is_beam

        if props:
            b, h = props['dims']
            rebar = props['rebar']
            
            # Basic Geometry & Material
            feat[2] = b
            feat[3] = h
            feat[4] = abs(params['fc']) / FC_SCALE
            feat[5] = params['Fy'] / FY_SCALE
            feat[6] = params['cover']
            
            # Rebar Details
            feat[7] = rebar['area'] * AREA_SCALE
            
            # Engineered Features
            Ag = b * h
            I_val = (b * h**3) / 12.0
            
            if feat[0] == 1.0: # Column
                feat[8] = rebar['nx']
                feat[9] = rebar['nz']
                total_steel = rebar['area'] * (2*(rebar['nx'] + rebar['nz']) - 4)
            else: # Beam
                feat[8] = rebar['top']
                feat[9] = rebar['bot']
                total_steel = rebar['area'] * (rebar['top'] + rebar['bot'])
            
            feat[10] = I_val * I_SCALE
            feat[11] = total_steel / Ag # rho (Reinforcement Ratio)
            
            is_valid = True

        if is_valid:
            # Undirected Graph (Add both directions)
            edge_list.append([u, v])
            edge_attr_list.append(feat)
            edge_list.append([v, u])
            edge_attr_list.append(feat)

    if not edge_list:
        print(f"Warning: No edges created for {params['analysis_name']}")
        return None

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # ---------------------------------------------------------
    # 3. Global Features (dim=4, One-hot)
    # ---------------------------------------------------------
    # direction: 'X' or 'Z'
    # Check analysis_name for pos/neg suffix (e.g., ..._X_pos)
    name_parts = params['analysis_name'].split('_')
    sign = name_parts[-1] # 'pos' or 'neg'
    
    # [X+, X-, Z+, Z-]
    global_vec = [0.0, 0.0, 0.0, 0.0]
    
    if direction == 'X':
        if sign == 'pos': global_vec[0] = 1.0
        else: global_vec[1] = 1.0
    else: # Z
        if sign == 'pos': global_vec[2] = 1.0
        else: global_vec[3] = 1.0
        
    u_tensor = torch.tensor([global_vec], dtype=torch.float)

    # ---------------------------------------------------------
    # Construct Data
    # ---------------------------------------------------------
    data = Data(x=x_tensor, edge_index=edge_index, edge_attr=edge_attr, u=u_tensor)
    
    try:
        data.validate(raise_on_error=True)
    except Exception as e:
        print(f"Validation Error: {e}")
        return None
        
    return data

def process_pushover_curve(df_curve, max_roof_disp, num_points=100, total_weight=None):
    """
    푸쉬오버 곡선 정규화:
    X축: Drift Ratio (Disp / H) -> H는 각자 다르므로 여기서는 Disp 자체를 0~max로 보간 (나중에 H로 나누는게 맞지만 일단 유지)
    Y축: Base Shear Coefficient (Base Shear / Total Weight)
    """
    if df_curve is None or df_curve.empty or len(df_curve) < 2:
        return None
    
    if total_weight is None:
        # Fallback if weight is missing (should not happen in new logic)
        print("Warning: Total weight is None. Using raw shear.")
        norm_factor = 1.0
    else:
        norm_factor = total_weight

    standard_displacements = np.linspace(0, max_roof_disp, num_points)
    original_displacements = df_curve['Roof_Displacement_m'].values
    original_shears = df_curve['Base_Shear_N'].values
    
    if not np.all(np.diff(original_displacements) >= 0):
        sort_indices = np.argsort(original_displacements)
        original_displacements = original_displacements[sort_indices]
        original_shears = original_shears[sort_indices]

    interpolated_shears = np.interp(standard_displacements, original_displacements, original_shears)
    
    # Normalize by Total Weight
    normalized_shears = interpolated_shears / norm_factor
    
    return torch.tensor(normalized_shears, dtype=torch.float)
