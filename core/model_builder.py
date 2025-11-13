import openseespy.opensees as ops
import math
import sys
from pathlib import Path 

def build_model(params):
    """
    OpenSeesPy 모델을 구축합니다.
    [수정] ele_tag가 올바르게 증가하도록 수정하고, Tapering 로직을 다시 적용합니다.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # --- 기하정보 저장용 변수 ---
    all_node_coords = {}
    all_line_elements = {}
    all_column_tags = []
    all_beam_tags = []
    
    # --- Geometry 파라미터 ---
    num_bays_x = params['num_bays_x']
    num_bays_z = params['num_bays_z']
    num_nodes_x = num_bays_x + 1
    num_nodes_z = num_bays_z + 1
    bay_width_x = params['bay_width_x']
    bay_width_z = params['bay_width_z']
    story_height = params['story_height']
    num_stories = params['num_stories']
    
    # --- 노드 생성 ---
    node_tags = {} 
    for i in range(num_stories + 1):
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                node_tag = (i * num_nodes_x * num_nodes_z) + (j * num_nodes_x) + k + 1
                x = k * bay_width_x; y = i * story_height; z = j * bay_width_z
                ops.node(node_tag, x, y, z)
                node_tags[(i, j, k)] = node_tag
                all_node_coords[node_tag] = (x, y, z)

    # --- 경계 조건 ---
    for j in range(num_nodes_z):
        for k in range(num_nodes_x):
            ops.fix(node_tags[(0, j, k)], 1, 1, 1, 1, 1, 1) 

    # --- 재료 (Materials) ---
    fc = params['fc']; Fy = params['Fy']; E_steel = params['E_steel']
    f_c_mpa = abs(fc / 1e6)
    E_conc = (4700 * math.sqrt(f_c_mpa)) * 1e6
    G_conc = E_conc / (2 * (1 + 0.2))
    fce = fc * 1.1; fye = Fy * 1.1
    
    ops.uniaxialMaterial('Concrete04', 1, fce, -0.002, -0.004, E_conc, -0.1 * fce, 0.1)
    ops.uniaxialMaterial('Concrete04', 2, fce * 1.3, -0.003, -0.02, E_conc, -0.1 * (fce * 1.3), 0.1)
    ops.uniaxialMaterial('Steel02', 3, fye, E_steel, 0.01, 18, 0.925, 0.15)
    ops.uniaxialMaterial('Elastic', 4, G_conc)

    # --- Tapering을 위한 단면 생성 ---
    col_props_list = params['col_props_list']
    beam_props = params['beam_props']
    
    col_section_tags = []
    for i, col_props in enumerate(col_props_list):
        section_tag = 101 + i
        col_section_tags.append(section_tag)
        
        ops.section('Fiber', section_tag, '-torsion', 4)
        col_width = col_props['dims'][0]; col_depth = col_props['dims'][1]
        cover = col_props['cover']; As = col_props['rebar_Area']
        num_bars_col_x = col_props['num_bars_x']; num_bars_col_z = col_props['num_bars_z']
        y_core = col_depth/2.0 - cover; z_core = col_width/2.0 - cover
        ops.patch('rect', 1, 12, 12, -col_depth/2.0, -col_width/2.0, col_depth/2.0, col_width/2.0)
        ops.patch('rect', 2, 10, 10, -y_core, -z_core, y_core, z_core)
        ops.layer('straight', 3, num_bars_col_x, As, -y_core, z_core, y_core, z_core)
        ops.layer('straight', 3, num_bars_col_x, As, -y_core, -z_core, y_core, -z_core)
        ops.layer('straight', 3, num_bars_col_z - 2, As, -y_core, -z_core+cover, -y_core, z_core-cover)
        ops.layer('straight', 3, num_bars_col_z - 2, As, y_core, -z_core+cover, y_core, z_core-cover)

    ops.section('Fiber', 201, '-torsion', 4) # 보 단면 태그
    beam_width = beam_props['dims'][0]; beam_depth = beam_props['dims'][1]
    cover = beam_props['cover']; As = beam_props['rebar_Area']
    num_bars_beam_top = beam_props['num_bars_top']; num_bars_beam_bot = beam_props['num_bars_bot']
    y_core_b = beam_depth/2.0 - cover; z_core_b = beam_width/2.0 - cover
    ops.patch('rect', 1, 12, 12, -beam_depth/2.0, -beam_width/2.0, beam_depth/2.0, beam_width/2.0)
    ops.patch('rect', 2, 10, 10, -y_core_b, -z_core_b, y_core_b, z_core_b)
    ops.layer('straight', 3, num_bars_beam_top, As, y_core_b, -z_core_b, y_core_b, z_core_b)
    ops.layer('straight', 3, num_bars_beam_bot, As, -y_core_b, -z_core_b, -y_core_b, z_core_b)

    # --- 강체 다이어프램 & 질량 ---
    total_width_x = num_bays_x * bay_width_x; total_width_z = num_bays_z * bay_width_z
    cx = total_width_x / 2.0; cz = total_width_z / 2.0
    floor_mass = (params['dead_load_pa'] / params['g']) * (total_width_x * total_width_z)
    mass_moment_inertia = floor_mass * (total_width_x**2 + total_width_z**2) / 12.0
    master_nodes = []
    for i in range(1, num_stories + 1):
        master_node_tag = (num_stories + i + 1) * 1000 
        y = i * story_height
        ops.node(master_node_tag, cx, y, cz)
        master_nodes.append(master_node_tag)
        all_node_coords[master_node_tag] = (cx, y, cz)
        ops.mass(master_node_tag, floor_mass, 0.0, floor_mass, 0.0, mass_moment_inertia, 0.0)
        slave_nodes = [node_tags[(i, j, k)] for j in range(num_nodes_z) for k in range(num_nodes_x)]
        ops.rigidDiaphragm(2, master_node_tag, *slave_nodes) 
        ops.fix(master_node_tag, 0, 1, 0, 1, 0, 1)

    # --- 요소 생성 ---
    ops.geomTransf('PDelta', 1, 0, 0, 1)
    ops.geomTransf('PDelta', 2, 0, 1, 0)
    ops.geomTransf('PDelta', 3, 0, 1, 0)
    
    num_int_pts = params['num_int_pts']
    
    col_integration_tags = []
    for i, sec_tag in enumerate(col_section_tags):
        integ_tag = 301 + i
        ops.beamIntegration('Lobatto', integ_tag, sec_tag, num_int_pts)
        col_integration_tags.append(integ_tag)
    
    ops.beamIntegration('Lobatto', 401, 201, num_int_pts) # 보 적분 태그

    ele_tag = 1 
    for i in range(num_stories):
        story_group_idx = min(i, len(col_integration_tags) - 1)
        col_integration_tag = col_integration_tags[story_group_idx]

        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                # 기둥
                ops.element('nonlinearBeamColumn', ele_tag, node_tags[(i,j,k)], node_tags[(i+1,j,k)], col_integration_tag, 1)
                all_line_elements[ele_tag] = (node_tags[(i,j,k)], node_tags[(i+1,j,k)]) 
                all_column_tags.append(ele_tag)
                ele_tag += 1
                # 보 (Z-방향)
                if k < num_bays_x:
                    ops.element('nonlinearBeamColumn', ele_tag, node_tags[(i+1,j,k)], node_tags[(i+1,j,k+1)], 401, 3)
                    all_line_elements[ele_tag] = (node_tags[(i+1,j,k)], node_tags[(i+1,j,k+1)])
                    all_beam_tags.append(ele_tag)
                    ele_tag += 1
                # 보 (X-방향)
                if j < num_bays_z:
                    ops.element('nonlinearBeamColumn', ele_tag, node_tags[(i+1,j,k)], node_tags[(i+1,j+1,k)], 401, 2)
                    all_line_elements[ele_tag] = (node_tags[(i+1,j,k)], node_tags[(i+1,j+1,k)])
                    all_beam_tags.append(ele_tag)
                    ele_tag += 1

    print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays (No Core).")
        
    return {
        'base_nodes': [node_tags[(0, j, k)] for j in range(num_nodes_z) for k in range(num_nodes_x)],
        'master_nodes': master_nodes,
        'control_node': master_nodes[-1], 
        'all_node_coords': all_node_coords,
        'all_line_elements': all_line_elements,   
        'all_column_tags': all_column_tags,
        'all_beam_tags': all_beam_tags,
        'story_height': story_height
    }