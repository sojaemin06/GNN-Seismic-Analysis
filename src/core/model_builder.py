import openseespy.opensees as ops
import math
import sys
from pathlib import Path 

def build_model(params):
    """
    [수정됨] 그룹화된 부재 속성을 사용하여 OpenSeesPy 모델을 구축합니다.
    [수정] forceBeamColumn 요소를 사용하여 모델 안정성을 향상시킵니다.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    model_info = {
        'all_node_coords': {}, 'all_line_elements': {}, 'all_shell_elements': {},
        'all_column_tags': [], 'all_beam_tags': [], 'all_shell_tags': [],
        'all_beam_tags_type2': [], 'all_beam_tags_type3': [],
        'base_nodes': [], 'master_nodes': [], 'control_node': None
    }

    num_stories = params['num_stories']
    num_bays_x, num_bays_z = params['num_bays_x'], params['num_bays_z']
    bay_width_x, bay_width_z = params['bay_width_x'], params['bay_width_z']
    story_height = params['story_height']
    num_nodes_x, num_nodes_z = num_bays_x + 1, num_bays_z + 1
    model_info['story_height'] = story_height

    node_tags = {}
    for i in range(num_stories + 1):
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                node_tag = (i * num_nodes_x * num_nodes_z) + (j * num_nodes_x) + k + 1
                x, y, z = k * bay_width_x, i * story_height, j * bay_width_z
                ops.node(node_tag, x, y, z)
                node_tags[(i, j, k)] = node_tag
                model_info['all_node_coords'][node_tag] = (x, y, z)

    for j in range(num_nodes_z):
        for k in range(num_nodes_x):
            node_tag = node_tags[(0, j, k)]
            model_info['base_nodes'].append(node_tag)
            ops.fix(node_tag, 1, 1, 1, 1, 1, 1)

    fc, Fy, E_steel = params['fc'], params['Fy'], params['E_steel']
    fce, fye = fc * 1.1, Fy * 1.1
    E_conc = (4700 * math.sqrt(abs(fc / 1e6))) * 1e6
    G_conc = E_conc / (2 * (1 + 0.2))
    ops.uniaxialMaterial('Concrete04', 1, fce, -0.002, -0.004, E_conc, -0.1 * fce, 0.1)
    ops.uniaxialMaterial('Concrete04', 2, fce * 1.3, -0.003, -0.02, E_conc, -0.1 * (fce * 1.3), 0.1)
    ops.uniaxialMaterial('Steel02', 3, fye, E_steel, 0.01, 18, 0.925, 0.15)
    ops.uniaxialMaterial('Elastic', 4, G_conc)

    # --- [수정] 그룹별/위치별 Fiber Section 및 단면 적분 동적 생성 ---
    section_tags = {}
    integration_tags = {}
    sec_tag_counter = 101
    integ_tag_counter = 1001
    num_int_pts = params['num_int_pts']
    
    num_story_groups = math.ceil(num_stories / 2)
    for group_idx in range(num_story_groups):
        for col_type in ['exterior', 'interior']:
            dims = params['col_props_by_group'][group_idx][col_type]
            ops.section('Fiber', sec_tag_counter, '-torsion', 4)
            y_core = dims[1]/2.0 - params['cover']; z_core = dims[0]/2.0 - params['cover']
            ops.patch('rect', 1, 12, 12, -dims[1]/2.0, -dims[0]/2.0, dims[1]/2.0, dims[0]/2.0)
            ops.patch('rect', 2, 10, 10, -y_core, -z_core, y_core, z_core)
            ops.layer('straight', 3, params['num_bars_z'], params['rebar_Area'], -y_core, z_core, y_core, z_core)
            ops.layer('straight', 3, params['num_bars_z'], params['rebar_Area'], -y_core, -z_core, y_core, -z_core)
            ops.layer('straight', 3, params['num_bars_x'] - 2, params['rebar_Area'], -y_core, -z_core+params['cover'], -y_core, z_core-params['cover'])
            ops.layer('straight', 3, params['num_bars_x'] - 2, params['rebar_Area'], y_core, -z_core+params['cover'], y_core, z_core-params['cover'])
            
            ops.beamIntegration('Lobatto', integ_tag_counter, sec_tag_counter, num_int_pts)
            integration_tags[(group_idx, f'col_{col_type}')] = integ_tag_counter
            sec_tag_counter += 1
            integ_tag_counter += 1

        for beam_type in ['exterior', 'interior']:
            dims = params['beam_props_by_group'][group_idx][beam_type]
            ops.section('Fiber', sec_tag_counter, '-torsion', 4)
            y_core_b = dims[1]/2.0 - params['cover']; z_core_b = dims[0]/2.0 - params['cover']
            ops.patch('rect', 1, 12, 12, -dims[1]/2.0, -dims[0]/2.0, dims[1]/2.0, dims[0]/2.0)
            ops.patch('rect', 2, 10, 10, -y_core_b, -z_core_b, y_core_b, z_core_b)
            # [수정] 상부 철근과 하부 철근을 올바르게 정의 (1단 배근)
            ops.layer('straight', 3, params['num_bars_top'], params['rebar_Area'], -z_core_b, y_core_b, z_core_b, y_core_b)
            ops.layer('straight', 3, params['num_bars_bot'], params['rebar_Area'], -z_core_b, -y_core_b, z_core_b, -y_core_b)

            # [신규] 2단 배근 (선택 사항)
            if 'num_bars_top_2nd' in params and params['num_bars_top_2nd'] > 0:
                # 1단 배근에서 50mm 안쪽으로 배치
                ops.layer('straight', 3, params['num_bars_top_2nd'], params['rebar_Area'], -z_core_b, y_core_b - 0.05, z_core_b, y_core_b - 0.05)
            if 'num_bars_bot_2nd' in params and params['num_bars_bot_2nd'] > 0:
                # 1단 배근에서 50mm 안쪽으로 배치
                ops.layer('straight', 3, params['num_bars_bot_2nd'], params['rebar_Area'], -z_core_b, -y_core_b + 0.05, z_core_b, -y_core_b + 0.05)

            ops.beamIntegration('Lobatto', integ_tag_counter, sec_tag_counter, num_int_pts)
            integration_tags[(group_idx, f'beam_{beam_type}')] = integ_tag_counter
            sec_tag_counter += 1
            integ_tag_counter += 1

    total_width_x = num_bays_x * bay_width_x; total_width_z = num_bays_z * bay_width_z
    cx, cz = total_width_x / 2.0, total_width_z / 2.0
    floor_mass = (params['dead_load_pa'] / params['g']) * (total_width_x * total_width_z)
    mass_moment_inertia = floor_mass * (total_width_x**2 + total_width_z**2) / 12.0
    for i in range(1, num_stories + 1):
        master_node_tag = (num_stories + i + 1) * 1000
        ops.node(master_node_tag, cx, i * story_height, cz)
        model_info['master_nodes'].append(master_node_tag)
        model_info['all_node_coords'][master_node_tag] = (cx, i * story_height, cz)
        ops.mass(master_node_tag, floor_mass, 0.0, floor_mass, 0.0, mass_moment_inertia, 0.0)
        slave_nodes = [node_tags[(i, j, k)] for j in range(num_nodes_z) for k in range(num_nodes_x)]
        ops.rigidDiaphragm(2, master_node_tag, *slave_nodes)
        ops.fix(master_node_tag, 0, 1, 0, 1, 0, 1)
    model_info['control_node'] = model_info['master_nodes'][-1]

    ops.geomTransf('PDelta', 1, 0, 0, -1)  # 기둥 (강축이 글로벌 X축을 향하도록)
    ops.geomTransf('PDelta', 2, 0, 0, 1)   # X축과 나란한 보 (강축이 글로벌 Y축을 향하도록)
    ops.geomTransf('PDelta', 3, 1, 0, 0)   # Z축과 나란한 보 (강축이 글로벌 Y축을 향하도록)

    # --- 요소 생성 (Elements) ---
    ele_tag = 1
    
    for i in range(num_stories):
        story_group_idx = i // 2
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                # 1. 기둥 생성
                is_edge_col = (j == 0 or j == num_nodes_z - 1) or (k == 0 or k == num_nodes_x - 1)
                col_type = 'exterior' if is_edge_col else 'interior'
                col_integ_tag = integration_tags[(story_group_idx, f'col_{col_type}')]
                node_i, node_j = node_tags[(i, j, k)], node_tags[(i+1, j, k)]
                ops.element('forceBeamColumn', ele_tag, node_i, node_j, 1, col_integ_tag)
                model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                model_info['all_column_tags'].append(ele_tag)
                ele_tag += 1
                
                # 2. 보 생성 (X축과 나란함)
                if k < num_bays_x:
                    is_ext_beam = (j == 0 or j == num_nodes_z - 1)
                    beam_type = 'exterior' if is_ext_beam else 'interior'
                    beam_integ_tag = integration_tags[(story_group_idx, f'beam_{beam_type}')]
                    node_i, node_j = node_tags[(i+1, j, k)], node_tags[(i+1, j, k+1)]
                    ops.element('forceBeamColumn', ele_tag, node_i, node_j, 2, beam_integ_tag)
                    model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                    model_info['all_beam_tags'].append(ele_tag)
                    model_info['all_beam_tags_type3'].append(ele_tag)
                    ele_tag += 1
                    
                # 3. 보 생성 (Z축과 나란함)
                if j < num_bays_z:
                    is_ext_beam = (k == 0 or k == num_nodes_x - 1)
                    beam_type = 'exterior' if is_ext_beam else 'interior'
                    beam_integ_tag = integration_tags[(story_group_idx, f'beam_{beam_type}')]
                    node_i, node_j = node_tags[(i+1, j, k)], node_tags[(i+1, j+1, k)]
                    ops.element('forceBeamColumn', ele_tag, node_i, node_j, 3, beam_integ_tag)
                    model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                    model_info['all_beam_tags'].append(ele_tag)
                    model_info['all_beam_tags_type2'].append(ele_tag)
                    ele_tag += 1

    print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays (Grouped Members).")
    return model_info