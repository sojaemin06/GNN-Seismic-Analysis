import openseespy.opensees as ops
import math
import sys
from pathlib import Path 

def build_model(params):
    """
    [수정됨] 그룹화된 부재 속성을 사용하여 OpenSeesPy 모델을 구축합니다.
    [수정] forceBeamColumn 요소를 사용하여 모델 안정성을 향상시킵니다.
    [수정] 부재별 철근 상세 정보(개수, 면적)를 params['col_props_by_group'] 등에서 직접 읽어오도록 개선되었습니다.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    model_info = {
        'all_node_coords': {}, 'all_line_elements': {}, 'all_shell_elements': {},
        'all_column_tags': [], 'all_beam_tags': [], 'all_shell_tags': [],
        'all_beam_tags_type2': [], 'all_beam_tags_type3': [],
        'base_nodes': [], 'master_nodes': [], 'control_node': None,
        'element_section_map': {} # [신규] 요소 태그와 단면 정보를 매핑
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

    # 재료 강도 정의 (기대강도/공칭강도 분리)
    fc = params['fc'] # 기대강도 (Pa, 음수)
    f_ck_nominal = params['f_ck_nominal'] # 공칭강도 (Pa, 음수)
    Fy = params['Fy']
    E_steel = params['E_steel']
    
    # 재료 강도 할당 (강도 자체는 기대강도 사용)
    fce, fye = fc, Fy
    
    # 콘크리트 탄성계수(E_conc) 계산 (KBC 기준, 공칭강도 기반)
    f_ck_nominal_mpa = abs(f_ck_nominal / 1e6)
    if f_ck_nominal_mpa <= 40:
        delta_f = 4.0
    elif f_ck_nominal_mpa >= 60:
        delta_f = 6.0
    else: # 40 < f_ck < 60
        delta_f = 4.0 + (f_ck_nominal_mpa - 40.0) * (6.0 - 4.0) / (60.0 - 40.0)
    
    f_cu_mpa = f_ck_nominal_mpa + delta_f
    E_conc = (8500 * (f_cu_mpa)**(1.0/3.0)) * 1e6 # Pa 단위로 변환
    G_conc = E_conc / (2 * (1 + 0.2))
    ops.uniaxialMaterial('Concrete04', 1, fce, -0.002, -0.003, E_conc, -0.1 * fce, 0.1)
    ops.uniaxialMaterial('Concrete04', 2, fce * 1.3, -0.003, -0.02, E_conc, -0.1 * (fce * 1.3), 0.1)
    
    # 철근 재료 모델 (MultiLinear: 좌굴 후 거동 상세 모델링)
    fye_expected = abs(params['Fy'])
    fye_nominal = abs(params['Fy_nominal']) # [신규] 공칭강도 분리
    e_yield_expected = fye_expected / E_steel
    e_yield_nominal = fye_nominal / E_steel

    e_buckling_start = -0.003
    e_post_buckling_end = -0.005
    e_ultimate_tension = 0.05
    e_ultimate_compression = -0.02

    # 응력-변형률 관계를 점들의 리스트로 정의
    strain_pts = [
        e_ultimate_compression,
        e_post_buckling_end,
        e_buckling_start,
        -e_yield_nominal, # 공칭강도 기준 항복
        0.0,
        e_yield_expected, # 기대강도 기준 항복
        e_ultimate_tension
    ]
    stress_pts = [
        -0.1 * fye_nominal, # 잔류강도 (공칭 기준)
        -0.1 * fye_nominal, # 좌굴 후 강도 (공칭 기준)
        -fye_nominal,       # 좌굴 시작 (공칭 기준)
        -fye_nominal,       # 압축 항복 (공칭 기준)
        0.0,
        fye_expected,       # 인장 항복 (기대강도 기준)
        fye_expected * 1.01 # 변형률 경화 (기대강도 기준)
    ]

    # 변형률을 기준으로 점들을 자동 정렬
    paired_points = sorted(zip(strain_pts, stress_pts))
    rebar_strain_points, rebar_stress_points = zip(*paired_points)
    
    # OpenSees MultiLinear 형식에 맞게 [s1, e1, s2, e2, ...] 형태의 단일 리스트로 변환
    multilinear_pts = [item for pair in paired_points for item in pair]

    ops.uniaxialMaterial('MultiLinear', 3, *multilinear_pts)
    ops.uniaxialMaterial('Elastic', 4, G_conc)

    # --- 그룹별/위치별 Fiber Section 및 단면 적분 동적 생성 ---
    section_tags = {}
    integration_tags = {}
    sec_tag_counter = 101
    integ_tag_counter = 1001
    num_int_pts = params['num_int_pts']
    
    num_story_groups = math.ceil(num_stories / 2)
    for group_idx in range(num_story_groups):
        for col_type in ['exterior', 'interior']:
            # [수정] params 구조 변경 반영: dims 뿐만 아니라 rebar 정보도 가져옴
            prop_data = params['col_props_by_group'][group_idx][col_type]
            dims = prop_data['dims']
            rebar_info = prop_data['rebar']
            
            rebar_area = rebar_info['area']
            nz = rebar_info['nz'] # Number of bars along Z-axis face (Width direction)
            nx = rebar_info['nx'] # Number of bars along X-axis face (Depth direction)

            ops.section('Fiber', sec_tag_counter, '-torsion', 4)
            y_core = dims[1]/2.0 - params['cover']; z_core = dims[0]/2.0 - params['cover']
            y_total = dims[1]/2.0; z_total = dims[0]/2.0
            
            # [Refactored] Define Core and Cover patches separately to avoid overlap
            
            # 1. Confined Core (Material 2)
            # Center rectangle
            ops.patch('rect', 2, 10, 10, -y_core, -z_core, y_core, z_core)
            
            # 2. Unconfined Cover (Material 1) - 4 Patches
            # Top Cover (+Y side)
            ops.patch('rect', 1, 10, 2, y_core, -z_total, y_total, z_total)
            # Bottom Cover (-Y side)
            ops.patch('rect', 1, 10, 2, -y_total, -z_total, -y_core, z_total)
            # Left Side Cover (-Z side, between Top/Bot)
            ops.patch('rect', 1, 2, 8, -y_core, -z_total, y_core, -z_core)
            # Right Side Cover (+Z side, between Top/Bot)
            ops.patch('rect', 1, 2, 8, -y_core, z_core, y_core, z_total)
            
            # [Refactored 2.0] Rebar Placement with strict axis definitions
            # Y-axis = Depth (Height in section), Z-axis = Width
            # nz = Number of bars along Z-axis (Width)
            # nx = Number of bars along Y-axis (Depth)
            
            # 1. Corner Bars (4 corners)
            # Top-Right (+Y, +Z)
            ops.layer('straight', 3, 1, rebar_area, y_core, z_core, y_core, z_core)
            # Top-Left (+Y, -Z)
            ops.layer('straight', 3, 1, rebar_area, y_core, -z_core, y_core, -z_core)
            # Bot-Right (-Y, +Z)
            ops.layer('straight', 3, 1, rebar_area, -y_core, z_core, -y_core, z_core)
            # Bot-Left (-Y, -Z)
            ops.layer('straight', 3, 1, rebar_area, -y_core, -z_core, -y_core, -z_core)
            
            # 2. Intermediate Bars along Z-axis (Top & Bottom faces)
            # These vary in Z, fixed Y. Using 'nz' count.
            if nz > 2:
                # Spacing along Z
                s_z = (2.0 * z_core) / (nz - 1)
                z_start = -z_core + s_z # Start from left (negative Z) + spacing
                z_end = z_core - s_z    # End at right (positive Z) - spacing
                
                # Top Face (+Y)
                ops.layer('straight', 3, nz - 2, rebar_area, y_core, z_start, y_core, z_end)
                # Bottom Face (-Y)
                ops.layer('straight', 3, nz - 2, rebar_area, -y_core, z_start, -y_core, z_end)

            # 3. Intermediate Bars along Y-axis (Left & Right faces)
            # These vary in Y, fixed Z. Using 'nx' count.
            if nx > 2:
                # Spacing along Y
                s_y = (2.0 * y_core) / (nx - 1)
                y_start = -y_core + s_y # Start from bottom (negative Y) + spacing
                y_end = y_core - s_y    # End at top (positive Y) - spacing
                
                # Right Face (+Z)
                ops.layer('straight', 3, nx - 2, rebar_area, y_start, z_core, y_end, z_core)
                # Left Face (-Z)
                ops.layer('straight', 3, nx - 2, rebar_area, y_start, -z_core, y_end, -z_core)
            
            ops.beamIntegration('Lobatto', integ_tag_counter, sec_tag_counter, num_int_pts)
            integration_tags[(group_idx, f'col_{col_type}')] = integ_tag_counter
            sec_tag_counter += 1
            integ_tag_counter += 1

        for beam_type in ['exterior', 'interior']:
            prop_data = params['beam_props_by_group'][group_idx][beam_type]
            dims = prop_data['dims']
            rebar_info = prop_data['rebar']
            
            rebar_area = rebar_info['area']
            num_top = rebar_info['top']
            num_bot = rebar_info['bot']

            ops.section('Fiber', sec_tag_counter, '-torsion', 4)
            y_core_b = dims[1]/2.0 - params['cover']; z_core_b = dims[0]/2.0 - params['cover']
            y_total_b = dims[1]/2.0; z_total_b = dims[0]/2.0

            # [Refactored] Beam Core and Cover patches
            
            # 1. Confined Core (Material 2)
            ops.patch('rect', 2, 10, 10, -y_core_b, -z_core_b, y_core_b, z_core_b)
            
            # 2. Unconfined Cover (Material 1) - 4 Patches
            # Top Cover (+Y)
            ops.patch('rect', 1, 10, 2, y_core_b, -z_total_b, y_total_b, z_total_b)
            # Bottom Cover (-Y)
            ops.patch('rect', 1, 10, 2, -y_total_b, -z_total_b, -y_core_b, z_total_b)
            # Left Side Cover (-Z)
            ops.patch('rect', 1, 2, 8, -y_core_b, -z_total_b, y_core_b, -z_core_b)
            # Right Side Cover (+Z)
            ops.patch('rect', 1, 2, 8, -y_core_b, z_core_b, y_core_b, z_total_b)
            
            # [수정] 동적 철근 개수 적용
            ops.layer('straight', 3, num_top, rebar_area, -z_core_b, y_core_b, z_core_b, y_core_b)
            ops.layer('straight', 3, num_bot, rebar_area, -z_core_b, -y_core_b, z_core_b, -y_core_b)
            
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

    ops.geomTransf('PDelta', 1, 0, 0, -1)
    ops.geomTransf('PDelta', 2, 0, 0, 1)
    ops.geomTransf('PDelta', 3, 1, 0, 0)

    ele_tag = 1
    for i in range(num_stories):
        story_group_idx = i // 2
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                is_edge_col = (j == 0 or j == num_nodes_z - 1) or (k == 0 or k == num_nodes_x - 1)
                col_type = 'exterior' if is_edge_col else 'interior'
                # [수정] dims 접근 방식 변경
                dims = params['col_props_by_group'][story_group_idx][col_type]['dims']
                
                col_integ_tag = integration_tags[(story_group_idx, f'col_{col_type}')]
                node_i, node_j = node_tags[(i, j, k)], node_tags[(i+1, j, k)]
                ops.element('forceBeamColumn', ele_tag, node_i, node_j, 1, col_integ_tag)
                model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                model_info['all_column_tags'].append(ele_tag)
                model_info['element_section_map'][ele_tag] = {'type': 'column', 'dims': dims}
                ele_tag += 1
                
                if k < num_bays_x:
                    is_ext_beam = (j == 0 or j == num_nodes_z - 1)
                    beam_type = 'exterior' if is_ext_beam else 'interior'
                    beam_integ_tag = integration_tags[(story_group_idx, f'beam_{beam_type}')]
                    dims = params['beam_props_by_group'][story_group_idx][beam_type]['dims']
                    
                    node_i, node_j = node_tags[(i+1, j, k)], node_tags[(i+1, j, k+1)]
                    ops.element('forceBeamColumn', ele_tag, node_i, node_j, 2, beam_integ_tag)
                    model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                    model_info['all_beam_tags'].append(ele_tag)
                    model_info['all_beam_tags_type3'].append(ele_tag)
                    model_info['element_section_map'][ele_tag] = {'type': 'beam', 'dims': dims}
                    ele_tag += 1
                    
                if j < num_bays_z:
                    is_ext_beam = (k == 0 or k == num_nodes_x - 1)
                    beam_type = 'exterior' if is_ext_beam else 'interior'
                    beam_integ_tag = integration_tags[(story_group_idx, f'beam_{beam_type}')]
                    dims = params['beam_props_by_group'][story_group_idx][beam_type]['dims']
                    
                    node_i, node_j = node_tags[(i+1, j, k)], node_tags[(i+1, j+1, k)]
                    ops.element('forceBeamColumn', ele_tag, node_i, node_j, 3, beam_integ_tag)
                    model_info['all_line_elements'][ele_tag] = (node_i, node_j)
                    model_info['all_beam_tags'].append(ele_tag)
                    model_info['all_beam_tags_type2'].append(ele_tag)
                    model_info['element_section_map'][ele_tag] = {'type': 'beam', 'dims': dims}
                    ele_tag += 1

    print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays (Grouped Members).")
    return model_info