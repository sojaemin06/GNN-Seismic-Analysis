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
    ops.uniaxialMaterial('Concrete04', 1, fce, -0.002, -0.004, E_conc, -0.1 * fce, 0.1)
    ops.uniaxialMaterial('Concrete04', 2, fce * 1.3, -0.003, -0.02, E_conc, -0.1 * (fce * 1.3), 0.1)
    # 철근 재료 모델 (이선형 + 압축부 좌굴 후 강도저하)
    fye_val = abs(fye)
    e_yield = fye_val / E_steel
    strain_hardening_ratio = 0.01 # 기존 Steel02 모델의 b 값

    # '내진성능 평가요령' 지침에 따른 변형률 한계
    e_ultimate_tension = 0.05
    e_ultimate_compression = -0.02
    e_buckling_start = -0.003

    # 인장부 정의 (변형률 경화 고려)
    stress_t1 = fye_val
    strain_t1 = e_yield
    # 항복 이후 극한 변형률까지 선형으로 경화한다고 가정
    stress_t2 = fye_val * (1 + strain_hardening_ratio) 
    strain_t2 = e_ultimate_tension

    # 압축부 정의 (좌굴 고려)
    stress_c1 = -fye_val
    strain_c1 = -e_yield
    stress_c2 = -fye_val # 항복 후 좌굴 변형률까지 강도 유지
    strain_c2 = e_buckling_start
    stress_c3 = -0.1 * fye_val # 좌굴 발생 후 강도 10%로 저하
    strain_c3 = e_buckling_start * 1.01 # 수치적 안정을 위해 약간의 변형률 증가
    stress_c4 = -0.1 * fye_val
    strain_c4 = e_ultimate_compression
    
    # OpenSees 'MultiLinear' 재료 정의
    # 주의: MultiLinear는 이력거동(Hysteresis)을 모델링하지 않으며 단조하중(Pushover)에 적합합니다.
    strains = [strain_c4, strain_c3, strain_c2, strain_c1, 0.0, strain_t1, strain_t2]
    stresses = [stress_c4, stress_c3, stress_c2, stress_c1, 0.0, stress_t1, stress_t2]
    ops.uniaxialMaterial('MultiLinear', 3, '-strain', *strains, '-stress', *stresses)
    ops.uniaxialMaterial('Elastic', 4, G_conc)

    # --- [수정] 그룹별/위치별 Fiber Section 및 단면 적분 동적 생성 ---
    section_tags = {}
    integration_tags = {}
    section_recorder_paths = {} # 섹션별 변형률 기록 파일 경로 저장
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
            
            recorder_filename = params['output_dir'] / f"sec_{sec_tag_counter}_maxStrain.out"
            ops.recorder('Section', '-file', str(recorder_filename), '-sec', sec_tag_counter, 'maxStrain')
            section_recorder_paths[sec_tag_counter] = str(recorder_filename)
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

            recorder_filename = params['output_dir'] / f"sec_{sec_tag_counter}_maxStrain.out"
            ops.recorder('Section', '-file', str(recorder_filename), '-sec', sec_tag_counter, 'maxStrain')
            section_recorder_paths[sec_tag_counter] = str(recorder_filename)
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

    model_info['section_recorder_paths'] = section_recorder_paths # 섹션별 Recorder 파일 경로 추가
    print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays (Grouped Members).")
    return model_info