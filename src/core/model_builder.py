import openseespy.opensees as ops
import math
import sys
from pathlib import Path 

def build_model(params):
    """
    OpenSeesPy 모델을 구축합니다.
    [수정] Matplotlib 및 소성힌지 플로팅을 위해 모든 노드/요소 정보를 Python 변수에 저장합니다.
    """
    ops.wipe()
    ops.model('basic', '-ndm', 3, '-ndf', 6)

    # --- [수정] Matplotlib 및 힌지 플로팅을 위한 기하정보 저장용 변수 ---
    all_node_coords = {}    # { node_tag: (x, y, z) }
    all_line_elements = {}  # { ele_tag: (node_i, node_j) }
    all_shell_elements = {} # { ele_tag: (n1, n2, n3, n4) }
    
    # [신규] 힌지 플로팅을 위한 요소 태그 분류
    all_column_tags = []
    all_beam_tags = []
    all_shell_tags = []
    
    # [신규] 힌지 애니메이션을 위한 보 태그 분류 (축 방향 기준)
    all_beam_tags_type2 = [] # Z-dir (X축과 나란함, ops.geomTransf(2, ...))
    all_beam_tags_type3 = [] # X-dir (Z축과 나란함, ops.geomTransf(3, ...))
    # --- [수정] 끝 ---

    # --- Geometry 파라미터 ---
    num_bays_x = params['num_bays_x']
    num_bays_z = params['num_bays_z']
    num_nodes_x = num_bays_x + 1
    num_nodes_z = num_bays_z + 1
    
    bay_width_x = params['bay_width_x']
    bay_width_z = params['bay_width_z']
    story_height = params['story_height']
    num_stories = params['num_stories']
    
    build_core = params.get('build_core', True) 
    
    cz_start_node = 0
    cx_start_node = 0
    cz_end_node = 0
    cx_end_node = 0
    
    if build_core:
        cz_start_bay = params['core_z_start_bay_idx']
        cx_start_bay = params['core_x_start_bay_idx']
        num_core_bays_z = params['num_core_bays_z']
        num_core_bays_x = params['num_core_bays_x']

        if (cx_start_bay < 0 or (cx_start_bay + num_core_bays_x) > num_bays_x or
            cz_start_bay < 0 or (cz_start_bay + num_core_bays_z) > num_bays_z):
            
            print(f"---! Error: Core bay indices are out of bounds !---")
            sys.exit()

        cz_start_node = cz_start_bay
        cx_start_node = cx_start_bay
        cz_end_node = cz_start_node + num_core_bays_z
        cx_end_node = cx_start_node + num_core_bays_x
        
    else:
        print("Info: 'build_core' is False. Building a moment frame model without a shear core.")
        

    node_tags = {} 
    
    # --- 4x4 그리드 노드 생성 ---
    for i in range(num_stories + 1): # 층 (Y축)
        for j in range(num_nodes_z): # Z축 그리드
            for k in range(num_nodes_x): # X축 그리드
                node_tag = (i * num_nodes_x * num_nodes_z) + (j * num_nodes_x) + k + 1
                x = k * bay_width_x
                y = i * story_height
                z = j * bay_width_z
                ops.node(node_tag, x, y, z)
                node_tags[(i, j, k)] = node_tag
                
                all_node_coords[node_tag] = (x, y, z) # 좌표 저장

    # --- 경계 조건 (Boundary Conditions) ---
    base_nodes = []
    for j in range(num_nodes_z):
        for k in range(num_nodes_x):
            node_tag = node_tags[(0, j, k)]
            base_nodes.append(node_tag)
            ops.fix(node_tag, 1, 1, 1, 1, 1, 1) 

    # --- 재료 (Materials) ---
    fc = params['fc']
    f_c_mpa = abs(fc / 1e6)
    E_conc = (4700 * math.sqrt(f_c_mpa)) * 1e6 # Pa
    Fy = params['Fy']; E_steel = params['E_steel']
    G_conc = E_conc / (2 * (1 + 0.2)) # 전단 탄성 계수

    # [수정] 평가요령에 따른 기대강도(Expected Strength) 적용
    # fck > 21MPa and fck <= 40MPa 이므로 보정계수 1.1 적용
    # Fy >= 400MPa and Fy < 500MPa 이므로 보정계수 1.1 적용
    fce = fc * 1.1
    fye = Fy * 1.1
    
    ops.uniaxialMaterial('Concrete04', 1, fce, -0.002, -0.004, E_conc, -0.1 * fce, 0.1) # 비구속
    ops.uniaxialMaterial('Concrete04', 2, fce * 1.3, -0.003, -0.02, E_conc, -0.1 * (fce * 1.3), 0.1) # 구속
    ops.uniaxialMaterial('Steel02', 3, fye, E_steel, 0.01, 18, 0.925, 0.15)
    ops.uniaxialMaterial('Elastic', 4, G_conc) # 비틀림 (Torsion)

    # --- 섬유 단면 (Fiber Sections) ---
    # [수정] 파라미터화된 단면 속성 가져오기
    col_width = params['col_dims'][0]; col_depth = params['col_dims'][1]
    beam_width = params['beam_dims'][0]; beam_depth = params['beam_dims'][1]
    cover = params['cover']; As = params['rebar_Area'] # generate_dataset.py와 키 이름 일치
    num_bars_col_x = params['num_bars_x']
    num_bars_col_z = params['num_bars_z']
    num_bars_beam_top = params['num_bars_top']
    num_bars_beam_bot = params['num_bars_bot']

    core_fib_y = 10; core_fib_z = 10; cover_layers = 1
    total_fib_y = core_fib_y + (2 * cover_layers); total_fib_z = core_fib_z + (2 * cover_layers)
    
    ops.section('Fiber', 101, '-torsion', 4) # 기둥 단면
    y_core = col_depth/2.0 - cover; z_core = col_width/2.0 - cover
    ops.patch('rect', 1, total_fib_y, total_fib_z, -col_depth/2.0, -col_width/2.0, col_depth/2.0, col_width/2.0)
    ops.patch('rect', 2, core_fib_y, core_fib_z, -y_core, -z_core, y_core, z_core)
    
    # [수정] 기둥 철근 배근을 파라미터화 (기존 버그 수정 포함)
    # Note: 이 배근은 모서리 철근이 중복 계산될 수 있는 간소화된 방식입니다.
    # y축과 평행한 면의 철근 (상부/하부 철근에 해당)
    ops.layer('straight', 3, num_bars_col_x, As, -y_core, z_core, y_core, z_core)
    ops.layer('straight', 3, num_bars_col_x, As, -y_core, -z_core, y_core, -z_core)
    # z축과 평행한 면의 철근 (좌측/우측 철근에 해당, 단부 철근 제외)
    ops.layer('straight', 3, num_bars_col_z - 2, As, -y_core, -z_core+cover, -y_core, z_core-cover)
    ops.layer('straight', 3, num_bars_col_z - 2, As, y_core, -z_core+cover, y_core, z_core-cover)
    
    ops.section('Fiber', 102, '-torsion', 4) # 보 단면
    y_core_b = beam_depth/2.0 - cover; z_core_b = beam_width/2.0 - cover
    ops.patch('rect', 1, total_fib_y, total_fib_z, -beam_depth/2.0, -beam_width/2.0, beam_depth/2.0, beam_width/2.0)
    ops.patch('rect', 2, core_fib_y, core_fib_z, -y_core_b, -z_core_b, y_core_b, z_core_b)
    
    # [수정] 보 철근 배근을 파라미터화
    # 상부근
    ops.layer('straight', 3, num_bars_beam_top, As, y_core_b, -z_core_b, y_core_b, z_core_b)
    # 하부근
    ops.layer('straight', 3, num_bars_beam_bot, As, -y_core_b, -z_core_b, -y_core_b, z_core_b)

    # --- 쉘 단면 (Shell Section - 전단벽) ---
    wall_thickness = params['wall_thickness']
    ft = 0.1 * abs(fce); Gf = 10000.0; Gc = 20000.0
    ops.nDMaterial('PlasticDamageConcretePlaneStress', 11, E_conc, 0.2, abs(fce), ft, Gc, Gf)
    ops.nDMaterial('PlateFromPlaneStress', 111, 11, G_conc)
    ops.nDMaterial('PlateRebar', 12, 3, 0.0); ops.nDMaterial('PlateRebar', 13, 3, 90.0)
    reinf_thick = wall_thickness * params['wall_reinf_ratio']; core_thick = wall_thickness - 2 * reinf_thick
    ops.section('LayeredShell', 201, 3, 12, reinf_thick, 111, core_thick, 13, reinf_thick)

    # --- 강체 다이어프램 & 질량 (Rigid Diaphragm & Masses) ---
    total_width_x = num_bays_x * bay_width_x
    total_width_z = num_bays_z * bay_width_z
    cx = total_width_x / 2.0; cz = total_width_z / 2.0
    g = params['g']; dead_load_pa = params['dead_load_pa']; live_load_pa = params['live_load_pa']
    floor_area = total_width_x * total_width_z
    mass_per_area = dead_load_pa / g
    floor_mass = mass_per_area * floor_area
    mass_moment_inertia = floor_mass * (total_width_x**2 + total_width_z**2) / 12.0

    master_nodes = []
    for i in range(1, num_stories + 1):
        master_node_tag = (num_stories + i + 1) * 1000 
        y = i * story_height
        ops.node(master_node_tag, cx, y, cz) # 마스터 노드 (층의 중심)
        master_nodes.append(master_node_tag)
        all_node_coords[master_node_tag] = (cx, y, cz) # 좌표 저장
        
        ops.mass(master_node_tag, floor_mass, 0.0, floor_mass, 0.0, mass_moment_inertia, 0.0)
        
        slave_nodes = []
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                # [수정] 다이어프램 정의 변경: 코어 노드도 다이어프램에 포함시킴
                # (이전 로직은 코어 노드를 제외했으나, 이는 일반적이지 않음)
                slave_nodes.append(node_tags[(i, j, k)])
        
        ops.rigidDiaphragm(2, master_node_tag, *slave_nodes) 
        ops.fix(master_node_tag, 0, 1, 0, 1, 0, 1) # 마스터 노드의 수직/회전 자유도 구속

    # --- 요소 생성 (Elements - Frame & Core) ---
    ops.geomTransf('PDelta', 1, 0, 0, 1)  # 기둥 (Y축 방향)
    ops.geomTransf('PDelta', 2, 0, 1, 0)  # 보 (X-dir, Z축과 나란함)
    ops.geomTransf('PDelta', 3, 0, 1, 0)  # 보 (Z-dir, X축과 나란함)
    
    num_int_pts = params['num_int_pts']
    ops.beamIntegration('Lobatto', 1, 101, num_int_pts) # 기둥 적분
    ops.beamIntegration('Lobatto', 2, 102, num_int_pts) # 보 적분

    ele_tag = 1 
    shell_ele_tag = (num_stories + 1) * 100 
    
    for i in range(num_stories):
        for j in range(num_nodes_z):
            for k in range(num_nodes_x):
                
                # 1. 기둥 생성
                is_core_node = False
                if build_core:
                    is_core_node = (j >= cz_start_node and j <= cz_end_node) and \
                                   (k >= cx_start_node and k <= cx_end_node)
                
                if not is_core_node:
                    node_i = node_tags[(i, j, k)]
                    node_j = node_tags[(i+1, j, k)]
                    ops.element('nonlinearBeamColumn', ele_tag, node_i, node_j, num_int_pts, 101, 1)
                    
                    all_line_elements[ele_tag] = (node_i, node_j) 
                    all_column_tags.append(ele_tag)
                    
                    ele_tag += 1
                
                # 2. 보 생성 (Z-방향, X축과 나란함) -> Type 3
                if k < num_bays_x:
                    node_i = node_tags[(i+1, j, k)]
                    node_j = node_tags[(i+1, j, k+1)]
                    ops.element('nonlinearBeamColumn', ele_tag, node_i, node_j, num_int_pts, 102, 3) # Transf 3

                    all_line_elements[ele_tag] = (node_i, node_j) 
                    all_beam_tags.append(ele_tag)
                    all_beam_tags_type3.append(ele_tag) # [신규] Type 3

                    ele_tag += 1
                    
                # 3. 보 생성 (X-방향, Z축과 나란함) -> Type 2
                if j < num_bays_z:
                    node_i = node_tags[(i+1, j, k)]
                    node_j = node_tags[(i+1, j+1, k)]
                    ops.element('nonlinearBeamColumn', ele_tag, node_i, node_j, num_int_pts, 102, 2) # Transf 2

                    all_line_elements[ele_tag] = (node_i, node_j)
                    all_beam_tags.append(ele_tag)
                    all_beam_tags_type2.append(ele_tag) # [신규] Type 2

                    ele_tag += 1

        # 4. 중앙 전단벽 코어 생성 (Shell 요소, build_core=True일 때만)
        if build_core:
            n1 = node_tags[(i, cz_start_node, cx_start_node)]; n5 = node_tags[(i+1, cz_start_node, cx_start_node)]
            n2 = node_tags[(i, cz_start_node, cx_end_node)];   n6 = node_tags[(i+1, cz_start_node, cx_end_node)]
            n3 = node_tags[(i, cz_end_node, cx_start_node)];   n7 = node_tags[(i+1, cz_end_node, cx_start_node)]
            n4 = node_tags[(i, cz_end_node, cx_end_node)];     n8 = node_tags[(i+1, cz_end_node, cx_end_node)]

            ops.element('ShellMITC4', shell_ele_tag, n1, n3, n7, n5, 201); # Wall 1 (Z-dir)
            all_shell_elements[shell_ele_tag] = (n1, n3, n7, n5) 
            all_shell_tags.append(shell_ele_tag);
            shell_ele_tag += 1
            
            ops.element('ShellMITC4', shell_ele_tag, n2, n4, n8, n6, 201); # Wall 2 (Z-dir)
            all_shell_elements[shell_ele_tag] = (n2, n4, n8, n6)
            all_shell_tags.append(shell_ele_tag);
            shell_ele_tag += 1
            
            ops.element('ShellMITC4', shell_ele_tag, n1, n2, n6, n5, 201); # Wall 3 (X-dir)
            all_shell_elements[shell_ele_tag] = (n1, n2, n6, n5)
            all_shell_tags.append(shell_ele_tag);
            shell_ele_tag += 1
            
            ops.element('ShellMITC4', shell_ele_tag, n3, n4, n8, n7, 201); # Wall 4 (X-dir)
            all_shell_elements[shell_ele_tag] = (n3, n4, n8, n7)
            all_shell_tags.append(shell_ele_tag);
            shell_ele_tag += 1

    if build_core:
        print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays + Core at (Z_Bay:{cz_start_bay}, X_Bay:{cx_start_bay}).")
    else:
        print(f"Model built successfully with {num_bays_x}x{num_bays_z} Bays (No Core).")
        
    return {
        'base_nodes': base_nodes,
        'master_nodes': master_nodes,
        'control_node': master_nodes[-1], 
        
        'all_node_coords': all_node_coords,
        'all_line_elements': all_line_elements,   
        'all_shell_elements': all_shell_elements, 
        
        'all_column_tags': all_column_tags,
        'all_beam_tags': all_beam_tags,
        'all_shell_tags': all_shell_tags,
        
        'all_beam_tags_type2': all_beam_tags_type2,
        'all_beam_tags_type3': all_beam_tags_type3
    }