import openseespy.opensees as ops
import math
import numpy as np

# ### 3. 모듈형 함수: 중력 해석 ###
def run_gravity_analysis(params):
    """
    중력 하중을 재하하고 정적 해석을 수행합니다.
    """
    print("\nRunning Gravity Analysis...")
    
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)

    num_bays_x = params['num_bays_x']
    num_bays_z = params['num_bays_z']
    num_nodes_x = num_bays_x + 1
    num_nodes_z = num_bays_z + 1
    total_width_x = num_bays_x * params['bay_width_x']
    total_width_z = num_bays_z * params['bay_width_z']
    floor_area = total_width_x * total_width_z
    
    gravity_load_pa = params['dead_load_pa'] + 0.25 * params['live_load_pa']
    floor_gravity_load = gravity_load_pa * floor_area
    
    nodal_gravity_load = -floor_gravity_load / (num_nodes_x * num_nodes_z) 

    num_stories = params['num_stories']
    for i in range(1, num_stories + 1):
        slave_nodes_on_floor = [ (i * num_nodes_x * num_nodes_z) + (j * num_nodes_x) + k + 1 
                                for j in range(num_nodes_z) for k in range(num_nodes_x) ]
        for node_tag in slave_nodes_on_floor:
            ops.load(node_tag, 0.0, nodal_gravity_load, 0.0, 0.0, 0.0, 0.0)

    # 해석 설정
    ops.constraints('Transformation')
    ops.numberer('RCM')
    ops.system('BandGeneral')
    ops.test('NormDispIncr', 1.0e-6, 2000, 0)
    ops.algorithm('KrylovNewton')
    ops.integrator('LoadControl', 0.1) 
    ops.analysis('Static')
    
    ok = ops.analyze(10)
    
    if ok != 0:
        print("Gravity analysis failed. Trying Newton algorithm...")
        ops.algorithm('Newton')
        ok = ops.analyze(10)
        if ok != 0:
            print("Gravity analysis failed completely.")
            return False

    print("Gravity analysis complete.")
    ops.loadConst('-time', 0.0)
    return True

# ### 4. 모듈형 함수: 고유치 해석 ###
def run_eigen_analysis(params):
    """
    고유치 해석을 수행하고 주기를 출력합니다.
    """
    print("\nRunning Eigenvalue Analysis...")
    num_modes = params['num_modes']
    
    try:
        eigenvalues = ops.eigen(num_modes)
    except Exception as e:
        print(f"Eigenvalue analysis failed. Error: {e}")
        print("Trying with 'fullGenLapack' solver...")
        try:
            eigenvalues = ops.eigen('-fullGenLapack', num_modes)
        except Exception as e2:
            print(f"Eigenvalue analysis failed again. Error: {e2}")
            return False
    
    if not eigenvalues or len(eigenvalues) == 0:
        print("Eigenvalue analysis failed to return values.")
        return False

    periods = []
    for val in eigenvalues:
        if val > 0:
            periods.append(2 * math.pi / math.sqrt(val))
        else:
            periods.append(float('inf')) # 0 또는 음의 고유치

    print(f"Eigenvalues: {eigenvalues}")
    print(f"Periods (T1-T{num_modes}): {periods}")
    
    print("\nModal Properties (Mass Participation Ratios):")
    try:
        ops.modalProperties('-print')
    except Exception as e:
        print(f"Could not print modal properties: {e}")
    
    return True

# ### 5. 모듈형 함수: 푸쉬오버 해석 실행 ###
def run_pushover_analysis(params, model_nodes):
    """
    푸쉬오버 해석을 실행하고 .out 레코더 파일을 생성합니다.
    [수정] 모든 기둥/보의 'plasticRotation'과 모든 쉘의 'forces'를 기록하도록 변경합니다.
    [수정] X방향 질량참여율이 가장 높은 모드를 찾아 해당 모드 형상으로 하중을 가력합니다.
    [신규] M-Phi 관계를 기록하기 위한 특정 요소 레코더를 추가합니다.
    """
    print("\nStarting Pushover Analysis...")
    
    # --- [수정] 1. X방향 지배 모드 탐색 ---
    print("Finding dominant mode in X-direction (DOF 1)...")
    master_nodes = model_nodes['master_nodes']
    num_modes = params['num_modes']
    
    best_mode = -1
    max_mpr_x = 0.0
    
    total_mass = [0,0,0,0,0,0]
    for node_tag in master_nodes:
        mass = ops.nodeMass(node_tag) 
        if mass:
            total_mass[0] += mass[0] # m_x
            
    if total_mass[0] < 1e-9:
        print("Error: Total mass in X-direction is zero.")
        return False

    modal_mpr_x = [] 
    for i in range(1, num_modes + 1):
        L_x = 0.0
        M_phi_x = 0.0
        for node_tag in master_nodes:
            vec = ops.nodeEigenvector(node_tag, i)
            mass = ops.nodeMass(node_tag)
            if vec and mass:
                phi_x = vec[0] 
                m_x = mass[0]
                L_x += m_x * phi_x
                M_phi_x += m_x * phi_x * phi_x
        if M_phi_x > 1e-9: 
            mpr_x = (L_x * L_x) / (M_phi_x * total_mass[0])
            modal_mpr_x.append(mpr_x)
            if mpr_x > max_mpr_x:
                max_mpr_x = mpr_x
                best_mode = i
        else:
            modal_mpr_x.append(0.0)

    if best_mode == -1:
        print("Error: Could not find a valid dominant mode. Defaulting to mode 1.")
        best_mode = 1
    
    print(f"Dominant X-Mode Found: Mode {best_mode} (MPR_X = {max_mpr_x*100:.2f}%)")
    print(f"X-Dir MPRs (Mode 1~{num_modes}): {[f'{m*100:.1f}%' for m in modal_mpr_x]}")

    # --- [수정] 2. 모드 기반 하중 계산 (찾아낸 'best_mode' 사용) ---
    phi_x_vec = []
    floor_masses = []
    for node_tag in master_nodes:
        eigenvector = ops.nodeEigenvector(node_tag, best_mode) 
        phi_x_vec.append(eigenvector[0] if eigenvector else 0.0)
        mass = ops.nodeMass(node_tag)
        floor_masses.append(mass[0] if mass else 0.0)

    force_dist = [m * phi for m, phi in zip(floor_masses, phi_x_vec)]
    if not force_dist or sum(abs(f) for f in force_dist) < 1e-9:
        print("Warning: Dominant mode shape is zero or invalid. Using mass-proportional load.")
        total_mass_sum = sum(floor_masses)
        if total_mass_sum == 0:
             print("Error: Total mass is zero. Cannot apply pushover load.")
             return False
        force_ratios = [m / total_mass_sum for m in floor_masses]
    else:
        total_dist = sum(force_dist)
        force_ratios = [f / total_dist for f in force_dist]
        
    print(f"Pushover Force Ratios (based on Mode {best_mode}): {force_ratios}")

    # --- 3. 푸쉬오버 하중 패턴 적용 ---
    ops.pattern('Plain', 2, 1)
    for i, node_tag in enumerate(master_nodes):
        if i < len(force_ratios):
            ops.load(node_tag, force_ratios[i], 0.0, 0.0, 0.0, 0.0, 0.0)

    # --- 4. 레코더 설정 ---
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    control_node = model_nodes['control_node']
    base_nodes = model_nodes['base_nodes']
    
    all_column_tags = model_nodes.get('all_column_tags', [])
    all_beam_tags = model_nodes.get('all_beam_tags', [])
    all_shell_tags = model_nodes.get('all_shell_tags', [])
    
    path_disp = output_dir / f"{analysis_name}_all_floor_disp.out"
    path_base = output_dir / f"{analysis_name}_base_shear.out"
    path_wall_forces = output_dir / f"{analysis_name}_all_wall_forces.out"
    path_col_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation.out"
    path_beam_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation.out"
    path_col_forces = output_dir / f"{analysis_name}_all_col_forces.out"
    
    # --- [신규] M-Phi 관계 레코더 ---
    path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele.out"
    if all_column_tags:
        target_element_for_Mphi = all_column_tags[0] # 1층 첫번째 기둥
        ops.recorder('Element', '-file', str(path_m_phi), '-time', 
                     '-ele', target_element_for_Mphi, 
                     'section', 1, 'forceAndDeformation')
        print(f"Recording Moment-Curvature for target element: {target_element_for_Mphi} at IP 1")
    # --- [신규] 끝 ---

    if all_shell_tags:
        # [수정] 'forces' 대신 'material stress'와 'material strain'을 기록하여 손상 평가
        # 콘크리트(재료 11)의 응력/변형률을 각 가우스 포인트에서 기록
        ops.recorder('Element', '-file', str(path_wall_forces), '-time', '-ele', *all_shell_tags, 
                     'material', '1', 'stressAndStrain')
    elif params.get('build_core', True):
        print("Warning: No shell elements found to record (build_core=True).")
    
    ops.recorder('Node', '-file', str(path_disp), '-time', '-node', *master_nodes, '-dof', 1, 'disp')
    
    if base_nodes:
        ops.recorder('Node', '-file', str(path_base), '-time', '-node', *base_nodes, '-dof', 1, 'reaction')
    else:
        print("Error: No base nodes found to record base shear.")
        return False
        
    if all_column_tags:
        ops.recorder('Element', '-file', str(path_col_rot), '-time', 
                     '-ele', *all_column_tags, 
                     'plasticRotation')
        ops.recorder('Element', '-file', str(path_col_forces), '-time', 
                     '-ele', *all_column_tags, 
                     'force')
    else:
        print("Warning: No column elements found to record.")

    if all_beam_tags:
        ops.recorder('Element', '-file', str(path_beam_rot), '-time',
                     '-ele', *all_beam_tags,
                     'plasticRotation')
    else:
        print("Warning: No beam elements found to record.")


    # --- 5. 변위 제어 해석 실행 ---
    control_dof = 1
    target_disp = (params['story_height'] * params['num_stories']) * params['target_drift'] 
    num_steps = params['num_steps'] 
    
    if num_steps <= 0:
        print("Error: num_steps must be greater than 0.")
        return False
        
    displacement_increment = target_disp / num_steps 
    
    ops.integrator('DisplacementControl', control_node, control_dof, displacement_increment)
    ops.numberer('RCM')
    ops.system('BandGeneral')
    
    main_tolerance = 1.0e-5
    main_iterations = 1000
    fallback_tolerance = 1.0e-3
    fallback_iterations = 2000

    ops.test('NormDispIncr', main_tolerance, main_iterations, 0) 
    ops.algorithm('NewtonLineSearch')
    ops.analysis('Static')

    print(f"Running Pushover to {target_disp:.4f} m ({params['num_steps']} steps)...")
    
    print_freq = max(1, num_steps // 20) 

    for i in range(params['num_steps']):
        ok = ops.analyze(1)
        
        if ok != 0:
            print(f"\nAnalysis failed at step {i+1}. Trying fallback algorithms...")
            ops.test('NormDispIncr', fallback_tolerance, fallback_iterations, 0)
            ops.algorithm('KrylovNewton')
            ok = ops.analyze(1)
            
            if ok != 0:
                 print("Fallback 2: Trying ModifiedNewton...")
                 ops.algorithm('ModifiedNewton')
                 ok = ops.analyze(1)
            
            if ok == 0:
                print("Fallback successful. Continuing with main algorithm.")
                ops.test('NormDispIncr', main_tolerance, main_iterations, 0) 
                ops.algorithm('NewtonLineSearch')
            else:
                print(f"All fallbacks failed at step {i+1}. Stopping analysis.")
                break 
        
        if (i+1) % print_freq == 0 or (i+1) == num_steps:
            try:
                current_disp = ops.nodeDisp(control_node, control_dof)
                print(f"Step {i+1}/{params['num_steps']} complete. Current Disp: {current_disp:.4f} m")
            except Exception:
                print(f"Step {i+1}/{params['num_steps']} complete. (Could not get disp)")

    print("Pushover analysis finished.")
    ops.wipeAnalysis()
    ops.remove('recorders') 
    return True