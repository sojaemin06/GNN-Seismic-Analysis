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
def run_eigen_analysis(params, model_nodes):
    """
    고유치 해석을 수행하고, X, Z 방향의 최종 누적 질량 참여율을 반환합니다.
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
            return False, 0.0, 0.0
    
    if not eigenvalues or len(eigenvalues) == 0:
        print("Eigenvalue analysis failed to return values.")
        return False, 0.0, 0.0

    periods = []
    for val in eigenvalues:
        if val > 0:
            periods.append(2 * math.pi / math.sqrt(val))
        else:
            periods.append(float('inf')) # 0 또는 음의 고유치

    print(f"Periods (T1-T{num_modes}): {periods}")
    
    # [신규] 누적 질량 참여율 계산
    master_nodes = model_nodes['master_nodes']
    total_mass_x = 0.0
    total_mass_z = 0.0
    for node_tag in master_nodes:
        mass = ops.nodeMass(node_tag)
        if mass:
            total_mass_x += mass[0]
            total_mass_z += mass[2]

    if total_mass_x < 1e-9 or total_mass_z < 1e-9:
        print("Error: Total mass is zero in one or more directions.")
        return False, 0.0, 0.0

    cumulative_mpr_x = 0.0
    cumulative_mpr_z = 0.0
    
    for i in range(1, num_modes + 1):
        L_x, M_phi_x = 0.0, 0.0
        L_z, M_phi_z = 0.0, 0.0
        for node_tag in master_nodes:
            vec = ops.nodeEigenvector(node_tag, i)
            mass = ops.nodeMass(node_tag)
            if vec and mass:
                phi_x, phi_z = vec[0], vec[2]
                m_x, m_z = mass[0], mass[2]
                L_x += m_x * phi_x
                M_phi_x += m_x * phi_x * phi_x
                L_z += m_z * phi_z
                M_phi_z += m_z * phi_z * phi_z
        
        if M_phi_x > 1e-9:
            mpr_x = (L_x * L_x) / (M_phi_x * total_mass_x)
            cumulative_mpr_x += mpr_x
        if M_phi_z > 1e-9:
            mpr_z = (L_z * L_z) / (M_phi_z * total_mass_z)
            cumulative_mpr_z += mpr_z

    print(f"Cumulative Mass Participation Ratio X: {cumulative_mpr_x*100:.2f}%")
    print(f"Cumulative Mass Participation Ratio Z: {cumulative_mpr_z*100:.2f}%")

    return True, cumulative_mpr_x, cumulative_mpr_z

def run_pushover_analysis(params, model_nodes, direction='X'):
    """
    푸쉬오버 해석을 실행하고 .out 레코더 파일을 생성합니다.
    [수정] 'direction' 파라미터를 추가하여 X축 또는 Z축 해석을 지원합니다.
    """
    print(f"\nStarting Pushover Analysis for direction: {direction}...")

    # --- 1. 방향에 따른 변수 설정 ---
    if direction == 'X':
        dof = 1
        mass_idx = 0
        eigen_idx = 0
    elif direction == 'Z':
        dof = 3
        mass_idx = 2
        eigen_idx = 2
    else:
        print(f"Error: Invalid direction '{direction}'. Must be 'X' or 'Z'.")
        return False

    # --- 2. 해당 방향의 지배 모드 탐색 ---
    print(f"Finding dominant mode in {direction}-direction (DOF {dof})...")
    master_nodes = model_nodes['master_nodes']
    num_modes = params['num_modes']
    
    best_mode = -1
    max_mpr = 0.0
    
    total_mass_in_dof = sum(ops.nodeMass(node_tag)[mass_idx] for node_tag in master_nodes if ops.nodeMass(node_tag))
            
    if total_mass_in_dof < 1e-9:
        print(f"Error: Total mass in {direction}-direction is zero.")
        return False

    modal_mprs = []
    for i in range(1, num_modes + 1):
        L, M_phi = 0.0, 0.0
        for node_tag in master_nodes:
            vec = ops.nodeEigenvector(node_tag, i)
            mass = ops.nodeMass(node_tag)
            if vec and mass:
                phi = vec[eigen_idx] 
                m = mass[mass_idx]
                L += m * phi
                M_phi += m * phi * phi
        if M_phi > 1e-9: 
            mpr = (L * L) / (M_phi * total_mass_in_dof)
            modal_mprs.append(mpr)
            if mpr > max_mpr:
                max_mpr = mpr
                best_mode = i
        else:
            modal_mprs.append(0.0)

    if best_mode == -1:
        print("Error: Could not find a valid dominant mode. Defaulting to mode 1.")
        best_mode = 1
    
    print(f"Dominant {direction}-Mode Found: Mode {best_mode} (MPR_{direction} = {max_mpr*100:.2f}%)")
    print(f"{direction}-Dir MPRs (Mode 1~{num_modes}): {[f'{m*100:.1f}%' for m in modal_mprs]}")

    # --- 3. 모드 기반 하중 계산 ---
    phi_vec = [ops.nodeEigenvector(node, best_mode)[eigen_idx] if ops.nodeEigenvector(node, best_mode) else 0.0 for node in master_nodes]
    floor_masses = [ops.nodeMass(node)[mass_idx] if ops.nodeMass(node) else 0.0 for node in master_nodes]

    force_dist = [m * phi for m, phi in zip(floor_masses, phi_vec)]
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

    # --- 4. 푸쉬오버 하중 패턴 적용 ---
    ops.pattern('Plain', 2, 1)
    load_vec = [0.0] * 6
    for i, node_tag in enumerate(master_nodes):
        if i < len(force_ratios):
            load_vec[dof-1] = force_ratios[i]
            ops.load(node_tag, *load_vec)

    # --- 5. 레코더 설정 (방향에 따라 파일명 변경) ---
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    control_node = model_nodes['control_node']
    base_nodes = model_nodes['base_nodes']
    
    all_column_tags = model_nodes.get('all_column_tags', [])
    all_beam_tags = model_nodes.get('all_beam_tags', [])
    
    # 파일명에 방향 추가
    path_disp = output_dir / f"{analysis_name}_all_floor_disp_{direction}.out"
    path_base = output_dir / f"{analysis_name}_base_shear_{direction}.out"
    path_col_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation_{direction}.out"
    path_beam_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation_{direction}.out"
    path_col_forces = output_dir / f"{analysis_name}_all_col_forces_{direction}.out"
    path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele_{direction}.out"

    if all_column_tags:
        target_element_for_Mphi = all_column_tags[0]
        ops.recorder('Element', '-file', str(path_m_phi), '-time', '-ele', target_element_for_Mphi, 'section', 1, 'forceAndDeformation')
    
    ops.recorder('Node', '-file', str(path_disp), '-time', '-node', *master_nodes, '-dof', dof, 'disp')
    
    if base_nodes:
        ops.recorder('Node', '-file', str(path_base), '-time', '-node', *base_nodes, '-dof', dof, 'reaction')
    else:
        print("Error: No base nodes found to record base shear.")
        return False
        
    if all_column_tags:
        ops.recorder('Element', '-file', str(path_col_rot), '-time', '-ele', *all_column_tags, 'plasticRotation')
        ops.recorder('Element', '-file', str(path_col_forces), '-time', '-ele', *all_column_tags, 'force')
    if all_beam_tags:
        ops.recorder('Element', '-file', str(path_beam_rot), '-time', '-ele', *all_beam_tags, 'plasticRotation')

    # --- 6. 변위 제어 해석 실행 ---
    target_disp = (params['story_height'] * params['num_stories']) * params['target_drift'] 
    num_steps = params['num_steps'] 
    
    if num_steps <= 0:
        print("Error: num_steps must be greater than 0.")
        return False
        
    displacement_increment = target_disp / num_steps 
    
    ops.integrator('DisplacementControl', control_node, dof, displacement_increment)
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
                current_disp = ops.nodeDisp(control_node, dof)
                print(f"Step {i+1}/{params['num_steps']} complete. Current Disp: {current_disp:.4f} m")
            except Exception:
                print(f"Step {i+1}/{params['num_steps']} complete. (Could not get disp)")

    print("Pushover analysis finished.")
    ops.wipeAnalysis()
    ops.remove('recorders') 
    return True