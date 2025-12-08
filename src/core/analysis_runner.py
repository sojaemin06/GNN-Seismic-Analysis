import openseespy.opensees as ops
import math
import numpy as np

# ### 3. 모듈형 함수: 중력 해석 ###
def run_gravity_analysis(params, model_nodes_info=None):
    """중력 하중을 재하하고 정적 해석을 수행합니다. (성공 여부, 총 반력 Y 반환)"""
    print("\nRunning Gravity Analysis...")
    ops.timeSeries('Linear', 1); ops.pattern('Plain', 1, 1)
    num_bays_x, num_bays_z = params['num_bays_x'], params['num_bays_z']
    num_nodes_x, num_nodes_z = num_bays_x + 1, num_bays_z + 1
    floor_area = (num_bays_x * params['bay_width_x']) * (num_bays_z * params['bay_width_z'])
    gravity_load_pa = params['dead_load_pa'] + 0.25 * params['live_load_pa']
    nodal_gravity_load = -gravity_load_pa * floor_area / (num_nodes_x * num_nodes_z)
    for i in range(1, params['num_stories'] + 1):
        slave_nodes = [(i * num_nodes_x * num_nodes_z) + (j * num_nodes_x) + k + 1 for j in range(num_nodes_z) for k in range(num_nodes_x)]
        for node_tag in slave_nodes: ops.load(node_tag, 0.0, nodal_gravity_load, 0.0, 0.0, 0.0, 0.0)
    ops.constraints('Transformation'); ops.numberer('RCM'); ops.system('UmfPack')
    ops.test('NormDispIncr', 1.0e-6, 2000, 0); ops.algorithm('KrylovNewton'); ops.integrator('LoadControl', 0.1); ops.analysis('Static')
    
    ok = ops.analyze(10)
    if ok != 0:
        print("Gravity analysis failed. Trying Newton algorithm...")
        ops.algorithm('Newton'); ok = ops.analyze(10)
        if ok != 0: print("Gravity analysis failed completely."); return False, 0.0
    
    # Calculate Total Base Reaction (Y-dir)
    ops.reactions()
    total_reaction_y = 0.0
    
    # Use base nodes from info if available, else infer (1 to num_base_nodes)
    base_nodes = []
    if model_nodes_info and 'base_nodes' in model_nodes_info:
        base_nodes = model_nodes_info['base_nodes']
    else:
        base_nodes = range(1, num_nodes_x * num_nodes_z + 1)

    for node in base_nodes:
        # Node reaction: 1=X, 2=Y, 3=Z...
        rxn = ops.nodeReaction(node)
        if rxn: total_reaction_y += rxn[1] # Add Y reaction (index 1)
        
    print(f"Gravity analysis complete. Total Base Reaction Y: {total_reaction_y:.2f} N")
    ops.loadConst('-time', 0.0)
    return True, total_reaction_y

# ### 4. 모듈형 함수: 고유치 해석 ###
def run_eigen_analysis(params, model_nodes_info, silent=False):
    """[수정] 고유치 해석을 수행하고, 계산된 모든 모드 속성(modal_props)을 반환합니다."""
    if not silent: print("\nRunning Eigenvalue Analysis...")
    num_modes = params['num_modes']
    master_nodes = model_nodes_info['master_nodes']
    
    try: eigenvalues = ops.eigen(num_modes)
    except Exception:
        if not silent: print("Default eigenvalue analysis failed. Trying with 'fullGenLapack' solver...")
        try: eigenvalues = ops.eigen('-fullGenLapack', num_modes)
        except Exception as e2:
            if not silent: print(f"Eigenvalue analysis failed completely. Error: {e2}")
            return False, None

    if not eigenvalues or len(eigenvalues) == 0:
        if not silent: print("Eigenvalue analysis failed to return values.")
        return False, None
    
    modal_props, floor_masses_x, floor_masses_z = [], [], []
    for node_tag in master_nodes:
        mass = ops.nodeMass(node_tag)
        if mass: floor_masses_x.append(mass[0]); floor_masses_z.append(mass[2])
        else: floor_masses_x.append(0); floor_masses_z.append(0)
    
    total_mass_x, total_mass_z = sum(floor_masses_x), sum(floor_masses_z)
    if total_mass_x < 1e-9 or total_mass_z < 1e-9: return False, None
    
    for i, eig_val in enumerate(eigenvalues):
        period = (2 * math.pi / math.sqrt(eig_val)) if eig_val > 0 else float('inf')
        mode_num = i + 1
        mode_shape = np.array([ops.nodeEigenvector(tag, mode_num) for tag in master_nodes])
        phi_x, phi_z = mode_shape[:, 0], mode_shape[:, 2]
        
        L_x, L_z = np.dot(floor_masses_x, phi_x), np.dot(floor_masses_z, phi_z)
        M_phi_x, M_phi_z = np.dot(floor_masses_x, phi_x**2), np.dot(floor_masses_z, phi_z**2)
        
        mpr_x = (L_x**2 / (M_phi_x * total_mass_x)) if M_phi_x * total_mass_x > 1e-9 else 0.0
        mpr_z = (L_z**2 / (M_phi_z * total_mass_z)) if M_phi_z * total_mass_z > 1e-9 else 0.0

        M_star_x, M_star_z = L_x**2 / M_phi_x if M_phi_x > 1e-9 else 0, L_z**2 / M_phi_z if M_phi_z > 1e-9 else 0
        gamma_x, gamma_z = L_x / M_phi_x if M_phi_x > 1e-9 else 0, L_z / M_phi_z if M_phi_z > 1e-9 else 0
        
        modal_props.append({
            'mode': mode_num, 'period': period, 'mpr_x': mpr_x, 'mpr_z': mpr_z,
            'gamma_x': gamma_x, 'gamma_z': gamma_z, 'phi_x': phi_x, 'phi_z': phi_z,
            'M_star_x': M_star_x, 'M_star_z': M_star_z
        })

    if not silent:
        cumulative_mpr_x = sum(p['mpr_x'] for p in modal_props)
        cumulative_mpr_z = sum(p['mpr_z'] for p in modal_props)
        print(f"Cumulative Mass Participation Ratio X: {cumulative_mpr_x*100:.2f}%")
        print(f"Cumulative Mass Participation Ratio Z: {cumulative_mpr_z*100:.2f}%")

    return True, modal_props

def run_pushover_analysis(params, model_nodes_info, modal_props, direction='X'):
    """
    [수정] 푸쉬오버 해석을 실행하고, 사용된 지배 모드의 속성을 반환합니다.
    """
    print(f"\nStarting Pushover Analysis for direction: {direction}...")
    if direction not in ['X', 'Z']: return False, None
    dof = 1 if direction == 'X' else 3

    # --- 1. 해당 방향의 지배 모드 탐색 ---
    dominant_mode = max(modal_props, key=lambda p: p[f'mpr_{direction.lower()}'])
    print(f"Dominant {direction}-Mode Found: Mode {dominant_mode['mode']} (MPR_{direction} = {dominant_mode[f'mpr_{direction.lower()}']*100:.2f}%)")

    # --- 2. 모드 기반 하중 계산 ---
    phi_vec = dominant_mode[f'phi_{direction.lower()}']
    masses = [ops.nodeMass(tag)[dof-1] if ops.nodeMass(tag) else 0.0 for tag in model_nodes_info['master_nodes']]
    force_dist = [m * p for m, p in zip(masses, phi_vec)]
    total_dist = sum(force_dist)
    if abs(total_dist) < 1e-9:
        print("Warning: Dominant mode shape is zero or invalid. Using mass-proportional load."); total_mass_sum = sum(masses)
        force_ratios = [m / total_mass_sum for m in masses] if total_mass_sum > 0 else []
    else: force_ratios = [f / total_dist for f in force_dist]
    if not force_ratios: print("Error: Cannot apply pushover load."); return False, None
    print(f"Pushover Force Ratios (based on Mode {dominant_mode['mode']}): {force_ratios}")

    # --- 3. 푸쉬오버 하중 패턴 및 레코더 설정 ---
    ops.pattern('Plain', 2, 1)
    for i, node_tag in enumerate(model_nodes_info['master_nodes']): ops.load(node_tag, *([force_ratios[i] if idx == dof-1 else 0.0 for idx in range(6)]))
    
    output_dir = params['output_dir']; analysis_name = params['analysis_name']
    control_node = model_nodes_info['control_node']; base_nodes = model_nodes_info['base_nodes']
    
    ops.recorder('Node', '-file', str(output_dir / f"{analysis_name}_all_floor_disp_{direction}.out"), '-time', '-node', *model_nodes_info['master_nodes'], '-dof', dof, 'disp')
    if base_nodes: ops.recorder('Node', '-file', str(output_dir / f"{analysis_name}_base_shear_{direction}.out"), '-time', '-node', *base_nodes, '-dof', dof, 'reaction')
    
    # [New] Recorders for Plastic Rotation (Section Deformations)
    # Record deformations for ALL sections to ensure data capture. 
    # Post-processor will handle extracting specific integration points.
    col_tags = model_nodes_info.get('all_column_tags', [])
    if col_tags:
        ops.recorder('Element', '-file', str(output_dir / f"{analysis_name}_all_col_plastic_rotation_{direction}.out"), 
                     '-time', '-ele', *col_tags, 'section', 'deformation')
                     
    beam_tags = model_nodes_info.get('all_beam_tags', [])
    if beam_tags:
        ops.recorder('Element', '-file', str(output_dir / f"{analysis_name}_all_beam_plastic_rotation_{direction}.out"), 
                     '-time', '-ele', *beam_tags, 'section', 'deformation')

    # --- 4. 변위 제어 해석 실행 ---
    target_disp = (params['story_height'] * params['num_stories']) * params['target_drift']
    ops.integrator('DisplacementControl', control_node, dof, target_disp / params['num_steps'])
    ops.numberer('RCM'); ops.system('UmfPack'); ops.test('NormDispIncr', 1.0e-4, 2000, 0); ops.algorithm('NewtonLineSearch'); ops.analysis('Static')
    
    print(f"Running Pushover to {target_disp:.4f} m ({params['num_steps']} steps)...")
    for i in range(params['num_steps']):
        if ops.analyze(1) != 0:
            print(f"\nAnalysis failed at step {i+1}. Trying fallback algorithms..."); ops.test('NormDispIncr', 1.0e-3, 2000, 0)
            if ops.algorithm('KrylovNewton') != 0 or ops.analyze(1) != 0:
                if ops.algorithm('ModifiedNewton') != 0 or ops.analyze(1) != 0:
                    print("All fallbacks failed. Stopping analysis."); break
            print("Fallback successful. Continuing..."); ops.test('NormDispIncr', 1.0e-5, 1000, 0); ops.algorithm('NewtonLineSearch')
        if (i+1) % max(1, params['num_steps'] // 20) == 0: print(f"Step {i+1}/{params['num_steps']} complete.")

    print("Pushover analysis finished."); ops.wipeAnalysis(); ops.remove('recorders')
    return True, dominant_mode
