import openseespy.opensees as ops
import math
import numpy as np
import matplotlib.pyplot as plt

def _plot_shear_verification(story_shears_multi, story_shears_first, num_stories, direction, output_path):
    """층전단력 검증 결과를 그래프로 저장합니다."""
    stories = np.arange(1, num_stories + 1)
    
    # kN 단위로 변환
    v_multi_kn = story_shears_multi / 1000
    v_first_kn = story_shears_first / 1000
    limit_kn = 1.3 * v_first_kn

    plt.figure(figsize=(8, 10))
    plt.plot(v_multi_kn, stories, 'r-o', label='Multi-Mode Shear (V_multi)')
    plt.plot(v_first_kn, stories, 'b-^', label='Dominant-Mode Shear (V_first)')
    plt.plot(limit_kn, stories, 'g--', label='130% Limit (1.3 * V_first)')

    plt.title(f'NSP Applicability Verification ({direction.upper()}-Direction)')
    plt.ylabel('Story')
    plt.xlabel('Story Shear (kN)')
    plt.grid(True)
    plt.legend()
    plt.yticks(stories)
    
    plt.savefig(output_path)
    plt.close()
    print(f"Verification plot saved to: {output_path}")

def _create_demand_spectrum(params):
    """요구 응답 스펙트럼 생성 함수를 반환합니다."""
    
    Z = params['seismic_zone_factor']
    I = params['hazard_factor']
    soil_type = params['soil_type']
    
    S = Z * I
    
    fa_map = {
        'S1': {'s<=0.1': 1.12, 's=0.2': 1.12, 's=0.3': 1.12}, 'S2': {'s<=0.1': 1.4, 's=0.2': 1.4, 's=0.3': 1.3},
        'S3': {'s<=0.1': 1.7, 's=0.2': 1.5, 's=0.3': 1.3}, 'S4': {'s<=0.1': 1.6, 's=0.2': 1.4, 's=0.3': 1.2},
        'S5': {'s<=0.1': 1.8, 's=0.2': 1.3, 's=0.3': 1.3}
    }
    fv_map = {
        'S1': {'s<=0.1': 0.84, 's=0.2': 0.84, 's=0.3': 0.84}, 'S2': {'s<=0.1': 1.5, 's=0.2': 1.4, 's=0.3': 1.3},
        'S3': {'s<=0.1': 1.7, 's=0.2': 1.6, 's=0.3': 1.5}, 'S4': {'s<=0.1': 2.2, 's=0.2': 2.0, 's=0.3': 1.8},
        'S5': {'s<=0.1': 3.0, 's=0.2': 2.7, 's=0.3': 2.4}
    }

    def interpolate(s_val, p1, v1, p2, v2):
        return v1 + (s_val - p1) * (v2 - v1) / (p2 - p1)

    def get_coeff(coeff_map, s_val):
        s_val = round(s_val, 4)
        if s_val <= 0.1: return coeff_map[soil_type]['s<=0.1']
        if s_val == 0.2: return coeff_map[soil_type]['s=0.2']
        if s_val >= 0.3: return coeff_map[soil_type]['s=0.3']
        if 0.1 < s_val < 0.2: return interpolate(s_val, 0.1, coeff_map[soil_type]['s<=0.1'], 0.2, coeff_map[soil_type]['s=0.2'])
        return interpolate(s_val, 0.2, coeff_map[soil_type]['s=0.2'], 0.3, coeff_map[soil_type]['s=0.3'])

    Fa = get_coeff(fa_map, S)
    Fv = get_coeff(fv_map, S)
    S_ds, S_d1 = S * 2.5 * Fa, S * Fv
    T0, Ts = 0.2 * S_d1 / S_ds, S_d1 / S_ds
    
    def get_sa(period):
        if period < T0: return S_ds * (0.4 + 0.6 * period / T0)
        elif T0 <= period <= Ts: return S_ds
        else: return S_d1 / period
            
    return get_sa

def _get_modal_properties(num_modes, master_nodes):
    """모드별 특성(주기, 질량참여율, 참여계수)을 계산합니다."""
    try:
        eigenvalues = ops.eigen(num_modes)
    except Exception:
        print("Default eigenvalue analysis failed. Trying with 'fullGenLapack' solver...")
        try:
            eigenvalues = ops.eigen('-fullGenLapack', num_modes)
        except Exception as e2:
            print(f"Eigenvalue analysis failed completely. Error: {e2}")
            return [], []

    if not eigenvalues:
        print("Eigenvalue analysis returned no values."); return [], []
        
    periods = [2 * math.pi / math.sqrt(val) if val > 0 else 0 for val in eigenvalues]
    
    masses = np.array([ops.nodeMass(tag) for tag in master_nodes])
    floor_masses_x = masses[:, 0]
    floor_masses_z = masses[:, 2]
    total_mass_x = np.sum(floor_masses_x)
    total_mass_z = np.sum(floor_masses_z)
    
    modal_props = []
    for i in range(len(periods)):
        mode_num = i + 1
        mode_shape = np.array([ops.nodeEigenvector(tag, mode_num) for tag in master_nodes])
        phi_x, phi_z = mode_shape[:, 0], mode_shape[:, 2]
        
        L_x, L_z = np.sum(floor_masses_x * phi_x), np.sum(floor_masses_z * phi_z)
        M_phi_x, M_phi_z = np.sum(floor_masses_x * phi_x**2), np.sum(floor_masses_z * phi_z**2)
        
        mpr_x = (L_x**2) / (M_phi_x * total_mass_x) if M_phi_x > 1e-9 else 0.0
        mpr_z = (L_z**2) / (M_phi_z * total_mass_z) if M_phi_z > 1e-9 else 0.0

        M_star = M_phi_x + M_phi_z # Simplified for 2D horizontal
        if M_star < 1e-9: continue

        gamma_x, gamma_z = L_x / M_star, L_z / M_star
        
        modal_props.append({
            'mode': mode_num, 'period': periods[i], 'mpr_x': mpr_x, 'mpr_z': mpr_z,
            'gamma_x': gamma_x, 'gamma_z': gamma_z, 'phi_x': phi_x, 'phi_z': phi_z
        })
        
    return modal_props, {'x': floor_masses_x, 'z': floor_masses_z}

def _run_rsa(modes_to_run, modal_props, floor_masses, get_sa_func, direction):
    """주어진 모드들에 대해 RSA를 수행하고 층전단력을 반환합니다."""
    num_stories = len(floor_masses[direction.lower()])
    story_shears_modal = []

    for mode_prop in modal_props:
        if mode_prop['mode'] not in modes_to_run: continue
            
        T, Sa = mode_prop['period'], get_sa_func(mode_prop['period'])
        gamma = mode_prop[f'gamma_{direction.lower()}']
        phi = mode_prop[f'phi_{direction.lower()}']
        mass = floor_masses[direction.lower()]
            
        inertia_forces = mass * phi * gamma * Sa
        modal_shears = np.array([np.sum(inertia_forces[i:]) for i in range(num_stories)])
        story_shears_modal.append(modal_shears)

    return np.sqrt(np.sum(np.square(story_shears_modal), axis=0))

def verify_nsp_applicability(params, model_nodes_info):
    """'내진성능 평가요령' 4.3.1절(3)항에 따라 비선형 정적해석(NSP) 적용 가능 여부를 검증합니다."""
    print("\n--- 비선형 정적해석(Pushover) 적용 타당성 검증 시작 ---")
    
    master_nodes, num_stories, num_modes = model_nodes_info['master_nodes'], params['num_stories'], params['num_modes']
    get_sa = _create_demand_spectrum(params)
    modal_props, floor_masses = _get_modal_properties(num_modes, master_nodes)
    if not modal_props: return False, False

    cmpr_x, cmpr_z = 0.0, 0.0
    modes_for_90p_x, modes_for_90p_z = [], []
    for prop in modal_props:
        if cmpr_x < 0.9: cmpr_x += prop['mpr_x']; modes_for_90p_x.append(prop['mode'])
        if cmpr_z < 0.9: cmpr_z += prop['mpr_z']; modes_for_90p_z.append(prop['mode'])
    
    print(f"X-dir: {len(modes_for_90p_x)} modes for {cmpr_x*100:.1f}% mass participation.")
    print(f"Z-dir: {len(modes_for_90p_z)} modes for {cmpr_z*100:.1f}% mass participation.")

    # X-Direction Verification
    shears_multi_x = _run_rsa(modes_for_90p_x, modal_props, floor_masses, get_sa, 'X')
    shears_first_x = _run_rsa([1], modal_props, floor_masses, get_sa, 'X') # 1차 모드는 항상 1번으로 가정
    
    is_valid_x = True
    print("\n[X-Direction Verification]\nStory | V_multi (kN) | V_first (kN) | Ratio | Result")
    for i in range(num_stories):
        v_m, v_f = shears_multi_x[i]/1000, shears_first_x[i]/1000
        ratio = v_m / v_f if v_f > 1e-6 else 0
        result = 'OK' if ratio <= 1.3 else 'NG'
        if ratio > 1.3: is_valid_x = False
        print(f"{i+1:^5} | {v_m:^12.2f} | {v_f:^12.2f} | {ratio:^5.2f} | {result:^6}")

    # Z-Direction Verification
    z_first_mode = max(modal_props, key=lambda p: p['mpr_z'])['mode'] if modal_props else 2
    shears_multi_z = _run_rsa(modes_for_90p_z, modal_props, floor_masses, get_sa, 'Z')
    shears_first_z = _run_rsa([z_first_mode], modal_props, floor_masses, get_sa, 'Z')
    
    is_valid_z = True
    print("\n[Z-Direction Verification]\nStory | V_multi (kN) | V_first (kN) | Ratio | Result")
    for i in range(num_stories):
        v_m, v_f = shears_multi_z[i]/1000, shears_first_z[i]/1000
        ratio = v_m / v_f if v_f > 1e-6 else 0
        result = 'OK' if ratio <= 1.3 else 'NG'
        if ratio > 1.3: is_valid_z = False
        print(f"{i+1:^5} | {v_m:^12.2f} | {v_f:^12.2f} | {ratio:^5.2f} | {result:^6}")

    if not params.get('skip_post_processing', False):
        output_dir, name = params['output_dir'], params['analysis_name']
        _plot_shear_verification(shears_multi_x, shears_first_x, num_stories, 'X', output_dir / f"{name}_NSP_verification_plot_X.png")
        _plot_shear_verification(shears_multi_z, shears_first_z, num_stories, 'Z', output_dir / f"{name}_NSP_verification_plot_Z.png")

    print("\n--- 타당성 검증 완료 ---")
    return is_valid_x, is_valid_z
