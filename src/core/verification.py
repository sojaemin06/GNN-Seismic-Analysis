import openseespy.opensees as ops
import math
import numpy as np
import matplotlib.pyplot as plt
import traceback

# --- Helper Functions ---

def _plot_shear_verification(story_shears_multi, story_shears_dominant, num_stories, direction, output_path):
    """층전단력 검증 결과를 그래프로 저장합니다."""
    stories = np.arange(1, num_stories + 1)
    v_multi_kn = story_shears_multi / 1000
    v_dominant_kn = story_shears_dominant / 1000
    limit_kn = 1.3 * v_dominant_kn

    plt.figure(figsize=(8, 10))
    plt.plot(v_multi_kn, stories, 'r-o', label='Multi-Mode Shear (V_multi)')
    plt.plot(v_dominant_kn, stories, 'b-^', label='Dominant-Mode Shear (V_dominant)')
    plt.plot(limit_kn, stories, 'g--', label='130% Limit (1.3 * V_dominant)')

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
    Z, I, soil_type = params['seismic_zone_factor'], params['hazard_factor'], params['soil_type']
    S = Z * I
    
    fa_map = {'S1': 1.12, 'S2': 1.4, 'S3': 1.5, 'S4': 1.4, 'S5': 1.3}
    fv_map = {'S1': 0.84, 'S2': 1.4, 'S3': 1.6, 'S4': 2.0, 'S5': 2.7}
    
    Fa, Fv = fa_map.get(soil_type, 1.0), fv_map.get(soil_type, 1.0)
    S_ds, S_d1 = S * 2.5 * Fa, S * Fv
    T0, Ts = 0.2 * S_d1 / S_ds, S_d1 / S_ds
    
    def get_sa(period):
        if period < T0: return S_ds * (0.4 + 0.6 * period / T0)
        return S_ds if T0 <= period <= Ts else S_d1 / period
            
    return get_sa

def _run_rsa(modes_to_run, modal_props, floor_masses, get_sa_func, direction):
    """주어진 모드들에 대해 RSA를 수행하고 층전단력을 반환합니다."""
    num_stories = len(floor_masses[direction.lower()])
    story_shears_modal = []

    dir_lower = direction.lower()
    for mode_prop in modal_props:
        if mode_prop['mode'] not in modes_to_run: continue
            
        T, Sa = mode_prop['period'], get_sa_func(mode_prop['period'])
        gamma = mode_prop[f'gamma_{dir_lower}']
        phi = mode_prop[f'phi_{dir_lower}']
        mass = floor_masses[dir_lower]
            
        inertia_forces = mass * phi * gamma * Sa
        modal_shears = np.array([np.sum(inertia_forces[i:]) for i in range(num_stories)])
        story_shears_modal.append(modal_shears)

    return np.sqrt(np.sum(np.square(story_shears_modal), axis=0)) if story_shears_modal else np.zeros(num_stories)

# --- Main Verification Function ---

def verify_nsp_applicability(params, model_nodes_info, modal_props):
    """[수정] 미리 계산된 modal_props를 사용하여 비선형 정적해석(NSP) 적용의 타당성을 검증합니다."""
    print("\n--- 비선형 정적해석(Pushover) 적용 타당성 검증 시작 ---")
    
    try:
        master_nodes, num_stories = model_nodes_info['master_nodes'], params['num_stories']
        get_sa = _create_demand_spectrum(params)

        masses = np.array([ops.nodeMass(tag) for tag in master_nodes])
        floor_masses = {'x': masses[:, 0], 'z': masses[:, 2]}

        cmpr_x, cmpr_z = 0.0, 0.0
        modes_for_90p_x, modes_for_90p_z = [], []
        for prop in modal_props:
            if cmpr_x < 0.9: cmpr_x += prop['mpr_x']; modes_for_90p_x.append(prop['mode'])
            if cmpr_z < 0.9: cmpr_z += prop['mpr_z']; modes_for_90p_z.append(prop['mode'])
        
        print(f"X-dir: {len(modes_for_90p_x)} modes for {cmpr_x*100:.1f}% mass participation.")
        print(f"Z-dir: {len(modes_for_90p_z)} modes for {cmpr_z*100:.1f}% mass participation.")

        shears_multi_x = _run_rsa(modes_for_90p_x, modal_props, floor_masses, get_sa, 'X')
        shears_multi_z = _run_rsa(modes_for_90p_z, modal_props, floor_masses, get_sa, 'Z')
        
        dominant_mode_x = max(modal_props, key=lambda p: p['mpr_x'])
        dominant_mode_z = max(modal_props, key=lambda p: p['mpr_z'])

        shears_dominant_x = _run_rsa([dominant_mode_x['mode']], modal_props, floor_masses, get_sa, 'X')
        shears_dominant_z = _run_rsa([dominant_mode_z['mode']], modal_props, floor_masses, get_sa, 'Z')

        is_valid_x = True
        print(f"\n[X-Direction Verification (Dominant Mode: {dominant_mode_x['mode']})]\nStory | V_multi (kN) | V_dominant (kN) | Ratio | Result")
        for i in range(num_stories):
            v_m, v_f = shears_multi_x[i]/1000, shears_dominant_x[i]/1000
            ratio = v_m / v_f if v_f > 1e-6 else 0
            result = 'OK' if ratio <= 1.3 else 'NG'
            if ratio > 1.3: is_valid_x = False
            print(f"{i+1:^5} | {v_m:^12.2f} | {v_f:^15.2f} | {ratio:^5.2f} | {result:^6}")

        is_valid_z = True
        print(f"\n[Z-Direction Verification (Dominant Mode: {dominant_mode_z['mode']})]\nStory | V_multi (kN) | V_dominant (kN) | Ratio | Result")
        for i in range(num_stories):
            v_m, v_f = shears_multi_z[i]/1000, shears_dominant_z[i]/1000
            ratio = v_m / v_f if v_f > 1e-6 else 0
            result = 'OK' if ratio <= 1.3 else 'NG'
            if ratio > 1.3: is_valid_z = False
            print(f"{i+1:^5} | {v_m:^12.2f} | {v_f:^15.2f} | {ratio:^5.2f} | {result:^6}")

        if not params.get('skip_post_processing', False):
            output_dir, name = params['output_dir'], params['analysis_name']
            _plot_shear_verification(shears_multi_x, shears_dominant_x, num_stories, 'X', output_dir / f"{name}_NSP_verification_plot_X.png")
            _plot_shear_verification(shears_multi_z, shears_dominant_z, num_stories, 'Z', output_dir / f"{name}_NSP_verification_plot_Z.png")

        print("\n--- 타당성 검증 완료 ---")
        return is_valid_x, is_valid_z
        
    except Exception as e:
        print(f"Error during NSP applicability verification: {e}")
        print(traceback.format_exc())
        return False, False
