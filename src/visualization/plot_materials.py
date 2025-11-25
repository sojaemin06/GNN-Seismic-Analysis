import matplotlib.pyplot as plt
import numpy as np

def plot_material_stress_strain(params):
    """
    [수정] 그래프 제목에 사용된 재료 모델의 학술적 근거와 OpenSees 모델명을 함께 명시합니다.
    """
    print("\nPlotting assumed material stress-strain curves...")
    
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    # --- 1. 콘크리트 그래프 (비구속 + 횡구속 통합) ---
    fig_conc, ax_conc = plt.subplots(figsize=(10, 7))
    
    # 1a. 비구속 콘크리트 (Unconfined Concrete)
    fpc_unc = params['fc']
    epsc0_unc = -0.002
    epscu_unc = -0.003
    fpcu_unc = 0.1 * fpc_unc
    
    strains_unc_parabola = np.linspace(0, epsc0_unc, 50)
    stresses_unc_parabola = fpc_unc * (2 * (strains_unc_parabola / epsc0_unc) - (strains_unc_parabola / epsc0_unc)**2)
    strains_unc_linear = np.array([epsc0_unc, epscu_unc])
    stresses_unc_linear = np.array([fpc_unc, fpcu_unc])

    ax_conc.plot(strains_unc_parabola, stresses_unc_parabola / 1e6, color='grey')
    ax_conc.plot(strains_unc_linear, stresses_unc_linear / 1e6, color='grey', label=f'Unconfined Concrete (f_c={abs(fpc_unc)/1e6:.1f} MPa)')
    
    # 1b. 횡구속 콘크리트 (Confined Concrete)
    fpc_c = params['fc'] * 1.3
    epsc0_c = -0.003
    epscu_c = -0.02
    fpcu_c = 0.1 * fpc_c

    strains_c_parabola = np.linspace(0, epsc0_c, 50)
    stresses_c_parabola = fpc_c * (2 * (strains_c_parabola / epsc0_c) - (strains_c_parabola / epsc0_c)**2)
    strains_c_linear = np.array([epsc0_c, epscu_c])
    stresses_c_linear = np.array([fpc_c, fpcu_c])

    ax_conc.plot(strains_c_parabola, stresses_c_parabola / 1e6, color='black')
    ax_conc.plot(strains_c_linear, stresses_c_linear / 1e6, color='black', label=f'Confined Concrete (f_cc={abs(fpc_c)/1e6:.1f} MPa)')

    ax_conc.set_title(f'Assumed Concrete Stress-Strain Curve\n(Popovics Model / uniaxialMaterial Concrete04)')
    ax_conc.set_xlabel('Strain (m/m)'); ax_conc.set_ylabel('Stress (MPa)'); ax_conc.grid(True); ax_conc.legend()
    ax_conc.axhline(0, color='black', lw=0.5); ax_conc.axvline(0, color='black', lw=0.5)
    plt.tight_layout()

    output_filename_conc = output_dir / f"{analysis_name}_material_concrete_combined.png"
    plt.savefig(output_filename_conc, dpi=300)
    plt.close(fig_conc)
    print(f"Combined concrete curve plot saved to: {output_filename_conc}")
    
    # --- 2. 철근 그래프 (개별) ---
    fig_rebar, ax_rebar = plt.subplots(figsize=(10, 7))
    fye_expected = abs(params['Fy'])
    fye_nominal = abs(params['Fy_nominal'])
    e_yield_expected = fye_expected / params['E_steel']
    e_yield_nominal = fye_nominal / params['E_steel']

    e_buckling_start = -0.003
    e_post_buckling_end = -0.005
    e_ultimate_tension = 0.05
    e_ultimate_compression = -0.02

    strain_pts = [
        e_ultimate_compression,
        e_post_buckling_end,
        e_buckling_start,
        -e_yield_nominal,
        0.0,
        e_yield_expected,
        e_ultimate_tension
    ]
    stress_pts = [
        -0.1 * fye_nominal,
        -0.1 * fye_nominal,
        -fye_nominal,
        -fye_nominal,
        0.0,
        fye_expected,
        fye_expected * 1.01
    ]
    
    paired_points = sorted(zip(strain_pts, stress_pts))
    rebar_strain_points, rebar_stress_points = zip(*paired_points)
    
    ax_rebar.plot(rebar_strain_points, np.array(rebar_stress_points) / 1e6, 'g-s', markerfacecolor='none', 
                  label=f'Rebar Backbone (Fy_exp={fye_expected/1e6:.0f}, Fy_nom={fye_nominal/1e6:.0f} MPa)')
    
    ax_rebar.set_title(f'Assumed Rebar Stress-Strain Curve\n(Guideline-based Buckling Model / uniaxialMaterial MultiLinear)')
    ax_rebar.set_xlabel('Strain (m/m)'); ax_rebar.set_ylabel('Stress (MPa)'); ax_rebar.grid(True); ax_rebar.legend()
    ax_rebar.axhline(0, color='black', lw=0.5); ax_rebar.axvline(0, color='black', lw=0.5)
    plt.tight_layout()
    
    output_filename_rebar = output_dir / f"{analysis_name}_material_rebar.png"
    plt.savefig(output_filename_rebar, dpi=300)
    plt.close(fig_rebar)
    print(f"Separate rebar curve plot saved to: {output_filename_rebar}")
