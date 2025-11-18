import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.colors as mcolors

# ### 9. 모듈형 함수: 애니메이션 및 플롯 저장 ###
def animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs):
    """
    [수정] 요청사항 반영: 2x2 플롯으로 변경 (Pushover Curve, Deformation, Plastic Hinges, Inter-story Drift)
    """
    print("\nCreating pushover animation (4-panel) with plastic hinges and drift...")

    from matplotlib.patches import Polygon
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    # --- 1. 데이터 준비 ---
    total_steps = len(df_curve) 
    if total_steps < 2:
        print("Warning: Not enough data points to create animation (< 2 steps).")
        return

    max_anim_frames = 200 
    step_size = max(1, total_steps // max_anim_frames) 
    num_frames = (total_steps - 1) // step_size + 1 
    
    x_data_roof = df_curve['Roof_Displacement_m'].values
    y_data_shear = (df_curve['Base_Shear_N'] / 1000).values # kN 단위
    
    num_stories = params['num_stories']
    story_height = params['story_height']
    building_y_coords = [0.0] + [(i + 1) * story_height for i in range(num_stories)]
    story_mid_heights = [story_height * (i + 0.5) for i in range(num_stories)]

    z_line_idx = params.get('plot_z_line_index', 0)
    target_z = z_line_idx * params['bay_width_z']
    print(f"Plotting animation hinges for Z-Line {z_line_idx} (Z = {target_z:.1f}m)...")
    z_tolerance = 1e-6

    # --- 2. 힌지 및 손상 데이터 로드 ---
    try:
        node_coords = model_nodes_info['all_node_coords']
        all_line_elements = model_nodes_info['all_line_elements']
        all_shell_elements = model_nodes_info.get('all_shell_elements', {})
        all_column_tags = model_nodes_info.get('all_column_tags', []) 
        all_beam_tags_type3 = model_nodes_info.get('all_beam_tags_type3', []) 

        df_col_rot = final_states_dfs.get('col_rot_df')
        df_beam_rot = final_states_dfs.get('beam_rot_df')
        df_wall_forces = final_states_dfs.get('wall_forces_df')
        
        num_int_pts = params.get('num_int_pts', 5)
        ip_start, ip_end = 1, num_int_pts
        
        ROT_IO, ROT_LS, ROT_CP = 0.005, 0.02, 0.04
        mkr_cp = {'marker': 's', 'color': 'red', 'markersize': 12, 'mew': 1.5, 'mec': 'black'}
        mkr_ls = {'marker': 'D', 'color': 'orange', 'markersize': 10, 'mew': 1.0, 'mec': 'black'}
        mkr_io = {'marker': 'o', 'color': 'blue', 'markersize': 8, 'mew': 0.5, 'mec': 'black'}

        wall_polygons, cmap, norm = [], None, None
        if df_wall_forces is not None and not df_wall_forces.empty:
            all_stresses = [
                -( (row[f'Ele{ele_tag}_GP{gp}_s11'] + row[f'Ele{ele_tag}_GP{gp}_s22']) / 2 - 
                   np.sqrt(((row[f'Ele{ele_tag}_GP{gp}_s11'] - row[f'Ele{ele_tag}_GP{gp}_s22']) / 2)**2 + row[f'Ele{ele_tag}_GP{gp}_s12']**2) )
                for _, row in df_wall_forces.iterrows() 
                for ele_tag in all_shell_elements.keys() 
                for gp in range(1, 5) 
                if f'Ele{ele_tag}_GP{gp}_s11' in row
            ]
            if all_stresses:
                max_stress = max(all_stresses)
                norm = mcolors.Normalize(vmin=0, vmax=max_stress if max_stress > 0 else 1.0)
                cmap = plt.cm.get_cmap('jet')
            else: df_wall_forces = None
    except Exception as e:
        print(f"Warning: Cannot load data for hinge/damage animation: {e}")
        df_col_rot, df_beam_rot, df_wall_forces = None, None, None

    # --- 3. 플롯 설정 (2x2) ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19, 14), gridspec_kw={'width_ratios': [1.2, 1], 'height_ratios': [1, 1]})
    fig.suptitle(f"{analysis_name} Pushover Analysis", fontsize=16, y=0.97)

    # 3-1. 축 1: 푸쉬오버 곡선
    max_x_lim = max(x_data_roof) * 1.05 if max(x_data_roof) > 0 else 0.1
    max_y_lim = max(y_data_shear) * 1.05 if max(y_data_shear) > 0 else 1.0
    ax1.set_xlim(0, max_x_lim); ax1.set_ylim(0, max_y_lim)
    ax1.set_xlabel('Roof Displacement (m)'); ax1.set_ylabel('Base Shear (kN)'); ax1.grid(True)
    line, = ax1.plot([], [], 'b-', lw=2, label='Pushover Curve') 
    point, = ax1.plot([], [], 'bo', markersize=8) 
    yield_point_artist, = ax1.plot([], [], 'go', markersize=10, label='Approx. Yield')
    peak_point_artist, = ax1.plot([], [], 'rs', markersize=10, label='Peak Strength')
    collapse_line_artist, = ax1.plot([], [], 'r--', linewidth=2, label='Collapse (80% peak)')
    ax1.legend(loc='lower right')

    # 3-2. 축 2: 입면 변형도
    max_disp_for_plot = max_x_lim * 1.2 
    ax2.set_xlim(-max_disp_for_plot * 0.1, max_disp_for_plot)
    ax2.set_ylim(0, building_y_coords[-1] * 1.2 if building_y_coords else 1.0)
    ax2.set_xlabel('Lateral Displacement (m)'); ax2.set_ylabel('Height (m)')
    ax2.set_title('Building Deformation'); ax2.grid(True)
    structure_line, = ax2.plot([], [], 'r-o', lw=3, markersize=8)
    time_text_ax2 = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, va='top')
    
    # 3-3. 축 3: 2D 소성 힌지 입면도
    max_x_geom = params['num_bays_x'] * params['bay_width_x']
    max_y_geom = params['num_stories'] * params['story_height']
    ax3.set_xlim(-max_x_geom * 0.1, max_x_geom * 1.1); ax3.set_ylim(-0.5, max_y_geom * 1.1)
    ax3.set_xlabel('X (m)'); ax3.set_ylabel('Y (Height) (m)')
    ax3.set_title(f'Plastic Hinge Formation (Frame at Z = {target_z:.1f}m)'); ax3.grid(True); ax3.axis('equal') 
    time_text_ax3 = ax3.text(0.05, 0.95, '', transform=ax3.transAxes, va='top')
    
    for (node_i_tag, node_j_tag) in all_line_elements.values():
        n_i, n_j = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
        if n_i and n_j and abs(n_i[2] - target_z) < z_tolerance and abs(n_j[2] - target_z) < z_tolerance:
            ax3.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], '-', color='black', linewidth=1, zorder=1)

    if all_shell_elements:
        for ele_tag, tags in all_shell_elements.items():
            nodes = [node_coords.get(t) for t in tags]
            if all(nodes) and any(abs(n[2] - target_z) < z_tolerance for n in nodes):
                poly = Polygon([[n[0], n[1]] for n in nodes], facecolor='gray', edgecolor='black', alpha=0.7, lw=0.5, zorder=2)
                ax3.add_patch(poly)
                wall_polygons.append({'tag': ele_tag, 'patch': poly})
    
    hinge_io_plot_ax3, = ax3.plot([], [], **mkr_io, linestyle='None', zorder=8)
    hinge_ls_plot_ax3, = ax3.plot([], [], **mkr_ls, linestyle='None', zorder=9)
    hinge_cp_plot_ax3, = ax3.plot([], [], **mkr_cp, linestyle='None', zorder=10)

    # [신규] 3-4. 축 4: 층간 변형각
    ax4.set_ylim(ax2.get_ylim()); ax4.set_xlim(0, 5) # 5% drift limit
    ax4.set_title('Inter-story Drift Ratio'); ax4.set_xlabel('Drift Ratio (%)'); ax4.set_ylabel('Height (m)'); ax4.grid(True)
    drift_profile_line, = ax4.plot([], [], 'g-s', lw=2, markersize=8, label='Current Drift')
    # 평가요령 표 4.6.1 RC 모멘트골조 허용치
    ax4.axvline(x=0.7, color='blue', linestyle='--', label='IO Limit (0.7%)')
    ax4.axvline(x=2.0, color='orange', linestyle='--', label='LS Limit (2.0%)')
    ax4.axvline(x=3.0, color='red', linestyle='--', label='CP Limit (3.0%)')
    ax4.legend(loc='lower right')

    # --- 4. 애니메이션 함수 정의 ---
    def init():
        artists = [line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                   peak_point_artist, collapse_line_artist, hinge_io_plot_ax3, hinge_ls_plot_ax3, 
                   hinge_cp_plot_ax3, drift_profile_line]
        for artist in artists:
            if isinstance(artist, plt.Line2D): artist.set_data([], [])
            else: artist.set_text('')
        for poly_info in wall_polygons: poly_info['patch'].set_facecolor('gray')
        return artists + [p['patch'] for p in wall_polygons]

    def animate(i):
        idx = min(i * step_size, total_steps - 1)
        text_info = f'Disp: {x_data_roof[idx]:.4f} m\nShear: {y_data_shear[idx]:.0f} kN'
        
        line.set_data(x_data_roof[:idx+1], y_data_shear[:idx+1]) 
        point.set_data([x_data_roof[idx]], [y_data_shear[idx]])
        
        current_disps = [0.0] + list(df_disp.iloc[idx, 1:].values)
        structure_line.set_data(current_disps, building_y_coords) 
        time_text_ax2.set_text(text_info)
        time_text_ax3.set_text(text_info)
        
        # 힌지 업데이트
        io_x, io_y, ls_x, ls_y, cp_x, cp_y = [], [], [], [], [], []
        if df_col_rot is not None:
            row = df_col_rot.iloc[idx]
            for tag in all_column_tags:
                ni_t, nj_t = all_line_elements[tag]
                ni_c, nj_c = node_coords[ni_t], node_coords[nj_t]
                if abs(ni_c[2] - target_z) > z_tolerance: continue
                for loc_idx, ip in enumerate([ip_start, ip_end]):
                    theta_p = abs(row.get(f'Ele{tag}_IP{ip}_ry', 0)) + abs(row.get(f'Ele{tag}_IP{ip}_rz', 0))
                    loc = ni_c if loc_idx == 0 else nj_c
                    if theta_p >= ROT_CP: cp_x.append(loc[0]); cp_y.append(loc[1])
                    elif theta_p >= ROT_LS: ls_x.append(loc[0]); ls_y.append(loc[1])
                    elif theta_p >= ROT_IO: io_x.append(loc[0]); io_y.append(loc[1])
        # (보 힌지 로직은 간결성을 위해 생략, 필요시 추가)
        hinge_io_plot_ax3.set_data(io_x, io_y)
        hinge_ls_plot_ax3.set_data(ls_x, ls_y)
        hinge_cp_plot_ax3.set_data(cp_x, cp_y)

        # [신규] 층간변형각 업데이트
        drifts = [(current_disps[i+1] - current_disps[i]) / story_height * 100 for i in range(num_stories)]
        drift_profile_line.set_data(drifts, story_mid_heights)

        # 성능점 표시
        if perf_points.get('yield_disp', 0) > 0 and x_data_roof[idx] >= perf_points['yield_disp']:
            yield_point_artist.set_data([perf_points['yield_disp']], [perf_points.get('yield_shear', 0) / 1000])
        if perf_points.get('peak_disp', 0) > 0 and x_data_roof[idx] >= perf_points['peak_disp']:
            peak_point_artist.set_data([perf_points['peak_disp']], [perf_points['peak_shear'] / 1000])
        if perf_points.get('collapse_disp') is not None and x_data_roof[idx] >= perf_points['collapse_disp']:
            collapse_line_artist.set_data([perf_points['collapse_disp'], perf_points['collapse_disp']], [0, max_y_lim])
        
        return (line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                peak_point_artist, collapse_line_artist, hinge_io_plot_ax3, hinge_ls_plot_ax3, 
                hinge_cp_plot_ax3, drift_profile_line) + tuple(p['patch'] for p in wall_polygons)

    # --- 5. 저장: 애니메이션 (MP4) ---
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=num_frames, interval=50, blit=False)
    try:
        output_filename = output_dir / f"{analysis_name}_pushover_animation.mp4"
        anim.save(str(output_filename), writer='ffmpeg', fps=20, dpi=150)
        print(f"\nAnimation saved successfully to: {output_filename}")
    except Exception as e:
        print(f"\n---! Error saving animation: {e}. Check if 'ffmpeg' is installed. !---")

    # --- 6. 저장: 정적 플롯 (PNG) ---
    try:
        static_image_path = output_dir / f"{analysis_name}_pushover_final_plot.png"
        animate(num_frames - 1) 
        line.set_data(x_data_roof, y_data_shear) 
        if df_wall_forces is not None and cmap is not None:
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            fig.colorbar(sm, ax=ax3, orientation='vertical', fraction=0.08, pad=0.04).set_label('Principal Compressive Stress (Pa)')
        fig.savefig(static_image_path, dpi=300, bbox_inches='tight')
        print(f"Final static pushover plot saved to: {static_image_path}")
    except Exception as e:
        print(f"\n---! Error saving static plot: {e} !---")
    
    plt.close(fig)

# ### [신규] 10. 모듈형 함수: 모멘트-곡률 애니메이션 ###
def animate_and_plot_M_phi(df_m_phi, params):
    """
    [신규] 특정 요소/적분점의 모멘트-곡률(M-Phi) 관계를
    애니메이션(mp4) 및 정적(png)으로 저장합니다.
    """
    print("\nCreating Moment-Curvature (M-Phi) animation...")
    
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    # --- 데이터 준비 ---
    if df_m_phi is None or df_m_phi.empty or len(df_m_phi) < 2:
        print("Warning: M-Phi data is missing or empty. Skipping M-Phi plot.")
        return

    num_frames = 200 
    total_steps = len(df_m_phi)
    step_size = max(1, total_steps // num_frames)
    
    x_data_phi = df_m_phi['Curvature_phi_y_rad/m'].values
    y_data_moment = (df_m_phi['Moment_My_N-m'] / 1000).values # kN-m 단위
    
    # --- 플롯 설정 ---
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{analysis_name}\nMoment-Curvature (M-Phi) for Target Element", fontsize=14)

    # 축 범위 설정
    max_x_lim = max(abs(x_data_phi)) * 1.05 if max(abs(x_data_phi)) > 1e-9 else 0.1
    max_y_lim = max(abs(y_data_moment)) * 1.05 if max(abs(y_data_moment)) > 1e-9 else 1.0
    ax.set_xlim(-max_x_lim, max_x_lim)
    ax.set_ylim(-max_y_lim, max_y_lim)
    
    ax.set_xlabel('Curvature ($\phi_y$) (rad/m)')
    ax.set_ylabel('Moment ($M_y$) (kN-m)')
    
    ax.grid(True)
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    
    line, = ax.plot([], [], 'b-', lw=2) 
    point, = ax.plot([], [], 'bo', markersize=8) 
    time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, va='top')

    # --- 애니메이션 함수 정의 ---
    def init_m_phi():
        line.set_data([], []) 
        point.set_data([], [])
        time_text.set_text('')
        return line, point, time_text

    def animate_m_phi(i):
        current_index = min(i * step_size, total_steps - 1) 
        
        line.set_data(x_data_phi[:current_index+1], y_data_moment[:current_index+1]) 
        current_phi = x_data_phi[current_index]
        current_moment = y_data_moment[current_index]
        point.set_data([current_phi], [current_moment])
        
        time_text.set_text(f'Step: {current_index}/{total_steps}\nCurv: {current_phi:.6f}\nMoment: {current_moment:.1f} kNm')
        
        return line, point, time_text

    # --- 저장: 애니메이션 (MP4) ---
    anim = animation.FuncAnimation(fig, animate_m_phi, init_func=init_m_phi,
                                   frames=num_frames, interval=50, blit=False)
    
    try:
        output_filename = output_dir / f"{analysis_name}_M_phi_animation.mp4"
        anim.save(str(output_filename), writer='ffmpeg', fps=20, dpi=150)
        print(f"\n M-Phi Animation saved successfully to: {output_filename}")
    except Exception as e:
        print(f"\n---! Error saving M-Phi animation !---")
        print(f"Error details: {e}")

    # --- 저장: 정적 플롯 (PNG) ---
    try:
        static_image_path = output_dir / f"{analysis_name}_M_phi_final_plot.png"
        
        print("Setting final frame for M-Phi static plot...")
        line.set_data(x_data_phi, y_data_moment) # 전체 곡선
        point.set_data([x_data_phi[-1]], [y_data_moment[-1]]) # 마지막 점
        time_text.set_text(f'Final State\nCurv: {x_data_phi[-1]:.6f}\nMoment: {y_data_moment[-1]:.1f} kNm')

        fig.savefig(static_image_path, dpi=300, bbox_inches='tight')
        print(f"\nFinal static M-Phi plot saved to: {static_image_path}")
    except Exception as e:
        print(f"\n---! Error saving static M-Phi plot !---")
        print(f"Error details: {e}")
    
    plt.close(fig)