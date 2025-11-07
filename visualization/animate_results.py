import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# ### 9. 모듈형 함수: 애니메이션 및 플롯 저장 ###
def animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs):
    """
    [수정] 요청사항 반영: 1x3 플롯으로 변경 (Pushover Curve, Deformation, Plastic Hinges)
    """
    print("\nCreating pushover animation (3-panel) with plastic hinges...")
    
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

    # --- [신규] 2번 요청: Z축 라인 선택 ---
    z_line_idx = params.get('plot_z_line_index', 0)
    target_z = z_line_idx * params['bay_width_z']
    print(f"Plotting animation hinges for Z-Line {z_line_idx} (Z = {target_z:.1f}m)...")
    z_tolerance = 1e-6
    # --- [신규] 끝 ---

    # --- 2. 힌지 애니메이션을 위한 데이터 로드 ---
    try:
        node_coords = model_nodes_info['all_node_coords']
        all_line_elements = model_nodes_info['all_line_elements']
        
        all_column_tags = model_nodes_info.get('all_column_tags', [])
        all_beam_tags_type2 = model_nodes_info.get('all_beam_tags_type2', [])
        all_beam_tags_type3 = model_nodes_info.get('all_beam_tags_type3', [])
        
        df_col_rot = final_states_dfs.get('col_rot_df')
        df_beam_rot = final_states_dfs.get('beam_rot_df')
        
        num_int_pts = params.get('num_int_pts', 5)
        ip_start = 1
        ip_end = num_int_pts
        
        ROT_IO = 0.005  # Immediate Occupancy
        ROT_LS = 0.02   # Life Safety
        ROT_CP = 0.04   # Collapse Prevention
        
        mkr_cp = {'marker': 's', 'color': 'red', 'markersize': 12, 'mew': 1.5, 'mec': 'black'}
        mkr_ls = {'marker': 'D', 'color': 'orange', 'markersize': 10, 'mew': 1.0, 'mec': 'black'}
        mkr_io = {'marker': 'o', 'color': 'blue', 'markersize': 8, 'mew': 0.5, 'mec': 'black'}

    except Exception as e:
        print(f"Warning: Cannot load data for hinge animation: {e}")
        df_col_rot = None
        df_beam_rot = None

    # --- 3. 플롯 설정 (1x3) ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 7), gridspec_kw={'width_ratios': [1, 0.8, 1]})
    fig.suptitle(f"{analysis_name} Pushover Animation", fontsize=16)

    # 3-1. 축 1: 푸쉬오버 곡선
    max_x_lim = max(x_data_roof) * 1.05 if max(x_data_roof) > 0 else 0.1
    max_y_lim = max(y_data_shear) * 1.05 if max(y_data_shear) > 0 else 1.0
    ax1.set_xlim(0, max_x_lim)
    ax1.set_ylim(0, max_y_lim)
    ax1.set_xlabel('Roof Displacement (m)')
    ax1.set_ylabel('Base Shear (kN)')
    ax1.grid(True)
    
    line, = ax1.plot([], [], 'b-', lw=2, label='Pushover Curve') 
    point, = ax1.plot([], [], 'bo', markersize=8) 
    yield_point_artist, = ax1.plot([], [], 'go', markersize=10, label=f'Approx. Yield')
    peak_point_artist, = ax1.plot([], [], 'rs', markersize=10, label=f'Peak Strength')
    collapse_line_artist, = ax1.plot([], [], 'r--', linewidth=2, label=f'Collapse (80% peak)')
    ax1.legend(loc='lower right')

    # 3-2. 축 2: 입면 변형도 (힌지 제거)
    max_disp_for_plot = max_x_lim * 1.2 
    ax2.set_xlim(-max_disp_for_plot * 0.1, max_disp_for_plot)
    ax2.set_ylim(0, building_y_coords[-1] * 1.2 if building_y_coords else 1.0)
    ax2.set_xlabel('Lateral Displacement (m)')
    ax2.set_ylabel('Height (m)')
    ax2.set_title('Building Deformation') 
    ax2.grid(True)
    
    structure_line, = ax2.plot([], [], 'r-o', lw=3, markersize=8)
    time_text_ax2 = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, va='top')
    
    # 3-3. 축 3: 2D 소성 힌지 입면도
    max_x_geom = params['num_bays_x'] * params['bay_width_x']
    max_y_geom = params['num_stories'] * params['story_height']
    ax3.set_xlim(-max_x_geom * 0.1, max_x_geom * 1.1)
    ax3.set_ylim(-0.5, max_y_geom * 1.1)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (Height) (m)')
    
    ax3.set_title(f'Plastic Hinge Formation (Frame at Z = {target_z:.1f}m)') 
    
    ax3.grid(True)
    ax3.axis('equal') 
    time_text_ax3 = ax3.text(0.05, 0.95, '', transform=ax3.transAxes, va='top')
    
    # ax3 배경 구조물 플로팅
    for (node_i_tag, node_j_tag) in all_line_elements.values():
        n_i = node_coords.get(node_i_tag)
        n_j = node_coords.get(node_j_tag)
        if n_i and n_j:
            if abs(n_i[2] - target_z) < z_tolerance and abs(n_j[2] - target_z) < z_tolerance:
                ax3.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], '-', color='black', linewidth=1, zorder=1)

    # ax3 힌지 아티스트 정의
    hinge_io_plot_ax3, = ax3.plot([], [], **mkr_io, linestyle='None', zorder=8)
    hinge_ls_plot_ax3, = ax3.plot([], [], **mkr_ls, linestyle='None', zorder=9)
    hinge_cp_plot_ax3, = ax3.plot([], [], **mkr_cp, linestyle='None', zorder=10)


    # --- 4. 애니메이션 함수 정의 ---
    def init():
        line.set_data([], []) 
        point.set_data([], [])
        structure_line.set_data([], []) 
        time_text_ax2.set_text('')
        time_text_ax3.set_text('')
        yield_point_artist.set_data([], [])
        peak_point_artist.set_data([], [])
        collapse_line_artist.set_data([], [])
        
        hinge_io_plot_ax3.set_data([], [])
        hinge_ls_plot_ax3.set_data([], [])
        hinge_cp_plot_ax3.set_data([], [])

        return (line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                peak_point_artist, collapse_line_artist, 
                hinge_io_plot_ax3, hinge_ls_plot_ax3, hinge_cp_plot_ax3)

    def animate(i):
        current_index = i * step_size
        if current_index >= total_steps:
            current_index = total_steps - 1
        
        current_roof_disp = x_data_roof[current_index]
        current_shear = y_data_shear[current_index]
        text_info = f'Disp: {current_roof_disp:.4f} m\nShear: {current_shear:.0f} kN'
        
        # 4-1. 축 1 업데이트
        line.set_data(x_data_roof[:current_index+1], y_data_shear[:current_index+1]) 
        point.set_data([current_roof_disp], [current_shear])
        
        # 4-2. 축 2 업데이트
        current_floor_disps_row = df_disp.iloc[current_index, 1:].values
        building_x_coords = [0.0] + list(current_floor_disps_row)
        structure_line.set_data(building_x_coords, building_y_coords) 
        time_text_ax2.set_text(text_info)
        
        # 4-3. 축 3 업데이트 (소성 힌지)
        time_text_ax3.set_text(text_info)
        
        io_x_ax3, io_y_ax3 = [], []
        ls_x_ax3, ls_y_ax3 = [], []
        cp_x_ax3, cp_y_ax3 = [], []
        
        # 기둥 힌지 (Z축 필터링)
        if df_col_rot is not None:
            current_col_rot_step = df_col_rot.iloc[current_index] 
            for ele_tag in all_column_tags:
                try:
                    n_i_tag, n_j_tag = all_line_elements[ele_tag]
                    n_i_coords = node_coords[n_i_tag]; n_j_coords = node_coords[n_j_tag]

                    if abs(n_i_coords[2] - target_z) > z_tolerance:
                        continue
                        
                    locations = [n_i_coords, n_j_coords]
                    
                    for loc_idx, ip in enumerate([ip_start, ip_end]):
                        rot_ry = current_col_rot_step[f'Ele{ele_tag}_IP{ip}_ry']
                        rot_rz = current_col_rot_step[f'Ele{ele_tag}_IP{ip}_rz']
                        theta_p = abs(rot_ry) + abs(rot_rz)
                        
                        loc_coords = locations[loc_idx]
                        
                        if theta_p >= ROT_CP:
                            cp_x_ax3.append(loc_coords[0]); cp_y_ax3.append(loc_coords[1])
                        elif theta_p >= ROT_LS:
                            ls_x_ax3.append(loc_coords[0]); ls_y_ax3.append(loc_coords[1])
                        elif theta_p >= ROT_IO:
                            io_x_ax3.append(loc_coords[0]); io_y_ax3.append(loc_coords[1])
                except KeyError: continue 

        # 보 힌지 (Z축 필터링)
        if df_beam_rot is not None:
            current_beam_rot_step = df_beam_rot.iloc[current_index]
            
            # Type 3 (X-dir, rz 휨)
            for ele_tag in all_beam_tags_type3: # Type 3만
                try:
                    n_i_tag, n_j_tag = all_line_elements[ele_tag]
                    n_i_coords = node_coords[n_i_tag]; n_j_coords = node_coords[n_j_tag]

                    if abs(n_i_coords[2] - target_z) > z_tolerance:
                        continue

                    locations = [n_i_coords, n_j_coords]
                    
                    for loc_idx, ip in enumerate([ip_start, ip_end]):
                        theta_p = abs(current_beam_rot_step[f'Ele{ele_tag}_IP{ip}_rz']) # Type 3 = rz
                        loc_coords = locations[loc_idx]

                        if theta_p >= ROT_CP:
                            cp_x_ax3.append(loc_coords[0]); cp_y_ax3.append(loc_coords[1])
                        elif theta_p >= ROT_LS:
                            ls_x_ax3.append(loc_coords[0]); ls_y_ax3.append(loc_coords[1])
                        elif theta_p >= ROT_IO:
                            io_x_ax3.append(loc_coords[0]); io_y_ax3.append(loc_coords[1])
                except KeyError: continue
        
        hinge_io_plot_ax3.set_data(io_x_ax3, io_y_ax3)
        hinge_ls_plot_ax3.set_data(ls_x_ax3, ls_y_ax3)
        hinge_cp_plot_ax3.set_data(cp_x_ax3, cp_y_ax3)
        
        # 4-4. 성능점 표시 (ax1)
        if perf_points.get('yield_disp', 0) > 0 and current_roof_disp >= perf_points['yield_disp']:
            yield_point_artist.set_data([perf_points['yield_disp']], [perf_points['yield_shear'] / 1000])
        if perf_points.get('peak_disp', 0) > 0 and current_roof_disp >= perf_points['peak_disp']:
            peak_point_artist.set_data([perf_points['peak_disp']], [perf_points['peak_shear'] / 1000])
        if perf_points.get('collapse_disp') is not None and current_roof_disp >= perf_points['collapse_disp']:
            collapse_line_artist.set_data([perf_points['collapse_disp'], perf_points['collapse_disp']], [0, max_y_lim])
        
        return (line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                peak_point_artist, collapse_line_artist, 
                hinge_io_plot_ax3, hinge_ls_plot_ax3, hinge_cp_plot_ax3)

    # --- 5. 저장: 애니메이션 (MP4) ---
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=num_frames, interval=50, blit=False)

    try:
        output_filename = output_dir / f"{analysis_name}_pushover_animation.mp4"
        anim.save(str(output_filename), writer='ffmpeg', fps=20, dpi=150)
        print(f"\nAnimation saved successfully to: {output_filename}")
    except Exception as e:
        print(f"\n---! Error saving animation !---")
        print("Could not save animation. Do you have 'ffmpeg' installed and accessible in your system's PATH?")
        print(f"Error details: {e}")

    # --- 6. 저장: 정적 플롯 (PNG) ---
    try:
        static_image_path = output_dir / f"{analysis_name}_pushover_final_plot.png"
        
        print("Setting final frame for static plot...")
        
        last_frame_index = num_frames - 1
        animate(last_frame_index) 
        
        line.set_data(x_data_roof, y_data_shear) 

        fig.savefig(static_image_path, dpi=300, bbox_inches='tight')
        print(f"\nFinal static pushover plot (3-panel) saved to: {static_image_path}")
    except Exception as e:
        print(f"\n---! Error saving static plot !---")
        print(f"Error details: {e}")
    
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