import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon

# ### 9. 모듈형 함수: 애니메이션 및 플롯 저장 ###
def animate_and_plot_pushover(df_curve, df_disp, perf_points, params, model_nodes_info, final_states_dfs, first_failure_event=None, direction='X'):
    """
    [수정] 2x2 플롯 애니메이션 생성. 
    [신규] first_failure_event를 받아 푸쉬오버 곡선에 재료 파괴 시점 표시.
    [신규] direction 파라미터에 따라 올바른 입면도 플로팅.
    """
    print("\nCreating pushover animation (4-panel) with plastic hinges and drift...")

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

    # [수정] 푸쉬오버 곡선(ax1)은 절대값으로 플로팅하여 1사분면에 표시 (방향 무관 비교 가능)
    x_data_roof_abs = np.abs(x_data_roof)
    y_data_shear_abs = np.abs(y_data_shear)
    
    num_stories = params['num_stories']
    story_height = params['story_height']
    building_y_coords = [0.0] + [(i + 1) * story_height for i in range(num_stories)]
    story_mid_heights = [story_height * (i + 0.5) for i in range(num_stories)]

    # --- 2. 힌지 및 손상 데이터 로드 ---
    df_col_rot, df_beam_rot, df_wall_forces = None, None, None
    try:
        node_coords = model_nodes_info['all_node_coords']
        all_line_elements = model_nodes_info['all_line_elements']
        all_shell_elements = model_nodes_info.get('all_shell_elements', {})
        all_column_tags = model_nodes_info.get('all_column_tags', []) 
        all_beam_tags = model_nodes_info.get('all_beam_tags', [])

        # 데이터 프레임 가져오기
        df_col_rot = final_states_dfs.get('col_rot_df')
        df_beam_rot = final_states_dfs.get('beam_rot_df')
        
        num_int_pts = params.get('num_int_pts', 5)
        ip_start, ip_end = 1, num_int_pts
        
        ROT_IO, ROT_LS, ROT_CP = 0.002, 0.02, 0.04
        mkr_cp = {'marker': 's', 'color': 'red', 'markersize': 12, 'mew': 1.5, 'mec': 'black'}
        mkr_ls = {'marker': 'D', 'color': 'orange', 'markersize': 10, 'mew': 1.0, 'mec': 'black'}
        mkr_io = {'marker': 'o', 'color': 'blue', 'markersize': 8, 'mew': 0.5, 'mec': 'black'}

        # 벽체 응력 데이터 처리 (있을 경우)
        # 주의: df_wall_forces는 final_states_dfs에 없을 수도 있음
        df_wall_forces = final_states_dfs.get('wall_forces_df')
        
        wall_polygons, cmap, norm = [], None, None
        if df_wall_forces is not None and not df_wall_forces.empty:
            try:
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
                else:
                    df_wall_forces = None
            except Exception as e:
                 print(f"Warning: Error processing wall forces: {e}")
                 df_wall_forces = None

    except Exception as e:
        print(f"Warning: Cannot load data for hinge/damage animation: {e}")
        # 여기서 df_col_rot를 None으로 만들면 안됨, 부분적으로 로드된 데이터라도 사용 시도
        if df_col_rot is None: print("  - Column rotation data missing")
        if df_beam_rot is None: print("  - Beam rotation data missing")

    # --- 3. 플롯 설정 (2x2) ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(19, 14), gridspec_kw={'width_ratios': [1.2, 1], 'height_ratios': [1, 1]})
    fig.suptitle(f"{analysis_name} Pushover Analysis ({direction}-dir)", fontsize=16, y=0.97)

    # 3-1. 축 1: 푸쉬오버 곡선 (절대값 사용)
    max_x_lim = max(x_data_roof_abs) * 1.05 if max(x_data_roof_abs) > 0 else 0.1
    max_y_lim = max(y_data_shear_abs) * 1.05 if max(y_data_shear_abs) > 0 else 1.0
    ax1.set_xlim(0, max_x_lim); ax1.set_ylim(0, max_y_lim)
    ax1.set_xlabel('Roof Displacement (m) [Abs]'); ax1.set_ylabel('Base Shear (kN) [Abs]'); ax1.grid(True)
    line, = ax1.plot([], [], 'b-', lw=2, label='Pushover Curve') 
    point, = ax1.plot([], [], 'bo', markersize=8) 
    yield_point_artist, = ax1.plot([], [], 'go', markersize=10, label='Approx. Yield')
    peak_point_artist, = ax1.plot([], [], 'rs', markersize=10, label='Peak Strength')
    collapse_line_artist, = ax1.plot([], [], 'r--', linewidth=2, label='Collapse (80% peak)')
    failure_marker_artist, = ax1.plot([], [], 'X', color='darkred', markersize=14, label='1st Material Failure', zorder=10)
    ax1.legend(loc='lower right')

    # 3-2. 축 2: 입면 변형도 (실제 변위, 대칭 축 설정)
    max_disp_abs = max(x_data_roof_abs)
    limit_disp = max_disp_abs * 1.2 if max_disp_abs > 0 else 1.0
    ax2.set_xlim(-limit_disp, limit_disp) # 대칭 설정
    ax2.set_ylim(0, building_y_coords[-1] * 1.2 if building_y_coords else 1.0)
    ax2.set_xlabel('Lateral Displacement (m)'); ax2.set_ylabel('Height (m)')
    ax2.set_title('Building Deformation'); ax2.grid(True)
    # 중심선 표시
    ax2.axvline(0, color='k', linestyle='--', alpha=0.3)
    structure_line, = ax2.plot([], [], 'r-o', lw=3, markersize=8)
    time_text_ax2 = ax2.text(0.05, 0.95, '', transform=ax2.transAxes, va='top')
    
    # 3-3. 축 3: 2D 소성 힌지 입면도
    max_y_geom = params['num_stories'] * params['story_height']
    ax3.set_ylim(-0.5, max_y_geom * 1.1)
    ax3.set_ylabel('Y (Height) (m)')
    ax3.grid(True); ax3.axis('equal') 
    time_text_ax3 = ax3.text(0.05, 0.95, '', transform=ax3.transAxes, va='top')
    
    # [수정] 배경 프레임 그리기 (방향 분기)
    target_coord = 0.0
    if direction == 'X':
        target_coord = params.get('plot_z_line_index', 0) * params['bay_width_z']
        tolerance = 0.1 # 10cm tolerance
        ax3.set_title(f'Plastic Hinge Formation (Frame at Z = {target_coord:.1f}m)')
        ax3.set_xlabel('X (m)')
        for (node_i_tag, node_j_tag) in all_line_elements.values():
            n_i, n_j = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if n_i and n_j and abs(n_i[2] - target_coord) < tolerance and abs(n_j[2] - target_coord) < tolerance:
                ax3.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], '-', color='black', linewidth=1, zorder=1)
    elif direction == 'Z':
        target_coord = params.get('plot_x_line_index', 0) * params['bay_width_x']
        tolerance = 0.1 # 10cm tolerance
        ax3.set_title(f'Plastic Hinge Formation (Frame at X = {target_coord:.1f}m)')
        ax3.set_xlabel('Z (m)')
        for (node_i_tag, node_j_tag) in all_line_elements.values():
            n_i, n_j = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if n_i and n_j and abs(n_i[0] - target_coord) < tolerance and abs(n_j[0] - target_coord) < tolerance:
                ax3.plot([n_i[2], n_j[2]], [n_i[1], n_j[1]], '-', color='black', linewidth=1, zorder=1)
    
    hinge_io_plot_ax3, = ax3.plot([], [], **mkr_io, linestyle='None', zorder=8)
    hinge_ls_plot_ax3, = ax3.plot([], [], **mkr_ls, linestyle='None', zorder=9)
    hinge_cp_plot_ax3, = ax3.plot([], [], **mkr_cp, linestyle='None', zorder=10)

    # 3-4. 축 4: 층간 변형각
    # 층간변형각은 항상 양수로 표현하는 것이 일반적
    ax4.set_ylim(ax2.get_ylim()); ax4.set_xlim(0, 5) # 5% drift limit
    ax4.set_title('Inter-story Drift Ratio'); ax4.set_xlabel('Drift Ratio (%)'); ax4.set_ylabel('Height (m)'); ax4.grid(True)
    drift_profile_line, = ax4.plot([], [], 'g-s', lw=2, markersize=8, label='Current Drift')
    # 평가요령 표 4.6.1 RC 모멘트골조 허용치
    ax4.axvline(x=0.7, color='blue', linestyle='--', label='IO Limit (0.7%)')
    ax4.axvline(x=2.0, color='orange', linestyle='--', label='LS Limit (2.0%)')
    ax4.axvline(x=3.0, color='red', linestyle='--', label='CP Limit (3.0%)')
    ax4.legend(loc='lower right')

    # --- 4. 애니메이션 함수 정의 ---
    
    # [수정] init 함수 추가 (NameError 해결)
    def init():
        artists = [line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                   peak_point_artist, collapse_line_artist, hinge_io_plot_ax3, hinge_ls_plot_ax3, 
                   hinge_cp_plot_ax3, drift_profile_line, failure_marker_artist]
        for artist in artists:
            if isinstance(artist, plt.Line2D): artist.set_data([], [])
            else: artist.set_text('')
        return artists

    def animate(i):
        # [Fix] Ensure the last frame shows the very last data point
        if i == num_frames - 1:
            idx = total_steps - 1
        else:
            idx = min(i * step_size, total_steps - 1)
            
        text_info = f'Disp: {x_data_roof[idx]:.4f} m\nShear: {y_data_shear[idx]:.0f} kN'
        
        # [수정] ax1에는 절대값 플로팅
        line.set_data(x_data_roof_abs[:idx+1], y_data_shear_abs[:idx+1]) 
        point.set_data([x_data_roof_abs[idx]], [y_data_shear_abs[idx]])
        
        current_disps = [0.0] + list(df_disp.iloc[idx, 1:].values)
        structure_line.set_data(current_disps, building_y_coords) 
        time_text_ax2.set_text(text_info)
        time_text_ax3.set_text(text_info)
        
        # 힌지 업데이트
        io_coords, ls_coords, cp_coords = [], [], []
        
        # [수정] 방향에 따른 변수 설정
        if direction == 'X':
            target_coord = params.get('plot_z_line_index', 0) * params['bay_width_z']
            coord_idx_check = 2 # Z 좌표를 확인
            coord_idx_plot = 0  # X 좌표를 플로팅
        else: # direction == 'Z'
            target_coord = params.get('plot_x_line_index', 0) * params['bay_width_x']
            coord_idx_check = 0 # X 좌표를 확인
            coord_idx_plot = 2  # Z 좌표를 플로팅
        
        # Tolerance for coordinate check
        tolerance = 0.1 # 10cm tolerance

        all_tags_to_plot = all_column_tags + all_beam_tags
        rot_df_map = {'col': df_col_rot, 'beam': df_beam_rot}
        
        for tag in all_tags_to_plot:
            df_to_use = rot_df_map['col'] if tag in all_column_tags else rot_df_map['beam']
            if df_to_use is None or tag not in all_line_elements: continue
            
            # [Fix] Handle case where rotation data is shorter than displacement data (due to zero-trimming)
            rot_idx = min(idx, len(df_to_use) - 1)
            
            ni_t, nj_t = all_line_elements[tag]
            ni_c, nj_c = node_coords.get(ni_t), node_coords.get(nj_t)
            if not ni_c or not nj_c: continue

            # 좌표 체크 (허용 오차 내에 있는 부재만 표시)
            if abs(ni_c[coord_idx_check] - target_coord) > tolerance:
                continue
            
            try:
                row = df_to_use.iloc[rot_idx]
            except IndexError:
                continue
            
            for loc_idx, ip in enumerate([ip_start, ip_end]):
                try:
                    # [수정] 곡률(kz, ky)을 읽어서 소성회전각(rad)으로 변환
                    # 가정: 소성힌지 길이 Lp = 0.5m (시각화용 근사값)
                    L_p_approx = 0.5 
                    kappa_z = row.get(f'Ele{tag}_IP{ip}_kz', 0)
                    kappa_y = row.get(f'Ele{tag}_IP{ip}_ky', 0)
                    
                    # 곡률(rad/m) * Lp(m) = 회전각(rad)
                    theta_p = (abs(kappa_z) + abs(kappa_y)) * L_p_approx
                    
                    loc = ni_c if loc_idx == 0 else nj_c
                    
                    if theta_p >= ROT_CP: cp_coords.append((loc[coord_idx_plot], loc[1]))
                    elif theta_p >= ROT_LS: ls_coords.append((loc[coord_idx_plot], loc[1]))
                    elif theta_p >= ROT_IO: io_coords.append((loc[coord_idx_plot], loc[1]))
                except KeyError:
                    continue
        
        hinge_io_plot_ax3.set_data(*zip(*io_coords) if io_coords else ([],[]))
        hinge_ls_plot_ax3.set_data(*zip(*ls_coords) if ls_coords else ([],[]))
        hinge_cp_plot_ax3.set_data(*zip(*cp_coords) if cp_coords else ([],[]))

        # [수정] 층간 변형각 절대값으로 표시
        drifts = [abs(current_disps[i+1] - current_disps[i]) / story_height * 100 for i in range(num_stories)]
        drift_profile_line.set_data(drifts, story_mid_heights)

        # [수정] 주요 이벤트 마커 절대값 좌표 사용
        if perf_points.get('yield_disp', 0) > 0 and x_data_roof_abs[idx] >= perf_points['yield_disp']:
            yield_point_artist.set_data([perf_points['yield_disp']], [perf_points.get('yield_shear', 0) / 1000])
        if perf_points.get('peak_disp', 0) > 0 and x_data_roof_abs[idx] >= perf_points['peak_disp']:
            peak_point_artist.set_data([perf_points['peak_disp']], [perf_points['peak_shear'] / 1000])
        if perf_points.get('collapse_disp') is not None and x_data_roof_abs[idx] >= perf_points['collapse_disp']:
            collapse_line_artist.set_data([perf_points['collapse_disp'], perf_points['collapse_disp']], [0, max_y_lim])
        
        if first_failure_event and x_data_roof_abs[idx] >= first_failure_event['roof_disp']:
            if not failure_marker_artist.get_xdata(): # 마커가 한번만 그려지도록
                failure_marker_artist.set_data([first_failure_event['roof_disp']], [first_failure_event['base_shear'] / 1000])
                ax1.text(first_failure_event['roof_disp'], first_failure_event['base_shear'] / 1000, 
                         f"  {first_failure_event['failure_type']}\n  Ele:{first_failure_event['element_tag']}",
                         color='darkred', va='bottom', ha='left', fontsize=9)

        return (line, point, structure_line, time_text_ax2, time_text_ax3, yield_point_artist, 
                peak_point_artist, collapse_line_artist, hinge_io_plot_ax3, hinge_ls_plot_ax3, 
                hinge_cp_plot_ax3, drift_profile_line, failure_marker_artist)

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
        animate(num_frames - 1) # 최종 상태로 업데이트
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
        animate_m_phi(num_frames) # 최종 상태로 업데이트
        fig.savefig(static_image_path, dpi=300, bbox_inches='tight')
        print(f"\nFinal static M-Phi plot saved to: {static_image_path}")
    except Exception as e:
        print(f"\n---! Error saving static M-Phi plot !---")
        print(f"Error details: {e}")
    
    plt.close(fig)