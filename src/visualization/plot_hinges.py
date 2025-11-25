import matplotlib.pyplot as plt
import numpy as np
import sys

# 3D 플롯 모듈 가용성 확인
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False

# ### 8. 모듈형 함수: 소성/손상 분포도 플로팅 ###
def plot_plastic_damage_distribution(params, model_nodes_info, final_states_dfs, direction='X'):
    """
    [수정] 해석 최종 단계에서의 소성힌지 분포도를 2D 입면도로 플로팅합니다.
           direction 파라미터에 따라 X-Y 또는 Y-Z 입면도를 선택적으로 그립니다.
    """
    print(f"\nPlotting Plastic Hinge / Damage Distribution for {direction}-direction...")
    
    if not MPL_3D_AVAILABLE or not final_states_dfs:
        print("Warning: Matplotlib or final_states_dfs data is not available. Skipping plotting.")
        return

    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    
    try:
        node_coords = model_nodes_info['all_node_coords']
        all_line_elements = model_nodes_info['all_line_elements']
        all_column_tags = model_nodes_info.get('all_column_tags', [])
        all_beam_tags = model_nodes_info.get('all_beam_tags', []) # 모든 보 태그
        
        col_rot_final = final_states_dfs.get('col_rot_df').iloc[-1:] if final_states_dfs.get('col_rot_df') is not None else None
        beam_rot_final = final_states_dfs.get('beam_rot_df').iloc[-1:] if final_states_dfs.get('beam_rot_df') is not None else None
        
        if col_rot_final is None and beam_rot_final is None:
             print("Warning: No final rotation data found. Skipping hinge plotting.")
             return

        num_int_pts = params.get('num_int_pts', 5)
        ip_start, ip_end = 1, num_int_pts
        
    except Exception as e:
        print(f"Error: 힌지 플로팅 데이터 로드 중 오류: {e}")
        return

    fig_2d, ax_2d = plt.subplots(figsize=(10, 12))
    base_color, base_lw = 'black', 1
    
    ROT_IO, ROT_LS, ROT_CP = 0.005, 0.02, 0.04
    mkr_cp = {'marker': 's', 'color': 'red', 'markersize': 12, 'mew': 1.5, 'zorder': 10, 'label': 'Collapse Prevention (CP)', 'mec': 'black'}
    mkr_ls = {'marker': 'D', 'color': 'orange', 'markersize': 10, 'mew': 1.0, 'zorder': 9, 'label': 'Life Safety (LS)', 'mec': 'black'}
    mkr_io = {'marker': 'o', 'color': 'blue', 'markersize': 8, 'mew': 0.5, 'zorder': 8, 'label': 'Immediate Occupancy (IO)', 'mec': 'black'}
    
    if direction == 'X':
        z_line_idx = params.get('plot_z_line_index', 0)
        target_coord = z_line_idx * params['bay_width_z']
        tolerance = 1e-6
        
        # 배경 프레임
        for (node_i_tag, node_j_tag) in all_line_elements.values():
            n_i, n_j = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if n_i and n_j and abs(n_i[2] - target_coord) < tolerance and abs(n_j[2] - target_coord) < tolerance:
                ax_2d.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], '-', color=base_color, linewidth=base_lw, zorder=1)

        # 힌지
        all_tags_to_plot = all_column_tags + all_beam_tags
        rot_df_map = {'col': col_rot_final, 'beam': beam_rot_final}
        
        for ele_tag in all_tags_to_plot:
            node_i_tag, node_j_tag = all_line_elements[ele_tag]
            n_i_coords, n_j_coords = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if not n_i_coords or not n_j_coords or abs(n_i_coords[2] - target_coord) > tolerance:
                continue

            df_to_use = rot_df_map['col'] if ele_tag in all_column_tags else rot_df_map['beam']
            if df_to_use is None: continue

            for loc_coords, ip in zip([n_i_coords, n_j_coords], [ip_start, ip_end]):
                try:
                    # 기둥은 양방향, 보는 주축 휨만 고려
                    if ele_tag in all_column_tags:
                        theta_p = abs(df_to_use[f'Ele{ele_tag}_IP{ip}_ry'].values[0]) + abs(df_to_use[f'Ele{ele_tag}_IP{ip}_rz'].values[0])
                    else: # 보
                        theta_p = abs(df_to_use[f'Ele{ele_tag}_IP{ip}_rz'].values[0])
                    
                    if theta_p >= ROT_CP: ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_cp)
                    elif theta_p >= ROT_LS: ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_ls)
                    elif theta_p >= ROT_IO: ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_io)
                except KeyError: continue
                
        ax_2d.set_title(f'Plastic Hinge Distribution (Frame at Z = {target_coord:.1f}m)')
        ax_2d.set_xlabel('X (m)')

    elif direction == 'Z':
        x_line_idx = params.get('plot_x_line_index', 0)
        target_coord = x_line_idx * params['bay_width_x']
        tolerance = 1e-6
        
        # 배경 프레임
        for (node_i_tag, node_j_tag) in all_line_elements.values():
            n_i, n_j = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if n_i and n_j and abs(n_i[0] - target_coord) < tolerance and abs(n_j[0] - target_coord) < tolerance:
                ax_2d.plot([n_i[2], n_j[2]], [n_i[1], n_j[1]], '-', color=base_color, linewidth=base_lw, zorder=1)
        
        # 힌지
        all_tags_to_plot = all_column_tags + all_beam_tags
        rot_df_map = {'col': col_rot_final, 'beam': beam_rot_final}

        for ele_tag in all_tags_to_plot:
            node_i_tag, node_j_tag = all_line_elements[ele_tag]
            n_i_coords, n_j_coords = node_coords.get(node_i_tag), node_coords.get(node_j_tag)
            if not n_i_coords or not n_j_coords or abs(n_i_coords[0] - target_coord) > tolerance:
                continue

            df_to_use = rot_df_map['col'] if ele_tag in all_column_tags else rot_df_map['beam']
            if df_to_use is None: continue

            for loc_coords, ip in zip([n_i_coords, n_j_coords], [ip_start, ip_end]):
                try:
                    if ele_tag in all_column_tags:
                        theta_p = abs(df_to_use[f'Ele{ele_tag}_IP{ip}_ry'].values[0]) + abs(df_to_use[f'Ele{ele_tag}_IP{ip}_rz'].values[0])
                    else: # 보
                        theta_p = abs(df_to_use[f'Ele{ele_tag}_IP{ip}_ry'].values[0])
                    
                    if theta_p >= ROT_CP: ax_2d.plot(loc_coords[2], loc_coords[1], **mkr_cp)
                    elif theta_p >= ROT_LS: ax_2d.plot(loc_coords[2], loc_coords[1], **mkr_ls)
                    elif theta_p >= ROT_IO: ax_2d.plot(loc_coords[2], loc_coords[1], **mkr_io)
                except KeyError: continue
                
        ax_2d.set_title(f'Plastic Hinge Distribution (Frame at X = {target_coord:.1f}m)')
        ax_2d.set_xlabel('Z (m)')

    # 공통 플롯 설정
    ax_2d.set_ylabel('Y (Height) (m)')
    ax_2d.axis('equal')
    ax_2d.grid(True)
    
    handles, labels = ax_2d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_2d.legend(by_label.values(), by_label.keys(), loc='lower right')

    try:
        path_2d = output_dir / f"{analysis_name}_model_2D_PlasticHinge_Plot.png"
        fig_2d.savefig(path_2d, dpi=300, bbox_inches='tight')
        print(f"Matplotlib 2D Plastic Hinge plot saved to: {path_2d}")
    except Exception as e:
        print(f"---! Error saving Plastic Hinge plot: {e}")

    plt.close(fig_2d)
    plt.close('all')