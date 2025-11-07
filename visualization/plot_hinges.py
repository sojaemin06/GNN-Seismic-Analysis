import matplotlib.pyplot as plt
import numpy as np
import sys

# 3D 플롯 모듈 가용성 확인
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Polygon
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False

# ### 8. 모듈형 함수: 소성/손상 분포도 플로팅 ###
def plot_plastic_damage_distribution(params, model_nodes_info, final_states_dfs):
    """
    [신규] 해석 최종 단계에서의 소성힌지(기둥/보) 및 손상(쉘) 분포도를
    논문 <그림 4-5>와 유사하게 2D 입면도로 플로팅합니다.
    """
    print("\nPlotting Plastic Hinge / Damage Distribution...")
    
    if not MPL_3D_AVAILABLE:
        print("Matplotlib이 없어 플로팅을 건너뜁니다.")
        return
        
    if not final_states_dfs:
        print("Warning: 'final_states_dfs' 데이터가 비어있습니다. 플로팅을 건너뜁니다.")
        return

    # --- [신규] 2번 요청: Z축 라인 선택 ---
    z_line_idx = params.get('plot_z_line_index', 0)
    target_z = z_line_idx * params['bay_width_z']
    print(f"Plotting static hinges for Z-Line {z_line_idx} (Z = {target_z:.1f}m)...")
    z_tolerance = 1e-6
    # --- [신규] 끝 ---

    # --- 데이터 로드 ---
    try:
        node_coords = model_nodes_info['all_node_coords']
        all_line_elements = model_nodes_info['all_line_elements']   # Dict {tag: (i,j)}
        all_shell_elements = model_nodes_info['all_shell_elements'] # Dict {tag: (n1,n2,n3,n4)}
        
        all_column_tags = model_nodes_info.get('all_column_tags', [])
        # [수정] 힌지 플로팅을 위해 build_model에서 반환된 축별 보 태그 사용
        all_beam_tags_type2 = model_nodes_info.get('all_beam_tags_type2', [])
        all_beam_tags_type3 = model_nodes_info.get('all_beam_tags_type3', [])
        
        col_rot_final = final_states_dfs.get('col_rot_df').iloc[-1:] if final_states_dfs.get('col_rot_df') is not None else None
        beam_rot_final = final_states_dfs.get('beam_rot_df').iloc[-1:] if final_states_dfs.get('beam_rot_df') is not None else None
        wall_force_final = final_states_dfs.get('wall_forces_df').iloc[-1:] if final_states_dfs.get('wall_forces_df') is not None else None
        
        if col_rot_final is None and beam_rot_final is None:
             print("Warning: No final rotation data found in 'final_states_dfs'.")
             return

        num_int_pts = params.get('num_int_pts', 5)
        ip_start = 1
        ip_end = num_int_pts
        
    except KeyError as e:
        print(f"Error: 힌지 플로팅에 필요한 모델 정보가 없습니다: {e}")
        return
    except Exception as e:
        print(f"Error: 힌지 플로팅 데이터 로드 중 오류: {e}")
        return

    # --- 2D 입면도 (X-Y) 플롯 설정 ---
    fig_2d, ax_2d = plt.subplots(figsize=(10, 12))
    
    # --- 1. 배경 구조물 플로팅 (회색) ---
    base_color = 'black'
    base_lw = 1
    
    # [수정] 2번 요청: Z축 필터링
    for (node_i_tag, node_j_tag) in all_line_elements.values():
        n_i = node_coords.get(node_i_tag)
        n_j = node_coords.get(node_j_tag)
        if n_i and n_j:
            if abs(n_i[2] - target_z) < z_tolerance and abs(n_j[2] - target_z) < z_tolerance:
                ax_2d.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], '-', color=base_color, linewidth=base_lw, zorder=1)

    # --- 2. 소성 힌지 심볼 플로팅 ---
    ROT_IO = 0.005  # Immediate Occupancy
    ROT_LS = 0.02   # Life Safety
    ROT_CP = 0.04   # Collapse Prevention
    
    mkr_cp = {'marker': 's', 'color': 'red', 'markersize': 12, 'mew': 1.5, 'zorder': 10, 'label': 'Collapse Prevention (CP)', 'mec': 'black'}
    mkr_ls = {'marker': 'D', 'color': 'orange', 'markersize': 10, 'mew': 1.0, 'zorder': 9, 'label': 'Life Safety (LS)', 'mec': 'black'}
    mkr_io = {'marker': 'o', 'color': 'blue', 'markersize': 8, 'mew': 0.5, 'zorder': 8, 'label': 'Immediate Occupancy (IO)', 'mec': 'black'}

    # 2-1. 기둥 힌지 플로팅
    if col_rot_final is not None:
        for ele_tag, (node_i_tag, node_j_tag) in all_line_elements.items():
            if ele_tag not in all_column_tags:
                continue
            
            n_i_coords = node_coords.get(node_i_tag)
            n_j_coords = node_coords.get(node_j_tag)
            if not n_i_coords or not n_j_coords: continue

            if abs(n_i_coords[2] - target_z) > z_tolerance:
                continue
            
            locations = [n_i_coords, n_j_coords]
            ips = [ip_start, ip_end]
            
            for loc_coords, ip in zip(locations, ips):
                try:
                    rot_ry = col_rot_final[f'Ele{ele_tag}_IP{ip}_ry'].values[0]
                    rot_rz = col_rot_final[f'Ele{ele_tag}_IP{ip}_rz'].values[0]
                    theta_p = abs(rot_ry) + abs(rot_rz) # [수정] 기둥은 양방향 휨을 모두 고려 (X-Pushover는 주로 ry)
                    
                    if theta_p >= ROT_CP:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_cp)
                    elif theta_p >= ROT_LS:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_ls) 
                    elif theta_p >= ROT_IO:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_io)
                except KeyError:
                    continue 
    
    # 2-2. 보 힌지 플로팅
    if beam_rot_final is not None:
        
        # 2-2a. 보 (Type 3, X-방향, rz 휨)
        for ele_tag in all_beam_tags_type3: # Type 3만
            if ele_tag not in all_line_elements: continue
            node_i_tag, node_j_tag = all_line_elements[ele_tag]
                
            n_i_coords = node_coords.get(node_i_tag)
            n_j_coords = node_coords.get(node_j_tag)
            if not n_i_coords or not n_j_coords: continue

            if abs(n_i_coords[2] - target_z) > z_tolerance:
                continue

            locations = [n_i_coords, n_j_coords]
            ips = [ip_start, ip_end]

            for loc_coords, ip in zip(locations, ips):
                try:
                    theta_p = abs(beam_rot_final[f'Ele{ele_tag}_IP{ip}_rz'].values[0]) # [수정] Type 3는 rz (Transf 3)
                    
                    if theta_p >= ROT_CP:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_cp)
                    elif theta_p >= ROT_LS:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_ls)
                    elif theta_p >= ROT_IO:
                        ax_2d.plot(loc_coords[0], loc_coords[1], **mkr_io)
                except KeyError:
                    continue
    
    # --- 3. 플롯 저장 ---
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    path_2d = output_dir / f"{analysis_name}_model_2D_PlasticHinge_Plot_Z{z_line_idx}.png" 

    ax_2d.set_xlabel('X (m)')
    ax_2d.set_ylabel('Y (Height) (m)')
    
    ax_2d.set_title(f'Plastic Damage Distribution (Frame at Z = {target_z:.1f}m)\n(Final Step: {params["target_drift"]*100:.1f}% Drift)')
    
    ax_2d.axis('equal')
    ax_2d.grid(True)
    
    handles, labels = ax_2d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax_2d.legend(by_label.values(), by_label.keys(), loc='lower right')

    try:
        fig_2d.savefig(path_2d, dpi=300, bbox_inches='tight')
        print(f"Matplotlib 2D Plastic Hinge plot saved to: {path_2d}")
    except Exception as e:
        print(f"---! Error saving Plastic Hinge plot: {e}")

    plt.close(fig_2d)
    plt.close('all')