import matplotlib.pyplot as plt
import numpy as np
import traceback
import sys

# [추가] 3D 플로팅 및 2D 패치(Polygon)를 위한 matplotlib 모듈 임포트
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Polygon
    MPL_3D_AVAILABLE = True
except ImportError:
    print("Warning: matplotlib 3D toolkit을 찾을 수 없습니다. 3D 플롯을 건너뜁니다.")
    MPL_3D_AVAILABLE = False

# ### [신규] 2. Matplotlib를 이용한 3D/2D 모델 시각화 (전단벽 포함) ###
def plot_model_matplotlib(params, model_nodes_info, direction='X'):
    """
    [수정] Matplotlib를 사용하여 3D 모델 형상과 2D 입면도를 PNG로 저장합니다.
           direction 파라미터에 따라 X-Y 또는 Y-Z 입면도를 선택적으로 그립니다.
    """
    print(f"\nPlotting model geometry using Matplotlib for {direction}-direction...")
    
    if not MPL_3D_AVAILABLE:
        print("Matplotlib 3D toolkit이 없어 플로팅을 건너뜁니다.")
        return

    try:
        node_coords_dict = model_nodes_info['all_node_coords']
        line_elements = model_nodes_info['all_line_elements']   
        shell_elements = model_nodes_info.get('all_shell_elements', {})
        
        if not node_coords_dict or (not line_elements and not shell_elements):
            print("Warning: 모델 기하정보가 비어있습니다. Matplotlib 플로팅을 건너뜁니다.")
            return
            
    except KeyError:
        print("Error: 'all_node_coords' 등 기하정보가 model_nodes_info에 없습니다.")
        return
    except Exception as e:
        print(f"Error: 모델 기하정보 로드 중 오류 발생: {e}")
        return
        
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']
    path_3d = output_dir / f"{analysis_name}_model_3D_Matplotlib.png"
    path_2d = output_dir / f"{analysis_name}_model_2D_Elevation_{direction}.png"

    # --- 3D 플롯 설정 ---
    fig_3d = plt.figure(figsize=(12, 12))
    ax_3d = fig_3d.add_subplot(111, projection='3d')

    # --- 2D 입면도 플롯 설정 ---
    fig_2d, ax_2d = plt.subplots(figsize=(8, 10))

    first_coord = list(node_coords_dict.values())[0]
    min_coords = np.array(first_coord)
    max_coords = np.array(first_coord)
    for coord in node_coords_dict.values():
        min_coords = np.minimum(min_coords, coord)
        max_coords = np.maximum(max_coords, coord)

    try:
        # --- 3D 선 요소(보/기둥) 플로팅 ---
        for (node_i_tag, node_j_tag) in line_elements.values():
            n_i = node_coords_dict.get(node_i_tag)
            n_j = node_coords_dict.get(node_j_tag)
            
            if n_i is None or n_j is None: continue
            ax_3d.plot([n_i[0], n_j[0]], [n_i[2], n_j[2]], [n_i[1], n_j[1]], 'b-', linewidth=0.5)

        # --- 2D 입면도 플로팅 (방향에 따라 분기) ---
        if direction == 'X':
            z_line_idx = params.get('plot_z_line_index', 0)
            target_z = z_line_idx * params['bay_width_z']
            z_tolerance = 1e-6
            
            for (node_i_tag, node_j_tag) in line_elements.values():
                n_i = node_coords_dict.get(node_i_tag)
                n_j = node_coords_dict.get(node_j_tag)
                if n_i and n_j and abs(n_i[2] - target_z) < z_tolerance and abs(n_j[2] - target_z) < z_tolerance:
                    ax_2d.plot([n_i[0], n_j[0]], [n_i[1], n_j[1]], 'b-', linewidth=0.5)
            
            ax_2d.set_xlabel('X (m)')
            ax_2d.set_ylabel('Y (Height) (m)')
            ax_2d.set_title(f'Model Elevation (Frame at Z={target_z:.1f}m)')

        elif direction == 'Z':
            x_line_idx = params.get('plot_x_line_index', 0)
            target_x = x_line_idx * params['bay_width_x']
            x_tolerance = 1e-6

            for (node_i_tag, node_j_tag) in line_elements.values():
                n_i = node_coords_dict.get(node_i_tag)
                n_j = node_coords_dict.get(node_j_tag)
                if n_i and n_j and abs(n_i[0] - target_x) < x_tolerance and abs(n_j[0] - target_x) < x_tolerance:
                    ax_2d.plot([n_i[2], n_j[2]], [n_i[1], n_j[1]], 'b-', linewidth=0.5)

            ax_2d.set_xlabel('Z (m)')
            ax_2d.set_ylabel('Y (Height) (m)')
            ax_2d.set_title(f'Model Elevation (Frame at X={target_x:.1f}m)')

        # --- 3D 쉘 요소(전단벽) 플로팅 (2D는 단순화를 위해 생략) ---
        for (n1_tag, n2_tag, n3_tag, n4_tag) in shell_elements.values():
            n1, n2, n3, n4 = (node_coords_dict.get(t) for t in [n1_tag, n2_tag, n3_tag, n4_tag])
            if n1 and n2 and n3 and n4:
                verts_3d = [[n1[0], n1[2], n1[1]], [n2[0], n2[2], n2[1]], [n3[0], n3[2], n3[1]], [n4[0], n4[2], n4[1]]]
                poly_3d = Poly3DCollection([verts_3d], facecolors='gray', edgecolors='black', alpha=0.5, linewidths=0.5)
                ax_3d.add_collection3d(poly_3d)

        # 3D 플롯 저장
        ax_3d.set_xlabel('X'); ax_3d.set_ylabel('Z'); ax_3d.set_zlabel('Y (Height)')
        max_range = np.array(max_coords - min_coords).max()
        if max_range == 0: max_range = 1.0 
        mid = (max_coords + min_coords) / 2.0
        ax_3d.set_xlim(mid[0] - max_range / 2, mid[0] + max_range / 2)
        ax_3d.set_zlim(mid[1] - max_range / 2, mid[1] + max_range / 2)
        ax_3d.set_ylim(mid[2] - max_range / 2, mid[2] + max_range / 2)
        ax_3d.view_init(elev=30, azim=-60)
        fig_3d.savefig(path_3d, dpi=300, bbox_inches='tight')
        print(f"Matplotlib 3D plot (with shells) saved to: {path_3d}")

        # 2D 플롯 저장
        ax_2d.axis('equal'); ax_2d.grid(True)
        fig_2d.savefig(path_2d, dpi=300, bbox_inches='tight')
        print(f"Matplotlib 2D Elevation plot saved to: {path_2d}")

    except Exception as e:
        print(f"\n---! Error saving Matplotlib geometry plot !---")
        print(traceback.format_exc())
    
    plt.close(fig_3d)
    plt.close(fig_2d)
    plt.close('all')