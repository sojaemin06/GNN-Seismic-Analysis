import matplotlib.pyplot as plt
import sys

# [수정] opsvis 임포트
try:
    import opsvis as ovs
    OPSVIS_AVAILABLE = True
except ImportError:
    print("Warning: opsvis를 찾을 수 없습니다. 'pip install opsvis'를 시도하세요.")
    OPSVIS_AVAILABLE = False


# ### 2.5. opsvis를 사용한 시각화 (Wireframe + Fiber Sections) ###
def plot_with_opsvis(params):
    """
    opsvis를 사용하여 모델 형상(와이어프레임)과 파이버 단면을 플로팅합니다.
    [수정] 축 라벨이 혼동되는 문제를 해결하기 위해 plt.gca()로 라벨을 강제 설정합니다.
    """
    if not OPSVIS_AVAILABLE:
        print("\nopsvis가 설치되어 있지 않아 시각화를 건너뜁니다.")
        return
    
    print("\nPlotting with opsvis (Wireframe & Fibers)...")
    output_dir = params['output_dir']
    analysis_name = params['analysis_name']

    # --- 1. 3D 모델 형상 플로팅 (Wireframe PNG) ---
    try:
        model_png_path = output_dir / f"{analysis_name}_opsvis_model_Wireframe.png"
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        ovs.plot_model(node_labels=0, element_labels=0, az_el=(-90, 15))
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y (Height)') 
        ax.set_zlabel('Z (Depth)') 
        
        plt.savefig(str(model_png_path), dpi=150, bbox_inches='tight')
        plt.close(fig) 
        print(f"opsvis 3D (Wireframe) model saved to: {model_png_path}")
        
    except Exception as e:
        print(f"---! Error plotting opsvis 3D wireframe model: {e}")
        plt.close('all') 

    # --- 2. 파이버 단면 플로팅 (모든 그룹 및 위치별 생성) ---
    try:
        # 공통 철근 정보
        cover = params['cover']
        As = params['rebar_Area']
        
        # 기본값 설정 (params에 없을 경우 대비)
        num_bars_x = params.get('num_bars_x', 3)
        num_bars_z = params.get('num_bars_z', 3)
        num_bars_top = params.get('num_bars_top', 4)
        num_bars_bot = params.get('num_bars_bot', 4)
        
        # 색상 설정
        matcolor = ['w', 'lightgrey', 'gold', 'r', 'w'] 

        # --- 기둥 단면 생성 ---
        col_groups = params.get('col_props_by_group', {})
        for grp_idx, props in col_groups.items():
            for loc in ['exterior', 'interior']:
                if loc not in props: continue
                
                dims = props[loc] # (width, depth) or similar
                col_width = dims[0]
                col_depth = dims[1]
                
                core_fib_y = 10; core_fib_z = 10; cover_layers = 1
                total_fib_y = core_fib_y + (2 * cover_layers)
                total_fib_z = core_fib_z + (2 * cover_layers)
                y_core = col_depth/2.0 - cover
                z_core = col_width/2.0 - cover
                
                # 단면 리스트 구성
                sec_list = [
                    ['section', 'Fiber', 101, '-torsion', 4], # Tag는 시각화용 임의값
                    ['patch', 'rect', 1, total_fib_y, total_fib_z, -col_depth/2.0, -col_width/2.0, col_depth/2.0, col_width/2.0],
                    ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core, -z_core, y_core, z_core],
                    ['layer', 'straight', 3, num_bars_x, As, -y_core, -z_core, -y_core, z_core], # Left
                    ['layer', 'straight', 3, num_bars_x, As, y_core, -z_core, y_core, z_core],   # Right
                    ['layer', 'straight', 3, num_bars_z, As, -y_core, z_core, y_core, z_core],   # Top
                    ['layer', 'straight', 3, num_bars_z, As, -y_core, -z_core, y_core, -z_core]  # Bottom
                ]
                
                # 파일명 예: ..._opsvis_sec_Col_Group0_Exterior.png
                filename = f"{analysis_name}_opsvis_sec_Col_Group{grp_idx}_{loc.capitalize()}.png"
                filepath = output_dir / filename
                
                ovs.plot_fiber_section(sec_list, matcolor=matcolor)
                plt.axis('equal')
                # 제목 추가 (선택사항)
                plt.title(f"Col Group {grp_idx} ({loc.capitalize()})\n{col_width:.2f}m x {col_depth:.2f}m")
                plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
                plt.clf()
                print(f"Saved Section: {filename}")

        # --- 보 단면 생성 ---
        beam_groups = params.get('beam_props_by_group', {})
        for grp_idx, props in beam_groups.items():
            for loc in ['exterior', 'interior']:
                if loc not in props: continue
                
                dims = props[loc]
                beam_width = dims[0]
                beam_depth = dims[1]
                
                core_fib_y = 10; core_fib_z = 10; cover_layers = 1
                total_fib_y = core_fib_y + (2 * cover_layers)
                total_fib_z = core_fib_z + (2 * cover_layers)
                
                y_core_b = beam_depth/2.0 - cover
                z_core_b = beam_width/2.0 - cover
                
                sec_list = [
                    ['section', 'Fiber', 102, '-torsion', 4],
                    ['patch', 'rect', 1, total_fib_y, total_fib_z, -beam_depth/2.0, -beam_width/2.0, beam_depth/2.0, beam_width/2.0],
                    ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core_b, -z_core_b, y_core_b, z_core_b],
                    ['layer', 'straight', 3, num_bars_top, As, y_core_b, -z_core_b, y_core_b, z_core_b],
                    ['layer', 'straight', 3, num_bars_bot, As, -y_core_b, -z_core_b, -y_core_b, z_core_b]
                ]
                
                filename = f"{analysis_name}_opsvis_sec_Beam_Group{grp_idx}_{loc.capitalize()}.png"
                filepath = output_dir / filename
                
                ovs.plot_fiber_section(sec_list, matcolor=matcolor)
                plt.axis('equal')
                plt.title(f"Beam Group {grp_idx} ({loc.capitalize()})\n{beam_width:.2f}m x {beam_depth:.2f}m")
                plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
                plt.clf()
                print(f"Saved Section: {filename}")
                
    except Exception as e:
        print(f"---! Error plotting opsvis fiber sections: {e}")
    
    plt.close('all') # 모든 잔여 플롯 닫기