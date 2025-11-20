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

    # --- 2. 파이버 단면 플로팅 (기둥: 101, 보: 102) ---
    try:
        # [수정] 그룹화된 파라미터에서 대표 단면 크기를 가져옴
        col_dims = params['col_props_by_group'][0]['exterior']
        beam_dims = params['beam_props_by_group'][0]['exterior']
        col_width = col_dims[0]; col_depth = col_dims[1]
        beam_width = beam_dims[0]; beam_depth = beam_dims[1]
        
        # [수정] params 딕셔너리에서 철근 개수 가져오기
        cover = params['cover']; As = params['rebar_Area']
        num_bars_x = params.get('num_bars_x', 3)
        num_bars_z = params.get('num_bars_z', 3)
        num_bars_top = params.get('num_bars_top', 4)
        num_bars_bot = params.get('num_bars_bot', 4)

        core_fib_y = 10; core_fib_z = 10; cover_layers = 1
        total_fib_y = core_fib_y + (2 * cover_layers); total_fib_z = core_fib_z + (2 * cover_layers)
        y_core = col_depth/2.0 - cover; z_core = col_width/2.0 - cover
        
        # [수정] 기둥 단면 철근 개수를 변수(num_bars_x, num_bars_z)로 수정
        # 참고: 현재 layer 정의는 각 코어 경계에 철근을 배치하는 방식입니다.
        col_section_list = [
            ['section', 'Fiber', 101, '-torsion', 4],
            ['patch', 'rect', 1, total_fib_y, total_fib_z, -col_depth/2.0, -col_width/2.0, col_depth/2.0, col_width/2.0],
            ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core, -z_core, y_core, z_core],
            ['layer', 'straight', 3, num_bars_x, As, -y_core, -z_core, -y_core, z_core], # Left
            ['layer', 'straight', 3, num_bars_x, As, y_core, -z_core, y_core, z_core],   # Right
            ['layer', 'straight', 3, num_bars_z, As, -y_core, z_core, y_core, z_core],   # Top
            ['layer', 'straight', 3, num_bars_z, As, -y_core, -z_core, y_core, -z_core]  # Bottom
        ]
        
        y_core_b = beam_depth/2.0 - cover; z_core_b = beam_width/2.0 - cover
        
        # [수정] 보 단면 철근 개수를 변수(num_bars_top, num_bars_bot)로 수정
        beam_section_list = [
            ['section', 'Fiber', 102, '-torsion', 4],
            ['patch', 'rect', 1, total_fib_y, total_fib_z, -beam_depth/2.0, -beam_width/2.0, beam_depth/2.0, beam_width/2.0],
            ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core_b, -z_core_b, y_core_b, z_core_b],
            ['layer', 'straight', 3, num_bars_top, As, y_core_b, -z_core_b, y_core_b, z_core_b],
            ['layer', 'straight', 3, num_bars_bot, As, -y_core_b, -z_core_b, -y_core_b, z_core_b]
        ]
        matcolor = ['w', 'lightgrey', 'gold', 'r', 'w'] 
        col_section_path = output_dir / f"{analysis_name}_opsvis_sec101_col.png"
        ovs.plot_fiber_section(col_section_list, matcolor=matcolor) 
        plt.axis('equal')
        plt.savefig(str(col_section_path), dpi=150, bbox_inches='tight') 
        plt.clf() 
        print(f"Column Fiber Section (101) saved to: {col_section_path}")
        beam_section_path = output_dir / f"{analysis_name}_opsvis_sec102_beam.png"
        ovs.plot_fiber_section(beam_section_list, matcolor=matcolor)
        plt.axis('equal')
        plt.savefig(str(beam_section_path), dpi=150, bbox_inches='tight')
        plt.clf() 
        print(f"Beam Fiber Section (102) saved to: {beam_section_path}")
    except Exception as e:
        print(f"---! Error plotting opsvis fiber sections: {e}")
    
    plt.close('all') # 모든 잔여 플롯 닫기