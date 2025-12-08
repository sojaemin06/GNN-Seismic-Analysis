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
    [수정] params 구조 변경에 따라 rebar 정보를 동적으로 불러오도록 수정합니다.
    [수정] Fiber Section 시각화 시 좌표 및 레이어 정의를 model_builder.py와 일치시켜 정확한 형상을 그립니다.
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
        cover = params['cover']
        # 색상 설정 (0:?, 1:ConcCore, 2:ConcCover, 3:Steel)
        # opsvis matcolor: [tag0_color, tag1_color, tag2_color, tag3_color, ...]
        # tag 1: unconfined conc (cover) -> lightgrey
        # tag 2: confined conc (core) ->  grey (or darker)
        # tag 3: steel -> red (or black)
        
        # model_builder.py의 재료 태그:
        # 1: Unconfined Concrete (Cover)
        # 2: Confined Concrete (Core)
        # 3: Steel
        matcolor = ['w', 'lightgrey', 'darkgrey', 'red', 'w'] 

        # --- 기둥 단면 생성 ---
        col_groups = params.get('col_props_by_group', {})
        for grp_idx, props in col_groups.items():
            for loc in ['exterior', 'interior']:
                if loc not in props: continue
                
                prop_data = props[loc]
                dims = prop_data['dims']
                rebar_info = prop_data['rebar']
                
                # dims = (Depth, Width) ?? -> model_builder.py: dims[1]=y(Depth?), dims[0]=z(Width?)
                # model_builder.py:
                # y_core = dims[1]/2.0 - cover  (Depth 방향)
                # z_core = dims[0]/2.0 - cover  (Width 방향)
                # patch rect ... -dims[1]/2.0 ... (y좌표가 Depth)
                
                # 따라서:
                col_depth = dims[1] # y축 길이
                col_width = dims[0] # z축 길이
                
                rebar_area = rebar_info['area']
                nz = rebar_info['nz'] # Z 방향 (Width) 면의 철근 개수 (Top/Bottom)
                nx = rebar_info['nx'] # X(Y) 방향 (Depth) 면의 철근 개수 (Left/Right)
                
                core_fib_y = 10; core_fib_z = 10; cover_layers = 1
                total_fib_y = core_fib_y + (2 * cover_layers)
                total_fib_z = core_fib_z + (2 * cover_layers)
                
                y_core = col_depth/2.0 - cover
                z_core = col_width/2.0 - cover
                
                # 단면 리스트 구성 (model_builder.py 로직 복사)
                sec_list = [
                    ['section', 'Fiber', 101, '-torsion', 4],
                    # Tag 1: Cover (Unconfined) - 전체 사각형
                    ['patch', 'rect', 1, total_fib_y, total_fib_z, -col_depth/2.0, -col_width/2.0, col_depth/2.0, col_width/2.0],
                    # Tag 2: Core (Confined) - 내부 사각형 (덮어쓰기)
                    ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core, -z_core, y_core, z_core],
                    
                    # Tag 3: Steel
                    # Top/Bottom Layers (along Z-axis, constant Y)
                    # nz 개수, y = +y_core (Top), y = -y_core (Bottom)
                    # 좌표: (y_start, z_start, y_end, z_end)
                    # model_builder: -y_core, z_core, y_core, z_core ??? -> model_builder 좌표 순서 확인 필요.
                    # OpenSees patch/layer 명령: y, z 좌표계 (Local)
                    # model_builder.py: 
                    # ops.layer('straight', 3, nz, area, -y_core, z_core, y_core, z_core) -> y가 변함?? 이건 Right side 인데?
                    
                    # [중요] model_builder.py 의 layer 정의 재확인
                    # ops.layer('straight', 3, nz, area, -y_core, z_core, y_core, z_core) 
                    # -> y: -y_core to y_core (Depth 방향 변함), z: z_core (Width 고정) => 이건 Right Face!
                    
                    # 아하! model_builder.py의 변수명 nz, nx가 헷갈리게 되어 있었을 수 있음.
                    # model_builder.py 코드:
                    # ops.layer(..., nz, ..., -y_core, z_core, y_core, z_core) -> Right Face (z=+z_core)
                    # ops.layer(..., nz, ..., -y_core, -z_core, y_core, -z_core) -> Left Face (z=-z_core)
                    # ops.layer(..., nx-2, ..., -y_core, -z_core+cov, -y_core, z_core-cov) -> Bottom Face (y=-y_core)
                    # ops.layer(..., nx-2, ..., y_core, -z_core+cov, y_core, z_core-cov) -> Top Face (y=+y_core)
                    
                    # [결론] 
                    # nz 변수는 "Z축 좌표가 고정된 면(Left/Right)에 배치된 철근 수" -> 즉 세로(Depth) 방향 철근 수
                    # nx 변수는 "Y축 좌표가 고정된 면(Top/Bottom)에 배치된 철근 수" -> 즉 가로(Width) 방향 철근 수
                    
                    # 따라서 opsvis도 이에 맞춰야 함.
                    
                    # Left Face (z = -z_core)
                    ['layer', 'straight', 3, nz, rebar_area, -y_core, -z_core, y_core, -z_core],
                    # Right Face (z = +z_core)
                    ['layer', 'straight', 3, nz, rebar_area, -y_core, z_core, y_core, z_core]
                ]
                
                # Top/Bottom (between corners)
                if nx > 2:
                    # Bottom Face (y = -y_core)
                    sec_list.append(['layer', 'straight', 3, nx - 2, rebar_area, -y_core, -z_core + cover, -y_core, z_core - cover])
                    # Top Face (y = +y_core)
                    sec_list.append(['layer', 'straight', 3, nx - 2, rebar_area, y_core, -z_core + cover, y_core, z_core - cover])
                
                filename = f"{analysis_name}_opsvis_sec_Col_Group{grp_idx}_{loc.capitalize()}.png"
                filepath = output_dir / filename
                
                ovs.plot_fiber_section(sec_list, matcolor=matcolor)
                plt.axis('equal')
                plt.title(f"Col Group {grp_idx} ({loc.capitalize()})\n{col_width:.2f}m(W) x {col_depth:.2f}m(D)\nMain Bars: {nx}(W) x {nz}(D)")
                plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
                plt.clf()
                print(f"Saved Section: {filename}")

        # --- 보 단면 생성 ---
        beam_groups = params.get('beam_props_by_group', {})
        for grp_idx, props in beam_groups.items():
            for loc in ['exterior', 'interior']:
                if loc not in props: continue
                
                prop_data = props[loc]
                dims = prop_data['dims']
                rebar_info = prop_data['rebar']
                
                beam_width = dims[0]
                beam_depth = dims[1]
                
                rebar_area = rebar_info['area']
                num_top = rebar_info['top']
                num_bot = rebar_info['bot']
                
                core_fib_y = 10; core_fib_z = 10; cover_layers = 1
                total_fib_y = core_fib_y + (2 * cover_layers)
                total_fib_z = core_fib_z + (2 * cover_layers)
                
                y_core_b = beam_depth/2.0 - cover
                z_core_b = beam_width/2.0 - cover
                
                sec_list = [
                    ['section', 'Fiber', 102, '-torsion', 4],
                    ['patch', 'rect', 1, total_fib_y, total_fib_z, -beam_depth/2.0, -beam_width/2.0, beam_depth/2.0, beam_width/2.0],
                    ['patch', 'rect', 2, core_fib_y, core_fib_z, -y_core_b, -z_core_b, y_core_b, z_core_b],
                    # Top bars (y = +y_core_b)
                    ['layer', 'straight', 3, num_top, rebar_area, y_core_b, -z_core_b, y_core_b, z_core_b],
                    # Bottom bars (y = -y_core_b)
                    ['layer', 'straight', 3, num_bot, rebar_area, -y_core_b, -z_core_b, -y_core_b, z_core_b]
                ]
                
                filename = f"{analysis_name}_opsvis_sec_Beam_Group{grp_idx}_{loc.capitalize()}.png"
                filepath = output_dir / filename
                
                ovs.plot_fiber_section(sec_list, matcolor=matcolor)
                plt.axis('equal')
                plt.title(f"Beam Group {grp_idx} ({loc.capitalize()})\n{beam_width:.2f}m(W) x {beam_depth:.2f}m(D)\nTop: {num_top}, Bot: {num_bot}")
                plt.savefig(str(filepath), dpi=100, bbox_inches='tight')
                plt.clf()
                print(f"Saved Section: {filename}")
                
    except Exception as e:
        print(f"---! Error plotting opsvis fiber sections: {e}")
    
    plt.close('all') # 모든 잔여 플롯 닫기