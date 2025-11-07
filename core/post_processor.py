import numpy as np
import pandas as pd
import traceback

# ### 6. 모듈형 함수: 결과 처리 및 CSV 저장 ###
def process_pushover_results(params, model_nodes):
    """
    Pushover 해석으로 생성된 .out 파일들을 읽어 CSV로 변환 및 저장합니다.
    [수정] M-Phi 레코더의 열 개수(13->9) 및 인덱스(6,11 -> 2,5)를 수정합니다.
    [수정] 애니메이션을 위해 df_disp와 '전체 부재 데이터 DataFrame'을 반환합니다.
    """
    print("\nProcessing results and exporting to CSV...")
    df_disp = None 
    df_pushover_curve = None
    df_m_phi = None
    full_element_state_dfs = {} 
    
    try:
        output_dir = params['output_dir']
        analysis_name = params['analysis_name']
        
        # --- 6-1. 푸쉬오버 곡선 (Roof Disp vs Base Shear) ---
        path_disp = output_dir / f"{analysis_name}_all_floor_disp.out"
        path_base = output_dir / f"{analysis_name}_base_shear.out"

        if not path_disp.exists() or not path_base.exists():
            print(f"Error: Recorder files missing. Cannot process results.")
            return None, None, None, None
            
        roof_disp_data = np.loadtxt(path_disp)
        base_shear_data = np.loadtxt(path_base)
        
        if roof_disp_data.size == 0 or base_shear_data.size == 0:
            print("Warning: Recorder files are empty. Analysis may have failed immediately.")
            return None, None, None, None
            
        master_nodes = model_nodes['master_nodes']
        num_stories = len(master_nodes)
        
        if roof_disp_data.ndim == 1: roof_disp_data = roof_disp_data.reshape(-1, 1 + num_stories)
        if base_shear_data.ndim == 1: base_shear_data = base_shear_data.reshape(-1, 1 + len(model_nodes['base_nodes']))

        disp_cols = ['Time'] + [f'Floor{i+1}_DispX' for i in range(num_stories)]
        df_disp = pd.DataFrame(roof_disp_data, columns=disp_cols)
        
        base_nodes = model_nodes['base_nodes']
        base_node_tags = [str(tag) for tag in base_nodes]
        base_cols = ['Time'] + [f'BaseReaction_Node{tag}' for tag in base_node_tags]
        df_base_reactions = pd.DataFrame(base_shear_data, columns=base_cols)
        
        roof_disp = df_disp[f'Floor{num_stories}_DispX'].values
        base_shear_total = -df_base_reactions.iloc[:, 1:].sum(axis=1).values
        
        df_pushover_curve = pd.DataFrame({
            'Roof_Displacement_m': roof_disp,
            'Base_Shear_N': base_shear_total,
            'Pseudo_Time': df_disp['Time'].values
        })
        
        csv_path_curve = output_dir / f"{analysis_name}_pushover_curve.csv"
        df_pushover_curve.to_csv(csv_path_curve, index=False, float_format='%.6f')
        print(f"Pushover curve data saved to: {csv_path_curve}")

        
        # --- 6-2. 전단벽 내력(Forces) (모든 쉘) ---
        path_wall_forces = output_dir / f"{analysis_name}_all_wall_forces.out"
        all_shell_tags = model_nodes.get('all_shell_tags', [])
        if path_wall_forces.exists() and all_shell_tags:
            try:
                wall_forces_data = np.loadtxt(path_wall_forces)
                if wall_forces_data.ndim == 1: wall_forces_data = wall_forces_data.reshape(1, -1) 
                headers = ['Time']
                for ele_tag in all_shell_tags:
                    for gp in range(1, 5): 
                        headers.extend([
                            f'Ele{ele_tag}_GP{gp}_Nxx', f'Ele{ele_tag}_GP{gp}_Nyy', f'Ele{ele_tag}_GP{gp}_Nxy', 
                            f'Ele{ele_tag}_GP{gp}_Mxx', f'Ele{ele_tag}_GP{gp}_Myy', f'Ele{ele_tag}_GP{gp}_Mxy',
                            f'Ele{ele_tag}_GP{gp}_Vxz', f'Ele{ele_tag}_GP{gp}_Vyz'
                        ])

                if wall_forces_data.ndim > 1 and wall_forces_data.shape[1] == len(headers):
                    df_wall_forces = pd.DataFrame(wall_forces_data, columns=headers)
                    csv_path_forces = output_dir / f"{analysis_name}_all_wall_forces.csv"
                    df_wall_forces.to_csv(csv_path_forces, index=False, float_format='%.6e')
                    print(f"Wall forces data saved to: {csv_path_forces}")
                    full_element_state_dfs['wall_forces_df'] = df_wall_forces 
                else: 
                    print(f"\n---! Warning: Wall force data column mismatch !---")
            except Exception as e: print(f"\n---! Warning: Error processing wall force data: {e}")
        else: print("Skipping wall force processing (no file or no elements).")
        

        # --- 6-3. 기둥 소성회전 (모든 기둥) ---
        path_col_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation.out"
        all_column_tags = model_nodes.get('all_column_tags', [])
        if path_col_rot.exists() and all_column_tags:
            try:
                col_rot_data = np.loadtxt(path_col_rot)
                if col_rot_data.ndim == 1: col_rot_data = col_rot_data.reshape(1, -1)
                
                num_int_pts = params.get('num_int_pts', 5) 
                headers = ['Time']
                for ele_tag in all_column_tags: 
                    for ip in range(1, num_int_pts + 1):
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])
                        
                if col_rot_data.ndim > 1 and col_rot_data.shape[1] == len(headers):
                    df_col_rot = pd.DataFrame(col_rot_data, columns=headers)
                    csv_path_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation.csv"
                    df_col_rot.to_csv(csv_path_rot, index=False, float_format='%.6e')
                    print(f"Column plastic rotation data saved to: {csv_path_rot}")
                    full_element_state_dfs['col_rot_df'] = df_col_rot 
                else: 
                    print(f"\n---! Warning: Column plastic rotation data column mismatch !---")
            except Exception as e: print(f"\n---! Warning: Error processing column plastic rotation data !--- \nError details: {e}")
        else: print("Skipping column plastic rotation processing (no file or no elements).")
        
        # --- 6-4. 보 소성회전 (모든 보) ---
        path_beam_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation.out"
        all_beam_tags = model_nodes.get('all_beam_tags', [])
        if path_beam_rot.exists() and all_beam_tags:
            try:
                beam_rot_data = np.loadtxt(path_beam_rot)
                if beam_rot_data.ndim == 1: beam_rot_data = beam_rot_data.reshape(1, -1)

                num_int_pts = params.get('num_int_pts', 5)
                headers = ['Time']
                for ele_tag in all_beam_tags:
                    for ip in range(1, num_int_pts + 1):
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])

                if beam_rot_data.ndim > 1 and beam_rot_data.shape[1] == len(headers):
                    df_beam_rot = pd.DataFrame(beam_rot_data, columns=headers)
                    csv_path_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation.csv"
                    df_beam_rot.to_csv(csv_path_rot, index=False, float_format='%.6e')
                    print(f"Beam plastic rotation data saved to: {csv_path_rot}")
                    full_element_state_dfs['beam_rot_df'] = df_beam_rot 
                else:
                    print(f"\n---! Warning: Beam plastic rotation data column mismatch !---")
            except Exception as e:
                print(f"\n---! Warning: Error processing beam plastic rotation data !--- \nError details: {e}")
        else:
            print("Skipping beam plastic rotation processing (no file or no elements).")

        # --- 6-5. 기둥 내력 (모든 기둥) ---
        path_col_forces = output_dir / f"{analysis_name}_all_col_forces.out"
        if path_col_forces.exists() and all_column_tags:
            try:
                col_forces_data = np.loadtxt(path_col_forces)
                if col_forces_data.ndim == 1: col_forces_data = col_forces_data.reshape(1, -1)
                headers = ['Time']
                for ele_tag in all_column_tags: headers.extend([f'Ele{ele_tag}_I_P', f'Ele{ele_tag}_I_Vy', f'Ele{ele_tag}_I_Vz', f'Ele{ele_tag}_I_T', f'Ele{ele_tag}_I_My', f'Ele{ele_tag}_I_Mz', f'Ele{ele_tag}_J_P', f'Ele{ele_tag}_J_Vy', f'Ele{ele_tag}_J_Vz', f'Ele{ele_tag}_J_T', f'Ele{ele_tag}_J_My', f'Ele{ele_tag}_J_Mz'])
                if col_forces_data.ndim > 1 and col_forces_data.shape[1] == len(headers):
                    df_col_forces = pd.DataFrame(col_forces_data, columns=headers)
                    csv_path_forces = output_dir / f"{analysis_name}_all_col_forces.csv"
                    df_col_forces.to_csv(csv_path_forces, index=False, float_format='%.6e')
                    print(f"Column element forces data saved to: {csv_path_forces}")
                    full_element_state_dfs['col_forces_df'] = df_col_forces 
                else: 
                    print(f"\n---! Warning: Column element force data column mismatch !---")
            except Exception as e: print(f"\n---! Warning: Error processing column element force data !--- \nError details: {e}")
        else: print("Skipping column force processing (no file or no elements).")
        
        # --- [수정] 6-6. M-Phi (모멘트-곡률) 관계 ---
        path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele.out"
        if path_m_phi.exists():
            try:
                m_phi_data = np.loadtxt(path_m_phi)
                if m_phi_data.ndim == 1: m_phi_data = m_phi_data.reshape(1, -1)
                
                # [수정] OpenSees 'section' 레코더는 9개 열을 반환합니다:
                # [Time(0), P(1), Mz(2), My(3), eps(4), phi_z(5), phi_y(6), Vz(7), Vy(8)]
                if m_phi_data.shape[1] == 9: 
                    df_m_phi = pd.DataFrame({
                        'Time': m_phi_data[:, 0],
                        'Moment_My_N-m': m_phi_data[:, 3], 
                        'Curvature_phi_y_rad/m': m_phi_data[:, 6] 
                    })
                    csv_path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele.csv"
                    df_m_phi.to_csv(csv_path_m_phi, index=False, float_format='%.6e')
                    print(f"Target Moment-Curvature data saved to: {csv_path_m_phi}")
                else:
                    print(f"\n---! Warning: M-Phi data column mismatch! Expected 9, got {m_phi_data.shape[1]} ---")

            except Exception as e:
                print(f"\n---! Warning: Error processing M-Phi data: {e}")
        else:
            print("Skipping M-Phi processing (no file found).")
        # --- [수정] 끝 ---

        return df_pushover_curve, df_disp, full_element_state_dfs, df_m_phi

    except Exception as e:
        print(f"Error processing recorder data: {e}")
        print(traceback.format_exc())
        return None, None, None, None

# ### 7. 모듈형 함수: 성능점 계산 ###
def calculate_performance_points(df_curve):
    """
    푸쉬오버 곡선에서 주요 성능점(항복, 최대, 붕괴)을 계산합니다.
    """
    perf_points = {}
    if df_curve.empty:
        return {'peak_disp': 0, 'peak_shear': 0, 'collapse_disp': None, 'yield_disp': 0, 'yield_shear': 0}
        
    try:
        max_strength_index = df_curve['Base_Shear_N'].idxmax()
        perf_points['peak_disp'] = df_curve.loc[max_strength_index, 'Roof_Displacement_m']
        perf_points['peak_shear'] = df_curve.loc[max_strength_index, 'Base_Shear_N']
        post_peak_df = df_curve.loc[max_strength_index:]
        collapse_shear_limit = perf_points['peak_shear'] * 0.80 
        collapse_point = post_peak_df[post_peak_df['Base_Shear_N'] <= collapse_shear_limit]
        perf_points['collapse_disp'] = collapse_point.iloc[0]['Roof_Displacement_m'] if not collapse_point.empty else None
        
        # 항복점 계산 (FEMA 기준 근사)
        yield_shear_approx = perf_points['peak_shear'] * 0.60
        yield_point_df = df_curve[df_curve['Base_Shear_N'] >= yield_shear_approx]
        
        if not yield_point_df.empty:
            # 60% 지점을 지나는 할선(secant)의 기울기 계산
            first_yield_approx_idx = yield_point_df.index[0]
            if first_yield_approx_idx == 0: # 0번 인덱스면 기울기 계산 불가
                 raise Exception("Yield point at index 0")
                 
            K_secant = df_curve.loc[first_yield_approx_idx, 'Base_Shear_N'] / df_curve.loc[first_yield_approx_idx, 'Roof_Displacement_m']
            
            # 유효 항복 변위 (D_y)
            D_y = perf_points['peak_shear'] / K_secant
            
            # D_y에 가장 가까운 변위 찾기
            yield_index = (df_curve['Roof_Displacement_m'] - D_y).abs().idxmin()
            
            perf_points['yield_disp'] = df_curve.loc[yield_index, 'Roof_Displacement_m']
            perf_points['yield_shear'] = df_curve.loc[yield_index, 'Base_Shear_N']
        
        else: # 60%에 도달 못한 경우
             raise Exception("Curve did not reach 60% of peak shear")
            
    except Exception as e:
        print(f"Warning: Could not calculate performance points. Defaulting to 0. Error: {e}")
        perf_points['peak_disp'] = perf_points.get('peak_disp', 0.0)
        perf_points['peak_shear'] = perf_points.get('peak_shear', 0.0)
        perf_points['collapse_disp'] = perf_points.get('collapse_disp', None)
        perf_points['yield_disp'] = 0.0
        perf_points['yield_shear'] = 0.0

    return perf_points