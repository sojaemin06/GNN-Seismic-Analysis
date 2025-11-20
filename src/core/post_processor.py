import numpy as np
import pandas as pd
import traceback

# ### 6. 모듈형 함수: 결과 처리 및 CSV 저장 ###
def process_pushover_results(params, model_nodes, direction='X'):
    """
    Pushover 해석으로 생성된 .out 파일들을 읽어 CSV로 변환 및 저장합니다.
    [수정] M-Phi 레코더의 열 개수(13->9) 및 인덱스(6,11 -> 2,5)를 수정합니다.
    [수정] 애니메이션을 위해 df_disp와 '전체 부재 데이터 DataFrame'을 반환합니다.
    [신규] 'direction' 파라미터를 추가하여 방향별 결과 파일을 처리합니다.
    """
    print(f"\nProcessing results for {direction}-direction and exporting to CSV...")
    df_disp = None 
    df_pushover_curve = None
    df_m_phi = None
    full_element_state_dfs = {} 
    
    try:
        output_dir = params['output_dir']
        analysis_name = params['analysis_name']
        
        # --- 6-1. 푸쉬오버 곡선 (Roof Disp vs Base Shear) ---
        path_disp = output_dir / f"{analysis_name}_all_floor_disp_{direction}.out"
        path_base = output_dir / f"{analysis_name}_base_shear_{direction}.out"

        if not path_disp.exists() or not path_base.exists():
            print(f"Error: Recorder files missing for {direction}-direction. Cannot process results.")
            return None, None, None, None
            
        roof_disp_data = np.loadtxt(path_disp)
        base_shear_data = np.loadtxt(path_base)
        
        if roof_disp_data.size == 0 or base_shear_data.size == 0:
            print(f"Warning: Recorder files for {direction}-direction are empty. Analysis may have failed immediately.")
            return None, None, None, None
            
        master_nodes = model_nodes['master_nodes']
        num_stories = len(master_nodes)
        
        # Determine the correct displacement column based on direction
        disp_col_name = f'Floor{num_stories}_Disp{direction}'
        
        if roof_disp_data.ndim == 1: roof_disp_data = roof_disp_data.reshape(-1, 1 + num_stories)
        if base_shear_data.ndim == 1: base_shear_data = base_shear_data.reshape(-1, 1 + len(model_nodes['base_nodes']))

        # Adjust column names based on direction
        disp_cols = ['Time'] + [f'Floor{i+1}_Disp{direction}' for i in range(num_stories)]
        df_disp = pd.DataFrame(roof_disp_data, columns=disp_cols)
        
        base_nodes = model_nodes['base_nodes']
        base_node_tags = [str(tag) for tag in base_nodes]
        base_cols = ['Time'] + [f'BaseReaction_Node{tag}' for tag in base_node_tags]
        df_base_reactions = pd.DataFrame(base_shear_data, columns=base_cols)
        
        roof_disp = df_disp[disp_col_name].values
        base_shear_total = -df_base_reactions.iloc[:, 1:].sum(axis=1).values
        
        df_pushover_curve = pd.DataFrame({
            'Roof_Displacement_m': roof_disp,
            'Base_Shear_N': base_shear_total,
            'Pseudo_Time': df_disp['Time'].values
        })
        
        csv_path_curve = output_dir / f"{analysis_name}_pushover_curve_{direction}.csv"
        df_pushover_curve.to_csv(csv_path_curve, index=False, float_format='%.6f')
        print(f"Pushover curve data saved to: {csv_path_curve}")

        # [신규] 해석이 조기 종료된 경우를 대비하여, 모든 후속 데이터프레임의 길이를
        # 푸쉬오버 곡선의 길이에 맞춤
        num_valid_steps = len(df_pushover_curve)

        
        # --- 6-2. 전단벽 내력(Forces) (모든 쉘) ---
        # Note: Wall forces are typically not direction-specific in their recorder output names,
        # but if they were, this would need adjustment. For now, assuming generic.
        path_wall_forces = output_dir / f"{analysis_name}_all_wall_forces.out"
        all_shell_tags = model_nodes.get('all_shell_tags', [])
        if path_wall_forces.exists() and all_shell_tags:
            try:
                wall_forces_data = np.loadtxt(path_wall_forces)
                if wall_forces_data.ndim == 1: wall_forces_data = wall_forces_data.reshape(1, -1) 
                headers = ['Time']
                # [수정] 'material stressAndStrain' 레코더는 GP당 6개 값(s11,s22,s12,e11,e22,e12)을 반환
                for ele_tag in all_shell_tags:
                    for gp in range(1, 5): 
                        headers.extend([
                            f'Ele{ele_tag}_GP{gp}_s11', f'Ele{ele_tag}_GP{gp}_s22', f'Ele{ele_tag}_GP{gp}_s12', 
                            f'Ele{ele_tag}_GP{gp}_e11', f'Ele{ele_tag}_GP{gp}_e22', f'Ele{ele_tag}_GP{gp}_e12'
                        ])

                if wall_forces_data.ndim > 1 and wall_forces_data.shape[1] == len(headers):
                    df_wall_forces = pd.DataFrame(wall_forces_data, columns=headers).head(num_valid_steps) # 길이 맞춤
                    csv_path_forces = output_dir / f"{analysis_name}_all_wall_forces.csv"
                    df_wall_forces.to_csv(csv_path_forces, index=False, float_format='%.6e')
                    print(f"Wall forces data saved to: {csv_path_forces}")
                    full_element_state_dfs['wall_forces_df'] = df_wall_forces 
                else: 
                    print(f"\n---! Warning: Wall material data column mismatch! Expected {len(headers)}, got {wall_forces_data.shape[1]} !---")
            except Exception as e: print(f"\n---! Warning: Error processing wall force data: {e}")
        else: print("Skipping wall force processing (no file or no elements).")
        

        # --- 6-3. 기둥 소성회전 (모든 기둥) ---
        path_col_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation_{direction}.out"
        all_column_tags = model_nodes.get('all_column_tags', [])
        if path_col_rot.exists() and all_column_tags:
            try:
                col_rot_data = np.loadtxt(path_col_rot)
                if col_rot_data.ndim == 1: col_rot_data = col_rot_data.reshape(1, -1)
                
                # [수정] 'plasticRotation' 레코더는 Lobatto 적분 사용 시 양단(2개)의 결과만 반환합니다.
                # num_int_pts 파라미터와 관계없이 2개의 적분점(ip=1, ip=num_int_pts)을 가정합니다.
                headers = ['Time']
                for ele_tag in all_column_tags: 
                    for ip in [1, params.get('num_int_pts', 5)]: # IP 1과 마지막 IP
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])
                        
                if col_rot_data.ndim > 1 and col_rot_data.shape[1] == len(headers):
                    df_col_rot = pd.DataFrame(col_rot_data, columns=headers).head(num_valid_steps) # 길이 맞춤
                    csv_path_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation_{direction}.csv"
                    df_col_rot.to_csv(csv_path_rot, index=False, float_format='%.6e')
                    print(f"Column plastic rotation data saved to: {csv_path_rot}")
                    full_element_state_dfs['col_rot_df'] = df_col_rot 
                else: 
                    print(f"\n---! Warning: Column plastic rotation data column mismatch! Expected {len(headers)}, got {col_rot_data.shape[1]} !---")
            except Exception as e: print(f"\n---! Warning: Error processing column plastic rotation data !--- \nError details: {e}")
        else: print("Skipping column plastic rotation processing (no file or no elements).")
        
        # --- 6-4. 보 소성회전 (모든 보) ---
        path_beam_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation_{direction}.out"
        all_beam_tags = model_nodes.get('all_beam_tags', [])
        if path_beam_rot.exists() and all_beam_tags:
            try:
                beam_rot_data = np.loadtxt(path_beam_rot)
                if beam_rot_data.ndim == 1: beam_rot_data = beam_rot_data.reshape(1, -1)

                # [수정] 'plasticRotation' 레코더는 Lobatto 적분 사용 시 양단(2개)의 결과만 반환합니다.
                # num_int_pts 파라미터와 관계없이 2개의 적분점(ip=1, ip=num_int_pts)을 가정합니다.
                headers = ['Time']
                for ele_tag in all_beam_tags:
                    for ip in [1, params.get('num_int_pts', 5)]: # IP 1과 마지막 IP
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])

                if beam_rot_data.ndim > 1 and beam_rot_data.shape[1] == len(headers):
                    df_beam_rot = pd.DataFrame(beam_rot_data, columns=headers).head(num_valid_steps) # 길이 맞춤
                    csv_path_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation_{direction}.csv"
                    df_beam_rot.to_csv(csv_path_rot, index=False, float_format='%.6e')
                    print(f"Beam plastic rotation data saved to: {csv_path_rot}")
                    full_element_state_dfs['beam_rot_df'] = df_beam_rot 
                else:
                    print(f"\n---! Warning: Beam plastic rotation data column mismatch! Expected {len(headers)}, got {beam_rot_data.shape[1]} !---")
            except Exception as e:
                print(f"\n---! Warning: Error processing beam plastic rotation data !--- \nError details: {e}")
        else:
            print("Skipping beam plastic rotation processing (no file or no elements).")

        # --- 6-5. 기둥 내력 (모든 기둥) ---
        path_col_forces = output_dir / f"{analysis_name}_all_col_forces_{direction}.out"
        if path_col_forces.exists() and all_column_tags:
            try:
                col_forces_data = np.loadtxt(path_col_forces)
                if col_forces_data.ndim == 1: col_forces_data = col_forces_data.reshape(1, -1)
                headers = ['Time']
                for ele_tag in all_column_tags: headers.extend([f'Ele{ele_tag}_I_P', f'Ele{ele_tag}_I_Vy', f'Ele{ele_tag}_I_Vz', f'Ele{ele_tag}_I_T', f'Ele{ele_tag}_I_My', f'Ele{ele_tag}_I_Mz', f'Ele{ele_tag}_J_P', f'Ele{ele_tag}_J_Vy', f'Ele{ele_tag}_J_Vz', f'Ele{ele_tag}_J_T', f'Ele{ele_tag}_J_My', f'Ele{ele_tag}_J_Mz'])
                if col_forces_data.ndim > 1 and col_forces_data.shape[1] == len(headers):
                    df_col_forces = pd.DataFrame(col_forces_data, columns=headers).head(num_valid_steps) # 길이 맞춤
                    csv_path_forces = output_dir / f"{analysis_name}_all_col_forces_{direction}.csv"
                    df_col_forces.to_csv(csv_path_forces, index=False, float_format='%.6e')
                    print(f"Column element forces data saved to: {csv_path_forces}")
                    full_element_state_dfs['col_forces_df'] = df_col_forces 
                else: 
                    print(f"\n---! Warning: Column element force data column mismatch !---")
            except Exception as e: print(f"\n---! Warning: Error processing column element force data !--- \nError details: {e}")
        else: print("Skipping column force processing (no file or no elements).")
        
        # --- [수정] 6-6. M-Phi (모멘트-곡률) 관계 ---
        path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele_{direction}.out"
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
                    }).head(num_valid_steps) # 길이 맞춤
                    csv_path_m_phi = output_dir / f"{analysis_name}_M_phi_target_ele_{direction}.csv"
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


def check_material_strain_failure(model_info, params):
    """
    섹션별 Recorder 파일에서 재료 변형률을 읽어, '내진성능 평가요령' 지침에 따른 한계치 초과 여부 확인.
    """
    failure_records = []
    
    # 지침에 따른 변형률 한계 정의
    # Concrete04 (비구속, 횡구속) - 압축 극한변형률 (양수 값으로 비교)
    CONCRETE_COMPRESSIVE_STRAIN_LIMIT = 0.003
    # MultiLinear (철근)
    REBAR_TENSILE_STRAIN_LIMIT = 0.05
    REBAR_COMPRESSIVE_STRAIN_LIMIT = 0.02 # 압축 변형률은 음수이므로 절대값으로 비교

    # 재료 태그 정보 (model_builder.py의 ops.uniaxialMaterial 정의 및 maxStrain recorder 출력 순서 참고)
    # Recorder 출력 컬럼: Time, mat1_max_strain, mat1_min_strain, mat2_max_strain, mat2_min_strain, ...
    # CSV로 읽을 때 Time이 인덱스가 되므로, 데이터 컬럼은 0부터 시작
    # mat_tag 1 (Concrete_Unconfined) -> col index 0 (max_strain), 1 (min_strain)
    # mat_tag 2 (Concrete_Confined)   -> col index 2 (max_strain), 3 (min_strain)
    # mat_tag 3 (Rebar)               -> col index 4 (max_strain), 5 (min_strain)
    MAT_INFO_COL_MAP = {
        1: {'type': 'Concrete_Unconfined', 'max_idx': 0, 'min_idx': 1},
        2: {'type': 'Concrete_Confined',   'max_idx': 2, 'min_idx': 3},
        3: {'type': 'Rebar',               'max_idx': 4, 'min_idx': 5}
    }

    if 'section_recorder_paths' not in model_info:
        # print("경고: model_info에 section_recorder_paths가 없습니다. 변형률 파괴 검사를 건너뜁니다.")
        return failure_records

    for sec_tag, recorder_path_str in model_info['section_recorder_paths'].items():
        recorder_path = Path(recorder_path_str)
        if not recorder_path.exists():
            # print(f"경고: Recorder 파일이 존재하지 않습니다: {recorder_path}")
            continue

        try:
            # Recorder 파일 읽기 (Time 컬럼을 인덱스로 사용)
            df_strains = pd.read_csv(recorder_path, sep='\s+', header=None, index_col=0)
            if df_strains.empty:
                continue

            # 각 재료별로 변형률 한계 초과 여부 확인
            for mat_tag, info in MAT_INFO_COL_MAP.items():
                mat_type = info['type']
                max_strain_col_idx = info['max_idx']
                min_strain_col_idx = info['min_idx']
                
                # 해당 컬럼이 존재하는지 확인
                if df_strains.shape[1] <= min_strain_col_idx:
                    continue

                if mat_type.startswith('Concrete'):
                    # Concrete는 압축 파괴가 중요하므로 min_strain (음수 값)의 절대값을 확인
                    concrete_comp_strains = df_strains.iloc[:, min_strain_col_idx].abs()
                    if (concrete_comp_strains > CONCRETE_COMPRESSIVE_STRAIN_LIMIT).any():
                        failure_step_time = df_strains.index[concrete_comp_strains > CONCRETE_COMPRESSIVE_STRAIN_LIMIT][0]
                        failure_records.append({
                            'section_tag': sec_tag,
                            'material_type': mat_type,
                            'failure_type': 'Concrete Compressive Strain Limit',
                            'exceeded_strain': df_strains.iloc[:, min_strain_col_idx].min(), # 실제 기록된 음수 값
                            'limit': -CONCRETE_COMPRESSIVE_STRAIN_LIMIT, # 지침을 음수로 표시
                            'at_time': failure_step_time
                        })
                elif mat_type == 'Rebar':
                    rebar_tensile_strains = df_strains.iloc[:, max_strain_col_idx] # max_strain (인장, 양수 값)
                    rebar_comp_strains = df_strains.iloc[:, min_strain_col_idx].abs() # abs(min_strain) (압축, 음수 값의 절대값)
                    
                    if (rebar_tensile_strains > REBAR_TENSILE_STRAIN_LIMIT).any():
                        failure_step_time = df_strains.index[rebar_tensile_strains > REBAR_TENSILE_STRAIN_LIMIT][0]
                        failure_records.append({
                            'section_tag': sec_tag,
                            'material_type': mat_type,
                            'failure_type': 'Rebar Tensile Strain Limit',
                            'exceeded_strain': rebar_tensile_strains.max(),
                            'limit': REBAR_TENSILE_STRAIN_LIMIT,
                            'at_time': failure_step_time
                        })
                    
                    if (rebar_comp_strains > REBAR_COMPRESSIVE_STRAIN_LIMIT).any():
                        failure_step_time = df_strains.index[rebar_comp_strains > REBAR_COMPRESSIVE_STRAIN_LIMIT][0]
                        failure_records.append({
                            'section_tag': sec_tag,
                            'material_type': mat_type,
                            'failure_type': 'Rebar Compressive Strain Limit',
                            'exceeded_strain': df_strains.iloc[:, min_strain_col_idx].min(), # 실제 기록된 음수 값
                            'limit': -REBAR_COMPRESSIVE_STRAIN_LIMIT, # 지침을 음수로 표시
                            'at_time': failure_step_time
                        })

        except Exception as e:
            print(f"경고: Recorder 파일 {recorder_path} 처리 중 오류 발생: {e}")
            # print(traceback.format_exc()) # 디버깅용
            continue

    return failure_records