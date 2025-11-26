import numpy as np
import pandas as pd
import traceback
from pathlib import Path

# ### 6. 모듈형 함수: 결과 처리 및 CSV 저장 ###
def process_pushover_results(params, model_nodes, dominant_mode, direction='X'):
    """
    Pushover 해석으로 생성된 .out 파일들을 읽어 CSV로 변환 및 저장합니다.
    """
    print(f"\nProcessing results for {direction}-direction and exporting to CSV...")
    df_disp = None 
    df_pushover_curve = None
    df_m_phi = None
    full_element_state_dfs = {}
    
    try:
        output_dir = params['output_dir']
        analysis_name = params['analysis_name']
        
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
        
        disp_col_name = f'Floor{num_stories}_Disp{direction}'
        
        if roof_disp_data.ndim == 1:
            roof_disp_data = roof_disp_data.reshape(1, -1)
        
        expected_disp_cols_count = 1 + num_stories
        actual_disp_cols_count = roof_disp_data.shape[1]

        if actual_disp_cols_count > expected_disp_cols_count:
            print(f"Warning: Floor displacement recorder output for {direction}-direction has more columns than expected. Expected {expected_disp_cols_count}, got {actual_disp_cols_count}. Truncating.")
            roof_disp_data = roof_disp_data[:, :expected_disp_cols_count]
        elif actual_disp_cols_count < expected_disp_cols_count:
            print(f"Error: Floor displacement recorder output column mismatch for {direction}-direction. Expected {expected_disp_cols_count}, got {actual_disp_cols_count}. Skipping.")
            return None, None, None, None

        disp_cols = ['Time'] + [f'Floor{i+1}_Disp{direction}' for i in range(num_stories)]
        df_disp = pd.DataFrame(roof_disp_data, columns=disp_cols)
        
        base_nodes = model_nodes['base_nodes']
        base_node_tags = [str(tag) for tag in base_nodes]
        base_cols = ['Time'] + [f'BaseReaction_Node{tag}' for tag in base_node_tags]
        
        if base_shear_data.ndim == 1:
            base_shear_data = base_shear_data.reshape(1, -1)
            
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

        num_valid_steps = len(df_pushover_curve)
        
        path_col_rot = output_dir / f"{analysis_name}_all_col_plastic_rotation_{direction}.out"
        all_column_tags = model_nodes.get('all_column_tags', [])
        if path_col_rot.exists() and all_column_tags:
            try:
                col_rot_data = np.loadtxt(path_col_rot)
                if col_rot_data.ndim == 1: col_rot_data = col_rot_data.reshape(1, -1)
                headers = ['Time']
                for ele_tag in all_column_tags: 
                    for ip in [1, params.get('num_int_pts', 5)]:
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])
                        
                if col_rot_data.ndim > 1 and col_rot_data.shape[1] == len(headers):
                    df_col_rot = pd.DataFrame(col_rot_data, columns=headers).head(num_valid_steps)
                    full_element_state_dfs['col_rot_df'] = df_col_rot 
                else:
                    print(f"\n---! Warning: Column plastic rotation data column mismatch! Expected {len(headers)}, got {col_rot_data.shape[1]} !---")
            except Exception as e: print(f"\n---! Warning: Error processing column plastic rotation data !--- \nError details: {e}")
        
        path_beam_rot = output_dir / f"{analysis_name}_all_beam_plastic_rotation_{direction}.out"
        all_beam_tags = model_nodes.get('all_beam_tags', [])
        if path_beam_rot.exists() and all_beam_tags:
            try:
                beam_rot_data = np.loadtxt(path_beam_rot)
                if beam_rot_data.ndim == 1: beam_rot_data = beam_rot_data.reshape(1, -1)
                headers = ['Time']
                for ele_tag in all_beam_tags:
                    for ip in [1, params.get('num_int_pts', 5)]:
                        headers.extend([f'Ele{ele_tag}_IP{ip}_ry', f'Ele{ele_tag}_IP{ip}_rz', f'Ele{ele_tag}_IP{ip}_rx'])

                if beam_rot_data.ndim > 1 and beam_rot_data.shape[1] == len(headers):
                    df_beam_rot = pd.DataFrame(beam_rot_data, columns=headers).head(num_valid_steps)
                    full_element_state_dfs['beam_rot_df'] = df_beam_rot 
                else:
                    print(f"\n---! Warning: Beam plastic rotation data column mismatch! Expected {len(headers)}, got {beam_rot_data.shape[1]} !---")
            except Exception as e:
                print(f"\n---! Warning: Error processing beam plastic rotation data !--- \nError details: {e}")

        return df_pushover_curve, df_disp, full_element_state_dfs, df_m_phi

    except Exception as e:
        print(f"Error processing recorder data: {e}")
        print(traceback.format_exc())
        return None, None, None, None

def calculate_performance_points(df_curve):
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
        
        yield_shear_approx = perf_points['peak_shear'] * 0.60
        yield_point_df = df_curve[df_curve['Base_Shear_N'] >= yield_shear_approx]
        
        if not yield_point_df.empty:
            first_yield_approx_idx = yield_point_df.index[0]
            if first_yield_approx_idx == 0:
                 raise Exception("Yield point at index 0")
                 
            K_secant = df_curve.loc[first_yield_approx_idx, 'Base_Shear_N'] / df_curve.loc[first_yield_approx_idx, 'Roof_Displacement_m']
            D_y = perf_points['peak_shear'] / K_secant
            yield_index = (df_curve['Roof_Displacement_m'] - D_y).abs().idxmin()
            
            perf_points['yield_disp'] = df_curve.loc[yield_index, 'Roof_Displacement_m']
            perf_points['yield_shear'] = df_curve.loc[yield_index, 'Base_Shear_N']
        else:
             raise Exception("Curve did not reach 60% of peak shear")
            
    except Exception as e:
        print(f"Warning: Could not calculate performance points. Defaulting to 0. Error: {e}")
        perf_points['peak_disp'] = perf_points.get('peak_disp', 0.0)
        perf_points['peak_shear'] = perf_points.get('peak_shear', 0.0)
        perf_points['collapse_disp'] = perf_points.get('collapse_disp', None)
        perf_points['yield_disp'] = 0.0
        perf_points['yield_shear'] = 0.0

    return perf_points

def find_first_material_failure(model_info, params, df_curve, direction='X'):
    """
    [수정] fiber 레코더의 출력을 파싱하여 재료 파괴를 감지하고, 수치 불안정성을 필터링합니다.
    """
    first_failure_event = None
    CONCRETE_COMPRESSIVE_STRAIN_LIMIT = 0.003
    MIN_DISP_THRESHOLD = 0.0001 # 0.1mm, 이 변위 이전의 파괴는 수치 오류로 간주

    element_map = model_info.get('element_section_map', {})
    if not element_map or df_curve is None or df_curve.empty:
        return None

    for ele_tag, ele_info in element_map.items():
        fiber_paths = ele_info.get('fiber_paths', [])
        if not fiber_paths:
            continue

        try:
            all_fiber_data = []
            for i, p_str in enumerate(fiber_paths):
                path = Path(p_str)
                if path.exists() and path.stat().st_size > 0:
                    df = pd.read_csv(path, sep='\s+', header=None, names=['Time', f'Strain_f{i+1}'])
                    if not df.empty:
                        all_fiber_data.append(df.set_index('Time'))
            
            if not all_fiber_data:
                continue

            df_strains = pd.concat(all_fiber_data, axis=1).reset_index()
            strain_cols = [col for col in df_strains.columns if 'Strain' in col]
            df_strains['min_strain'] = df_strains[strain_cols].min(axis=1)
            
            df_strains['roof_disp'] = np.interp(df_strains['Time'], df_curve['Pseudo_Time'], df_curve['Roof_Displacement_m'])
            
            df_strains_filtered = df_strains[df_strains['roof_disp'] >= MIN_DISP_THRESHOLD]

            failure_df = df_strains_filtered[df_strains_filtered['min_strain'] < -CONCRETE_COMPRESSIVE_STRAIN_LIMIT]

            if not failure_df.empty:
                failure_event = failure_df.iloc[0]
                current_time = failure_event['Time']
                
                if first_failure_event is None or current_time < first_failure_event['time']:
                    base_shear = np.interp(current_time, df_curve['Pseudo_Time'], df_curve['Base_Shear_N'])
                    first_failure_event = {
                        'time': current_time,
                        'roof_disp': failure_event['roof_disp'],
                        'base_shear': base_shear,
                        'element_tag': ele_tag,
                        'element_type': ele_info['type'],
                        'failure_type': 'Concrete Compression',
                        'strain': failure_event['min_strain'],
                        'limit': -CONCRETE_COMPRESSIVE_STRAIN_LIMIT
                    }

        except Exception as e:
            print(f"Warning: Could not process fiber deformation files for element {ele_tag}. Error: {e}")
            continue
            
    return first_failure_event
