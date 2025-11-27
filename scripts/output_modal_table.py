# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
from pathlib import Path

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def create_modal_table(results_dir: Path):
    """
    modal_properties.json 파일을 읽어 모드 특성 테이블을 CSV로 출력합니다.
    """
    modal_json_path = results_dir / 'modal_properties.json'
    
    if not modal_json_path.exists():
        print(f"Error: modal_properties.json not found in {results_dir}")
        return

    try:
        with open(modal_json_path, 'r') as f:
            data = json.load(f)
        
        modal_props = data.get('modal_properties', [])
        if not modal_props:
            print("No modal properties found in JSON.")
            return

        # 데이터프레임 변환
        df = pd.DataFrame(modal_props)
        
        # 필요한 컬럼 선택 및 이름 변경 (가독성 향상)
        # mode, period, mpr_x, mpr_z, M_star_x, M_star_z, gamma_x, gamma_z
        cols_to_keep = ['mode', 'period', 'mpr_x', 'mpr_z', 'M_star_x', 'M_star_z', 'gamma_x', 'gamma_z']
        
        # 존재하는 컬럼만 선택 (방어적 코드)
        existing_cols = [c for c in cols_to_keep if c in df.columns]
        df = df[existing_cols].copy()
        
        df.rename(columns={
            'mode': 'Mode',
            'period': 'Period (s)',
            'mpr_x': 'Mass Part. Ratio X',
            'mpr_z': 'Mass Part. Ratio Z',
            'M_star_x': 'Eff. Modal Mass X (kg)',
            'M_star_z': 'Eff. Modal Mass Z (kg)',
            'gamma_x': 'Part. Factor X',
            'gamma_z': 'Part. Factor Z'
        }, inplace=True)

        # 출력
        output_csv_path = results_dir / 'modal_properties_table.csv'
        df.to_csv(output_csv_path, index=False)
        print(f"Modal properties table saved to: {output_csv_path}")
        
        # 콘솔에도 일부 출력
        print("\n--- Modal Properties Table (Top 5 Modes) ---")
        print(df.head(5).to_string(index=False))
        print("-" * 50)

    except Exception as e:
        print(f"Error processing modal table: {e}")

if __name__ == '__main__':
    default_dir_x = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_X'
    default_dir_z = Path(project_root) / 'results' / 'Run_Single_RC_Moment_Frame_Sampled_Z'
    
    if default_dir_x.exists():
        print(f"\nProcessing directory: {default_dir_x.name}")
        create_modal_table(default_dir_x)
        
    if default_dir_z.exists():
        print(f"\nProcessing directory: {default_dir_z.name}")
        create_modal_table(default_dir_z)
