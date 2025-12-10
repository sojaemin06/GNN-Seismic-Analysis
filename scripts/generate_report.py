# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
from pathlib import Path
import datetime
import base64
import textwrap
import io
import numpy as np
import matplotlib.pyplot as plt
# GUI 백엔드 오류 방지
plt.switch_backend('Agg')

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.kds_2022_spectrum import get_site_coefficients, calculate_design_acceleration, generate_kds2022_demand_spectrum
from src.core.kds_performance_criteria import get_performance_objectives

def create_design_spectrum_plot_b64(design_config):
    """설계 응답 스펙트럼 이미지를 생성하고 base64로 반환합니다."""
    try:
        site_params = design_config.get('site_parameters', {})
        site_class = site_params.get('site_class', 'S4')
        Z = site_params.get('Z', 0.11)
        S_MCE_val = Z * 2.0
        S_DBE_val = S_MCE_val * (2.0/3.0)
        
        Sd, Sa, SDS, SD1 = generate_kds2022_demand_spectrum(S_DBE_val, site_class)
        # T_long default is 5.0 in generate_kds2022_demand_spectrum
        T = np.linspace(0.01, 5.0, 500)
        
        plt.figure(figsize=(6, 4))
        plt.plot(T, Sa, 'b-', linewidth=2, label='Design Spectrum (DBE, 5% Damping)')
        plt.title(f"Design Response Spectrum (Site: {site_class}, Z: {Z}g)")
        plt.xlabel("Period (sec)")
        plt.ylabel("Spectral Acceleration (g)")
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    except Exception as e:
        print(f"Error generating spectrum plot: {e}")
        return None

def load_image_as_base64(image_path):
    """이미지 파일을 읽어 base64 문자열로 변환합니다."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def load_file_as_base64(file_path):
    """파일을 읽어 base64 문자열로 변환합니다 (동영상 등)."""
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_html_report(results_root_dir):
    """
    상세 내진성능평가 종합 보고서를 생성합니다.
    """
    print(f"Generating Detailed Seismic Performance Evaluation Report...")
    
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_path = Path(results_root_dir) / "Seismic_Performance_Detailed_Report.html"
    
    # HTML 헤더 및 스타일
    html_content = textwrap.dedent(f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>상세 내진성능평가 보고서</title>
        <style>
            body {{ font-family: 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif; line-height: 1.6; color: #333; max-width: 1200px; margin: 0 auto; padding: 20px; background-color: #f4f6f9; }}
            h1, h2, h3, h4 {{ color: #2c3e50; margin-top: 30px; }}
            h1 {{ text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 20px; margin-bottom: 40px; }}
            h2 {{ border-left: 5px solid #3498db; padding-left: 15px; background-color: #fff; padding: 10px 15px; border-radius: 0 5px 5px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
            h3 {{ border-bottom: 1px solid #eee; padding-bottom: 5px; margin-top: 25px; }}
            .section {{ margin-bottom: 40px; background: #fff; padding: 30px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); border-radius: 10px; }}
            
            table {{ width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.95em; }}
            th, td {{ padding: 12px 15px; border: 1px solid #e0e0e0; text-align: center; }}
            th {{ background-color: #f1f3f5; font-weight: bold; color: #495057; }}
            tr:nth-child(even) {{ background-color: #f8f9fa; }}
            
            .img-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin: 20px 0; }}
            .img-box {{ flex: 1; min-width: 350px; max-width: 48%; text-align: center; background: #fff; border: 1px solid #eee; padding: 15px; border-radius: 8px; transition: transform 0.2s; }}
            .img-box:hover {{ transform: translateY(-5px); box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
            .img-box img, .img-box video {{ max-width: 100%; height: auto; border-radius: 4px; border: 1px solid #ddd; }}
            .img-caption {{ margin-top: 10px; font-size: 0.95em; color: #555; font-weight: 500; }}
            
            .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .info-card {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #2ecc71; }}
            .info-card h4 {{ margin-top: 0; color: #27ae60; font-size: 1.1em; }}
            .info-card ul {{ list-style: none; padding: 0; margin: 0; }}
            .info-card li {{ margin-bottom: 8px; font-size: 0.95em; }}
            .label {{ font-weight: bold; color: #555; width: 180px; display: inline-block; }}
            .note {{ font-size: 0.9em; color: #666; background: #e0f2f7; padding: 10px; border-radius: 4px; margin-top: 10px; border-left: 5px solid #3498db; }}
            .pass-text {{ color: #28a745; font-weight: bold; }}
            .fail-text {{ color: #dc3545; font-weight: bold; }}
            
            /* CSM 결과 레이아웃 */
            .csm-result-container {{ display: flex; gap: 20px; margin-bottom: 30px; align-items: center; }}
            .csm-chart {{ flex: 2; }}
            .csm-details {{ flex: 1; background: #f8f9fa; padding: 20px; border-radius: 8px; font-size: 0.9em; box-shadow: inset 0 0 5px rgba(0,0,0,0.05); }}
            .csm-details h4 {{ margin-top: 0; border-bottom: 2px solid #ddd; padding-bottom: 10px; margin-bottom: 15px; color: #333; }}
            .detail-row {{ display: flex; justify-content: space-between; margin-bottom: 8px; border-bottom: 1px dashed #eee; padding-bottom: 4px; }}
            .detail-label {{ color: #666; font-weight: 500; }}
            .detail-value {{ font-weight: bold; color: #333; }}
        </style>
    </head>
    <body>
        <div class="section">
            <h1>상세 내진성능평가 종합 보고서</h1>
            <p style="text-align: right; color: #777;">생성 일시: {current_date}</p>
        </div>
    """).strip()

    # --- 공통 설정 로드 ---
    config_path = Path(project_root) / 'scripts' / 'seismic_design_config.json'
    dataset_config_path = Path(project_root) / 'scripts' / 'dataset_config.json'
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            design_config = json.load(f)
    except Exception:
        design_config = {}

    site_params = design_config.get('site_parameters', {})
    
    # --- 1. 성능 목표 ---
    importance_class = site_params.get('importance_class', 'I')
    performance_objectives = get_performance_objectives(importance_class)

    html_block = textwrap.dedent(f"""
    <div class="section">
        <h2>1. 성능 목표</h2>
        <p>본 내진성능평가는 <strong>구조물 중요도 {importance_class}등급</strong>에 해당하는 성능 목표를 기준으로 수행되었습니다.</p>
        <table>
            <thead>
                <tr>
                    <th>성능 수준</th>
                    <th>재현 주기</th>
                    <th>허용 층간 변위비 (%)</th>
                    <th>주요 손상 허용 수준</th>
                </tr>
            </thead>
            <tbody>
    """).strip()
    html_content += html_block
    for obj in performance_objectives:
        html_content += f"""
                <tr>
                    <td>{obj['description']}</td>
                    <td>{obj['repetition_period']}</td>
                    <td>{obj['target_drift_ratio_limit']*100:.2f}</td>
                    <td>{obj['name'].split(" ")[-1]}</td>
                </tr>
        """
    html_block = textwrap.dedent("""
            </tbody>
        </table>
    </div>
    """).strip()
    html_content += html_block

    # --- 2. 평가 지진 ---
    site_class = site_params.get('site_class', 'S4')
    Z = site_params.get('Z', 0.11)
    S_MCE_val = Z * 2.0
    S_DBE_val = S_MCE_val * (2.0/3.0)
    Fa, Fv = get_site_coefficients(S_DBE_val, site_class)
    calc_SDS, calc_SD1 = calculate_design_acceleration(S_DBE_val, site_class)
    Ts = calc_SD1 / calc_SDS if calc_SDS > 0 else 0
    T0 = 0.2 * Ts

    # [NEW] Spectrum Image
    spectrum_b64 = create_design_spectrum_plot_b64(design_config)
    spectrum_html = ""
    if spectrum_b64:
        spectrum_html = f"""
        <div class="img-box" style="max-width: 600px; margin: 20px auto;">
            <img src="data:image/png;base64,{spectrum_b64}" alt="Design Response Spectrum">
            <div class="img-caption">설계 응답 스펙트럼 (Design Spectrum)</div>
        </div>
        """

    html_block = textwrap.dedent(f"""
    <div class="section">
        <h2>2. 평가 지진 (요구 스펙트럼 제원)</h2>
        <p class="note">※ 본 설계 스펙트럼은 <strong>5% 감쇠비</strong>를 기준으로 작성되었습니다.</p>
        <div class="info-grid">
            <div class="info-card" style="border-left-color: #3498db;">
                <h4 style="color: #2980b9;">[입력 파라미터]</h4>
                <ul>
                    <li><span class="label">지역 구분:</span> 지진구역 I</li>
                    <li><span class="label">지반 등급:</span> {site_class}</li>
                    <li><span class="label">지진구역계수 (Z):</span> {Z}g</li>
                    <li><span class="label">구조물 중요도 계수:</span> {importance_class}</li>
                </ul>
            </div>
            <div class="info-card" style="border-left-color: #e67e22;">
                <h4 style="color: #d35400;">[계산된 설계 스펙트럼 계수]</h4>
                <ul>
                    <li><span class="label">단주기 지반증폭계수 (Fa):</span> {Fa:.2f}</li>
                    <li><span class="label">1초주기 지반증폭계수 (Fv):</span> {Fv:.2f}</li>
                    <li><span class="label">단주기 설계스펙트럼가속도 (SDS):</span> {calc_SDS:.3f}g</li>
                    <li><span class="label">1초주기 설계스펙트럼가속도 (SD1):</span> {calc_SD1:.3f}g</li>
                    <li><span class="label">설계스펙트럼 천이주기 (T0):</span> {T0:.3f} sec</li>
                    <li><span class="label">설계스펙트럼 천이주기 (Ts):</span> {Ts:.3f} sec</li>
                </ul>
            </div>
        </div>
        {spectrum_html}
    </div>
    """).strip()
    html_content += html_block

    # --- 3. 대상 구조물 정보 (공통) ---
    # 첫 번째 유효한 결과 폴더에서 모드 정보 가져오기
    result_dirs = sorted([d for d in Path(results_root_dir).iterdir() if d.is_dir() and "Run_Single" in d.name])
    first_res_dir = result_dirs[0] if result_dirs else None
    
    if first_res_dir:
        modal_path = first_res_dir / 'modal_properties.json'
        if modal_path.exists():
            with open(modal_path, 'r') as f:
                modal_data = json.load(f)
                all_modes = modal_data.get('modal_properties', [])
    
    html_block = textwrap.dedent(f"""
    <div class="section">
        <h2>3. 대상 구조물 정보</h2>
        
        <h3>3.1 구조물 형상 및 재료</h3>
        <div class="img-container">
    """).strip()
    html_content += html_block

    # 구조물 형상 및 재료 이미지 (X_pos 결과 폴더 기준)
    target_dir = next((d for d in result_dirs if '_X_pos' in d.name), first_res_dir)
    
    if target_dir:
        img_patterns = [
            (f"*model_3D_Matplotlib.png", "3D 와이어프레임 모델"),
            (f"*material_concrete_combined.png", "콘크리트 응력-변형률 관계"),
            (f"*material_rebar.png", "철근 응력-변형률 관계")
        ]
        for pattern, caption in img_patterns:
            found_imgs = list(target_dir.glob(pattern))
            if found_imgs:
                b64_str = load_image_as_base64(found_imgs[0])
                if b64_str:
                    html_block = textwrap.dedent(f"""
                    <div class="img-box" style="max-width: 30%;">
                        <img src="data:image/png;base64,{b64_str}" alt="{caption}">
                        <div class="img-caption">{caption}</div>
                    </div>
                    """).strip()
                    html_content += html_block

    html_block = textwrap.dedent("""
        </div>
        
        <h3>3.2 부재 상세 정보 (단면 및 철근)</h3>
        <p>본 해석 모델에 적용된 각 부재 그룹별 단면 상세(Section Schedule)는 다음과 같습니다.</p>

    """).strip()
    html_content += html_block

    # [NEW] 단면 일람표 생성 로직
    if target_dir:
        # 1. 파일 스캔 및 분류
        col_images = {} # {group_idx: {'Exterior': path, 'Interior': path}}
        beam_images = {}
        
        for img_path in target_dir.glob("*opsvis_sec_*.png"):
            # Filename ex: ..._opsvis_sec_Col_Group0_Exterior.png
            parts = img_path.stem.split('_')
            if 'Col' in parts:
                target_dict = col_images
            elif 'Beam' in parts:
                target_dict = beam_images
            else:
                continue
                
            # Extract Group Index and Location
            try:
                group_part = next(p for p in parts if p.startswith('Group'))
                group_idx = int(group_part.replace('Group', ''))
                
                loc_part = parts[-1] # Exterior or Interior
                
                if group_idx not in target_dict: target_dict[group_idx] = {}
                target_dict[group_idx][loc_part] = img_path
            except:
                continue

        # 2. 기둥 일람표 HTML 생성
        html_block = textwrap.dedent("""
        <h4>3.2.1 기둥 일람표 (Column Schedule)</h4>
        <table>
            <thead>
                <tr>
                    <th>구분 (Group)</th>
                    <th>외부 (Exterior)</th>
                    <th>내부 (Interior)</th>
                </tr>
            </thead>
            <tbody>
        """).strip()
        html_content += html_block
        
        sorted_col_groups = sorted(col_images.keys())
        if not sorted_col_groups:
             html_content += "<tr><td colspan='3'>단면 이미지가 없습니다.</td></tr>"
        
        for grp_idx in sorted_col_groups:
            ext_img = col_images[grp_idx].get('Exterior')
            int_img = col_images[grp_idx].get('Interior')
            
            ext_html = "N/A"
            if ext_img:
                b64 = load_image_as_base64(ext_img)
                if b64: ext_html = f'<img src="data:image/png;base64,{b64}" style="max-width: 200px;"><br>Group {grp_idx} (Ext)'
            
            int_html = "N/A"
            if int_img:
                b64 = load_image_as_base64(int_img)
                if b64: int_html = f'<img src="data:image/png;base64,{b64}" style="max-width: 200px;"><br>Group {grp_idx} (Int)'
                
            html_content += f"""
            <tr>
                <td><strong>Group {grp_idx}</strong><br>(Story {grp_idx*2+1}~{(grp_idx+1)*2})</td>
                <td>{ext_html}</td>
                <td>{int_html}</td>
            </tr>
            """
            
        html_content += "</tbody></table>"

        # 3. 보 일람표 HTML 생성
        html_block = textwrap.dedent("""
        <h4>3.2.2 보 일람표 (Beam Schedule)</h4>
        <table>
            <thead>
                <tr>
                    <th>구분 (Group)</th>
                    <th>외부 (Exterior)</th>
                    <th>내부 (Interior)</th>
                </tr>
            </thead>
            <tbody>
        """).strip()
        html_content += html_block
        
        sorted_beam_groups = sorted(beam_images.keys())
        if not sorted_beam_groups:
             html_content += "<tr><td colspan='3'>단면 이미지가 없습니다.</td></tr>"

        for grp_idx in sorted_beam_groups:
            ext_img = beam_images[grp_idx].get('Exterior')
            int_img = beam_images[grp_idx].get('Interior')
            
            ext_html = "N/A"
            if ext_img:
                b64 = load_image_as_base64(ext_img)
                if b64: ext_html = f'<img src="data:image/png;base64,{b64}" style="max-width: 200px;"><br>Group {grp_idx} (Ext)'
            
            int_html = "N/A"
            if int_img:
                b64 = load_image_as_base64(int_img)
                if b64: int_html = f'<img src="data:image/png;base64,{b64}" style="max-width: 200px;"><br>Group {grp_idx} (Int)'
                
            html_content += f"""
            <tr>
                <td><strong>Group {grp_idx}</strong><br>(Story {grp_idx*2+1}~{(grp_idx+1)*2})</td>
                <td>{ext_html}</td>
                <td>{int_html}</td>
            </tr>
            """
        
        html_content += "</tbody></table>"

    html_block = textwrap.dedent("""
        <p>본 해석 모델에 적용된 주요 부재의 설정 범위(Configuration)는 다음과 같습니다.</p>
        <table>
            <thead>
                <tr>
                    <th>구분</th>
                    <th>위치</th>
                    <th>단면 크기 범위 (m)</th>
                    <th>주철근 범위</th>
                </tr>
            </thead>
            <tbody>
    """).strip()
    html_content += html_block
    
    # 부재 상세 정보 (dataset_config.json)
    try:
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            ds_config = json.load(f)
            mem_props = ds_config.get('member_properties', {})
            
            # 기둥 정보
            col_sec = mem_props.get('col_section_tiers_m', {})
            rebar_col = mem_props.get('rebar_col_list', [])
            rebar_col_names = ", ".join([r['name'] for r in rebar_col]) if rebar_col else "D22~D32"
            
            for loc, tiers in col_sec.items():
                range_str = f"{tiers[0][0]}m ~ {tiers[-1][0]}m" if tiers else "N/A"
                html_content += f"""
                <tr>
                    <td>기둥 (Column)</td>
                    <td>{loc} (외부/내부 등)</td>
                    <td>{range_str}</td>
                    <td>{rebar_col_names}</td>
                </tr>
                """
            
            # 보 정보
            beam_sec = mem_props.get('beam_section_tiers_m', {})
            rebar_beam = mem_props.get('rebar_beam_list', [])
            rebar_beam_names = ", ".join([r['name'] for r in rebar_beam]) if rebar_beam else "D19~D25"

            for loc, tiers in beam_sec.items():
                range_str = f"{tiers[0][0]}m ~ {tiers[-1][0]}m" if tiers else "N/A"
                html_content += f"""
                <tr>
                    <td>보 (Beam)</td>
                    <td>{loc} (외부/내부 등)</td>
                    <td>{range_str}</td>
                    <td>{rebar_beam_names}</td>
                </tr>
                """

    except Exception as e:
         html_content += f"""
                <tr><td colspan="4">부재 상세 정보를 불러오는 중 오류 발생: {e}</td></tr>
            """

    html_block = textwrap.dedent("""
            </tbody>
        </table>

        <h3>3.3 구조물 고유주기 및 질량참여율</h3>
        <p>구조물의 고유 주기와 방향별 질량 참여율은 다음과 같습니다.</p>
        <table>
            <thead>
                <tr>
                    <th>모드</th>
                    <th>고유 주기 (sec)</th>
                    <th>X방향 MPR (%)</th>
                    <th>X방향 누적 MPR (%)</th>
                    <th>Z방향 MPR (%)</th>
                    <th>Z방향 누적 MPR (%)</th>
                </tr>
            </thead>
            <tbody>
    """).strip()
    html_content += html_block
    
    cum_x, cum_z = 0.0, 0.0
    if first_res_dir and modal_path.exists():
        for mode in all_modes:
            cum_x += mode['mpr_x']
            cum_z += mode['mpr_z']
            
            html_content += f"""
                    <tr>
                        <td>{mode['mode']}</td>
                        <td>{mode['period']:.4f}</td>
                        <td>{mode['mpr_x']*100:.2f}</td>
                        <td>{cum_x*100:.2f}</td>
                        <td>{mode['mpr_z']*100:.2f}</td>
                        <td>{cum_z*100:.2f}</td>
                    </tr>
            """
    
    html_block = textwrap.dedent("""
            </tbody>
        </table>
    </div>
    """).strip()
    html_content += html_block

    # --- 4. 푸쉬오버 해석 적용성 검토 ---
    html_block = textwrap.dedent("""
    <div class="section">
        <h2>4. 푸쉬오버 해석 적용성 검토 (130% 룰)</h2>
        <p>각 방향별로 1차 모드(지배 모드)만으로 수행한 비선형 정적 해석의 층 전단력이, <strong>질량 참여율 합계 90% 이상을 만족하는 모드들을 합산한 RSA 해석 결과의 130% 이상</strong>인지 검증합니다.</p>
        <div class="img-container">
    """).strip()
    html_content += html_block

    for direction in ['X', 'Z']:
        # 해당 방향의 아무 결과나 하나 가져오기 (검증 결과는 동일하므로)
        target_dir = next((d for d in result_dirs if f'_{direction}_pos' in d.name), None)
        if not target_dir: target_dir = next((d for d in result_dirs if f'_{direction}_neg' in d.name), None)
        
        if target_dir:
            verif_img = list(target_dir.glob(f"*NSP_verification_plot_{direction}.png"))
            if verif_img:
                b64_str = load_image_as_base64(verif_img[0])
                if b64_str:
                    html_block = textwrap.dedent(f"""
                    <div class="img-box">
                        <img src="data:image/png;base64,{b64_str}" alt="130% 검증 ({direction})">
                        <div class="img-caption">[{direction}방향] 비선형 정적 해석 타당성 검증</div>
                    </div>
                    """).strip()
                    html_content += html_block

    html_block = textwrap.dedent("""
        </div>
    </div>
    """).strip()
    html_content += html_block

    # --- 5. 성능점 산정 및 평가 결과 (4방향) ---
    html_block = textwrap.dedent("""
    <div class="section">
        <h2>5. 성능점 산정 및 평가 결과</h2>
        <p>각 방향(X, Z) 및 가력 방향(+, -)에 대해 산정된 성능점과 평가 결과를 요약합니다.</p>
    """).strip()
    html_content += html_block

    # 4가지 케이스 순회
    cases = [('X', 'pos', '+'), ('X', 'neg', '-'), ('Z', 'pos', '+'), ('Z', 'neg', '-')]
    
    for direction, sign_str, sign in cases:
        # [수정] 음방향 결과 플립 설정 제거 (텍스트 반전 문제 등 방지)
        flip_style = ''
        flip_note = ''

        target_dir_name = f"Run_Single_RC_Moment_Frame_Sampled_{direction}_{sign_str}"
        # target_dir = next((d for d in result_dirs if target_dir_name in d.name), None) # 단순 매칭
        # 정확한 매칭을 위해
        target_dir = next((d for d in result_dirs if d.name.endswith(f"_{direction}_{sign_str}")), None)

        if not target_dir: continue
        
        summary_json_path = target_dir / f"csm_evaluation_summary_{direction}_{sign_str}.json"
        csm_summary = []
        if summary_json_path.exists():
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                csm_summary = json.load(f)

        html_block = textwrap.dedent(f"""
        <h3>5-{direction}({sign}). 해석 결과: {direction}방향 ({sign})</h3>
        {flip_note}
        
        <h4>5-{direction}({sign})-1. 푸쉬오버 곡선 및 힌지 분포</h4>
        <div class="img-container">
        """).strip()
        html_content += html_block
        
        # 푸쉬오버 동영상
        video_path = list(target_dir.glob(f"*pushover_animation.mp4"))
        if video_path:
            b64_vid = load_file_as_base64(video_path[0])
            if b64_vid:
                html_block = textwrap.dedent(f"""
                <div class="img-box" style="max-width: 800px;">
                    <video controls loop muted playsinline style="width: 100%;" {flip_style}>
                        <source src="data:video/mp4;base64,{b64_vid}" type="video/mp4">
                        브라우저 미지원
                    </video>
                    <div class="img-caption">푸쉬오버 해석 애니메이션 ({direction}{sign})</div>
                </div>
                """).strip()
                html_content += html_block

        # [NEW] 푸쉬오버 최종 플롯 (이미지)
        final_plot_path = list(target_dir.glob(f"*pushover_final_plot.png"))
        if final_plot_path:
            b64_img = load_image_as_base64(final_plot_path[0])
            if b64_img:
                html_block = textwrap.dedent(f"""
                <div class="img-box" style="max-width: 800px;">
                    <img src="data:image/png;base64,{b64_img}" alt="Pushover Final Plot" {flip_style}>
                    <div class="img-caption">푸쉬오버 최종 상태 ({direction}{sign})</div>
                </div>
                """).strip()
                html_content += html_block
        
        html_block = textwrap.dedent(f"""
        </div>
        <h4>5-{direction}({sign})-2. CSM 성능점 평가</h4>
        <div class="img-container">
        """).strip()
        html_content += html_block

        # CSM 결과
        csm_imgs = sorted(list(target_dir.glob(f"CSM_performance_point_{direction}_*.png")))
        
        for img_path in csm_imgs:
            # 파일명: CSM_performance_point_X_pos_Collapse_Prevention.png
            # 파싱
            suffix = img_path.stem.split(f"_{direction}_{sign_str}_")[-1] # Collapse_Prevention
            obj_name_clean = suffix.replace("_", " ")
            
            b64_str = load_image_as_base64(img_path)
            
            # 요약 데이터 매칭
            target_summary = next((item for item in csm_summary if item["objective_name"].replace(" ", "_") == obj_name_clean.replace(" ", "_")), None)
            
            if b64_str and target_summary:
                status_class = "pass-text" if target_summary['status'] == "PASS" else "fail-text"
                
                html_block = textwrap.dedent(f"""
                <div class="csm-result-container">
                    <div class="csm-chart img-box">
                        <img src="data:image/png;base64,{b64_str}" alt="{obj_name_clean}">
                    </div>
                    <div class="csm-details">
                        <h4>{obj_name_clean} 평가 결과</h4>
                        <div class="detail-row"><span class="detail-label">재현 주기:</span> <span class="detail-value">{target_summary['repetition_period']}</span></div>
                        <div class="detail-row"><span class="detail-label">성능점 (Sd, Sa):</span> <span class="detail-value">({target_summary['perf_point_Sd_m']:.4f}m, {target_summary['perf_point_Sa_g']:.4f}g)</span></div>
                        <div class="detail-row"><span class="detail-label">유효 주기/감쇠:</span> <span class="detail-value">{target_summary['effective_period_s']:.3f}s / {target_summary['effective_damping_pct']:.1f}%</span></div>
                        <div class="detail-row"><span class="detail-label">계산 층간변위비:</span> <span class="detail-value">{target_summary['calculated_drift_pct']:.3f} %</span></div>
                        <div class="detail-row"><span class="detail-label">허용 층간변위비:</span> <span class="detail-value">{target_summary['allowed_drift_pct']:.3f} %</span></div>
                        
                        <!-- [NEW] 붕괴 부재 수 표시 -->
                        <div class="detail-row">
                            <span class="detail-label">붕괴 부재 수:</span> 
                            <span class="detail-value {'fail-text' if target_summary.get('collapsed_member_count', 0) > 0 else ''}">
                                {target_summary.get('collapsed_member_count', 0)} 개
                            </span>
                        </div>

                        <div class="detail-row" style="border-bottom: none; margin-top: 10px;">
                            <span class="detail-label">최종 판정:</span> 
                            <span class="detail-value {status_class}">{target_summary['status']}</span>
                        </div>
                    </div>
                </div>
                """).strip()
                html_content += html_block

        html_block = textwrap.dedent("""
        </div>
        """).strip()
        html_content += html_block

    html_block = textwrap.dedent("""
    </div>
    <div class="section">
        <h2>6. 최종 종합 결론</h2>
        <p>본 보고서는 KDS 41 17 00 (건축물 내진설계기준)에 따라 비선형 정적 해석(Pushover Analysis) 및 역량스펙트럼법(CSM)을 이용하여 대상 구조물의 내진 성능을 평가하였습니다.</p>
        
        <h3>6.1 평가 결과 요약</h3>
        <table>
            <thead>
                <tr>
                    <th>해석 방향</th>
                    <th>성능 목표 (재현 주기)</th>
                    <th>층간변위비 (허용/계산)</th>
                    <th>중력하중 저항능력</th>
                    <th>붕괴 부재 수</th>
                    <th>최종 판정</th>
                </tr>
            </thead>
            <tbody>
    """).strip()
    html_content += html_block

    # [NEW] 모든 결과 수집하여 표 생성
    overall_status = "PASS"
    
    for direction, sign_str, sign in cases:
        # csm_evaluation_summary_X_pos.json 읽기
        target_dir = next((d for d in result_dirs if d.name.endswith(f"_{direction}_{sign_str}")), None)
        if not target_dir: continue
        
        summary_json_path = target_dir / f"csm_evaluation_summary_{direction}_{sign_str}.json"
        if summary_json_path.exists():
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                csm_summary = json.load(f)
                
            for item in csm_summary:
                status_class = "pass-text" if item['status'] == "PASS" else "fail-text"
                if "FAIL" in item['status']: overall_status = "FAIL"
                
                # 안정성 지표 및 붕괴 부재 수 (이전 버전 json에 없을 경우 대비)
                stability_ratio = item.get('stability_ratio', 1.0)
                stability_text = f"{stability_ratio*100:.0f}% (OK)" if stability_ratio >= 0.8 else f"{stability_ratio*100:.0f}% (WARNING)"
                collapsed_count = item.get('collapsed_member_count', 0)
                
                html_content += f"""
                <tr>
                    <td>{direction}({sign})</td>
                    <td>{item['objective_name']} ({item['repetition_period']})</td>
                    <td>{item['allowed_drift_pct']:.2f}% / {item['calculated_drift_pct']:.3f}%</td>
                    <td>{stability_text}</td>
                    <td>{collapsed_count}개</td>
                    <td class="{status_class}">{item['status']}</td>
                </tr>
                """

    html_block = textwrap.dedent(f"""
            </tbody>
        </table>

        <h3>6.2 종합 판단</h3>
        <ul>
            <li><strong>적용성 검토:</strong> 130% 룰 검증을 통해 1차 모드 기반 해석의 타당성을 확인하였습니다.</li>
            <li><strong>최종 결과:</strong> 모든 성능 목표에 대해 <span class="{'pass-text' if overall_status == 'PASS' else 'fail-text'}">{overall_status}</span> 하였습니다.</li>
        </ul>
        <div class="note">
            <p><strong>※ 참고 사항:</strong></p>
            <ul>
                <li>중력하중 저항능력은 성능점에서의 전단강도 보유율(잔존 강도 비율)로 평가하였으며, 80% 이상일 경우 'OK'로 판정합니다.</li>
                <li>붕괴 부재 수는 성능점에서의 소성회전각이 <strong>붕괴방지(CP) 한계 (0.04 rad)</strong>를 초과하는 부재(기둥/보)의 총 개수입니다.</li>
            </ul>
        </div>
    </div>
    </body>
    </html>
    """).strip()
    html_content += html_block

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\nSuccessfully generated detailed report: {report_path}")

if __name__ == '__main__':
    results_dir = Path(project_root) / 'results'
    if results_dir.exists():
        generate_html_report(results_dir)
    else:
        print(f"Error: Results directory not found at {results_dir}")