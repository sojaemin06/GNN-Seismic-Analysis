# -*- coding: utf-8 -*-
import sys
import os
import json
import pandas as pd
from pathlib import Path
import datetime
import base64
import textwrap

# --- 프로젝트 루트 경로 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.kds_2022_spectrum import get_site_coefficients, calculate_design_acceleration
from src.core.kds_performance_criteria import get_performance_objectives

def load_image_as_base64(image_path):
    """이미지 파일을 읽어 base64 문자열로 변환합니다."""
    if not os.path.exists(image_path):
        return None
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

def load_file_as_base64(file_path):
    """파일을 읽어 base64 문자열로 변환합니다."""
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
    design_config = {}
    config_path = Path(project_root) / 'scripts' / 'seismic_design_config.json'
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            design_config = json.load(f)
    except Exception:
        html_content += "<div class='section'><p>설계 설정 파일(seismic_design_config.json)을 로드할 수 없습니다.</p></div>"

    site_params = design_config.get('site_parameters', {})
    
    # --- 섹션 1: 성능 목표 ---
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
        <div class="note">
            <strong>※ 평가 기준:</strong><br>
            각 재현주기(2400년, 1400년 등)별 지진에 대해 구조물의 성능점(Performance Point)을 산정하고, 
            다음 세 가지 조건을 모두 만족해야 해당 성능 목표를 달성한 것으로 판정합니다.<br>
            1. <strong>층간 변위비 만족:</strong> 산정된 성능점에서의 층간 변위비가 각 성능 수준별 허용 한계 이내일 것.<br>
            2. <strong>중력 하중 저항 능력 유지:</strong> 성능점 도달 시까지 구조물이 붕괴되지 않고 중력을 지지할 수 있을 것. (푸쉬오버 동영상 및 결과 참조)<br>
            3. <strong>주요 부재 붕괴 미발생:</strong> 기둥 등 주요 구조 부재의 소성 회전각이 붕괴 한계(CP)를 초과하지 않을 것. (푸쉬오버 결과 참조)
        </div>
    </div>
    """).strip()
    html_content += html_block

    # --- 섹션 2: 평가 지진 (요구 스펙트럼 제원) ---
    site_class = site_params.get('site_class', 'S4')
    Z = site_params.get('Z', 0.11)
    
    # S_DBE 계산 (run_csm_evaluation.py와 동일 로직)
    S_MCE_val = Z * 2.0
    S_DBE_val = S_MCE_val * (2.0/3.0)
    
    Fa, Fv = get_site_coefficients(S_DBE_val, site_class)
    calc_SDS, calc_SD1 = calculate_design_acceleration(S_DBE_val, site_class)
    Ts = calc_SD1 / calc_SDS if calc_SDS > 0 else 0
    T0 = 0.2 * Ts

    html_block = textwrap.dedent(f"""
    <div class="section">
        <h2>2. 평가 지진 (요구 스펙트럼 제원)</h2>
        <p>평가 대상 구조물이 위치한 지역 및 지반 조건에 따른 설계 응답 스펙트럼 생성에 사용된 파라미터는 다음과 같습니다.</p>
        <div class="info-grid">
            <div class="info-card" style="border-left-color: #3498db;">
                <h4 style="color: #2980b9;">[입력 파라미터]</h4>
                <ul>
                    <li><span class="label">지역 구분:</span> 지진구역 I</li>
                    <li><span class="label">지반 등급:</span> {site_class}</li>
                    <li><span class="label">지진구역계수 (Z):</span> {Z}g</li>
                    <li><span class="label">구조물 중요도 계수 (Ie):</span> {importance_class}</li>
                </ul>
            </div>
            <div class="info-card" style="border-left-color: #e67e22;">
                <h4 style="color: #d35400;">[계산된 설계 스펙트럼 계수]</h4>
                <ul>
                    <li><span class="label">단주기 증폭계수 (Fa):</span> {Fa:.2f}</li>
                    <li><span class="label">1초주기 증폭계수 (Fv):</span> {Fv:.2f}</li>
                    <li><span class="label">설계스펙트럼가속도(SDS):</span> {calc_SDS:.3f}g</li>
                    <li><span class="label">설계스펙트럼가속도(SD1):</span> {calc_SD1:.3f}g</li>
                    <li><span class="label">단부 유효 주기 (T0):</span> {T0:.3f} sec</li>
                    <li><span class="label">장주기 유효 주기 (Ts):</span> {Ts:.3f} sec</li>
                </ul>
            </div>
        </div>
    </div>
    """).strip()
    html_content += html_block

    # --- 각 해석 결과 순회 ---
    result_dirs = sorted([d for d in Path(results_root_dir).iterdir() if d.is_dir() and "Run_Single" in d.name])
    
    for res_dir in result_dirs:
        dir_name = res_dir.name
        direction = 'X' if '_X' in dir_name else ('Z' if '_Z' in dir_name else 'Unknown')
        
        modal_path = res_dir / 'modal_properties.json'
        summary_json_path = res_dir / f"csm_evaluation_summary_{direction}.json"
        
        if not modal_path.exists(): continue

        with open(modal_path, 'r') as f:
            modal_data = json.load(f)
            all_modes = modal_data.get('modal_properties', [])
            dom_mode = modal_data.get('dominant_mode', {})

        # Calculate modes for RSA
        modes_for_rsa = 0
        cumulative_mpr_x = 0.0
        cumulative_mpr_z = 0.0
        for mode in all_modes:
            if direction == 'X':
                cumulative_mpr_x += mode['mpr_x']
                if cumulative_mpr_x * 100 >= 90.0:
                    modes_for_rsa = mode['mode']
                    break
            else: # direction == 'Z'
                cumulative_mpr_z += mode['mpr_z']
                if cumulative_mpr_z * 100 >= 90.0:
                    modes_for_rsa = mode['mode']
                    break
        
        # --- 3. 대상 구조물 정보 ---
        html_block = textwrap.dedent(f"""
        <div class="section">
            <h2>3-{direction}. 대상 구조물 정보 ({direction}방향)</h2>
            
            <h3>3-{direction}-1. 구조물 형상 및 재료</h3>
            <div class="img-container">
        """).strip()
        html_content += html_block


        # 구조물 형상 이미지 (3D 모델, 입면도)
        img_patterns = [
            (f"*model_3D_Matplotlib.png", "3D 와이어프레임 모델"),
            (f"*model_2D_Elevation_{direction}.png", f"{direction}방향 입면도")
        ]
        
        for pattern, caption in img_patterns:
            found_imgs = list(res_dir.glob(pattern))
            if found_imgs:
                b64_str = load_image_as_base64(found_imgs[0])
                if b64_str:
                    html_block = textwrap.dedent(f"""
                    <div class="img-box" style="max-width: 48%;">
                        <img src="data:image/png;base64,{b64_str}" alt="{caption}">
                        <div class="img-caption">{caption}</div>
                    </div>
                    """).strip()
                    html_content += html_block
        
        # 재료 모델 그래프
        mat_patterns = [
            (f"*material_concrete_combined.png", "콘크리트 응력-변형률 관계"),
            (f"*material_rebar.png", "철근 응력-변형률 관계")
        ]
        for pattern, caption in mat_patterns:
            found = list(res_dir.glob(pattern))
            if found:
                b64 = load_image_as_base64(found[0])
                if b64:
                    html_block = textwrap.dedent(f"""
                    <div class="img-box" style="max-width: 48%;">
                        <img src="data:image/png;base64,{b64}" alt="{caption}">
                        <div class="img-caption">{caption}</div>
                    </div>
                    """).strip()
                    html_content += html_block

        html_block = textwrap.dedent(f"""
            </div>

            <h3>3-2. 구조물 고유주기 및 질량참여율</h3>
            <p>해석에 사용된 모든 모드의 고유 주기와 질량 참여율은 다음과 같습니다. 누적 질량 참여율이 90% 이상이 되는 모드까지 해석에 고려되었습니다.</p>
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
        for mode in all_modes:
            cum_x += mode['mpr_x']
            cum_z += mode['mpr_z']
            is_dom = "style='background-color: #e8f8f5; font-weight: bold;'" if mode['mode'] == dom_mode['mode'] else ""
            
            html_content += f"""
                    <tr {is_dom}>
                        <td>{mode['mode']}</td>
                        <td>{mode['period']:.4f}</td>
                        <td>{mode['mpr_x']*100:.2f}</td>
                        <td>{cum_x*100:.2f}</td>
                        <td>{mode['mpr_z']*100:.2f}</td>
                        <td>{cum_z*100:.2f}</td>
                    </tr>
            """
        
        html_block = textwrap.dedent(f"""
                </tbody>
            </table>
            <p class="img-caption" style="text-align: left;">* 음영 처리된 행은 {direction}방향 지배 모드입니다.</p>
        </div>
        """).strip()
        html_content += html_block
        
        # --- 4. 푸쉬오버 해석 적용성 검토 ---
        html_block = textwrap.dedent(f"""
        <div class="section">
            <h2>4-{direction}. 푸쉬오버 해석 적용성 검토 ({direction}방향)</h2>
            <p>1차 모드(지배 모드)만으로 수행한 비선형 정적 해석의 층 전단력이, <strong>질량 참여율 합계 90% 이상을 만족하는 {modes_for_rsa}차 모드까지 고려한 다중 모드 응답 스펙트럼 해석(RSA) 결과의 130% 이상</strong>인지 검증합니다.</p>
            <div class="img-container">
        """).strip()
        html_content += html_block

        # 130% 룰 검증 이미지
        verif_img = list(res_dir.glob(f"*NSP_verification_plot_{direction}.png"))
        if verif_img:
            b64_str = load_image_as_base64(verif_img[0])
            if b64_str:
                html_block = textwrap.dedent(f"""
                <div class="img-box" style="max-width: 600px;">
                    <img src="data:image/png;base64,{b64_str}" alt="130% 검증 그래프">
                    <div class="img-caption">비선형 정적 해석 타당성 검증 ({direction}방향)</div>
                </div>
                """).strip()
                html_content += html_block
        
        html_block = textwrap.dedent("""
            </div>

            <h3>4.1 푸쉬오버 해석 결과 (동영상)</h3>
            <p>횡하중 증가에 따른 구조물의 변형 형상과 소성 힌지 발생 과정을 보여줍니다. (재생 버튼을 눌러 확인)</p>
            <div class="img-container">
        """).strip()
        html_content += html_block

        # 푸쉬오버 동영상 (MP4) - Base64 임베딩
        video_path = list(res_dir.glob(f"*pushover_animation.mp4"))
        if video_path:
            b64_vid = load_file_as_base64(video_path[0])
            if b64_vid:
                html_block = textwrap.dedent(f"""
                <div class="img-box" style="max-width: 800px;">
                    <video controls loop muted playsinline style="width: 100%;">
                        <source src="data:video/mp4;base64,{b64_vid}" type="video/mp4">
                        브라우저가 동영상을 지원하지 않습니다.
                    </video>
                    <div class="img-caption">푸쉬오버 해석 애니메이션 ({direction}방향)</div>
                </div>
                """).strip()
                html_content += html_block
        else:
            # 동영상이 없으면 정적 이미지 대체
            static_push = list(res_dir.glob("*pushover_final_plot.png"))
            if static_push:
                b64 = load_image_as_base64(static_push[0])
                html_block = textwrap.dedent(f"""
                <div class="img-box">
                    <img src="data:image/png;base64,{b64}" alt="푸쉬오버 결과">
                    <div class="img-caption">푸쉬오버 곡선 및 최종 힌지 분포 (동영상 없음)</div>
                </div>
                """).strip()
                html_content += html_block

        html_block = textwrap.dedent("""
            </div>
        </div>
        """).strip()
        html_content += html_block

        # --- 5. 성능점 산정 및 평가 결과 ---
        html_block = textwrap.dedent(f"""
        <div class="section">
            <h2>5-{direction}. 성능점 산정 및 평가 결과 ({direction}방향)</h2>
            <p>역량스펙트럼법(CSM)을 통해 각 재현주기별 성능 목표에 대한 성능점과 해당 층간 변위비를 산정하고, 목표 성능 만족 여부를 판정합니다.</p>
            
            <h3>5.1 CSM 성능점 그래프</h3>
            <div class="img-container">
        """).strip()
        html_content += html_block

        # CSM 평가 요약 데이터 로드
        csm_summary = []
        if summary_json_path.exists():
            with open(summary_json_path, 'r', encoding='utf-8') as f:
                csm_summary = json.load(f)

        # CSM 결과 이미지와 상세 정보를 나란히 표시
        csm_imgs = sorted(list(res_dir.glob(f"CSM_performance_point_{direction}_*.png")))
        
        for img_path in csm_imgs:
            # 파일명 파싱
            obj_name_from_file = img_path.stem.split(f"_{direction}_")[-1].replace("_", " ")
            b64_str = load_image_as_base64(img_path)
            
            # 해당 목표에 대한 요약 데이터 찾기
            target_summary = next((item for item in csm_summary if item["objective_name"].replace(" ", "_") == obj_name_from_file.replace(" ", "_")), None)
            
            if b64_str and target_summary:
                status_class = "pass-text" if target_summary['status'] == "PASS" else "fail-text"
                
                html_block = textwrap.dedent(f"""
                <div class="csm-result-container">
                    <div class="csm-chart img-box">
                        <img src="data:image/png;base64,{b64_str}" alt="{obj_name_from_file}">
                        <div class="img-caption"><strong>[{target_summary['direction']}방향] {target_summary['repetition_period']} {target_summary['objective_name']} 성능 평가 그래프</strong></div>
                    </div>
                    <div class="csm-details">
                        <h4>평가 상세 결과</h4>
                        <div class="detail-row"><span class="detail-label">성능 목표:</span> <span class="detail-value">{target_summary['objective_name']}</span></div>
                        <div class="detail-row"><span class="detail-label">재현 주기:</span> <span class="detail-value">{target_summary['repetition_period']}</span></div>
                        <div class="detail-row"><span class="detail-label">성능점 변위 (Sd):</span> <span class="detail-value">{target_summary['perf_point_Sd_m']:.4f} m</span></div>
                        <div class="detail-row"><span class="detail-label">성능점 가속도 (Sa):</span> <span class="detail-value">{target_summary['perf_point_Sa_g']:.4f} g</span></div>
                        <div class="detail-row"><span class="label">유효 주기 (Teff):</span> <span class="detail-value">{target_summary['effective_period_s']:.3f} sec</span></div>
                        <div class="detail-row"><span class="label">유효 감쇠비:</span> <span class="detail-value">{target_summary['effective_damping_pct']:.1f} %</span></div>
                        <div class="detail-row"><span class="label">계산 층간변위비:</span> <span class="detail-value">{target_summary['calculated_drift_pct']:.3f} %</span></div>
                        <div class="detail-row"><span class="label">허용 층간변위비:</span> <span class="detail-value">{target_summary['allowed_drift_pct']:.3f} %</span></div>
                        <div class="detail-row" style="border-bottom: none; margin-top: 15px;">
                            <span class="detail-label" style="font-size: 1.1em;">최종 판정:</span> 
                            <span class="detail-value {status_class}" style="font-size: 1.2em;">{target_summary['status']}</span>
                        </div>
                    </div>
                </div>
                """).strip()
                html_content += html_block

        html_block = textwrap.dedent("""
            </div>
        </div>
        """).strip()
        html_content += html_block

    html_block = textwrap.dedent("""
    <div class="section">
        <h2>6. 최종 종합 결론</h2>
        <p>본 보고서는 KDS 41 17 00 (건축물 내진설계기준)에 따라 비선형 정적 해석(Pushover Analysis) 및 역량스펙트럼법(CSM)을 이용하여 대상 구조물의 내진 성능을 평가하였습니다.</p>
        <p><strong>주요 평가 결과 요약:</strong></p>
        <ul>
            <li>대상 구조물은 <strong>[여기에 구조물 개요 요약]</strong>입니다.</li>
            <li>설계 응답 스펙트럼은 <strong>[여기에 평가 지진 요약]</strong> 조건을 기반으로 생성되었습니다.</li>
            <li>각 성능 목표에 대한 평가 결과는 <strong>[여기에 성능점 요약 및 판정 결과 요약]</strong>입니다.</li>
        </ul>
        <p>위의 각 섹션별 분석 결과와 그래프, 표를 종합적으로 검토하시어, 해당 구조물이 목표로 하는 내진 성능 수준을 만족하는지 최종 판단하시기 바랍니다.</p>
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