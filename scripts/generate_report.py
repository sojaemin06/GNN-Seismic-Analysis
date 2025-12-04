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

    html_block = textwrap.dedent(f"""
    <div class="section">
        <h2>2. 평가 지진 (요구 스펙트럼 제원)</h2>
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
                    <li><span class="label">Fa:</span> {Fa:.2f}</li>
                    <li><span class="label">Fv:</span> {Fv:.2f}</li>
                    <li><span class="label">SDS:</span> {calc_SDS:.3f}g</li>
                    <li><span class="label">SD1:</span> {calc_SD1:.3f}g</li>
                    <li><span class="label">T0:</span> {T0:.3f} sec</li>
                    <li><span class="label">Ts:</span> {Ts:.3f} sec</li>
                </ul>
            </div>
        </div>
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
        
        <h3>3.2 부재 상세 정보</h3>
        <p>본 해석 모델에 적용된 주요 부재의 단면 및 철근 상세 정보입니다.</p>
        <table>
            <thead>
                <tr>
                    <th>구분</th>
                    <th>층 그룹</th>
                    <th>위치</th>
                    <th>단면 크기 (mm)</th>
                    <th>주철근 상세</th>
                </tr>
            </thead>
            <tbody>
    """).strip()
    html_content += html_block
    
    # 부재 상세 정보 (dataset_config.json 로드 필요)
    try:
        with open(dataset_config_path, 'r', encoding='utf-8') as f:
            ds_config = json.load(f)
            mem_props = ds_config.get('member_properties', {})
            # 실제 해석에 사용된 부재 정보는 run_single_analysis에서 랜덤하게 결정되므로
            # 여기서는 가능한 범위나 대표값을 표시하는 것이 좋음.
            # 하지만 정확한 정보를 위해선 run_single_analysis 결과에 부재 정보를 저장해야 함.
            # 현재는 간단히 범위만 표시하거나, "상세 정보는 해석 로그 참조"로 대체.
            # 또는 member_properties.json 내용을 간단히 요약.
            
            # 임시: 대표적인 정보만 출력 (실제 해석값은 아닐 수 있음 주의)
            html_content += f"""
                <tr><td colspan="5">상세 부재 정보는 해석 설정 파일(dataset_config.json)을 참조하십시오.</td></tr>
            """
    except:
        pass

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
                    <video controls loop muted playsinline style="width: 100%;">
                        <source src="data:video/mp4;base64,{b64_vid}" type="video/mp4">
                        브라우저 미지원
                    </video>
                    <div class="img-caption">푸쉬오버 해석 애니메이션 ({direction}{sign})</div>
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
        <ul>
            <li><strong>적용성 검토:</strong> 130% 룰 검증을 통해 1차 모드 기반 해석의 타당성을 확인하였습니다.</li>
            <li><strong>성능 만족 여부:</strong> 각 방향(X, Z) 및 가력 부호(+, -)에 대해 산정된 성능점에서의 층간 변위비가 허용 기준을 만족하는지 확인하였습니다.</li>
            <li><strong>종합 판단:</strong> 모든 해석 케이스에서 'PASS' 판정을 받은 경우, 해당 구조물은 목표 내진 성능을 확보한 것으로 판단할 수 있습니다.</li>
        </ul>
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
