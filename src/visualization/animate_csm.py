# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from PIL import Image
import glob

def animate_csm_process(csm_results, results_dir, direction, filename_suffix=""):
    """
    CSM 반복 과정을 시각화하는 애니메이션(GIF) 및 최종 성능점 플롯(PNG)을 생성합니다.
    (FuncAnimation 대신 개별 프레임 저장 후 병합 방식 사용)
    """
    print(f"--- Creating CSM Animation and Static Plot ({filename_suffix}) ---")

    # 데이터 추출
    capacity_adrs = csm_results.get('capacity_adrs', {'Sd': [], 'Sa': []})
    demand_5pct = csm_results.get('demand_spectrum_5pct', {'Sd': [], 'Sa': []})
    history = csm_results.get('iteration_history', [])
    perf_point = csm_results.get('performance_point', {'Sd': 0.0, 'Sa': 0.0})

    # 안전한 파일명 접미사 생성
    safe_suffix = filename_suffix.replace(" ", "_").replace("(", "").replace(")", "").replace(".", "_")

    # 축 범위 계산
    max_sd_cap = np.max(capacity_adrs['Sd']) if capacity_adrs and capacity_adrs['Sd'] and len(capacity_adrs['Sd']) > 0 else 0.1
    max_sd_dem = np.max(demand_5pct['Sd']) if demand_5pct and demand_5pct['Sd'] and len(demand_5pct['Sd']) > 0 else 0.1
    max_sd = max(max_sd_cap, max_sd_dem) * 1.1

    max_sa_cap = np.max(capacity_adrs['Sa']) if capacity_adrs and capacity_adrs['Sa'] and len(capacity_adrs['Sa']) > 0 else 0.1
    max_sa_dem = np.max(demand_5pct['Sa']) if demand_5pct and demand_5pct['Sa'] and len(demand_5pct['Sa']) > 0 else 0.1
    max_sa = max(max_sa_cap, max_sa_dem) * 1.1

    temp_filenames = []
    
    # 헬퍼 함수: 프레임 그리기 및 저장
    def save_frame(iter_data, is_final=False, frame_idx=0):
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 1. Capacity Curve (Blue)
        if capacity_adrs and capacity_adrs['Sd'] and len(capacity_adrs['Sd']) > 0:
            ax.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], 'b-', lw=2, label='Capacity')
        
        # 2. 5% Demand Spectrum (Red Dashed)
        if demand_5pct and demand_5pct['Sd'] and len(demand_5pct['Sd']) > 0:
            ax.plot(demand_5pct['Sd'], demand_5pct['Sa'], 'r--', lw=1, label='5% Demand')

        # 3. Previous Iterations (Damped Demand Curves) - 누적 표시
        # 현재 프레임 이전까지의 모든 history 데이터를 순회하며 흐리게 표시
        if history:
            # frame_idx는 0부터 시작. is_final=True일 때는 len(history)와 같음.
            limit_idx = frame_idx if not is_final else len(history)
            
            for i in range(limit_idx):
                prev_iter_data = history[i]
                p_sd, p_sa = prev_iter_data['demand_curve_damped']
                if p_sd and len(p_sd) > 0:
                    # 범례에 중복 추가되지 않도록 첫 번째만 라벨링하거나 라벨 생략
                    label = 'Previous Iterations' if i == 0 else None
                    ax.plot(p_sd, p_sa, color='gray', alpha=0.3, linestyle=':', lw=1, label=label)

        title_str = ""
        info_str = ""

        if is_final:
            # 최종 수렴 상태
            if iter_data: # 마지막 반복 데이터가 있다면 감쇠 곡선 그리기
                damped_sd, damped_sa = iter_data['demand_curve_damped']
                if damped_sd and len(damped_sd) > 0:
                    ax.plot(damped_sd, damped_sa, 'g-.', lw=2.0, label='Final Damped Demand') # 두께 강조
            
            ax.plot([perf_point['Sd']], [perf_point['Sa']], 'm*', markersize=15, label='Perf. Point')
            title_str = f"CSM Converged ({filename_suffix})"
            info_str = (f"Sd={perf_point['Sd']:.4f}m, Sa={perf_point['Sa']:.4f}g")
        
        elif iter_data:
            # 반복 과정 (현재 단계)
            damped_sd, damped_sa = iter_data['demand_curve_damped']
            trial_sd, trial_sa = iter_data['trial_Sd'], iter_data['trial_Sa']
            
            if damped_sd and len(damped_sd) > 0:
                ax.plot(damped_sd, damped_sa, 'g-.', lw=2.0, label='Current Damped Demand') # 두께 강조
            
            if trial_sd is not None:
                ax.plot([trial_sd], [trial_sa], 'yo', markersize=10, label='Trial Point')
                if trial_sd > 0:
                    ax.plot([0, trial_sd], [0, trial_sa], 'k--', lw=0.5) # Secant stiffness line

            title_str = f"CSM Iteration {iter_data['iter']} ({filename_suffix})"
            info_str = (f"μ={iter_data['mu']:.2f}, β={iter_data['beta_eff']*100:.1f}%")

        ax.set_title(title_str)
        ax.set_xlabel('Spectral Displacement (m)')
        ax.set_ylabel('Spectral Acceleration (g)')
        ax.set_xlim(0, max_sd)
        ax.set_ylim(0, max_sa)
        ax.legend(loc='upper right')
        ax.text(0.05, 0.95, info_str, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        
        # 임시 파일 저장
        temp_filename = results_dir / f"temp_csm_{direction}_{safe_suffix}_{frame_idx:03d}.png"
        fig.savefig(temp_filename, dpi=100)
        plt.close(fig)
        return str(temp_filename)

    # --- 프레임 생성 루프 ---
    try:
        if not history:
            # 반복 기록이 없으면 최종 결과만 1장 저장
            temp_filenames.append(save_frame(None, is_final=True, frame_idx=0))
        else:
            # 각 반복 단계 저장
            for i, iter_data in enumerate(history):
                temp_filenames.append(save_frame(iter_data, is_final=False, frame_idx=i))
            
            # 최종 결과 프레임 추가 (마지막 반복 데이터 기반)
            temp_filenames.append(save_frame(history[-1], is_final=True, frame_idx=len(history)))

        # --- GIF 생성 ---
        images = [Image.open(fn) for fn in temp_filenames]
        output_gif_path = results_dir / f"CSM_animation_{direction}_{safe_suffix}.gif"
        
        # 첫 번째 이미지를 기준으로 저장 (duration=1000ms, loop=0(무한))
        images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=1000, loop=0)
        print(f"Animation saved successfully to: {output_gif_path}")

    except Exception as e:
        print(f"---! Error saving animation: {e} !---")
    
    finally:
        # 임시 파일 삭제
        for fn in temp_filenames:
            try:
                os.remove(fn)
            except OSError:
                pass

    # 2. 최종 성능점 정적 플롯 저장 (PNG)
    try:
        static_image_filename = f"CSM_performance_point_{direction}_{safe_suffix}.png"
        static_image_path = results_dir / static_image_filename
        
        # 고해상도용 헬퍼 함수 호출 (파일 저장을 위해 plt.subplots 다시 생성됨)
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # --- 주기 가이드라인 추가 (T = 0.5, 1.0, 1.5, 2.0 sec) ---
        g = 9.81
        periods = [0.5, 1.0, 1.5, 2.0]
        colors = ['gray'] * 4
        
        for T in periods:
            # Sa(g) = (4 * pi^2 / T^2) * Sd(m) / g
            K_radial = (4 * np.pi**2) / (T**2) / g
            
            # 그래프 범위 내에서 선 그리기
            x_guide = np.array([0, max_sd])
            y_guide = K_radial * x_guide
            
            # y값이 max_sa를 넘지 않도록 클리핑
            if y_guide[1] > max_sa:
                x_guide[1] = max_sa / K_radial
                y_guide[1] = max_sa
                
            ax.plot(x_guide, y_guide, linestyle='--', linewidth=0.8, color='gray', alpha=0.5, zorder=0)
            ax.text(x_guide[1], y_guide[1], f'T={T}s', fontsize=8, color='gray', ha='right', va='bottom')

        if capacity_adrs and capacity_adrs['Sd'] and len(capacity_adrs['Sd']) > 0:
            ax.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], 'b-', lw=2, label='Capacity')
        if demand_5pct and demand_5pct['Sd'] and len(demand_5pct['Sd']) > 0:
            ax.plot(demand_5pct['Sd'], demand_5pct['Sa'], 'r--', lw=1, label='5% Demand')
            
        # 최종 상태
        if history:
            last_iter = history[-1]
            d_sd, d_sa = last_iter['demand_curve_damped']
            if d_sd and len(d_sd) > 0: 
                ax.plot(d_sd, d_sa, 'g-.', lw=1.5, label='Final Damped Demand')
            
        ax.plot([perf_point['Sd']], [perf_point['Sa']], 'm*', markersize=15, label='Perf. Point')
        
        ax.set_title(f"CSM Converged ({filename_suffix})")
        ax.set_xlabel('Spectral Displacement (m)')
        ax.set_ylabel('Spectral Acceleration (g)')
        ax.set_xlim(0, max_sd)
        ax.set_ylim(0, max_sa)
        ax.legend(loc='upper right')
        info_str = (f"Sd={perf_point['Sd']:.4f}m, Sa={perf_point['Sa']:.4f}g\nT_eff={perf_point['T_eff']:.3f}s")
        ax.text(0.05, 0.95, info_str, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

        fig.savefig(static_image_path, dpi=300, bbox_inches='tight')
        print(f"Final performance point plot saved to: {static_image_path}")
    except Exception as e:
        print(f"---! Error saving static plot: {e} !---")