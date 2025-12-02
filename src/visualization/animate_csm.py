# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

def animate_csm_process(csm_results, results_dir, direction, filename_suffix=""):
    """
    [수정] CSM 반복 과정을 시각화하는 애니메이션을 생성합니다.
    filename_suffix: 파일명에 추가할 식별자 (예: 성능목표 이름)
    """
    # print(f"--- Creating CSM Animation ({filename_suffix}) ---") # 로그 과다 방지

    # 데이터 추출
    capacity_adrs = csm_results['capacity_adrs']
    demand_5pct = csm_results['demand_spectrum_5pct']
    history = csm_results['iteration_history']
    perf_point = csm_results['performance_point']

    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_sd = max(np.max(capacity_adrs['Sd']), np.max(demand_5pct['Sd'])) * 1.1
    max_sa = max(np.max(capacity_adrs['Sa']), np.max(demand_5pct['Sa'])) * 1.1

    # 애니메이션 업데이트 함수
    def update(frame_num):
        ax.clear()
        
        # 정적 배경
        ax.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], 'b-', lw=2, label='Capacity')
        ax.plot(demand_5pct['Sd'], demand_5pct['Sa'], 'r--', lw=1, label='5% Demand')

        if frame_num < len(history):
            iter_data = history[frame_num]
            damped_sd, damped_sa = iter_data['demand_curve_damped']
            trial_sd, trial_sa = iter_data['trial_Sd'], iter_data['trial_Sa']
            
            ax.plot(damped_sd, damped_sa, 'g-.', lw=1.5, label='Damped Demand')
            ax.plot([trial_sd], [trial_sa], 'yo', markersize=10, label='Trial Point')
            # 할선 강성
            if trial_sd > 0:
                ax.plot([0, trial_sd], [0, trial_sa], 'k--', lw=0.5)
            
            title_str = f"CSM Iteration {iter_data['iter']} ({filename_suffix})"
            info_str = (f"μ={iter_data['mu']:.2f}, β={iter_data['beta_eff']*100:.1f}%")
        else:
            # 수렴 결과
            last_iter = history[-1]
            damped_sd, damped_sa = last_iter['demand_curve_damped']
            ax.plot(damped_sd, damped_sa, 'g-.', lw=1.5, label='Final Damped Demand')
            ax.plot([perf_point['Sd']], [perf_point['Sa']], 'm*', markersize=15, label='Perf. Point')
            
            title_str = f"CSM Converged ({filename_suffix})"
            info_str = (f"Sd={perf_point['Sd']:.4f}m, Sa={perf_point['Sa']:.4f}g")

        ax.set_title(title_str)
        ax.set_xlabel('Spectral Displacement (m)')
        ax.set_ylabel('Spectral Acceleration (g)')
        ax.set_xlim(0, max_sd)
        ax.set_ylim(0, max_sa)
        ax.legend(loc='upper right')
        ax.text(0.05, 0.95, info_str, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', fc='white', alpha=0.8))

    total_frames = len(history) + 1
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000, repeat=False)
    
    try:
        safe_suffix = filename_suffix.replace(" ", "_").replace("(", "").replace(")", "")
        output_filename = f"CSM_{safe_obj_name}_{direction}.gif" if "safe_obj_name" in locals() else f"CSM_animation_{direction}_{safe_suffix}.gif"
        output_path = results_dir / output_filename
        
        writer = animation.PillowWriter(fps=1)
        ani.save(output_path, writer=writer)
        # print(f"Saved: {output_path.name}")
    except Exception as e:
        print(f"Animation save failed: {e}")
    
    plt.close(fig)
