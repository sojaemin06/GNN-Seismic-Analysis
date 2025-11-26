# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from pathlib import Path

def animate_csm_process(csm_results, results_dir, direction):
    """
    [수정] CSM 반복 과정을 시각화하는 애니메이션을 생성합니다. (단순화 및 안정화 버전)
    """
    print(f"\n--- Creating CSM Animation for {direction}-Direction ---")

    # 데이터 추출
    capacity_adrs = csm_results['capacity_adrs']
    demand_5pct = csm_results['demand_spectrum_5pct']
    history = csm_results['iteration_history']
    perf_point = csm_results['performance_point']

    fig, ax = plt.subplots(figsize=(12, 10))
    
    max_sd = max(np.max(capacity_adrs['Sd']), np.max(demand_5pct['Sd'])) * 1.1
    max_sa = max(np.max(capacity_adrs['Sa']), np.max(demand_5pct['Sa'])) * 1.1

    # 애니메이션 업데이트 함수 (매 프레임 다시 그리기)
    def update(frame_num):
        ax.clear() # 매 프레임마다 축을 초기화

        # 정적 배경 플롯
        ax.plot(capacity_adrs['Sd'], capacity_adrs['Sa'], 'b-', lw=2.5, label='Capacity Spectrum')
        ax.plot(demand_5pct['Sd'], demand_5pct['Sa'], 'r--', lw=1.5, label='5% Demand Spectrum')

        if frame_num < len(history):
            # 반복 과정
            iter_data = history[frame_num]
            damped_sd, damped_sa = iter_data['demand_curve_damped']
            trial_sd, trial_sa = iter_data['trial_Sd'], iter_data['trial_Sa']
            
            ax.plot(damped_sd, damped_sa, 'g-.', lw=2, label='Damped Demand Spectrum')
            ax.plot([trial_sd], [trial_sa], 'yo', markersize=12, label='Trial Point')
            ax.plot([0, trial_sd], [0, trial_sa], 'k--', lw=1, label='Secant Stiffness')
            
            text = (f"Iteration: {iter_data['iter']}\n"
                    f"μ = {iter_data['mu']:.2f}\n"
                    f"β_eff = {iter_data['beta_eff']*100:.1f}%\n"
                    f"Trial Sd = {trial_sd:.4f} m")
        else: # 마지막 프레임
            # 마지막 반복의 감쇠 스펙트럼을 보여줌
            last_iter_data = history[-1]
            damped_sd, damped_sa = last_iter_data['demand_curve_damped']
            ax.plot(damped_sd, damped_sa, 'g-.', lw=2, label='Damped Demand Spectrum')

            # 최종 성능점 표시
            ax.plot([perf_point['Sd']], [perf_point['Sa']], 'm*', markersize=18, label='Performance Point')
            text = (f"Converged!\n"
                    f"Perf. Point Sd = {perf_point['Sd']:.4f} m\n"
                    f"Perf. Point Sa = {perf_point['Sa']:.4f} g\n"
                    f"T_eff = {perf_point['T_eff']:.3f} s")

        # 공통 플롯 속성 설정
        ax.set_title(f'CSM Performance Point Calculation ({direction}-Direction)')
        ax.set_xlabel('Spectral Displacement, Sd (m)')
        ax.set_ylabel('Spectral Acceleration, Sa (g)')
        ax.grid(True, which='both', linestyle=':')
        ax.text(0.6, 0.95, text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))
        ax.legend(loc='lower right')
        ax.set_xlim(0, max_sd)
        ax.set_ylim(0, max_sa)

    # 애니메이션 생성
    total_frames = len(history) + 1
    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1200, repeat=False)
    
    try:
        output_path = results_dir / f"CSM_animation_{direction}.gif"
        writer = animation.PillowWriter(fps=1)
        ani.save(output_path, writer=writer)
        print(f"CSM animation saved to: {output_path}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Please ensure you have 'Pillow' installed (`pip install Pillow`).")
    
    plt.close(fig)