# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- 프로젝트 루트 경로를 sys.path에 추가 ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def get_site_coefficients(S, site_class):
    """
    KDS 41 17 00:2022 [표 4.2-1], [표 4.2-2]에 따라 지반증폭계수를 계산합니다.
    선형 보간법을 사용합니다.
    """
    fa_data = {
        'S1': [(0.1, 1.12), (0.2, 1.12), (0.3, 1.12)],
        'S2': [(0.1, 1.4), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.5), (0.3, 1.3)],
        'S4': [(0.1, 1.6), (0.2, 1.4), (0.3, 1.2)],
        'S5': [(0.1, 1.8), (0.2, 1.3), (0.3, 1.3)],
    }
    fv_data = {
        'S1': [(0.1, 0.84), (0.2, 0.84), (0.3, 0.84)],
        'S2': [(0.1, 1.5), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.6), (0.3, 1.5)],
        'S4': [(0.1, 2.2), (0.2, 2.0), (0.3, 1.8)],
        'S5': [(0.1, 3.0), (0.2, 2.7), (0.3, 2.4)],
    }

    fa_points = fa_data.get(site_class)
    fv_points = fv_data.get(site_class)

    if not fa_points or not fv_points:
        raise ValueError(f"Invalid site class: {site_class}")

    s_coords, fa_coords = zip(*fa_points)
    s_coords, fv_coords = zip(*fv_points)
    
    Fa = np.interp(S, s_coords, fa_coords)
    Fv = np.interp(S, s_coords, fv_coords)
    
    return Fa, Fv

def generate_design_spectrum(S, site_class, T_long=5.0):
    """
    KDS 41 17 00:2022 기준에 따라 설계응답스펙트럼을 생성합니다.
    """
    Fa, Fv = get_site_coefficients(S, site_class)
    
    SDS = S * Fa * (2/3)
    SD1 = S * Fv * (2/3)
    
    Ts = SD1 / SDS
    T0 = 0.2 * Ts
    
    T = np.linspace(0, T_long, 500)
    Sa = np.zeros_like(T)
    
    # Spectrum definition
    Sa[T <= T0] = (0.4 + 0.6 * T[T <= T0] / T0) * SDS
    Sa[(T > T0) & (T <= Ts)] = SDS
    Sa[T > Ts] = SD1 / T[T > Ts]
    
    print("\n--- Demand Spectrum Parameters (KDS 41 17 00:2022) ---")
    print(f"Assumed S (from map): {S:.3f}g")
    print(f"Assumed Site Class: {site_class}")
    print(f"Interpolated Fa: {Fa:.3f}")
    print(f"Interpolated Fv: {Fv:.3f}")
    print("----------------------------------------------------")
    print(f"SDS: {SDS:.4f}g")
    print(f"SD1: {SD1:.4f}g")
    print(f"T0: {T0:.4f}s")
    print(f"Ts: {Ts:.4f}s")
    print("----------------------------------------------------")

    return T, Sa, SDS, SD1

def plot_spectrum(T, Sa, S, site_class):
    """
    생성된 설계응답스펙트럼을 그래프로 그리고 저장합니다.
    """
    plt.figure(figsize=(12, 8))
    plt.plot(T, Sa, label=f'Design Spectrum (S={S}g, Site Class={site_class})')
    plt.title('Design Response Spectrum (KDS 41 17 00:2022)')
    plt.xlabel('Period, T (s)')
    plt.ylabel('Spectral Acceleration, Sa (g)')
    plt.grid(True, which='both', linestyle='--')
    plt.legend()
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    # Save plot
    output_dir = Path(project_root) / 'results'
    output_dir.mkdir(exist_ok=True)
    plot_path = output_dir / f"demand_spectrum_S_{str(S).replace('.', '')}_{site_class}.png"
    plt.savefig(plot_path)
    print(f"\nDemand spectrum plot saved to: {plot_path}")
    plt.close()

if __name__ == '__main__':
    # 1. 임의의 값 설정 (서울, S4 지반)
    S_value = 0.16  # 유효지반가속도 (2400년 재현주기)
    site_class_value = 'S4'
    
    # 2. 설계응답스펙트럼 생성
    T_coords, Sa_coords, _, _ = generate_design_spectrum(S_value, site_class_value)
    
    # 3. 그래프 출력 및 저장
    plot_spectrum(T_coords, Sa_coords, S_value, site_class_value)
