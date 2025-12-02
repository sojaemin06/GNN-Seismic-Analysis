# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_site_coefficients(S, site_class):
    """
    KDS 41 17 00:2022 [표 4.2-1], [표 4.2-2]에 따라 지반증폭계수 Fa, Fv를 계산합니다.
    """
    fa_table = {
        'S1': [(0.1, 1.12), (0.2, 1.12), (0.3, 1.12)],
        'S2': [(0.1, 1.4), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.5), (0.3, 1.3)],
        'S4': [(0.1, 1.6), (0.2, 1.4), (0.3, 1.2)],
        'S5': [(0.1, 1.8), (0.2, 1.3), (0.3, 1.3)],
    }
    
    fv_table = {
        'S1': [(0.1, 0.84), (0.2, 0.84), (0.3, 0.84)],
        'S2': [(0.1, 1.5), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.6), (0.3, 1.5)],
        'S4': [(0.1, 2.2), (0.2, 2.0), (0.3, 1.8)],
        'S5': [(0.1, 3.0), (0.2, 2.7), (0.3, 2.4)],
    }

    if site_class not in fa_table:
        raise ValueError(f"Invalid site class: {site_class}")

    def interpolate_coeff(S_val, table):
        points = table[site_class]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        return np.interp(S_val, x, y)

    Fa = interpolate_coeff(S, fa_table)
    Fv = interpolate_coeff(S, fv_table)
    
    return Fa, Fv

def calculate_design_acceleration(S, site_class):
    """
    설계스펙트럼가속도 SDS, SD1을 계산합니다.
    """
    Fa, Fv = get_site_coefficients(S, site_class)
    SDS = S * Fa * (2.0 / 3.0)
    SD1 = S * Fv * (2.0 / 3.0)
    return SDS, SD1

def determine_seismic_design_category(SDS, SD1, importance_class):
    """
    내진설계범주를 결정합니다.
    """
    col_idx_map = {'Teuk': 0, 'I': 1, 'II': 2}
    if importance_class not in col_idx_map:
        raise ValueError(f"Invalid importance class: {importance_class}")
    
    col_idx = col_idx_map[importance_class]

    def get_cat_sds(sds_val, idx):
        if sds_val >= 0.50: return ['D', 'D', 'D'][idx]
        elif 0.33 <= sds_val < 0.50: return ['D', 'C', 'C'][idx]
        elif 0.17 <= sds_val < 0.33: return ['C', 'B', 'B'][idx]
        else: return ['A', 'A', 'A'][idx]

    def get_cat_sd1(sd1_val, idx):
        if sd1_val >= 0.20: return ['D', 'D', 'D'][idx]
        elif 0.14 <= sd1_val < 0.20: return ['D', 'C', 'C'][idx]
        elif 0.07 <= sd1_val < 0.14: return ['C', 'B', 'B'][idx]
        else: return ['A', 'A', 'A'][idx]

    cat_sds = get_cat_sds(SDS, col_idx)
    cat_sd1 = get_cat_sd1(SD1, col_idx)
    
    return max(cat_sds, cat_sd1)

def generate_kds2022_demand_spectrum(S, site_class, T_long=5.0):
    """
    [수정] 유효지반가속도 S를 직접 받아 설계응답스펙트럼(ADRS)을 생성합니다.
    """
    SDS, SD1 = calculate_design_acceleration(S, site_class)
    
    Ts = SD1 / SDS if SDS > 0 else 0
    T0 = 0.2 * Ts
    
    T = np.linspace(0.01, T_long, 500)
    Sa = np.zeros_like(T)
    
    if SDS > 0:
        Sa[T <= T0] = SDS * (0.4 + 0.6 * T[T <= T0] / T0)
        Sa[(T > T0) & (T <= Ts)] = SDS
        Sa[T > Ts] = SD1 / T[T > Ts]
    
    Sd = (T**2 / (4 * np.pi**2)) * Sa * 9.81
    
    return Sd, Sa, SDS, SD1