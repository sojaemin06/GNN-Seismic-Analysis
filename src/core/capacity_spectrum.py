# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def get_site_coefficients_kds2022(S, site_class):
    """
    KDS 41 17 00:2022 [표 4.2-1], [표 4.2-2]에 따라 지반증폭계수를 계산합니다.
    """
    fa_data = {
        'S1': [(0.1, 1.12), (0.2, 1.12), (0.3, 1.12)], 'S2': [(0.1, 1.4), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.5), (0.3, 1.3)], 'S4': [(0.1, 1.6), (0.2, 1.4), (0.3, 1.2)],
        'S5': [(0.1, 1.8), (0.2, 1.3), (0.3, 1.3)],
    }
    fv_data = {
        'S1': [(0.1, 0.84), (0.2, 0.84), (0.3, 0.84)], 'S2': [(0.1, 1.5), (0.2, 1.4), (0.3, 1.3)],
        'S3': [(0.1, 1.7), (0.2, 1.6), (0.3, 1.5)], 'S4': [(0.1, 2.2), (0.2, 2.0), (0.3, 1.8)],
        'S5': [(0.1, 3.0), (0.2, 2.7), (0.3, 2.4)],
    }
    fa_points = fa_data.get(site_class); fv_points = fv_data.get(site_class)
    if not fa_points or not fv_points: raise ValueError(f"Invalid site class: {site_class}")
    s_coords, fa_coords = zip(*fa_points); _, fv_coords = zip(*fv_points)
    Fa = np.interp(S, s_coords, fa_coords); Fv = np.interp(S, s_coords, fv_coords)
    return Fa, Fv

def generate_kds2022_demand_spectrum(S, site_class, T_long=5.0):
    """
    KDS 41 17 00:2022 기준에 따라 5% 감쇠 설계응답스펙트럼을 ADRS 형식으로 생성합니다.
    """
    Fa, Fv = get_site_coefficients_kds2022(S, site_class)
    SDS = S * Fa * (2/3); SD1 = S * Fv * (2/3)
    Ts = SD1 / SDS; T0 = 0.2 * Ts
    
    T = np.linspace(0.01, T_long, 500)
    Sa = np.zeros_like(T)
    
    Sa[T <= T0] = (0.4 + 0.6 * T[T <= T0] / T0) * SDS
    Sa[(T > T0) & (T <= Ts)] = SDS
    Sa[T > Ts] = SD1 / T[T > Ts]
    
    Sd = (T**2 / (4 * np.pi**2)) * Sa * 9.81
    return Sd, Sa

def pushover_to_adrs(df_pushover, pf1, m_eff, phi_roof):
    """
    푸쉬오버 곡선을 ADRS 형식으로 변환하고, 곡선을 단조 증가하도록 보정합니다.
    [수정] Sd 계산 시 지붕층 모드 형상 값(phi_roof)을 반영합니다.
    """
    g = 9.81
    if abs(pf1 * phi_roof) < 1e-9 or abs(m_eff) < 1e-9:
        return pd.DataFrame({'Sd': [], 'Sa': []})

    Sd = np.abs(df_pushover['Roof_Displacement_m'] / (pf1 * phi_roof))
    Sa = np.abs(df_pushover['Base_Shear_N'] / (m_eff * g))
    
    df_adrs = pd.DataFrame({'Sd': Sd, 'Sa': Sa}).sort_values(by='Sd').drop_duplicates(subset=['Sd']).reset_index(drop=True)
    
    if not df_adrs.empty:
        df_adrs['Sa_cummax'] = df_adrs['Sa'].cummax()
        df_adrs = df_adrs[df_adrs['Sa'] >= df_adrs['Sa_cummax'] * 0.95]
    
    if len(df_adrs) > 1 and df_adrs.iloc[0]['Sd'] < 1e-6:
        df_adrs = df_adrs.iloc[1:]

    return df_adrs[['Sd', 'Sa']].reset_index(drop=True)

def _find_intersection(cap_sd, cap_sa, dem_sd, dem_sa):
    """ 두 곡선의 교차점을 선형 보간으로 찾습니다. """
    interp_dem_sa = np.interp(cap_sd, dem_sd, dem_sa, left=0, right=0)
    diff = cap_sa - interp_dem_sa
    sign_change_idx = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_change_idx) == 0: return None
    idx = sign_change_idx[0]
    x1, y1 = cap_sd[idx], cap_sa[idx]; x2, y2 = cap_sd[idx + 1], cap_sa[idx + 1]
    x3, y3 = cap_sd[idx], interp_dem_sa[idx]; x4, y4 = cap_sd[idx + 1], interp_dem_sa[idx + 1]
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-12: return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
    if 0 <= t <= 1 and 0 <= u <= 1:
        return pd.Series({'Sd': x1 + t * (x2 - x1), 'Sa': y1 + t * (y2 - y1)})
    return None

def calculate_performance_point_csm(df_pushover, modal_properties, design_params, max_iter=30, tolerance=0.01):
    """
    [KDS 2022 최종] 역량스펙트럼법(CSM)으로 성능점을 계산합니다.
    """
    pf1, m_eff, phi_roof = modal_properties['pf1'], modal_properties['m_eff_t1'], modal_properties['phi_roof']
    capacity_adrs = pushover_to_adrs(df_pushover, pf1, m_eff, phi_roof)
    if capacity_adrs.empty or len(capacity_adrs) < 2:
        print("Warning: Capacity curve is empty or too short for CSM analysis."); return None

    Sd_demand_5pct, Sa_demand_5pct = generate_kds2022_demand_spectrum(S=design_params['S'], site_class=design_params['site_class'])
    
    try:
        Sa_max = capacity_adrs['Sa'].max()
        yield_point = capacity_adrs[capacity_adrs['Sa'] >= Sa_max * 0.6].iloc[0]
        Sd_y, Sa_y = yield_point['Sd'], yield_point['Sa']
        if Sd_y < 1e-9: raise IndexError
    except (IndexError, KeyError, ValueError):
        Sd_y, Sa_y = capacity_adrs.iloc[-1]['Sd'], capacity_adrs.iloc[-1]['Sa']
        print("Warning: Could not robustly determine yield point. Using last point.")

    pi_trial = capacity_adrs.iloc[-1]
    iteration_history = []
    
    for i in range(max_iter):
        mu = pi_trial['Sd'] / Sd_y if pi_trial['Sd'] > Sd_y and Sd_y > 1e-9 else 1.0
        beta_eff = 0.05 + 0.565 / np.pi * (mu - 1) / mu
        beta_eff = max(0.05, min(beta_eff, 0.35))
        
        B_S = 2.12 / (3.21 - 0.68 * np.log(beta_eff * 100)) if beta_eff > 0.05 else 1.0
        B_1 = 1.62 / (2.31 - 0.41 * np.log(beta_eff * 100)) if beta_eff > 0.05 else 1.0
        B_S, B_1 = max(1.0, B_S), max(1.0, B_1)

        Sa_demand_damped = Sa_demand_5pct / B_S
        Sd_demand_damped = Sd_demand_5pct / B_1

        new_pi_trial = _find_intersection(capacity_adrs['Sd'], capacity_adrs['Sa'], Sd_demand_damped, Sa_demand_damped)
        if new_pi_trial is None:
            print(f"Warning: No intersection found in iter {i+1}. Using last point."); new_pi_trial = capacity_adrs.iloc[-1]

        iteration_history.append({
            'iter': i + 1, 'mu': mu, 'beta_eff': beta_eff, 
            'trial_Sd': pi_trial['Sd'], 'new_Sd': new_pi_trial['Sd'],
            'demand_curve_damped': (Sd_demand_damped.tolist(), Sa_demand_damped.tolist())
        })
        
        if pi_trial['Sd'] > 1e-9 and abs(new_pi_trial['Sd'] - pi_trial['Sd']) / pi_trial['Sd'] < tolerance:
            performance_point = new_pi_trial; print(f"CSM converged after {i + 1} iterations."); break
        
        pi_trial = new_pi_trial
    else:
        performance_point = pi_trial; print("CSM did not converge. Using last trial point.")

    final_mu = performance_point['Sd'] / Sd_y if Sd_y > 1e-9 else 1.0
    final_beta_eff = max(0.05, min(0.05 + 0.565 / np.pi * (final_mu - 1) / final_mu, 0.35))
    k_final = performance_point['Sa'] / performance_point['Sd'] if performance_point['Sd'] > 1e-9 else 0
    T_final = 2 * np.pi / np.sqrt(k_final * 9.81) if k_final > 1e-9 else float('inf')

    return {'performance_point': {'Sd': performance_point['Sd'], 'Sa': performance_point['Sa'], 'T_eff': T_final, 'beta_eff': final_beta_eff},
            'capacity_adrs': capacity_adrs.to_dict('list'),
            'demand_spectrum_5pct': {'Sd': Sd_demand_5pct.tolist(), 'Sa': Sa_demand_5pct.tolist()},
            'iteration_history': iteration_history}