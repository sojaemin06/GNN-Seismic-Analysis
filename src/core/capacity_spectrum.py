# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from src.core.kds_2022_spectrum import generate_kds2022_demand_spectrum

def pushover_to_adrs(df_pushover, pf1, m_eff, phi_roof):
    """
    푸쉬오버 곡선을 ADRS 형식으로 변환하고, 곡선을 단조 증가하도록 보정합니다.
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

def calculate_performance_point_csm(df_pushover, modal_properties, design_params_list: list, max_iter=30, tolerance=0.01):
    """
    [KDS 2022 최종] 역량스펙트럼법(CSM)으로 여러 성능목표에 대한 성능점을 계산합니다.
    지원 방식:
    1. 'scaling': 기본설계지진(S_DBE) 스펙트럼을 생성 후 중요도계수(I_E)를 곱함.
    2. 'direct': 유효지반가속도(S)를 직접 사용하여 스펙트럼 생성.
    """
    pf1, m_eff, phi_roof = modal_properties['pf1'], modal_properties['m_eff_t1'], modal_properties['phi_roof']
    capacity_adrs = pushover_to_adrs(df_pushover, pf1, m_eff, phi_roof)
    if capacity_adrs.empty or len(capacity_adrs) < 2:
        print("Warning: Capacity curve is empty or too short for CSM analysis."); return None

    all_csm_results = []

    for design_params in design_params_list:
        objective_name = design_params.get('objective_name', 'Unknown Objective')
        method = design_params.get('method', 'direct') # default to direct if not specified

        # --- 수요 스펙트럼 생성 분기 ---
        if method == 'scaling':
            # 기본설계지진(S_DBE) 기준 스펙트럼 생성
            S_DBE = design_params['S_DBE']
            I_E = design_params['I_E']
            Sd_base, Sa_base, SDS, SD1 = generate_kds2022_demand_spectrum(S=S_DBE, site_class=design_params['site_class'])
            
            # 중요도계수로 스케일링
            Sd_demand_5pct = Sd_base * I_E
            Sa_demand_5pct = Sa_base * I_E
            
            # SDS, SD1도 스케일링하여 리포트용으로 저장
            SDS *= I_E
            SD1 *= I_E
            
        else: # 'direct'
            # S값으로 직접 생성
            S_target = design_params['S']
            Sd_demand_5pct, Sa_demand_5pct, SDS, SD1 = generate_kds2022_demand_spectrum(S=S_target, site_class=design_params['site_class'])

        # --- 이하 CSM 로직 동일 ---
        try:
            Sa_max = capacity_adrs['Sa'].max()
            yield_point = capacity_adrs[capacity_adrs['Sa'] >= Sa_max * 0.6].iloc[0]
            Sd_y, Sa_y = yield_point['Sd'], yield_point['Sa']
            if Sd_y < 1e-9: raise IndexError
        except (IndexError, KeyError, ValueError):
            Sd_y, Sa_y = capacity_adrs.iloc[-1]['Sd'], capacity_adrs.iloc[-1]['Sa']
            
        try:
            initial_part = capacity_adrs[capacity_adrs['Sa'] < capacity_adrs['Sa'].max() * 0.2]
            if len(initial_part) < 2: initial_part = capacity_adrs.head(2)
            K_e = initial_part['Sa'].iloc[-1] / initial_part['Sd'].iloc[-1]
            
            elastic_line_sa_at_demand_sd = K_e * Sd_demand_5pct
            pi_trial_initial = _find_intersection(Sd_demand_5pct, elastic_line_sa_at_demand_sd, Sd_demand_5pct, Sa_demand_5pct)

            if pi_trial_initial is None:
                pi_trial = capacity_adrs.iloc[-1]
            else:
                pi_trial = pi_trial_initial

        except (IndexError, ZeroDivisionError):
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
                new_pi_trial = capacity_adrs.iloc[-1]

            iteration_history.append({
                'iter': i + 1, 'mu': mu, 'beta_eff': beta_eff, 
                'trial_Sd': pi_trial['Sd'], 'trial_Sa': pi_trial['Sa'], 'new_Sd': new_pi_trial['Sd'],
                'demand_curve_damped': (Sd_demand_5pct.tolist(), Sa_demand_5pct.tolist())
            })
            
            if pi_trial['Sd'] > 1e-9 and abs(new_pi_trial['Sd'] - pi_trial['Sd']) / pi_trial['Sd'] < tolerance:
                performance_point = new_pi_trial; break
            
            pi_trial = new_pi_trial
        else:
            performance_point = pi_trial

        final_mu = performance_point['Sd'] / Sd_y if Sd_y > 1e-9 else 1.0
        final_beta_eff = max(0.05, min(0.05 + 0.565 / np.pi * (final_mu - 1) / final_mu, 0.35))
        k_final = performance_point['Sa'] / performance_point['Sd'] if performance_point['Sd'] > 1e-9 else 0
        T_final = 2 * np.pi / np.sqrt(k_final * 9.81) if k_final > 1e-9 else float('inf')

        all_csm_results.append({
            'objective_name': objective_name,
            'performance_point': {'Sd': performance_point['Sd'], 'Sa': performance_point['Sa'], 'T_eff': T_final, 'beta_eff': final_beta_eff},
            'capacity_adrs': capacity_adrs.to_dict('list'),
            'demand_spectrum_5pct': {'Sd': Sd_demand_5pct.tolist(), 'Sa': Sa_demand_5pct.tolist()},
            'iteration_history': iteration_history,
            'SDS': SDS,
            'SD1': SD1
        })
    return all_csm_results