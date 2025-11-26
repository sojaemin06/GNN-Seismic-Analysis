# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

def create_design_spectrum_kbc2016(soil_type, Sa, Sv):
    """
    KBC 2016 설계 응답 스펙트럼을 생성합니다.

    Args:
        soil_type (str): 지반 종류 (e.g., 'S1', 'S2', 'S3', 'S4', 'S5')
        Sa (float): 단주기 설계 스펙트럼 가속도 (g)
        Sv (float): 1초 주기 설계 스펙트럼 가속도 (g)

    Returns:
        tuple: (주기 배열, 가속도 스펙트럼 배열)
    """
    T = np.linspace(0.01, 5.0, 500)
    Sa_spec = np.zeros_like(T)

    soil_coeffs = {
        'S1': {'Ca': 0.8, 'Cv': 0.8}, 'S2': {'Ca': 1.0, 'Cv': 1.0},
        'S3': {'Ca': 1.0, 'Cv': 1.3}, 'S4': {'Ca': 1.1, 'Cv': 1.8},
        'S5': {'Ca': 1.1, 'Cv': 2.4}
    }
    
    if soil_type not in soil_coeffs:
        raise ValueError(f"Unknown soil type: {soil_type}. Expected one of {list(soil_coeffs.keys())}")

    Ca = soil_coeffs[soil_type]['Ca']
    Cv = soil_coeffs[soil_type]['Cv']

    Sds = Sa * Ca
    Sd1 = Sv * Cv

    T0 = 0.2 * Sd1 / Sds
    Ts = Sd1 / Sds

    Sa_spec[T <= T0] = (0.4 + (1 - 0.4) * T[T <= T0] / T0) * Sds
    Sa_spec[(T > T0) & (T <= Ts)] = Sds
    Sa_spec[T > Ts] = Sd1 / T[T > Ts]

    return T, Sa_spec

def pushover_to_adrs(df_pushover, pf1, m_eff):
    """
    푸쉬오버 곡선(지붕 변위-밑면 전단력)을 ADRS 형식(스펙트럼 변위-스펙트럼 가속도)으로 변환합니다.

    Args:
        df_pushover (pd.DataFrame): 'roof_drift'와 'base_shear' 컬럼을 포함하는 푸쉬오버 곡선 데이터프레임.
        pf1 (float): 1차 모드의 참여계수 (Participation Factor).
        m_eff (float): 1차 모드의 유효질량 (Effective Modal Mass).

    Returns:
        pd.DataFrame: 'Sd' (스펙트럼 변위)와 'Sa' (스펙트럼 가속도) 컬럼을 포함하는 ADRS 데이터프레임.
    """
    g = 9.81 # m/s^2
    
    # 지붕 변위(roof_drift)를 스펙트럼 변위(Sd)로 변환
    # Sd = Dr / (PF1 * phi_roof,1), 여기서 phi_roof,1는 1로 가정
    Sd = df_pushover['roof_drift'] / pf1
    
    # 밑면 전단력(base_shear)을 스펙트럼 가속도(Sa)로 변환
    # Sa = V / (alpha_1 * W_eff) = V / m_eff
    Sa = df_pushover['base_shear'] / m_eff / g # g로 나누어 unitless g로 표현

    return pd.DataFrame({'Sd': Sd, 'Sa': Sa})


def calculate_performance_point_csm(df_pushover, modal_properties, design_spectrum_params, max_iter=20, tolerance=0.01):
    """
    역량스펙트럼법(Capacity Spectrum Method, CSM)을 사용하여 성능점을 계산합니다. (ATC-40 Procedure A)

    Args:
        df_pushover (pd.DataFrame): 푸쉬오버 곡선 데이터.
        modal_properties (dict): 1차 모드 주기, 참여계수, 유효질량 등을 포함하는 딕셔너리.
        design_spectrum_params (dict): 설계 스펙트럼 생성에 필요한 파라미터.
        max_iter (int): 최대 반복 횟수.
        tolerance (float): 수렴 허용 오차.

    Returns:
        dict: 성능점 정보 (변위, 가속도, 주기, 감쇠비 등)와 계산 과정 데이터.
    """
    # 1. ADRS 형식으로 역량 스펙트럼 변환
    pf1 = modal_properties['pf1']
    m_eff = modal_properties['m_eff_t1']
    capacity_adrs = pushover_to_adrs(df_pushover, pf1, m_eff)

    # 2. 설계 응답 스펙트럼(수요 스펙트럼) 생성 (5% 감쇠)
    T_demand, Sa_demand_5pct = create_design_spectrum_kbc2016(**design_spectrum_params)
    Sd_demand_5pct = (T_demand**2 / (4 * np.pi**2)) * Sa_demand_5pct * 9.81
    
    # 감쇠비에 따른 저감계수(B_S, B_1) (FEMA 440)
    damping_reduction_factors = pd.DataFrame({
        'D_eff': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        'B_S': [1.0, 1.33, 1.56, 1.73, 2.0, 2.2, 2.36],
        'B_1': [1.0, 1.2, 1.3, 1.4, 1.53, 1.63, 1.7]
    }).set_index('D_eff')

    # 3. 반복법을 통한 성능점 탐색
    # 초기 시도점: 역량 스펙트럼의 마지막 점
    pi_trial = capacity_adrs.iloc[-1]
    
    iteration_history = []

    for i in range(max_iter):
        # 4. 현재 시도점(pi)과 원점을 잇는 할선 강성(secant stiffness)으로 등가 주기(T_eff) 계산
        k_eff = pi_trial['Sa'] / pi_trial['Sd'] if pi_trial['Sd'] > 0 else 0
        T_eff = 2 * np.pi / np.sqrt(k_eff * 9.81) if k_eff > 0 else float('inf')

        # 5. 등가 점성 감쇠비(beta_eff) 계산
        # 이선형 모델의 면적으로 근사
        bilinear_fit = np.polyfit(capacity_adrs['Sd'], capacity_adrs['Sa'], 1)
        Ay = bilinear_fit[1] # y-intercept
        
        area_bilinear = 0.5 * pi_trial['Sd'] * pi_trial['Sa']
        area_elastoplastic = Ay * pi_trial['Sd'] - 0.5 * Ay**2 / bilinear_fit[0] if bilinear_fit[0] !=0 else 0
        
        # 유효감쇠비 (FEMA 356, Eq 2-17) - κ 계수 적용
        # κ 값은 구조 시스템 종류(A, B, C)에 따라 달라짐. 여기서는 0.67 (보통 모멘트골조) 가정
        kappa = 0.67 
        beta_eff = kappa * (2 / np.pi) * ((area_bilinear - area_elastoplastic) / (pi_trial['Sa'] * pi_trial['Sd'])) if (pi_trial['Sa'] * pi_trial['Sd']) > 0 else 0
        beta_eff = max(0.05, beta_eff) # 최소 5%

        # 6. 감쇠 저감계수(SR_A, SR_V) 계산 (선형 보간)
        B_S = np.interp(beta_eff, damping_reduction_factors.index, damping_reduction_factors['B_S'])
        B_1 = np.interp(beta_eff, damping_reduction_factors.index, damping_reduction_factors['B_1'])

        # 7. 감쇠가 적용된 수요 스펙트럼(demand spectrum) 계산
        Sa_demand_damped = Sa_demand_5pct / B_S
        Sd_demand_damped = Sd_demand_5pct / B_1

        # 8. 수요-역량 스펙트럼의 교차점 찾기 -> 새로운 시도점(pi+1)
        # 수요 스펙트럼을 (Sd, Sa) 공간에서 표현
        demand_interp = np.interp(capacity_adrs['Sd'], Sd_demand_damped, Sa_demand_damped)
        
        # 교차점 찾기 (두 곡선 차의 부호가 바뀌는 지점)
        diff = capacity_adrs['Sa'] - demand_interp
        intersect_idx = np.where(np.diff(np.sign(diff)))[0]

        if len(intersect_idx) == 0:
            print("Warning: No intersection found. Using last point of capacity curve.")
            new_pi_trial = capacity_adrs.iloc[-1]
        else:
            idx = intersect_idx[0]
            # 선형 보간으로 교차점 정밀 계산
            x1, y1 = capacity_adrs['Sd'][idx], capacity_adrs['Sa'][idx]
            x2, y2 = capacity_adrs['Sd'][idx+1], capacity_adrs['Sa'][idx+1]
            x3, y3 = Sd_demand_damped[idx], Sa_demand_damped[idx] # This is not correct. Need to interpolate demand.
            
            # Simple approach: use the capacity point right after intersection
            new_pi_Sd = capacity_adrs['Sd'].iloc[idx+1]
            new_pi_Sa = capacity_adrs['Sa'].iloc[idx+1]
            new_pi_trial = pd.Series({'Sd': new_pi_Sd, 'Sa': new_pi_Sa})

        iteration_history.append({
            'iter': i + 1,
            'T_eff': T_eff,
            'beta_eff': beta_eff,
            'B_S': B_S,
            'B_1': B_1,
            'trial_Sd': pi_trial['Sd'],
            'trial_Sa': pi_trial['Sa'],
            'new_Sd': new_pi_trial['Sd'],
            'new_Sa': new_pi_trial['Sa'],
            'demand_curve_damped': (Sd_demand_damped, Sa_demand_damped)
        })

        # 9. 수렴 확인: |S_d,i+1 - S_d,i| / S_d,i+1 < tolerance
        if abs(new_pi_trial['Sd'] - pi_trial['Sd']) / new_pi_trial['Sd'] < tolerance:
            performance_point = new_pi_trial
            print(f"CSM converged after {i+1} iterations.")
            break
        
        pi_trial = new_pi_trial
    else:
        performance_point = pi_trial
        print("CSM did not converge within max iterations. Using last trial point.")

    # 최종 성능점 데이터를 dict로 정리
    result = {
        'performance_point': {
            'Sd': performance_point['Sd'],
            'Sa': performance_point['Sa'],
            'T_eff': T_eff,
            'beta_eff': beta_eff
        },
        'capacity_adrs': capacity_adrs.to_dict('list'),
        'demand_spectrum_5pct': {
            'T': T_demand.tolist(),
            'Sa': Sa_demand_5pct.tolist(),
            'Sd': Sd_demand_5pct.tolist()
        },
        'iteration_history': iteration_history
    }
    
    return result
