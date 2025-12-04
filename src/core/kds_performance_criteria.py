# -*- coding: utf-8 -*-

def get_performance_objectives(importance_class):
    """
    내진등급(importance_class)에 따라 성능목표 세트와 계산 방식을 반환합니다.
    
    [계산 방식 (method)]
    1. 'scaling': KDS 41 17 00에 따라 기본설계지진 스펙트럼에 중요도계수(I_E)를 곱하여 산정.
       - 대상: 2400년, 1400년, 1000년 재현주기
       - factor: 중요도계수 (I_E)
       
    2. 'direct': 내진성능 평가요령에 따라 해당 재현주기의 유효지반가속도(S)를 직접 산정하여 스펙트럼 작성.
       - 대상: 100년, 50년 재현주기
       - factor: 위험도계수 (I) (S = Z * I)
    
    [허용 층간변위비]
    '내진성능 평가요령(2021)' 표 4.6.1 RC 모멘트골조 기준
    """
    
    # --- 기본 정의 ---
    # 1. 2400년 재현주기 (특등급 인명보호 / 1등급 붕괴방지)
    # 기준: 기본설계지진 * 1.5 (특) 또는 * 1.5 (붕괴방지시 I_E=1.5 등가?) 
    # 주의: 2400년은 MCE이므로 기본설계지진(2/3 MCE) * 1.5가 맞음.
    obj_cp_2400 = {
        "name": "Collapse Prevention",
        "method": "scaling",
        "factor": 1.5, # 중요도계수 (특등급 기준 1.5) -> 2/3 * 1.5 = 1.0 MCE
        "repetition_period": "2400년",
        "target_drift_ratio_limit": 0.030, 
        "description": "건축물 성능수준: 붕괴방지 (구조요소: 붕괴방지) / 지진크기: 2400년 (MCE)"
    }
    
    # 특등급용 2400년 (인명보호)
    obj_ls_2400 = {
        "name": "Life Safety",
        "method": "scaling",
        "factor": 1.5, # 중요도계수 (특등급)
        "repetition_period": "2400년",
        "target_drift_ratio_limit": 0.020,
        "description": "건축물 성능수준: 인명보호 (구조요소: 인명안전) / 지진크기: 2400년"
    }

    # 2. 1400년 재현주기 (1등급 인명보호)
    obj_ls_1400 = {
        "name": "Life Safety",
        "method": "scaling",
        "factor": 1.2, # 중요도계수 (1등급)
        "repetition_period": "1400년",
        "target_drift_ratio_limit": 0.020,
        "description": "건축물 성능수준: 인명보호 (구조요소: 인명안전) / 지진크기: 1400년"
    }
    
    # 3. 1000년 재현주기 (2등급 인명보호 / 특등급 기능수행)
    obj_ls_1000 = {
        "name": "Life Safety",
        "method": "scaling",
        "factor": 1.0, # 중요도계수 (2등급)
        "repetition_period": "1000년",
        "target_drift_ratio_limit": 0.020,
        "description": "건축물 성능수준: 인명보호 (구조요소: 인명안전) / 지진크기: 1000년"
    }
    obj_op_1000 = {
        "name": "Operational",
        "method": "scaling",
        "factor": 1.0, # 중요도계수 (2등급 수준)
        "repetition_period": "1000년",
        "target_drift_ratio_limit": 0.007,
        "description": "건축물 성능수준: 기능수행 (구조요소: 거주가능) / 지진크기: 1000년"
    }

    # 4. 100년 재현주기 (직접 산정)
    obj_op_100 = {
        "name": "Operational",
        "method": "direct",
        "factor": 0.57, # 위험도계수 I
        "repetition_period": "100년",
        "target_drift_ratio_limit": 0.007,
        "description": "건축물 성능수준: 기능수행 (구조요소: 거주가능) / 재현주기 100년"
    }
    
    # 5. 50년 재현주기 (직접 산정)
    obj_op_50 = {
        "name": "Operational",
        "method": "direct",
        "factor": 0.4, # 위험도계수 I
        "repetition_period": "50년",
        "target_drift_ratio_limit": 0.007,
        "description": "건축물 성능수준: 기능수행 (구조요소: 거주가능) / 재현주기 50년"
    }

    # --- 등급별 목표 매핑 ---
    if importance_class == "Teuk": # 특등급
        return [obj_ls_2400, obj_op_1000]
    
    elif importance_class == "I": # 1등급
        return [obj_cp_2400, obj_ls_1400, obj_op_100]
        
    elif importance_class == "II": # 2등급
        return [obj_cp_2400, obj_ls_1000, obj_op_50]
    
    else:
        raise ValueError(f"Invalid importance class: {importance_class}")