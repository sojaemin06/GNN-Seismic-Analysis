# GNN 모델 설계 명세서 (Model Design Spec)

## 1. 목표
구조물의 그래프 정보와 해석 방향을 입력받아, 비선형 **푸쉬오버 곡선(밑면전단력 vs 지붕층변위)**을 예측하는 GNN 모델 구현.

## 2. 입력 명세 (Input Specifications)

### 2.1. 노드 특성 (Node Features)
- **차원:** 6
- **구성:** `[x, y, z, is_base, mass_norm, node_degree_norm]`
    - `x, y, z`: 절점 좌표 (m 단위). 건물의 상대적 비례(Aspect Ratio)를 학습.
    - `is_base`: 지지점 여부 (1.0 or 0.0). 경계 조건 정보.
    - `mass_norm`: 절점 질량 (ton 단위, 정규화됨). 지진 하중($F=ma$)의 크기를 결정하는 핵심 인자.
    - `node_degree_norm`: 노드에 연결된 부재 수 (정규화됨). 노드의 연결성 및 중요도를 GNN에 명시적으로 제공.

### 2.2. 엣지 특성 (Edge Features)
- **차원:** 12 (기존 10 + 2 파생 변수)
- **구성:**
  `[is_col, is_beam, width, depth, fc_norm, Fy_norm, cover, rebar_Area_norm, num_bars_1, num_bars_2, I_norm, rho]`
    - **기본 기하 정보:** `is_col`, `is_beam`, `width`, `depth`
    - **재료 정보 (정규화 필수):**
        - `fc_norm`: $f_c / 50$ (MPa 단위 기준)
        - `Fy_norm`: $F_y / 600$ (MPa 단위 기준)
    - **철근 상세:** `cover`, `num_bars_1`, `num_bars_2`
    - **공학적 파생 변수 (Engineered Features):**
        - `rebar_Area_norm`: $Area_{bar} \times 1000$ (단위 스케일 보정)
        - `I_norm`: 단면 2차 모멘트 ($bh^3/12$), $10^{-4} m^4$ 단위로 스케일링. 강성(Stiffness) 정보 제공.
        - `rho` ($\rho$): 철근비 ($A_s / A_g$). 부재의 연성 및 항복 강도 결정 인자.

### 2.3. 전역 특성 (Global Features)
- **차원:** 4
- **구성:** One-hot Vector for Direction
    - `[1, 0, 0, 0]`: X+ (Positive X)
    - `[0, 1, 0, 0]`: X- (Negative X)
    - `[0, 0, 1, 0]`: Z+ (Positive Z)
    - `[0, 0, 0, 1]`: Z- (Negative Z)

## 3. 출력 명세 (Output Specifications)

### 3.1. 타겟 (Target)
- **유형:** 정규화된 푸쉬오버 곡선 (Normalized Pushover Curve)
- **차원:** 100 (고정된 보간 포인트)
- **Y값 (예측 대상):** **밑면 전단력 계수 (Base Shear Coefficient)**
    - $V_{coeff} = \frac{V_{base}}{W_{total}}$
    - $V_{base}$: 밑면 전단력 (N)
    - $W_{total}$: 구조물 전체 유효 중량 (N)
    - **범위:** 대략 0.1 ~ 0.5 (건물 규모에 상관없이 안정적인 범위 유지)
- **X값 (입력 조건):** 지붕층 변위비 (Roof Drift Ratio)
    - $\delta_{drift} = \frac{\delta_{roof}}{H_{total}}$
    - $H_{total}$: 건물 전체 높이
    - **범위:** 0.0 ~ 0.03 (3% 변형률까지)

## 4. 모델 아키텍처 (제안)

### 4.1. 백본 (Backbone)
- **유형:** GATv2Conv (Graph Attention Network v2)
- **Edge Feature 활용:** 엣지의 강성($I$)과 강도($\rho$, $F_y$) 정보가 Attention 가중치 산정에 직접 반영되도록 설계.

### 4.2. 조건부 메커니즘 (Conditioning)
- **전략:** Global Pooling된 그래프 임베딩 벡터에 4차원 방향 벡터(One-hot)를 Concatenate.

### 4.3. Readout (Decoder)
- **구조:** `[Graph_Embedding + Direction_Vector]` -> MLP -> `Output(100)`


## 5. 학습 전략

### 5.1. 손실 함수 (Loss)
- **주요 손실:** MSE (Mean Squared Error)
- **보조 손실 (선택):** MAE

### 5.2. 최적화 (Optimizer)
- **종류:** Adam
- **학습률:** 0.001 (Scheduler 적용 권장)

## 6. 구현 계획
- **파일:** `src/gnn/models.py` (모델 클래스 정의)
- **파일:** `src/gnn/train.py` (학습 루프, 검증, 모델 저장)
