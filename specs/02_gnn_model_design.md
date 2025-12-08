# GNN 모델 설계 명세서 (Model Design Spec)

## 1. 목표
구조물의 그래프 정보와 해석 방향을 입력받아, 비선형 **푸쉬오버 곡선(밑면전단력 vs 지붕층변위)**을 예측하는 GNN 모델 구현.

## 2. 입력 명세 (Input Specifications)

### 2.1. 노드 특성 (Node Features)
- **차원:** 4
- **구성:** `[x, y, z, is_base]`
    - `x, y, z`: 절점 좌표 (m 단위)
    - `is_base`: 지지점 여부 (y ≈ 0 이면 1.0, 아니면 0.0)

### 2.2. 엣지 특성 (Edge Features)
- **차원:** 10
- **구성:** `[is_col, is_beam, width, depth, fc, Fy, cover, rebar_Area, num_bars_1, num_bars_2]`
    - `is_col`: 기둥이면 1.0
    - `is_beam`: 보이면 1.0
    - `width`, `depth`: 단면 치수 (m)
    - `fc`: 콘크리트 압축강도 (MPa, 스케일링됨)
    - `Fy`: 철근 항복강도 (MPa, 스케일링됨)
    - `cover`: 피복 두께 (m)
    - `rebar_Area`: 개별 철근 단면적
    - `num_bars_1`: (기둥) X방향 철근 수 / (보) 상부 철근 수
    - `num_bars_2`: (기둥) Z방향 철근 수 / (보) 하부 철근 수

### 2.3. 전역 특성 (Global Features)
- **차원:** 2
- **구성:** `[is_X_dir, is_Z_dir]`
    - `[1.0, 0.0]`: X방향 해석
    - `[0.0, 1.0]`: Z방향 해석
    - **용도:** 모델의 예측 조건을 결정하는 핵심 정보.

## 3. 출력 명세 (Output Specifications)

### 3.1. 타겟 (Target)
- **유형:** 푸쉬오버 곡선 (밑면 전단력)
- **차원:** 100 (고정된 보간 포인트)
- **값:** 지붕층 변위가 0부터 최대 변위까지 선형 증가할 때의 대응되는 밑면 전단력 값 (표준화됨).

## 4. 모델 아키텍처 (제안)

### 4.1. 백본 (Backbone)
- **유형:** 메시지 패싱 신경망 (GAT, GCN 등)
- **깊이:** 3~5 레이어 (전체 구조 거동 포착)
- **은닉층 차원:** 64 또는 128

### 4.2. 조건부 메커니즘 (Conditioning)
- 전역 특성(해석 방향)을 반영하는 방법:
- **전략:** 노드 임베딩을 Global Pooling한 후, 전역 특성 벡터와 결합(Concatenate)하여 최종 MLP(Decoder)에 입력.

### 4.3. Readout (Decoder)
- **구조:** `[Pool(Node_Embeddings), Global_Feature]` -> MLP -> Output
- **MLP 구성:**
    - 입력: 풀링된 그래프 특징 + 방향 벡터
    - 은닉층: 2~3개 (ReLU 활성화)
    - 출력: 100 (푸쉬오버 곡선 예측값)

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
