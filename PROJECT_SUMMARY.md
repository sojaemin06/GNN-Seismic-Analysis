# GNN 기반 푸쉬오버 곡선 예측 프로젝트

## 1. 프로젝트 목표
- Graph Neural Network(GNN)를 활용하여 저층 RC 구조물의 비선형 정적해석 결과인 **푸쉬오버 곡선(Pushover Curve)을 신속하게 예측**하는 딥러닝 모델을 개발한다.
- 최종 연구 결과는 **SCI급 해외 학술지 게재**를 목표로 한다.

## 2. 기술 스택 및 모델링

### 2.1. OpenSees 해석 모델링
- **재료 모델 (Material Models):**
  - 비구속 콘크리트: `Concrete04` (Karsan-Jirsa 모델)
  - 구속 콘크리트: `Concrete04` (Karsan-Jirsa 모델, 강도/변형률 1.3배 증가)
  - 철근: `Steel02` (Giuffré-Menegotto-Pinto 모델)
- **요소 및 단면 (Element & Section):**
  - 부재 모델링: `nonlinearBeamColumn` (분포 소성 모델)
  - 단면 정의: `Fiber` Section
  - 수치 적분: `Lobatto` (5-point)
- **모델링 기타:**
  - 층별 `rigidDiaphragm` 적용하여 강체 바닥 가정

### 2.2. GNN 모델 (`gnn_code/models.py`)
- **모델 타입:** `GATv2` (`GNN_Pushover`)
- **입력 특성 (Input Features):**
  - **노드 특성 (Node Features) - 3개:** 3D 좌표 (x, y, z)
  - **엣지 특성 (Edge Features) - 10개:** `[is_column, is_beam, width, depth, fc, Fy, cover, As, num_bars_1, num_bars_2]`
- **출력 특성 (Output Features):**
  - X, Z 두 방향의 표준화된 푸쉬오버 곡선 (각 100개 지점, 총 200개)
- **아키텍처:** 3개의 GATv2 레이어 + Global Mean Pooling + 2개의 완전 연결(FC) 레이어

### 2.3. 데이터셋 생성 제약조건
- 생성된 모델이 아래의 두 조건을 만족하는 경우에만 데이터셋에 포함됩니다.
- **질량 참여율 (Mass Participation Ratio):** X, Z 두 방향 모두 누적 질량 참여율 90% 이상 (`>= 0.9`).
- **비선형 정적해석 적용성 (NSP Applicability):** 다중모드/1차모드 층전단력 비 130% 이하 (`<= 1.3`).

### 2.4. 데이터셋 생성 규칙 (해석 안정성 확보)
- **기둥 Tapering:** 건물의 층수에 따라 기둥 단면을 그룹(예: 저층부, 중층부, 고층부)으로 나누어 샘플링하여 실제 구조물의 거동을 모사하고 해석 안정성을 높입니다.
- **강한 기둥-약한 보 (Strong Column-Weak Beam) 원칙:** 접합부에서 기둥의 휨강도가 보의 휨강도보다 충분히 크도록 부재 단면 및 배근을 샘플링하여 바람직한 파괴 메커니즘을 유도하고 해석 수렴성을 확보합니다.
- **재시도 메커니즘:** 특정 구조 레이아웃에 대한 해석이 실패할 경우, 다른 랜덤 재료/단면 속성을 샘플링하여 해석 성공 시까지 자동으로 재시도합니다 (최대 재시도 횟수 설정).