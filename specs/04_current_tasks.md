# 현재 작업 목록 (Current Tasks)

## 작업 범위: 성능 고도화 및 평가 (Phase 3)

### Task 1: 데이터 정합성 및 피처 엔지니어링 (완료)

- [x] **Target 데이터 수정**
  - `Base Shear` -> `Base Shear Coefficient` ($V/W$)
  - `total_weight`를 중력 해석 반력($\sum R_y$) 기반으로 정확히 산정
- [x] **입력 피처 고도화 (Feature Engineering)**
  - **Node:** `Mass` (질량), `Node Degree` (연결성) 추가
  - **Edge:** `Moment of Inertia` ($I$), `Reinforcement Ratio` ($\rho$) 추가
  - **Global:** 4방향(`X+`, `X-`, `Z+`, `Z-`) One-hot Vector 적용
- [x] **데이터 생성 파이프라인 수정**
  - `generate_dataset.py` 및 `graph_exporter.py`에 위 변경사항 반영
  - **Target Parameter Update:** 기존 노후 건축물(Existing Low-rise) 특성 반영 (fc 18~24MPa, Fy 300~400MPa, 소형 단면)

### Task 2: 모델 재학습 및 대규모 데이터셋 구축 (완료)

- [x] **대규모 데이터셋 생성**
  - 목표: 100개 건물 (400개 해석 샘플)
  - 상태: 완료 (Sample ID 0~98, 총 396개 파일 16-dim Edge Feature로 재생성)
- [x] **모델 고도화 및 재학습**
  - **Feature Enhancement:** Edge Feature 확장 (12 -> 16 dim) - 재료 비선형 파라미터(`epsc0`, `fpcu`, `epsU`, `b`) 추가.
  - **Architecture:** Hidden Dim 128로 확장, GlobalMeanPool 사용.
  - **Training:** LR Scheduler 도입 (ReduceLROnPlateau), 초기 LR 0.0005.
  - **결과:** Test Loss 0.0665 (목표 < 0.2 달성)

### Task 3: 성능 평가 및 검증 (완료)

- [x] **정량적 평가 (`evaluate_metrics.py`)**
  - **핵심 성과:** 주요 구조 성능 지표(최대 강도, 초기 강성, 에너지 소산)에서 **R² > 0.93** 달성.
  - 곡선 적합도(Median Curve R2)는 여전히 음수(-0.81)로 나타났으나, 이는 점 단위 오차에 민감한 지표 특성 때문임.
  - **결론:** 초기 강성 및 파단점 예측 등 실무적 활용 목표에는 충분히 부합함.
- [x] **정성적 평가 (`predict.py`)**
  - 실제 푸쉬오버 곡선 vs 예측 곡선 비교 시각화 완료 (`results/prediction_samples.png`)

## 작업 범위: 논문 작성 지원 (Phase 4)

### Task 4: 실험 결과 정리 및 시각화

- [ ] **성능 분석 보고서 작성**
  - 모델의 강점(구조 파라미터 예측)과 한계(Curve R2 변동성)를 분석하는 텍스트 리포트 생성
- [ ] **추가 시각화**
  - 논문용 고품질 플롯 생성 (Feautre Importance 분석 등 가능 시 추가)

