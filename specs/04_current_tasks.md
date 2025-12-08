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

### Task 2: 모델 재학습 및 대규모 데이터셋 구축 (진행 중)

- [ ] **대규모 데이터셋 생성**
  - 목표: 100개 건물 (400개 해석 샘플)
  - 상태: 사용자 로컬 환경에서 `scripts/generate_dataset.py` 실행 필요
- [ ] **모델 재학습**
  - 새로운 12차원 Edge Feature와 6차원 Node Feature를 반영하여 학습
  - 목표 성능: Test Loss < 0.2 (MSE 기준)

### Task 3: 성능 평가 및 검증

- [ ] **정량적 평가 (`evaluate_metrics.py`)**
  - R² Score, MAE 측정
  - 초기 강성, 최대 강도, 에너지 소산 능력 상관관계 분석
- [ ] **정성적 평가 (`predict.py`)**
  - 실제 푸쉬오버 곡선 vs 예측 곡선 비교 시각화
