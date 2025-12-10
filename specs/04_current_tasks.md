# 현재 작업 목록 (Current Tasks)

## 작업 범위: 성능 고도화 및 평가 (Phase 3)

### Task 1: 데이터 정합성 및 피처 엔지니어링 (완료)
- [x] **Target 데이터 수정** 및 **입력 피처 고도화** (완료)
- [x] **데이터 품질 검증 로직 추가** (Flat Curve, Low Strength 필터링)

### Task 2: 대규모 고품질 데이터셋 재구축 (진행 중)
- [ ] **기존 데이터 정리:** `data/processed/*.pt` 및 로그 파일 삭제 (Clean Start)
- [ ] **데이터 생성:** 500개 유효 샘플 생성 (`scripts/generate_dataset.py`, 품질 검증 적용)
- [ ] **생성 시간 측정:** 로그 파싱을 통한 정확한 데이터 생성 비용($T_{gen}$) 산출

### Task 3: 모델 재학습 및 실험 검증 (Planned)
- [ ] **평가 로직 고도화:**
  - `run_csm_evaluation.py`: 붕괴 부재 수(Collapsed Count)를 최종 `PASS/FAIL` 판정에 반영.
  - `generate_report.py`: 안정성 지표 및 붕괴 부재 정보를 포함한 상세 보고서 포맷 개선.
- [ ] **확장성 실험 (Scalability Experiment):**
  - 데이터 수(N=100~500)에 따른 R2 및 총 소요 시간($T_{gen} + T_{train}$) 분석 (`scripts/run_data_scalability_experiment.py`)
- [ ] **CSM 성능점 정확도 검증:**
  - `scripts/verify_gnn_csm.py` 재실행하여 $S_d$ 오차율 개선 확인
  - 목표: $S_d$ 오차 < 20% 달성 (필요 시 Loss Function에 CSM 오차 반영 고려)

## 작업 범위: 논문 작성 지원 및 시각화 데이터 생성 (Phase 4)

### Task 4: 논문용 Figure 데이터 생성 및 추출

**목표:** 논문 게재를 위해 각 단계별(데이터셋, 학습, 평가) 결과를 고해상도 이미지 및 원본 데이터(CSV)로 추출.

- [ ] **데이터셋 통계 시각화 (Dataset Statistics)**
  - [ ] 입력 변수 분포도 (Histograms): $f_c$, $f_y$, 기둥/보 단면 치수, 층수 등
  - [ ] 상관관계 분석 (Correlation Matrix): 입력 구조 변수 vs 구조 성능 지표($V_{base}$, $K_{initial}$)
  - [ ] 대표 구조 모델링 형상 3D Plot (OpenSees 시각화)

- [ ] **학습 과정 시각화 (Training Process)**
  - [ ] Loss Curve 데이터 추출 (Train vs Validation MSE)
  - [ ] Learning Rate 변화 추이
  - [ ] Epoch별 주요 Metric(R2) 변화

- [ ] **성능 평가 시각화 (Evaluation Results)**
  - [ ] **Parity Plots (Scatter):** 실제값 vs 예측값 ($x=y$ line 비교)
    - 초기 강성 ($K_{initial}$)
    - 최대 강도 ($V_{max}$)
    - 항복 변위 ($D_{yield}$)
    - 에너지 소산 능력 ($Energy$)
  - [ ] **Representative Pushover Curves:**
    - Best Case (상위 10%)
    - Average Case (중위값)
    - Worst Case (하위 10% - 오차 원인 분석용)
  - [ ] 오차 분포도 (Error Histograms)

- [ ] **해석 가능성 분석 (Optional)**
  - [ ] Feature Importance (GNN Explainer 또는 가중치 분석)
  - [ ] Latent Space t-SNE 시각화 (건물 유형별 군집화 확인)

### Task 5: 논문 초안 작성 지원

- [ ] **성능 분석 보고서 작성**
  - 정량적 지표 ($R^2$, MAE, MSE) 요약 테이블 생성
  - 기존 머신러닝 방법론 대비 GNN의 강점/한계 기술

