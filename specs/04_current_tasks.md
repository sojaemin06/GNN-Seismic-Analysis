# 현재 작업 목록 (Current Tasks)

## 작업 범위: 성능 고도화 및 평가 (Phase 3 - Completed)

### Task 1: 데이터 정합성 및 피처 엔지니어링 (완료)
- [x] **Target 데이터 수정** 및 **입력 피처 고도화** (완료)
- [x] **데이터 품질 검증 로직 추가** (Flat Curve, Low Strength 필터링)

### Task 2: 대규모 고품질 데이터셋 재구축 (완료)
- [x] **기존 데이터 정리:** Legacy(`data/processed`)와 CSM(`data/processed_csm`) 분리 운영.
- [x] **데이터 생성:** CSM 검증용 고품질 데이터셋(1차 모드 지배형) 구축 완료 (~600 Samples).
- [x] **Global Feature 확장:** 4차원(방향) -> 8차원(방향+모드) 업그레이드.

### Task 3: 모델 비교 및 선정 (완료)
- [x] **모델 벤치마크:** Legacy(v1) vs CSM(v2) 성능 비교 ($R^2$ 0.815 vs 0.831).
- [x] **최종 모델 선정:** **CSM 모델(v2)**을 최종 프로덕션 모델로 확정 (성능점 산정 기능 보유).
- [x] **CSM 성능점 정확도 검증:** $S_d$ 오차율 검토 및 정성적 그래프 비교 완료.

## 작업 범위: 논문 작성 지원 및 시각화 (Phase 4 - In Progress)

### Task 4: 논문용 Figure 데이터 생성 및 추출

**목표:** 논문 게재를 위해 선정된 **CSM 모델**의 결과를 고해상도 이미지 및 원본 데이터(CSV)로 추출.

- [x] **모델 비교 그래프:** `results/paper_figures/model_comparison_600_samples.png` 생성 완료.
- [ ] **데이터셋 통계 시각화 (Dataset Statistics)**
  - [ ] 입력 변수 분포도 (Histograms): $f_c$, $f_y$, $T_1$, $PF_1$ 등
  - [ ] 상관관계 분석 (Correlation Matrix): 모드 변수 vs 구조 성능 지표($V_{base}$)
- [ ] **성능 평가 시각화 (Evaluation Results)**
  - [ ] **Parity Plots (Scatter):** 실제값 vs 예측값 ($x=y$ line 비교)
    - $S_d$ (Spectral Displacement)
    - $V_{base}$ (Base Shear)
  - [ ] **Representative Pushover Curves:** Best/Average/Worst Case 추출.

### Task 5: 프로젝트 마무리 및 패키징
- [ ] **CLI 통합:** `verify_gnn_csm.py` 등을 사용자가 쉽게 쓸 수 있는 명령어로 정리.
- [ ] **최종 보고서:** 전체 실험 결과를 요약한 HTML/PDF 리포트 생성.


