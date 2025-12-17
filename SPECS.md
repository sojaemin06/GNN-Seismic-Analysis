# 프로젝트 명세서: 기존 RC 모멘트 골조의 내진성능평가를 위한 GNN 기반 대리 모델

## 1. 프로젝트 개요 (Project Overview)
*   **연구 목표:** 비선형 정적 해석(Pushover Analysis)을 대체할 수 있는 고속, 고정밀의 Graph Neural Network (GNN) 기반 대리 모델(Surrogate Model) 개발.
*   **타겟:** 비내진 상세를 가진 국내 기존 저층형 RC 건물 (3~5층).
*   **최종 목표:** SCI급 저널 게재 (*Engineering Structures*, *Computer-Aided Civil and Infrastructure Engineering* 등).
*   **현재 상태:** 모든 핵심 실험 완료 및 결과 분석 단계.

## 2. 데이터 생성 방법론 (Data Generation Methodology)
본 연구는 OpenSeesPy를 활용한 Monte Carlo Simulation(MCS) 기반의 합성 데이터셋을 사용함.

### 2.1 해석 엔진 및 절차 (`scripts/run_single_analysis.py`)
1.  **모델링:** `dataset_config.json`에 정의된 확률 변수에 따라 무작위 RC 골조 생성.
2.  **중력 해석:** 고정하중 및 활하중 재하.
3.  **고유치 해석:** 고유주기 및 모드 형상 도출.
4.  **유효성 검증:** 1차 모드 질량 참여율 및 130% 질량 참여 규칙(MPR)을 만족하는지 검증.
5.  **Pushover 해석:** 변위 제어(Displacement Control) 방식으로 목표 층간변위(Drift) 4%까지 가력.
6.  **결과 처리:** Base Shear vs. Roof Drift 곡선을 100개 포인트로 정규화하여 추출.

### 2.2 변수 범위 (`dataset_config.json`)
*   **기하학적 형상 (Geometry):** 층수(3~5), 스팬 수(2~4), 스팬 길이(3.5~6.0m).
*   **부재 단면 (Sections):** 기둥/보 치수 및 철근비(비내진 상세 반영).
*   **재료 물성 (Materials):** $f_{ck}$ (18~24 MPa), $f_y$ (300, 400 MPa).

## 3. GNN 모델 아키텍처 (`src/gnn1/models.py`)
구조 시스템의 위상(Topology)과 재료적 비선형성을 학습하기 위해 **PushoverGNN (GATv2)** 모델을 제안함.

*   **Proposed:** `PushoverGNN` (GATv2 - Graph Attention Network v2)
    *   **입력 (Input Graph):** Node(좌표, 질량, 구속 조건), Edge(부재 타입, 단면, 철근비, 재료 강도), Global(해석 방향, 고유치 결과).
    *   **Backbone:** `GATv2Conv` (4 Layers, 128 Hidden Dim, 4 Heads).
    *   **Pooling:** Hybrid Global Pooling (Mean + Add).
    *   **Decoder:** MLP (Predicts 100-point Pushover Curve).
*   **Baseline 1:** `BaselineGCN` (Graph Convolutional Network)
    *   `GCNConv` 사용. PushoverGNN과 유사한 Layer 및 Hidden Dim.
*   **Baseline 2:** `SimpleMLP` (Multi-Layer Perceptron)
    *   노드 평균 특성 + Global 특성 입력.

## 4. 학습 전략 (`src/gnn1/train.py`)
*   **Loss:** MSE Loss.
*   **Optimizer:** Adam + ReduceLROnPlateau.
*   **Metrics:** $ R^2 $ Score, Test Loss, **Inference Time (ms/sample)**.

## 5. 현재 진행 상황 (Status)
*   [x] **Pushover 해석 파이프라인 구축:** OpenSeesPy 연동 및 자동화 완료.
*   [x] **데이터셋 생성기 구현:** 무작위 설계 생성 및 검증 로직 적용 완료.
*   [x] **GAT 모델 고도화:** `PushoverGNN`의 Hidden Dim, Layer, Head 확장 완료.
*   [x] **비교 모델 구현:** `BaselineGCN` 및 `SimpleMLP` 구현 완료.
*   [x] **비교 실험 완료:** GAT의 우수성($R^2 \approx 0.92$) 입증.
*   [x] **Scalability 실험 완료:** 최적 데이터 수 도출 ($N_{opt} = 500$).
*   [x] **OOD 테스트 완료:** 6층 건물에 대한 일반화 성능 검증 (GAT > MLP).

## 6. SCI 논문 완성을 위한 핵심 과업 및 산출물 (Key Tasks & Deliverables)

### 6.1 비교 모델 연구 (Comparative Study: GAT vs. GCN vs. MLP)
*   **목적:** GNN의 위상 정보 활용 능력과 GAT의 Attention 메커니즘 우수성 입증.
*   **결과 (200 Epochs):**
    *   **GAT:** $R^2=0.9239$, Inference Time=3.78 ms
    *   **GCN:** $R^2=0.8099$, Inference Time=0.40 ms
    *   **MLP:** $R^2=0.8239$, Inference Time=0.08 ms
*   **결론:** GAT가 가장 우수한 성능을 보이며, 구조적 위상 정보와 Attention이 비선형 거동 예측에 필수적임.
*   **논문용 산출물:**
    *   `results/experiments/comparative_study_results.csv`
    *   `results/experiments/comparative_study_bars.png`, `comparative_scatter_plots.png`, `comparative_curve_samples.png`

### 6.2 최적 훈련 데이터 수 도출 실험 (Data Scalability Experiment)
*   **목적:** 데이터 효율성 입증 및 최적 데이터 수($ N_{opt} $) 선정.
*   **결과:** 훈련 데이터가 **500개**에 도달했을 때 $R^2 \approx 0.86$으로 성능이 수렴하기 시작함. 1,000개까지 늘려도 유의미한 성능 향상은 미미함.
*   **결론:** 본 연구에서 제안하는 GAT 모델 학습을 위한 **최적 데이터 수는 500개 ($N_{opt} = 500$)**로 선정. 이는 데이터 생성 비용 대비 효율성을 극대화한 결과임.
*   **논문용 산출물:**
    *   `results/experiments/data_scalability_experiment_results.csv`
    *   `results/experiments/data_scalability_plots.png` (Learning Curve)

### 6.3 일반화 성능 검증 (Generalization & Extrapolation Test)
*   **목적:** 훈련 데이터 분포(3~5층)를 벗어난 **6층 건물(OOD)**에 대한 예측 강건성 검증.
*   **결과:** MLP는 OOD 데이터에서 $R^2 \approx 0.06$으로 예측에 실패했으나, GAT는 상대적으로 높은 성능($R^2 \approx 0.48$)을 유지하며 구조적 일반화 가능성을 보임.
*   **논문용 산출물:**
    *   `results/experiments/ood_test_results.csv`
    *   `results/experiments/ood_generalization_plot.png`

### 6.4 해석 가속화 효과 정량화 (Quantification of Acceleration)
*   **목적:** GAT 모델의 압도적인 속도 우위 입증.
*   **결과:** GAT의 추론 속도(약 3.78ms)는 기존 OpenSees 해석(수십 초~수분) 대비 **약 $10^4$배 이상의 가속화**를 달성함.

### 6.5 결과 시각화 및 해석 (Visualization & Interpretation)
*   **Fig 6:** GAT 모델의 Attention Weight 시각화 (구조적 취약부위 집중 여부 확인).
*   **Fig 7:** t-SNE Embedding 시각화 (구조 특성에 따른 데이터 군집화).

### 6.6 논문 원고 작성 (Manuscript Preparation)
*   **구조:** Introduction -> Methodology (Proposed GAT) -> Experimental Setup -> Results (Comparative, Scalability, OOD) -> Conclusion.

---
**Last Updated:** 2025-12-15
**Author:** Gemini CLI Agent