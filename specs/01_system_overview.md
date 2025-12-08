# 시스템 개요 (System Overview)

## 1. 프로젝트 목표
- **Graph Neural Network (GNN)**를 활용하여 저층 RC **모멘트 골조** 구조물의 비선형 정적해석(푸쉬오버 해석) 결과인 **푸쉬오버 곡선(Pushover Curve)을 신속하게 예측**하는 모델 개발.
- **SCI급 해외 학술지 게재**를 최종 목표로 함.

## 2. 연구 범위
- **대상 구조물:** 1차 모드 지배형 RC 모멘트 골조.
- **해석 종류:** 비선형 정적 해석 (Pushover Analysis).
- **데이터 일관성:** GNN 평가의 명확성을 위해 1차 모드 거동이 지배적인 데이터만 엄선하여 사용.

## 3. 기술 스택
- **언어:** Python 3.10+
- **구조 해석:** OpenSeesPy
- **딥러닝:** PyTorch, PyTorch Geometric
- **데이터 처리:** NumPy, Pandas

## 4. 데이터 파이프라인
### 4.1. 데이터 생성 (`src/core/`, `scripts/generate_dataset.py`)
- **모델링:** 비선형 거동 정밀 모사를 위한 섬유요소모델(Fiber Element Model) 사용.
- **재료:** 공칭강도가 아닌 **기대강도(`fc`, `Fy`)** 적용.
- **유효성 검증:**
    - 1차 모드 지배 여부 확인.
    - 질량 참여율 90% 이상.
    - 조기 재료 파괴(콘크리트 변형률 0.003, 철근 0.05 초과)가 없는 모델만 사용.

### 4.2. 데이터 표현 (`src/data/`)
- **그래프 구조:**
    - 노드(Node): 절점 (Joint)
    - 엣지(Edge): 구조 부재 (기둥, 보)
- **전역 특성 (Global Feature):** 해석 방향 (X 또는 Z)

## 5. 디렉토리 구조
```
GNN_Project/
├── data/                # 데이터셋 저장소
├── results/             # 해석 및 예측 결과
├── scripts/             # 실행 스크립트
├── src/                 # 소스 코드
│   ├── core/            # OpenSees 해석 로직
│   ├── data/            # 그래프 변환 로직
│   ├── gnn/             # GNN 모델 및 학습 (구현 예정)
│   └── visualization/   # 시각화
└── specs/               # 프로젝트 명세서 (Spec Kit)
```
