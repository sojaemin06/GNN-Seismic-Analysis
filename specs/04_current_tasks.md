# 현재 작업 목록 (Current Tasks)

## 작업 범위: GNN 모델 구현 (Phase 2)

### Task 1: 모델 클래스 구현 (`src/gnn/models.py`)
- [ ] **`PushoverGNN` 클래스 정의**
    - 상속: `torch.nn.Module`
    - 입력: `data` (PyG Batch 객체)
- [ ] **레이어 구성**
    - `GATv2Conv` (또는 `GINEConv`) 3층 적층 (Edge Feature 활용)
    - `GlobalMeanPool`
    - `MLP` (Decoder): `[Hidden + Global_Feature]` -> `Output(100)`
- [ ] **Forward 메서드 구현**
    - Node Feature와 Edge Feature를 함께 처리하는 로직 포함

### Task 2: 학습 스크립트 구현 (`src/gnn/train.py`)
- [ ] **데이터 로더 설정**
    - `data/processed/`의 `.pt` 파일 로딩
    - Train/Val/Test 분할 (예: 8:1:1)
- [ ] **학습 루프(Training Loop)**
    - Loss 계산 (MSE)
    - Backpropagation
    - Optimizer Step
- [ ] **검증 및 로깅**
    - Epoch마다 Val Loss 기록
    - Best Model 저장 (`results/models/`)

### Task 3: 동작 테스트
- [ ] Dummy 데이터를 생성하여 모델 입출력 차원(`Shape`) 검증
