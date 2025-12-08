# 프로젝트 헌법 (Constitution)

## 1. 기본 원칙 (Core Principles)
- **목표 지향:** 모든 코드는 "1차 모드 지배 RC 모멘트 골조의 푸쉬오버 곡선 예측"이라는 목표에 기여해야 한다.
- **품질 최우선:** SCI급 논문 게재를 목표로 하므로, 데이터의 신뢰성과 모델의 재현성이 최우선이다.
- **Spec 준수:** `specs/` 디렉토리에 정의된 명세(Spec)는 코드보다 상위 권한을 가진다. 명세와 코드가 충돌하면 명세를 따른다.

## 2. 코딩 컨벤션 (Coding Convention)
- **언어:** Python 3.10+
- **스타일:** PEP 8 준수. (Black Formatter 스타일 지향)
- **타입 힌트:** 모든 함수와 메서드에 Type Hint를 명시한다.
- **주석:** 복잡한 로직에는 반드시 한글 주석으로 'Why'를 설명한다. Docstring은 필수다.

## 3. 기술적 제약 (Technical Constraints)
- **라이브러리:**
    - 구조해석: `openseespy`
    - 딥러닝: `torch`, `torch_geometric`
    - 데이터: `numpy`, `pandas`
- **재현성:** 모든 랜덤 프로세스(데이터 샘플링, 모델 초기화 등)에는 `seed`를 고정해야 한다.

## 4. 워크플로우 (Workflow)
1. **Specify:** 변경 사항이 생기면 Spec 문서를 먼저 수정한다.
2. **Plan:** Spec을 바탕으로 작업 계획을 수립한다.
3. **Implement:** 계획에 따라 코드를 작성하고 테스트한다.
