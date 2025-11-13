# Community-based Graph Continual Learning (Dynamic Graph Representation Learning)
본 프로젝트는 클래스 증분 시나리오에 동적으로 대응하기 위한 커뮤니티 기반 그래프 연속 학습 (Graph Continual Learning) 프레임워크 SAGE-CoG를 구현하였습니다. 소스 코드는 모두 Python 언어로 구현되었습니다.

---

## Directories

```
src/
├── backbones/     # GNN 기반의 Backbone 모델 레이어
├── methods/       # SAGE-CoG 구현
├── run.sh         # 실행 파일
├── train.py       # 학습 및 추론
└── utilities.py   # 각종 유틸리티
```

---

## Details
### 📁 Backbones
`backbones/` 디렉토리에는 본 프로젝트에서 사용하는 다양한 GNN(Graph Neural Network) 기반 모델들의 레이어 구조가 구현되어 있습니다. 예컨대 GCN, GAT, SGC 등 여러 그래프 신경망 아키텍처를 모듈화하여 제공하며, SAGE-CoG 프레임워크가 입력 그래프를 효과적으로 표현할 수 있도록 특징 인코딩 역할을 수행합니다. 해당 모듈들은 모델 선택 및 실험 환경에 따라 유연하게 교체하여 사용할 수 있도록 설계되어 있습니다.
### 📁 Methods 
`methods/` 디렉토리에는 커뮤니티 기반 그래프 연속 학습을 구현한 SAGE-CoG 알고리즘의 핵심 로직이 포함되어 있습니다. 동적 그래프 환경에서의 표현 업데이트, 리플레이 전략, 클래스 증분 상황 대응 등 연속 학습을 위한 주요 메커니즘이 이곳에서 처리됩니다.

주요 구성 요소인 `sage_cog.py`는 커뮤니티 기반 그래프 압축과 리플레이 전략을 결합한 연속 학습 알고리즘을 구현한 소스코드로, 그래프 데이터를 커뮤니티 단위로 분석하여 구조를 유지한 채 중요한 노드 정보를 선별·축약하는 방식으로 메모리 버짓을 효율적으로 활용합니다. 입력 그래프에서 커뮤니티를 탐지한 뒤, 크기·밀도·라벨 분포 등을 기반으로 고품질 커뮤니티를 선택하거나 병합하며, 이후 커뮤니티 단위 그래프 축약과 클래스 비율을 고려한 균형 조정을 수행하여 ‘슈퍼노드(supernode)’ 기반의 압축 그래프를 생성합니다. 이렇게 생성된 그래프는 PyG 포맷으로 변환되어 리플레이 데이터로 사용되며, Catastrophic Forgetting을 줄이기 위해 클래스 다양성, 커뮤니티 품질, 소수/다수 클래스 비율 등을 동적으로 반영하는 것이 특징입니다.
### 📄 Train ([train.py](./train.py))
`train.py`는 전체 학습 파이프라인을 담당하며, 데이터 로딩, 모델 초기화, 학습 루프, 평가 등의 과정을 일괄 관리합니다. 실험 설정값을 기반으로 백본 모델과 SAGE-CoG 메서드를 연결해 단일 실행만으로 실험을 재현할 수 있도록 구성되어 있습니다.
### 📄 Utilities ([utilities.py](./utilities.py))
`utilities.py`는 실험 전반에서 공통적으로 사용되는 핵심 기능들을 모아둔 모듈로, 데이터셋 로딩, 백본 모델 선택, CGL 알고리즘 초기화, 결과 파일명 생성, 성능 매트릭스 출력 등 실험 실행에 필요한 주요 로직들을 제공합니다. SAGE-CoG 설정을 자동으로 초기화할 수 있도록 구성되어 있으며, 데이터셋 구조나 실험 설정에 맞추어 모델과 학습 파이프라인을 일관성 있게 구성할 수 있도록 돕는 역할을 합니다.

## Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

## Dataset
데이터셋은 [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) 라이브러리에서 제공하는 데이터셋을 사용하였습니다. 

## Acknowledgment
이 코드는 CaT-CGL(https://github.com/superallen13/CaT-CGL.git)와 CGLB(https://github.com/QueuQ/CGLB.git)을 기반으로 구현되었습니다. 더 많은 베이스라인들과 구현 상세 내용은 이 둘을 참고해주세요.

## Run
아래 스크립트 파일을 실행하면 CoraFull dataset에 대한 실험을 진행할 수 있습니다.

```sh run.sh```
