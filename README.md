# Technical Porfolio
24510111 Boyoung Lim

### Python, SQL, R, Java, Scala, Go, C/C++, Javascript 등 데이터 처리 언어 활용 능력
**[./src/src.md](./src/src.md)**: Python
### Linux, Docker, Virtual Machines, Kubernetes 등을 활용한 데이터 활용 및 분석을 위한 환경 구축 여부
**[./yaml/docker.md](./yaml/docekr.md)**: linux 환경에서 Docker를 활용하여 Pytorch 기반의 데이터 처리 및 학습 환경을 구축했습니다.
### 머신러닝 라이브러리를 이용한 재현 가능한 개발 결과물 공개 여부
**[./results/results.md](./results/results.md)**: 

</br>
</br>


## Abtract
이 프로젝트는 시간의 흐름에 따라 변화하는 그래프 데이터에 대응하기 위한 새로운 그래프 연속 학습 프레임워크를 개발한 프로젝트 소스 코드입니다. 대표적인 Benchmark datasets인 CoraFull, Arxiv, Reddit, Products 네 가지 데이터셋을 사용하여 시간의 흐름에 따라 클래스가 증가하는 환경에서 높은 정확도와 효율성을 달성할 수 있음을 보였습니다. 데이터는 Pytorch Geometric에서 제공하고 있는 데이터셋을 활용해 연속 학습 시나리오에 맞추어 전처리를 하였습니다. 파이프라인은 다음과 같은 단계로 이루어져있습니다:

1. **Data Preprocessing**: 사용자가 지정한 데이터셋을 클래스 증분 연속 학습 시나리오에 맞추어 전처리를 수행합니다.
2. **Community-based Partitioning**: Louvain, Leiden 등의 Community Detection 알고리즘을 통해 Partitioning을 수행합니다.
3. **Community-Aware Coarsened Graph Construction**: Partitioning을 통해 생성된 커뮤니티의 수를 budget의 제한적인 한도 내로 줄이기 위해 중요도 기반 선별과 유사도 기반의 병합을 수행합니다. 병합된 각 커뮤니티는 슈퍼노드로 변환하여 원본 그래프의 구조를 보존하는 coarsed graph를 구성합니다.
4. **Class-Balanced Graph Refinement**: 커뮤니티 기반으로 생성된 슈퍼노드들에서 클래스 불균형을 측정하고, 각 노드의 커뮤니티 밀집도·규모를 반영한 품질 점수를 계산해 목표 비율에 맞게 고품질 노드를 선별함으로써 클래스 불균형을 제한하고 의미 있는 구조 정보를 유지한 균형 잡힌 coarsed graph를 생성합니다.

</br>

## Requirements
1. Docker, Docker-compose
2. Python 3.8, [Python Libraries](./requirements.txt)

## Directories
1. src: 프레임워크 소스 코드 
2. results: 결과 파일
2. yaml: Docker container 구성을 위한 docker-compose 파일이 있는 디렉토리