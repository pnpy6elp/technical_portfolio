# Results for Reproducibility  
본 디렉토리에서는 실제 실험 과정에서 생성된 각 데이터셋별 모델 결과물과 성능 측정 파일을 확인하실 수 있습니다. 연구 재현성을 보장하기 위해 모든 산출물과 실행 스크립트를 함께 제공합니다.

---

## Details  
이 디렉토리에는 CoraFull, Arxiv, Reddit, Products 네 가지 데이터셋에 대해 SAGE-CoG 기반 그래프 연속 학습 실험을 수행하며 생성된 결과가 포함되어 있습니다.  
각 실험에서는 메모리 예산(budget)에 따라 생성된 **슈퍼노드 기반 압축 그래프(memory bank)** 와 **평가 결과(performance 파일)** 가 저장되어 있어, 동일 설정으로 실험을 다시 수행하거나 결과를 검증하는 데 활용하실 수 있습니다.

- `memory_bank/`  
  각 데이터셋별로 생성된 **압축 그래프(supernode graph)** 가 저장됩니다. 실험 과정에서 커뮤니티 기반 그래프 축약 및 리플레이 전략을 통해 생성된 메모리 구조를 그대로 확인하실 수 있습니다.

- `performance/`  
  연속 학습 평가 과정에서 생성된 **최종 성능 파일(`*_result.pt`)** 이 포함됩니다. 평가 지표는 아래의 average accuracy와 average forgetting을 기반으로 산출하였습니다.

---

## Evaluation Metrics  

### **Average Performance (AP)**  
Average Performance는 연속 학습 시나리오에서 **각 단계(Task t)** 를 학습한 후 해당 시점까지 등장한 모든 작업에 대한 정확도를 평균낸 지표입니다.  
즉, 모델의 전반적인 성능 유지 능력을 평가하며 다음과 같이 계산됩니다:

\[
AA = \frac{1}{T} \sum_{t=1}^{T} Acc_{t,t'}
\]

여기서 \(Acc_{t,t'}\)는 *t번째 작업 학습 후* 이전 모든 작업에 대한 정확도를 의미합니다.

### **Average Forgetting (AF)**  
Average Forgetting은 이전에 학습한 작업(Task i)의 성능이 시간이 지남에 따라 얼마나 감소했는지를 측정하는 지표입니다.  
각 작업별로 **최고 정확도와 최종 정확도의 차이** 를 측정하여 평균을 계산합니다:

\[
AF = \frac{1}{T-1} \sum_{i=1}^{T-1} \left( \max_{t \leq T} Acc_{t,i} - Acc_{T,i} \right)
\]

이 지표는 모델이 Catastrophic Forgetting 문제를 얼마나 잘 해결하는지를 평가하는 핵심 값입니다.

---

## Reproducibility  
`reproducibility.sh` 스크립트를 실행하시면 네 가지 데이터셋(CoraFull, Arxiv, Reddit, Products)에 대한 전체 실험을 동일한 조건으로 반복 수행할 수 있습니다.  
각 실험은 반복 횟수(repeat), 학습 epoch, 메모리 budget 등이 고정된 설정으로 실행되므로, 제안한 SAGE-CoG 방법의 재현성을 손쉽게 확인하실 수 있습니다.
아래와 같이 스크립트 파일을 실행할 수 있습니다.

```sh reproducibility.sh```