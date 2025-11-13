# SAGE-CoG

# Abstract
> Graph representation learning has demonstrated remarkable success in various domains. Recently, to respond to dynamic environments where graph structures and node labels change over time, continual graph learning (CGL) methods have been proposed to learn from sequential tasks without catastrophic forgetting. Existing state-of-the-art (SOTA) approaches are designed exclusively for either transductive settings, where each task involves disjoint node classes within a shared graph, or inductive settings, where tasks comprise separate graphs with overlapping classes. This narrow focus limits their effectiveness in real-world applications, which often exhibit a mixture of both settings. We propose a Setting-Agnostic continual Graph lEarning framework with Community-aware Graph coarsening, simply SAGE-CoG, designed to seamlessly support both transductive and inductive tasks as well as their combination. Our method uses community detection to partition graphs and constructs coarsened graph representations that retain structural characteristics while significantly reducing memory requirements. We evaluate our SAGE-CoG on four transductive and three inductive benchmark datasets, comparing it with eight SOTA baselines that are specific to transductive and inductive settings. While some existing methods may perform better in the setting they are tailored for, they fail to generalize across settings. In contrast, our SAGE-CoG achieves consistently strong performance across both individual and combined settings, demonstrating clear superiority in realistic continual learning scenarios where setting boundaries are blurred. This makes our approach highly applicable to practical deployments that demand robustness, generality, and efficiency.

# Dependencies
- Python 3.9
- torch_Geometric 2.1
- networkx 3.1
- igraph 0.9.9

# Dataset
Datasets can be obtained from [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html) library. 

# Acknowledgment
This code is implemented based on CaT-CGL(https://github.com/superallen13/CaT-CGL.git) and CGLB(https://github.com/QueuQ/CGLB.git). Please refer to both for more baselines and implementation details.

# Run
