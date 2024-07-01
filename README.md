<p align="center">
<img src="figs/logo.png" width="30%"> <br>
</p>
<div align="center">
  <h1>🔭LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference</h1>
  📰<a href="https://arxiv.org/pdf/2406.18139" target="_blank"><strong>Paper</strong></a>
</div>

## Overview
Long-context Multimodal Large Language Models (MLLMs) demand substantial computational resources for inference as the growth of their multimodal Key-Value (KV) cache, in response to increasing input lengths, challenges memory and time efficiency.

Unlike single-modality LLMs that manage only textual contexts, the KV cache of long-context MLLMs includes representations from multiple images with temporal and spatial relationships and related textual contexts. The predominance of image tokens means traditional optimizations for LLMs' KV caches are unsuitable for multimodal long-context settings, and no prior works have addressed this challenge.

In this work, we introduce LOOK-M, a pioneering, fine-tuning-free approach that efficiently reduces the multimodal KV cache size while maintaining performance comparable to a full cache. We observe that during prompt prefilling phase, the model prioritizes more textual attention over image features, and based on the multimodal interaction observation, a new proposed text-prior method is explored to compress the KV cache.

Furthermore, to mitigate the degradation of image contextual information, we propose several compensatory strategies using KV pairs merging. LOOK-M demonstrates that with a significant reduction in KV Cache memory usage, such as reducing it by 80% in some cases, it not only achieves up to 1.5x faster decoding but also maintains or even enhances performance across a variety of long context multimodal tasks.
<div style="text-align: center;">
    <img src="figs/pipeline.png">
</div>

## Usage

### Environment Setup
The Environments Setup is consistent with Milebench
```
conda create -n LOOK-M
pip install -r requirements.txt
```

### Test LOOK-M and other KVCache Eviction Strategy
Example 1. test LOOK-M
```
conda activate LOOK-M
bash ./scripts/text-prior-pivot-merge_eval.sh
```
Exampke 2. test model without strategy
```
conda activate LOOK-M
bash ./scripts/origin_eval.sh
```
## TODO

- [ ] reorgnize the code for better using experience
- [ ] support more models

## Citation

#### If you find our work valuable, we would appreciate your citation: 🎈


```bibtex
@article{wan2024look,
  title={LOOK-M: Look-Once Optimization in KV Cache for Efficient Multimodal Long-Context Inference},
  author={Wan, Zhongwei and Wu, Ziang and Liu, Che and Huang, Jinfa and Zhu, Zhihong and Jin, Peng and Wang, Longyue and Yuan, Li},
  journal={arXiv preprint arXiv:2406.18139},
  year={2024}
}
```


#### The code is still being organized.🚧
