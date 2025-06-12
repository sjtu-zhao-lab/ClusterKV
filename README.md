# ClusterKV

ClusterKV: Manipulating LLM KV Cache in  Semantic Space for Recallable Compression (DAC'25)

## Requirements

ClusterKV uses [uv](https://github.com/astral-sh/uv) for package management and cmake for building. 
We tested on platforms with CUDA 12.8 (CUDA >= 11.0 should be generally supported).

- uv >= 0.7.3
- cmake >= 3.26.4
- CUDA >= 12.8

## Installation

0. Clone with submodules

```bash
git clone --recursive https://github.com/sjtu-zhao-lab/ClusterKV.git
cd ClusterKV
```

1. Sync enviroments and activate the virtual environment:

```bash
uv sync
source .venv/bin/activate
```

2. Install flash-attn individually
```bash
pip install flash-attn==2.6.3 --no-build-isolation
```

3. Install kernels

```bash
# Build and install libraft
pushd 3rdparty/raft/
INSTALL_PREFIX=$VIRTUAL_ENV ./build.sh libraft
popd && pushd kernel
bash setup.sh
popd
```

4. Install the package

```bash
pip install -e .
```

## Project Structure

- `clusterkv/`: Main package directory
  - `clusterkv_models/`: ClusterKV model implementations
  - `clusterkv_utils/`: Utility functions specified for ClusterKV
  - `quest_models/`: Quest model implementations
  - `utils/`: General utilities
- `kernel/`: CUDA kernel implementations
- `3rdparty/`: Third-party dependencies
- `efficiency/`: Efficiency benchmarking 
- `accuracy/`: Accuracy benchmarking

## Usage

- For the accuracy testing

```bash
# LongBench
cd accuracy/LongBench
CUDA_VISIBLE_DEVICES=0 python pred.py --model glm4-9b-chat-8k --task hotpotqa --cluster # or
CUDA_VISIBLE_DEVICES=0 python pred.py --model glm4-9b-chat-8k --task hotpotqa --quest   # or    
CUDA_VISIBLE_DEVICES=0 python pred.py --model glm4-9b-chat-8k --task hotpotqa           # full kv
# Perplexity
cd accuracy/pg19
CUDA_VISIBLE_DEVICES=0 python ppl_eval.py --model glm4-9b-chat-8k --cluster # or
CUDA_VISIBLE_DEVICES=0 python ppl_eval.py --model glm4-9b-chat-8k --quest   # or
CUDA_VISIBLE_DEVICES=0 python ppl_eval.py --model glm4-9b-chat-8k           # full kv
```
Model name and context length are configured in `accuracy/config`.

- For the efficiency testing

```bash
cd efficiency
CUDA_VISIBLE_DEVICES=0 python bench_textgen.py --model llama3-8b --iteration 5 --warmup 3 --method clusterkv    # or
CUDA_VISIBLE_DEVICES=0 python bench_textgen.py --model llama3-8b --iteration 5 --warmup 3 --method quest        # or
CUDA_VISIBLE_DEVICES=0 python bench_textgen.py --model llama3-8b --iteration 5 --warmup 3 --method full         # or
```

## Citation

If you find ClusterKV useful in your research, please consider citing our work.
```
@misc{liu2024clusterkvmanipulatingllmkv,
      title={ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression}, 
      author={Guangda Liu and Chengwei Li and Jieru Zhao and Chenqi Zhang and Minyi Guo},
      year={2024},
      eprint={2412.03213},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.03213}, 
}
```

## Acknowledgements

This project uses the following open-source libraries:
- [flashinfer](https://github.com/flashinfer-ai/flashinfer)
- [Quest](https://github.com/mit-han-lab/Quest)
- [raft](https://github.com/rapidsai/raft)

We are grateful to their great works!
