# Based on Punica Project
# Check: https://github.com/efeslab/Atom/blob/main/e2e/punica-atom/benchmarks/bench_textgen.py

import argparse
import dataclasses
import time
import numpy as np
import torch
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer

# from quest import LlamaForCausalLM
from clusterkv.quest_models.llama import LlamaForCausalLM as QuestLlamaForCausalLM
from clusterkv.clusterkv_models.llama import LlamaForCausalLM as ClusterKVLlamaForCausalLM
import os

c = torch.cuda.get_device_capability()
os.environ["TORCH_CUDA_ARCH_LIST"] = f"{c[0]}.{c[1]}"

@dataclasses.dataclass
class ModelConfig:
  model_path: str
  dtype: str = dataclasses.field(default="float16")
  device: str = dataclasses.field(default="cuda:0")

MODEL_CFGS = {
    "llama2-7b":
        ModelConfig(
            model_path="meta-llama/Llama-2-7b-chat-hf"
        ),
    "llama3-8b":
        ModelConfig(
            model_path="meta-llama/Meta-Llama-3-8B-Instruct"
        ),
}

def load_model(model_cfg: ModelConfig, method: str):
    device = torch.device(model_cfg.device)
    dtype = getattr(torch, model_cfg.dtype)
    torch.set_default_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_path, trust_remote_code=True)

    with device:
        if method == "quest":
            model = QuestLlamaForCausalLM.from_pretrained(
                model_cfg.model_path, device_map=device, torch_dtype=dtype,
            )
        elif method in ["clusterkv", "full"]:
            model = ClusterKVLlamaForCausalLM.from_pretrained(
                model_cfg.model_path, device_map=device, torch_dtype=dtype,
            )
    return model, tokenizer

@torch.inference_mode()
def benchmark_quest():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODEL_CFGS.keys(), default="llama3-8b")
    parser.add_argument("--context_len", type=int, default=4*1024, help="Prefill length")
    parser.add_argument("--decode_len", type=int, default=256, help="Generation length")
    parser.add_argument("--page_size", type=int, default=16, help="Page size for Quest")
    parser.add_argument("--token_budget", type=int, default=512, help="Token budget for ClusterKV and Quest")
    parser.add_argument("--iteration", type=int, default=3, help="Number of iterations")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup iterations")
    parser.add_argument("--method", type=str, 
                        choices=['quest', 'clusterkv', 'full'], required=True)
    parser.add_argument("--nlist", type=int, default=200, help="Number of clusters")
    parser.add_argument("--niter", type=int, default=20, help="Number of max cluster iterations")
    parser.add_argument("--sink", type=int, default=16, help="Sink size")
    parser.add_argument("--window", type=int, default=320, help="Window size")
    parser.add_argument("--window_nlist", type=int, default=8, help="Number of clusters in a window")
    parser.add_argument("--offload", action="store_true", help="Offloading cache to CPU")
    args = parser.parse_args()
    assert args.warmup < args.iteration, "Warmup iterations must be less than total iterations"

    assert args.model in MODEL_CFGS, f"Model {args.model} not found in MODEL_CFGS"
    model_cfg = MODEL_CFGS[args.model]

    if args.offload:
        assert args.method == "clusterkv", "Offloading is only supported for clusterkv"
    
    max_seq_len = args.context_len + args.decode_len + 512
    page_size = args.page_size
    method = args.method
    token_budget = 102400 if "full" in method else args.token_budget
    context_len = args.context_len
    decode_len = args.decode_len
    nlist = args.nlist
    niter = args.niter

    model, tokenizer = load_model(model_cfg, method)
    
    dtype = getattr(torch, model_cfg.dtype)
    device = torch.device(model_cfg.device)
    if method == "quest":
        model.quest_init(
            page_size=page_size,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device
        )
    elif method in ["clusterkv", "full"]:
        model.clusterkv_init(
            nlist=nlist,
            niter=niter,
            max_seq_len=max_seq_len,
            token_budget=token_budget,
            dtype=dtype,
            device=device,
            full=(method=="full"),
            sink=args.sink,
            window=args.window,
            window_nlist=args.window_nlist,
            offload=True if args.offload else False
        )

    hidden_size = model._config.hidden_size

    prefill_latency = []
    decode_latency = []
    data = load_dataset('THUDM/LongBench', "triviaqa", split='test')
    json_obj = data[1]
    prompt_format = "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}" 
    prompt = prompt_format.format(**json_obj)
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if len(tokenized_prompt) > context_len:
        half = int(context_len / 2)
        prompt = tokenizer.decode(
            tokenized_prompt[:half], skip_special_tokens=True
        ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
    # print(prompt)
    print("="*100)
    input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
    input_ids = input.input_ids

    generated_content = []
    for _ in tqdm(range(args.iteration)):
        # clear cuda cache
        torch.cuda.empty_cache()

        # Prefill Stage
        # hidden_states = torch.randn(1, context_len, hidden_size, dtype=dtype, device=device)
        ts = time.perf_counter()
        output = model(
            # inputs_embeds=hidden_states,
            input_ids=input_ids
        )
        te = time.perf_counter()
        prefill_latency.append(te - ts)
        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
        generated_content += [pred_token_idx.item()]

        # Start decoding decode_len tokens
        # hidden_states = torch.randn(1, 1, hidden_size, dtype=dtype, device=device)
        for _ in range(decode_len):
            ts = time.perf_counter()
            output = model(
                # inputs_embeds=hidden_states,
                input_ids=pred_token_idx,
            )
            te = time.perf_counter()
            decode_latency.append(te - ts)
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
            generated_content += [pred_token_idx.item()]
        
        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        print(pred)
        if method == "quest":
            model.quest_clear()
        elif method in ["clusterkv", "full"]:
            model.clusterkv_clear()
    
    warmup = args.warmup
    avg_prefill_latency = np.mean(prefill_latency[warmup:])
    avg_decode_latency = np.mean(decode_latency[warmup*decode_len:])

    print("page_size,token_budget,context_len,decode_len,avg_prefill_latency,avg_decode_latency")
    print(f"{page_size},{token_budget},{context_len},{decode_len},{avg_prefill_latency},{avg_decode_latency}")

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    seed_everything(42)
    benchmark_quest()
