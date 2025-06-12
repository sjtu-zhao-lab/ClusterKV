import torch
from tqdm import tqdm
import os, json
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaAttention
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from evaluation.patch import parse_common_args, enable_attention_eval, get_config_output_affix
from evaluation.cluster_attention import cluster_reset
import argparse

device = "cuda"
parser = argparse.ArgumentParser()
parser = parse_common_args(parser)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--dump", action="store_true", help="Dump once to avoid frequent I/O")
parser.add_argument("--num_eval_tokens", type=int, default=32000)
parser.add_argument("--start", type=int, default=0)


def load(model_name_or_path):
    print(f"Loading model from {model_name_or_path} ...")
    # however, tensor parallel for running falcon will occur bugs
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    model.eval()

    return model, tokenizer

def set_prompt_len(model_name, model, prompt_len):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            set_prompt_len(model_name, module, prompt_len)
        if isinstance(module, LlamaAttention) or \
            ("glm4" in model_name and module.__class__.__name__ == "SelfAttention"):
            module.prompt_len = prompt_len

args = parser.parse_args()

data = load_dataset("emozilla/pg19-test", split="test")

model2path = json.load(open("../config/model2path.json", "r"))
model_name = args.model
model, tokenizer = load(model2path[model_name])

loss_fn = CrossEntropyLoss(reduction="none")
past_key_values = DynamicCache.from_legacy_cache()

if args.quest or args.cluster:
    enable_attention_eval(model_name, model, args)

output_dir = args.output_dir if args.output_dir is not None else f"output/{model_name}/"
os.makedirs(output_dir, exist_ok=True)
config_affix = get_config_output_affix(args)
ppl_f = open(f"{output_dir}/ppl{config_affix}.txt", "w")
nll_f = open(f"{output_dir}/nnl{config_affix}.txt", "w")
all_nlls = []
all_ppls = []

num_eval_tokens = 0
ckpt = 0
for text in data["text"][:1]:
    encodings = tokenizer(text, return_tensors="pt")

    print(encodings.input_ids[:, :10])

    seq_len = encodings.input_ids.size(1)
    print(f"seq_len: {seq_len}")
    pbar = tqdm(range(1, seq_len - 1))

    for idx in pbar:
        if idx < args.start:
            continue
        if args.cluster:
            cluster_reset(model)
        set_prompt_len(model_name, model, 0)
        input_ids = encodings.input_ids[:, : idx + 1].to(device)
        with torch.no_grad():
            if "glm4" in model_name:
                model_kwargs = {
                    "past_key_values": None,
                    "return_last_logit": True,
                    "use_cache": True,
                    "is_first_forward": True,
                }
                model_inputs = model.prepare_inputs_for_generation(input_ids[:, :-1], 
                                                                   **model_kwargs)
                outputs = model(
                    **model_inputs, return_dict=True, 
                    output_attentions=False, output_hidden_states=False,
                )
                model_kwargs = model._update_model_kwargs_for_generation(outputs, 
                                                                         model_kwargs)
                model_inputs = model.prepare_inputs_for_generation(input_ids, 
                                                                   **model_kwargs)
                outputs = model(
                    **model_inputs, return_dict=True, 
                    output_attentions=False, output_hidden_states=False,
                )
            else:
                outputs = model(
                    input_ids[:, :-1],
                    past_key_values=DynamicCache.from_legacy_cache(),
                )
                past_key_values = outputs.past_key_values
                outputs = model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[:, -1:, :].view(-1, model.config.vocab_size)
            past_key_values = outputs.past_key_values
            label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
            neg_log_likelihood = loss_fn(logits, label)

        nll = neg_log_likelihood.item()
        all_nlls.append(neg_log_likelihood)
        pbar.set_description(
            f"nll: {neg_log_likelihood.item():.2f}, ppl: {torch.exp(neg_log_likelihood).item():.2f}"
        )
        ppl = torch.exp(torch.stack(all_nlls).mean()).item()
        if args.dump:
            all_ppls.append(ppl)
            if idx > 0 and idx % 1000 == 0:
                for n in all_nlls[ckpt:]:
                    print(n.item(), file=nll_f)
                for p in all_ppls[ckpt:]:
                    print(p, file=ppl_f)
                nll_f.flush()
                ppl_f.flush()
                ckpt = idx
        else:
            print(nll, file=nll_f, flush=True)
            print(ppl, file=ppl_f, flush=True)

        num_eval_tokens += 1
        if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
            break
    if args.num_eval_tokens is not None and num_eval_tokens >= args.num_eval_tokens:
        break

if args.dump:
    for n in all_nlls[ckpt:]:
        print(n.item(), file=nll_f)
    for p in all_ppls[ckpt:]:
        print(p, file=ppl_f)

nll_f.close()
ppl_f.close()

