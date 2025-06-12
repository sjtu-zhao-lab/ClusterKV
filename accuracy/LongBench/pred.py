import os, time
from requests.exceptions import ProxyError, SSLError
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)
from transformers.cache_utils import DynamicCache
from tqdm import tqdm
import numpy as np
import random
import argparse
from accuracy.patch import parse_common_args, enable_attention_eval, get_config_output_affix
from accuracy.cluster_attention import cluster_reset


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_idx", type=int, default=None)
    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "glm4" in model_name or "intern" in model_name or "llama3" in model_name:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                add_generation_prompt=True, tokenize=False)
    return prompt


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
):
    preds = []
    for _, json_obj in enumerate(tqdm(data)):
        if args.cluster:
            cluster_reset(model)
        prompt = prompt_format.format(**json_obj)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "glm4" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)
        # print(prompt)
        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa", "2wikimqa", "musique"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report", "qmsum"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        elif dataset in ["passage_retrieval_en"]:
            q_pos = prompt.rfind("The following is an abstract.")
        else:
            assert False

        # max simulation length is 100
        max_sim_len = 100
        q_pos = max(len(prompt) - max_sim_len, q_pos)

        if q_pos != None:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        q_input = tokenizer(question, truncation=False, return_tensors="pt").to("cuda")
        q_input.input_ids = q_input.input_ids[:, 1:]

        # print(input.input_ids.shape[-1], q_input.input_ids.shape[-1])
        # context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            assert False
        else:
            with torch.no_grad():
                if "glm4" in model_name:
                    model_kwargs = {
                        "past_key_values": None,
                        "return_last_logit": True,
                        "use_cache": True,
                        "is_first_forward": True,
                    }
                    input_ids = input.input_ids
                    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                    output = model(
                        **model_inputs, return_dict=True, 
                        output_attentions=False, output_hidden_states=False,
                    )
                    for q_input_id in q_input.input_ids[0]:
                        input_ids = torch.cat([input_ids, q_input_id.unsqueeze(0).unsqueeze(0)], dim=-1)
                        model_kwargs = model._update_model_kwargs_for_generation(output, model_kwargs)
                        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                        output = model(
                            **model_inputs, return_dict=True, 
                            output_attentions=False, output_hidden_states=False,
                        )
                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content = [pred_token_idx.item()]
                    input_ids = torch.cat([input_ids, pred_token_idx], dim=-1)
                    for _ in range(max_gen - 1):
                        model_kwargs = model._update_model_kwargs_for_generation(output, model_kwargs)
                        model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
                        output = model(
                            **model_inputs, return_dict=True, 
                            output_attentions=False, output_hidden_states=False,
                        )
                        pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(-1)
                        generated_content += [pred_token_idx.item()]
                        input_ids = torch.cat([input_ids, pred_token_idx], dim=-1)
                        if pred_token_idx.item() in [151329, 151336, 151338]:
                            break
                else:
                    output = model(
                        input_ids=input.input_ids,
                        past_key_values=DynamicCache.from_legacy_cache(),
                    )
                    past_key_values = output.past_key_values
                    for input_id in q_input.input_ids[0]:
                        output = model(
                            input_ids=input_id.unsqueeze(0).unsqueeze(0),
                            past_key_values=past_key_values,
                        )
                        past_key_values = output.past_key_values

                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content = [pred_token_idx.item()]
                    for _ in range(max_gen - 1):
                        outputs = model(
                            input_ids=pred_token_idx,
                            past_key_values=past_key_values,
                        )

                        past_key_values = outputs.past_key_values
                        pred_token_idx = (
                            outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                        )
                        generated_content += [pred_token_idx.item()]
                        if "glm4" in model_name:    # glm4 has 3 stop tokens
                            if pred_token_idx.item() in [151329, 151336, 151338]:
                                break
                        if pred_token_idx.item() == tokenizer.eos_token_id:
                            break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if "intern" in model_name or "qwen" in model_name or "glm4" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            path, trust_remote_code=True, torch_dtype=torch.float16,
            device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        ).to(device)
    elif "llama" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = LlamaForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True,
            attn_implementation="flash_attention_2", use_cache=True
        )
    else:
        assert False
    model = model.eval()

    if args.quest or args.cluster:
        enable_attention_eval(model_name, model, args)

    return model, tokenizer

def load_model_with_retry(model_path, model_name, device, retries=3, delay=1):
    for attempt in range(retries):
        try:
            model, tokenizer = load_model_and_tokenizer(model_path, model_name, device)
            return model, tokenizer
        except (ProxyError, SSLError) as e:
            print(f"Attempt {attempt + 1} failed due to network error: {e}")
            if attempt < retries - 1:
                time.sleep(delay)  # Wait before retrying
            else:
                raise  # Re-raise the last exception if all retries fail

if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    assert not (args.quest and args.cluster)     # cannot be enabled at same time
    if args.dist_t != "cosine":
        assert args.debug
    
    model2path = json.load(open("../config/model2path.json", "r"))
    model2maxlen = json.load(open("../config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_with_retry(
        model2path[model_name], model_name, device
    )
    max_length = model2maxlen[model_name]
    if args.task is not None:
        datasets = [args.task]
    else:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    if not os.path.exists("debug"):
        os.makedirs("debug")
    for dataset in datasets:
        if args.e:
            data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            res_dir = "debug" if args.debug or args.data_idx is not None else "pred_e"
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"{res_dir}/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"{res_dir}/{model_name}/{dataset}.jsonl"
        else:
            data = load_dataset("THUDM/LongBench", f"{dataset}", split="test")
            res_dir = "debug" if args.debug or args.data_idx is not None else "pred"
            if not os.path.exists(f"{res_dir}/{model_name}"):
                os.makedirs(f"{res_dir}/{model_name}")
            config_affix = get_config_output_affix(args)
            out_path = f"{res_dir}/{model_name}/{dataset}{config_affix}.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        if args.debug:
            data = data.select(range(1))
        elif args.data_idx is not None:
            data = data.select(range(args.data_idx, args.data_idx+1))
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
