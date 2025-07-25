import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="llama2-7b-chat-4k")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--n2eos', action='store_true', help="Using '\n\n' as eos")
    parser.add_argument('--method', type=str, default=None)
    parser.add_argument('--task', type=str, default=None)
    return parser.parse_args(args)

def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for (prediction, ground_truths, length) in zip(predictions, answers, lengths):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores

def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, answers):
        score = 0.
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip('\n').split('\n')[0]
        for ground_truth in ground_truths:
            score = max(score, dataset2metric[dataset](prediction, ground_truth, all_classes=all_classes))
        total_score += score
    return round(100 * total_score / len(predictions), 2)

if __name__ == '__main__':
    args = parse_args()
    scores = dict()
    if args.e:
        path = f"pred_e/{args.model}/"
    else:
        path = f"pred/{args.model}/"
    if args.n2eos:
        out_path = f"{path}/result-n2eos.json"
    else:
        out_path = f"{path}/result.json"
    if os.path.exists(out_path):
        if args.task is not None or args.method is not None:
            # Update incrementally
            with open(out_path) as f:
                scores = json.load(f)

    all_files = sorted(os.listdir(path))
    print("Evaluating on:", all_files)
    for filename in all_files:
        if not filename.endswith("jsonl"):
            continue
        predictions, answers, lengths = [], [], []
        if '-' in filename:
            dataset = filename.split('-')[0]
            method = filename.split('-')[1].split('.')[0]
        else:
            dataset = filename.split('.')[0]
            method = 'full'
        if args.task is not None and dataset != args.task:
            continue
        if args.method is not None and method != args.method:
            continue
        with open(f"{path}{filename}", "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if args.n2eos and dataset2metric[dataset] != rouge_score:
                    truc_pred = data["pred"].split('\n\n')[0]
                    predictions.append(truc_pred)
                else:
                    predictions.append(data["pred"])
                answers.append(data["answers"])
                all_classes = data["all_classes"]
                if "length" in data:
                    lengths.append(data["length"])
        if args.e:
            score = scorer_e(dataset, predictions, answers, lengths, all_classes)
        else:
            score = scorer(dataset, predictions, answers, all_classes)
        if dataset not in scores:
            scores[dataset] = {}
        scores[dataset][method] = score
    with open(out_path, "w") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
