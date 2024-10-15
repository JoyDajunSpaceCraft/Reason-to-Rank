import copy
import datasets
import json
import torch

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = [json.loads(line) for line in file]
    return data

def get_preprocessed_custom(dataset_config, tokenizer, file_path, model_mode="base_reason"):
    data = load_data(file_path)
    
    def apply_prompt_template(sample):
        
        query = sample['query']
        unsorted_docs = sample['unsorted_docs']
        reason_sample = sample["reason"]
        ranked_passages = reason_sample.get('ranked_passages', [])

        int_rank_id = [int(i) for i in sample["re_rank_id"]]
        re_rank_ids = torch.tensor(int_rank_id, dtype=torch.long)  
        query_docs = f"I will provide you with {len(unsorted_docs)} passages, each with a special token representing the passage enclosed in [], followed by original text.\nGenerate the reason based on Rank the passages based on their relevance to the search query: {query}\n"
        
        for i, doc in enumerate(unsorted_docs):
            query_docs += f"[Passage {i+1}] {doc}\n"

        direct_reasons = []
        listwise_reasons = []

        for i in range(len(unsorted_docs)):
            for j in ranked_passages:
                # print("j.get(identifier)", j.get("identifier"))
                if j.get("identifier") == i + 1:
                    direct_reasons.append(j.get('direct_reason', 'No direct reason provided'))
                    listwise_reasons.append(j.get('listwise_reason', 'No listwise reason provided'))

        return {
            "prompt": query_docs,
            "direct_reason": direct_reasons,
            "listwise_reason": listwise_reasons,
            "rerank_id": re_rank_ids.tolist(),
        }

    data = [apply_prompt_template(sample) for sample in data]

    def tokenize_add_label(sample, model_mode=model_mode):
        prompt = tokenizer.encode(tokenizer.bos_token + sample["prompt"], add_special_tokens=False)
        direct_reason = tokenizer.encode(" ".join(sample["direct_reason"]) + tokenizer.eos_token, add_special_tokens=False)
        listwise_reason = tokenizer.encode(" ".join(sample["listwise_reason"]) + tokenizer.eos_token, add_special_tokens=False)
        if model_mode == "base_reason":
            sample = {
                "input_ids": prompt + direct_reason + listwise_reason,
                "attention_mask": [1] * (len(prompt) + len(direct_reason) + len(listwise_reason)),
                "labels": [-100] * len(prompt) + direct_reason + listwise_reason,
                "extra_input_ids": None,
                "rank_id_labels": sample["rerank_id"],
                "reason_labels":direct_reason,
            }
        elif model_mode =="base_no_reason":
            sample = {
                "input_ids": prompt,
                "attention_mask": [1] * (len(prompt) + len(direct_reason) + len(listwise_reason)),
                "labels": [-100] * len(prompt) + direct_reason + listwise_reason,
                "extra_input_ids": None,
                "rank_id_labels": sample["rerank_id"],
                "reason_labels":direct_reason,
            }
            
        else:
            sample = {
                "input_ids": prompt + direct_reason + listwise_reason,
                "attention_mask": [1] * (len(prompt) + len(direct_reason) + len(listwise_reason)),
                "labels": [-100] * len(prompt) + direct_reason + listwise_reason,
                "extra_input_ids": listwise_reason,
                "rank_id_labels": sample["rerank_id"],
                "reason_labels":direct_reason,
            }

        return sample

    dataset = datasets.Dataset.from_list(data)
    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
from dataclasses import dataclass
from transformers.file_utils import ModelOutput
from typing import Optional

