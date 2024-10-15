# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

from tqdm import tqdm
from itertools import chain

from torch.utils.data import Dataset

class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=4096, model_mode=None):
        self.dataset = dataset
        self.chunk_size = chunk_size

        self.samples = []
        if model_mode=="rank":
            buffer = {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
                "extra_input_ids": [],
                "rank_id_labels":[]
                }
        elif model_mode=="reason":
            buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "extra_input_ids": [],
            "rank_id_labels":[],
            "reason_labels":[]
            }
        elif model_mode in ["base_reason","base_no_reason"]:
             buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "rank_id_labels":[],
            "reason_labels":[]
            }
        else:
            buffer = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "extra_input_ids": [],
            }
        for sample in tqdm(self.dataset, desc="Preprocessing dataset", dynamic_ncols=True):
            buffer = {k: v + sample[k] for k,v in buffer.items()}

            while len(next(iter(buffer.values()))) > self.chunk_size:
                self.samples.append({k: v[:self.chunk_size] for k,v in buffer.items()})
                buffer = {k: v[self.chunk_size:] for k,v in buffer.items()}

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)