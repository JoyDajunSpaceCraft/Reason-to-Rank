from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, default_data_collator
import torch 
from torch.utils.data import DistributedSampler
import torch.distributed as dist
from llama_recipes.data.sampler import LengthBasedBatchSampler, DistributedLengthBasedBatchSampler

def custom_collate_fn(batch, tokenizer, model_mode=None):
    # Get input_ids, attention_mask and extra_input_ids from the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    labels = [item['labels'] for item in batch]
    if model_mode not in ["base_reason", "base_no_reason"]:
        extra_input_ids = [item['extra_input_ids'] for item in batch]
    
    
    # Find the max length in the batch for padding
    max_length = max(len(ids) for ids in input_ids)
    
    # Pad input_ids and extra_input_ids to the same length
    input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in input_ids]
    if model_mode not in ["base_reason", "base_no_reason"]:
        extra_input_ids = [ids + [tokenizer.pad_token_id] * (max_length - len(ids)) for ids in extra_input_ids]
    attention_mask = [mask + [0] * (max_length - len(mask)) for mask in attention_mask]
    labels = [lbl + [-100] * (max_length - len(lbl)) for lbl in labels]  # Assuming -100 is the ignore index for labels
    
   
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    if model_mode not in ["base_reason", "base_no_reason"]:
        extra_input_ids = torch.tensor(extra_input_ids, dtype=torch.long)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    
    if model_mode=="rank":
        rank_id_labels = [item['rank_id_labels'] for item in batch]
        rank_id_labels = torch.tensor(rank_id_labels, dtype=torch.long)
        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'extra_input_ids': extra_input_ids,
        'rank_id_labels': rank_id_labels
        }
    elif model_mode=="reason":
        rank_id_labels = [item['rank_id_labels'] for item in batch]
        rank_id_labels = torch.tensor(rank_id_labels, dtype=torch.long)
        
        reason_labels = [item['reason_labels'] for item in batch]
        reason_labels = [lbl + [-100] * (max_length - len(lbl)) for lbl in reason_labels]  # Assuming -100 is the ignore index for labels
        reason_labels = torch.tensor(reason_labels, dtype=torch.long)
    
        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'extra_input_ids': extra_input_ids,
        'rank_id_labels': rank_id_labels,
        'reason_labels':reason_labels
        }
    elif model_mode in ["base_reason", "base_no_reason"]:
        rank_id_labels = [item['rank_id_labels'] for item in batch]
        rank_id_labels = torch.tensor(rank_id_labels, dtype=torch.long)
        reason_labels = [item['reason_labels'] for item in batch]
        reason_labels = [lbl + [-100] * (max_length - len(lbl)) for lbl in reason_labels]  # Assuming -100 is the ignore index for labels
        reason_labels = torch.tensor(reason_labels, dtype=torch.long)
    
        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'rank_id_labels': rank_id_labels,
        'reason_labels':reason_labels
        }
    else:
        return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'extra_input_ids': extra_input_ids,
        }
    

def get_dataloader_kwargs(train_config, dataset, tokenizer, mode, model_mode):
    kwargs = {}
    batch_size = train_config.batch_size_training if mode == "train" else train_config.val_batch_size
    if train_config.batching_strategy == "padding":
        if train_config.enable_fsdp:
            kwargs["batch_sampler"] = DistributedLengthBasedBatchSampler(
                dataset,
                batch_size=batch_size,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
            )
        else:
            kwargs["batch_sampler"] = LengthBasedBatchSampler(dataset, batch_size, drop_last=True, shuffle=mode == "train")
        kwargs["collate_fn"] = lambda batch: custom_collate_fn(batch, tokenizer, model_mode)
    elif train_config.batching_strategy == "packing":
        if train_config.enable_fsdp:
            kwargs["sampler"] = DistributedSampler(
                dataset,
                rank=dist.get_rank(),
                num_replicas=dist.get_world_size(),
                shuffle=mode == "train",
                drop_last=True,
            )
        kwargs["batch_size"] = batch_size
        kwargs["drop_last"] = True
        kwargs["collate_fn"] = lambda batch: custom_collate_fn(batch, tokenizer, model_mode)
    else:
        raise ValueError(f"Unknown batching strategy: {train_config.batching_strategy}")

    return kwargs
