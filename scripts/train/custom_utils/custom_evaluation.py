from contextlib import nullcontext
from pathlib import Path
from datetime import datetime
import contextlib
from tqdm import tqdm
from contextlib import nullcontext
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn as nn
from sklearn.metrics import ndcg_score
import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import json


from llama_recipes.model_checkpointing import save_model_checkpoint, save_model_and_optimizer_sharded, save_optimizer_checkpoint
from llama_recipes.policies import fpSixteen,bfSixteen, get_llama_wrapper
from llama_recipes.utils.memory_utils import MemoryTrace
from accelerate.utils import is_xpu_available, is_ccl_available
from llama_recipes.utils.flop_utils import FlopMeasure

def custom_evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, model_mode=None, device="cuda:0"):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, listwise_embedding_similarities
    """
    if train_config.enable_fsdp:
        # world_size = int(os.environ["WORLD_SIZE"])
        world_size =  1
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    listwise_embedding_similarities = []

    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            total_eval_steps += 1
            # Stop when the maximum number of eval steps is reached
            if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                if not train_config.enable_fsdp or local_rank == 0:
                    print("Max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                break

            for key in batch.keys():
                # batch[key] = batch[key].to(local_rank if train_config.enable_fsdp else 'cuda:0')
                batch[key] = batch[key].to(local_rank if train_config.enable_fsdp else device)
                

            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                if model_mode=="rank":
                # Forward pass and compute loss
                    outputs = model(input_ids=batch["input_ids"], extra_input_ids=batch["extra_input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], rank_id_labels = batch["rank_id_labels"],output_hidden_states=True)
                    loss = outputs.loss
                    # listwise_reason_embedding = outputs["listwise_reason_embedding"]
                elif model_mode=="reason":
                    outputs = model(input_ids=batch["input_ids"], extra_input_ids=batch["extra_input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], rank_id_labels = batch["rank_id_labels"],reason_labels = batch["reason_labels"],output_hidden_states=True)
                    loss = outputs.loss
                elif model_mode == "base_reason":
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], rank_id_labels = batch["rank_id_labels"],reason_labels = batch["reason_labels"],output_hidden_states=True)
                    loss = outputs.loss
                elif model_mode == "base_no_reason":
                    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], rank_id_labels = batch["rank_id_labels"],reason_labels = batch["reason_labels"],output_hidden_states=True)
                    loss = outputs.loss
                else:
                    outputs = model(input_ids=batch["input_ids"], extra_input_ids=batch["extra_input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"], output_hidden_states=True)
                    loss = outputs["loss"]
                    listwise_reason_embedding = outputs["listwise_reason_embedding"]
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))
                
                eval_loss += loss.detach().float()


    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss / world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank == 0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log({
            'eval/perplexity': eval_ppl,
            'eval/loss': eval_epoch_loss,
        }, commit=False)

    # Calculate the average similarity score across all batches
    # avg_similarity = np.mean(listwise_embedding_similarities)
    # print(f"Average Listwise Embedding Similarity: {avg_similarity}")

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
