import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from accelerate.utils import is_xpu_available, is_ccl_available
import torch.distributed as dist
import time
from llama_recipes.utils.memory_utils import MemoryTrace
from llama_recipes.utils.train_utils import profile
from custom_utils.custom_evaluation import custom_evaluation
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,  # general model non-sharded, non-flattened params
    LocalStateDictConfig,  # flattened params, usable only by FSDP
    # ShardedStateDictConfig, # un-flattened param but shards, usable by other parallel schemes.
)

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import os
import torch

def save_peft_checkpoint(model, model_path):
    """保存 PEFT 模型或 FSDP 模型"""
    
    # 检查 model_path 是否是一个目录，如果是，则添加一个文件名
    # if os.path.isdir(model_path):
    #     model_path = os.path.join(model_path, "model.pth")
    torch.cuda.synchronize()
    
    # 手动结束模型训练状态
    # fsdp_state = FSDP.get_fsdp_state(model)
    # fsdp_state._training_state = 'IDLE'

    if isinstance(model, FSDP):
        # FSDP 模型处理，确保没有未完成的任务
        torch.cuda.synchronize()  # 确保所有计算任务完成
        try:
            # 保存 FSDP 模型的状态字典，而不是整个模型
            state_dict = model.state_dict()
            torch.save(state_dict, model_path)
            print(f"FSDP 模型的状态字典已保存至 {model_path}")
        except Exception as e:
            print(f"保存 FSDP 模型时发生错误: {e}")
    else:
        # 对于 PEFT 模型，使用 save_pretrained 保存
        try:
            model.save_pretrained(model_path)
            print(f"PEFT 模型已保存至 {model_path}")
        except Exception as e:
            print(f"保存 PEFT 模型时发生错误: {e}")


# def save_peft_checkpoint(model, model_path, fsdp_config=None):
#     """Save pretrained PEFT model with handling for FSDP."""
    
#     if isinstance(model, FSDP):
#         # Ensure we are saving the model's full state dict
#         state_dict_type = fsdp_config.checkpoint_type if fsdp_config else 'full'
        
#         if state_dict_type == 'full':
#             # If we want to save the full model state dict
#             with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
#                 state_dict = model.state_dict()
#                 # Save the state dict normally
#                 torch.save(state_dict, f"{model_path}/pytorch_model.bin")
#                 print(f"Saved full FSDP state dict at {model_path}")
#         elif state_dict_type == 'sharded':
#             # If we want to save sharded model state dict
#             with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
#                 state_dict = model.state_dict()
#                 # Save the state dict
#                 torch.save(state_dict, f"{model_path}/sharded_model.bin")
#                 print(f"Saved sharded FSDP state dict at {model_path}")
#         else:
#             print(f"Unsupported FSDP state dict type: {state_dict_type}")
    
#     else:
#         # For non-FSDP models, save as usual
#         model.save_pretrained(model_path)
#         print(f"Saved non-FSDP model at {model_path}")

def train_custom_model(model, train_dataloader, eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, fsdp_config=None, local_rank=None, rank=None, wandb_run=None, model_mode=None, device="cuda:0",model_id="llama3_8B",datasize="1000"):
    # print("model.dtype", model.dtype)
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        # world_size = int(os.environ["WORLD_SIZE"])
        world_size = 1
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = []
    train_loss = []
    val_prep = []
    val_loss = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir+f"_{model_mode}"):
            os.makedirs(train_config.output_dir+f"_{model_mode}", exist_ok=True)
        metrics_filename = f"{train_config.output_dir}_{model_mode}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = []
        train_step_loss = []
        val_step_loss = []
        val_step_perplexity = []

    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False

    for epoch in range(train_config.num_epochs):
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader) // gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config, local_rank) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank == 0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break

                    input_ids = batch["input_ids"].to(local_rank if train_config.enable_fsdp else device)
                    
                    if model_mode not in ["base_reason", "base_no_reason"]:
                        extra_input_ids = batch["extra_input_ids"].to(local_rank if train_config.enable_fsdp else device)
                        
                    attention_mask = batch["attention_mask"].to(local_rank if train_config.enable_fsdp else device)
                    labels = batch["labels"].to(local_rank if train_config.enable_fsdp else device)
                    if model_mode=="rank":
                        rank_id_labels = batch["rank_id_labels"].to(local_rank if train_config.enable_fsdp else device)
                        
                    else:
                        rank_id_labels = batch["rank_id_labels"].to(local_rank if train_config.enable_fsdp else device)
                        reason_labels = batch["reason_labels"].to(local_rank if train_config.enable_fsdp else device)
                    loss = None
                    
                    with autocast():
                        if model_mode=="rank":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=None, 
                                            attention_mask=attention_mask, 
                                            labels=labels,
                                            output_hidden_states=True, 
                                            rank_id_labels=rank_id_labels,
                                            output_attentions=True)
                            
                        elif model_mode=="reason":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=extra_input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                                            rank_id_labels=rank_id_labels,
                                            reason_labels=reason_labels,
                                            output_hidden_states=True,
                                            output_attentions=True)
                            
                        elif model_mode == "base_reason":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=None,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                                            rank_id_labels=rank_id_labels,
                                            reason_labels=reason_labels,
                                            output_hidden_states=True,
                                            output_attentions=True)
                            
                        elif model_mode == "base_no_reason":
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=None,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                                            rank_id_labels=rank_id_labels,
                                            # reason_labels=reason_labels,
                                            output_hidden_states=True,
                                            output_attentions=True)
                        else:
                            outputs = model(input_ids=input_ids, 
                                            extra_input_ids=extra_input_ids,
                                            attention_mask=attention_mask,
                                            labels=labels, 
                      
                                            output_hidden_states=True,
                                            output_attentions=True)
                        try:
                            loss = outputs["loss"]
                        except Exception as e:
                            loss =outputs.loss
                    loss = loss / gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()

                    if train_config.use_fp16:
                        try:
                            scaler.scale(loss).backward()
                            # fsdp_state = FSDP.get_fsdp_state(model)
                            # print(f"After backward, FSDP state: {fsdp_state._handles[0]._training_state}")
                            if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                                if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                    scaler.unscale_(optimizer)
                                    if train_config.enable_fsdp:
                                        model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                    else:
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                                scaler.step(optimizer)
                                


                                scaler.update()
                                optimizer.zero_grad()  # 确保清除梯度
                                # fsdp_state = FSDP.get_fsdp_state(model)
                                # print(f"After optimizer step, FSDP state: {fsdp_state._handles[0]._training_state}")
                                torch.cuda.synchronize()  # 确保同步设备计算

                                pbar.update(1)
                        except Exception as e:
                            print(e)
                    else:
                        loss.backward()
                        # 打印模型状态
                        
                        # print(f"After backward model: {model._fsdp_handles[0]._training_state}")
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                            optimizer.step()
                            optimizer.zero_grad()  # 确保清除梯度
                            torch.cuda.synchronize()  # 确保同步设备计算
                            pbar.update(1)

                        
                    # optimizer.zero_grad()
                    # torch.cuda.synchronize()
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank == 0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })

                    pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")

                    if train_config.save_metrics:
                        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)

        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank == 0:
            memtrace.print_stats()

        lr_scheduler.step()
        save_peft_checkpoint(model, train_config.output_dir+f"_{model_mode}_{model_id}_{datasize}")
        # return None
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = custom_evaluation(model, train_config, eval_dataloader, local_rank, tokenizer, wandb_run, model_mode=model_mode, device=device)

            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model:
                if train_config.enable_fsdp:
                    dist.barrier()
                if train_config.use_peft:
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print(f"we are about to save the PEFT modules")
                    else:
                        print(f"we are about to save the PEFT modules")
                    
                    save_peft_checkpoint(model, train_config.output_dir+f"_{model_mode}_{model_id}_{datasize}")
                    if train_config.enable_fsdp:
                        if rank == 0:
                            print(f"PEFT modules are saved in {train_config.output_dir}_{model_mode}_{model_id}_{datasize} directory")
                    else:
                        print(f"PEFT modules are saved in {train_config.output_dir}_{model_mode}_{model_id}_{datasize} directory")

                else:
                    if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                        save_model_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
                    elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                        print("Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        save_model_and_optimizer_sharded(model, rank, train_config)
                        if train_config.save_optimizer:
                            save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                            print("Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")

                    if not train_config.use_peft and train_config.save_optimizer:
                        save_optimizer_checkpoint(model, optimizer, rank, train_config, epoch=epoch)
                        print("Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                if train_config.enable_fsdp:
                    dist.barrier()
            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                    print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(best_val_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank == 0:
                print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        if train_config.save_metrics:
            save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"] = TFlops
    if train_config.enable_fsdp and not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results
