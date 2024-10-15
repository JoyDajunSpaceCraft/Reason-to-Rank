from multi_task_load_data import get_preprocessed_custom
from llama_model import CustomLlamaForMultiTask
from mistral_model import CustomMistralForMultiTask

from collections import Counter
import os
import argparse
import dataclasses

import random
import torch
import torch.optim as optim
from peft import get_peft_model, PeftModel
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy
)

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import StepLR
from transformers import (
    AutoTokenizer,
    LlamaForCausalLM,   
    LlamaConfig,         
    MistralForCausalLM,  
    MistralConfig,       
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from configs.fsdp import fsdp_config as FSDP_CONFIG
from configs.training import train_config as TRAIN_CONFIG
from configs.quantization import quantization_config  as QUANTIZATION_CONFIG
from custom_utils.custom_concatdataset import ConcatDataset
from llama_recipes.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing
from llama_recipes.utils import fsdp_auto_wrap_policy
from llama_recipes.utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from custom_utils.custom_dataloder_kwargs import get_dataloader_kwargs

from llama_recipes.utils.train_utils import (
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    print_model_size,
    get_policies,
)
from custom_utils.custom_train_utils import train_custom_model
import argparse
from accelerate.utils import is_xpu_available
from warnings import warn

def setup_wandb(train_config, fsdp_config, **kwargs):
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "You are trying to use wandb which is not currently installed. "
            "Please install it using pip install wandb"
        )
    from llama_recipes.configs import wandb_config as WANDB_CONFIG
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    return run

def main(args):
    # Update the configuration for the training and sharding process
    train_config, fsdp_config = TRAIN_CONFIG(), FSDP_CONFIG()
    kwargs = {"model_name":args.model_id}
    update_config((train_config, fsdp_config),**kwargs)
    device = args.device
    model_mode = args.model_mode
    
    # Set the seeds for reproducibility
    if is_xpu_available():
        torch.xpu.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp:
        setup()
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if is_xpu_available():
            torch.xpu.set_device(local_rank)
        elif torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    wandb_run = None
    if train_config.use_wandb:
        if not train_config.enable_fsdp or rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, **kwargs)
    
    # Set quantization configs
    bnb_config = None
    if train_config.quantization:
        if isinstance(train_config.quantization, bool):
            warn("Quantization (--quantization) is a boolean, please specify quantization as '4bit' or '8bit'. Defaulting to '8bit'.", FutureWarning)
            train_config.quantization = "8bit"

        if train_config.quantization == "8bit" and train_config.enable_fsdp:
            raise ValueError("8bit quantization is not supported with FSDP, please use 4bit quantization")

        quant_config = QUANTIZATION_CONFIG()
        update_config(quant_config, **kwargs)
        bnb_config = quant_config.create_bnb_config(train_config.quantization)

    # Load the pre-trained model and setup its configuration
    use_cache = False if train_config.enable_fsdp else None

    if args.model_type == "llama":
        print(f"Loading LLaMA model: {args.model_id}")
        model = CustomLlamaForMultiTask.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    elif args.model_type == "mistral":
        print(f"Loading Mistral model: {args.model_id}")
        model = CustomMistralForMultiTask.from_pretrained(
            args.model_id,
            quantization_config=bnb_config,
            use_cache=use_cache,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            device_map="auto" if train_config.quantization and not train_config.enable_fsdp else None,
            torch_dtype=torch.float16 if train_config.use_fp16 else torch.bfloat16,
        )
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")

    # Load the tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name if train_config.tokenizer_name is None else train_config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    print_model_size(model, train_config, rank if train_config.enable_fsdp else 0)

    if train_config.enable_fsdp and fsdp_config.pure_bf16 and not train_config.quantization:
        model.to(torch.bfloat16)
        
    if train_config.use_peft:
        if train_config.from_peft_checkpoint:
            model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
            peft_config = model.peft_config()
        else:
            peft_config = generate_peft_config(train_config, {})
            model = get_peft_model(model, peft_config)
        if wandb_run:
            wandb_run.config.update(peft_config)
        model.print_trainable_parameters()

    # Setup FSDP if enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:
            freeze_transformer_layers(model, train_config.num_freeze_layers)

        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer if args.model_type == "llama" else MistralDecoderLayer)

        device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
        model = FSDP(
            model,
            auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy,
            sharding_strategy=fsdp_config.sharding_strategy,
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
        )
        
        if fsdp_config.fsdp_activation_checkpointing:            
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable()
            apply_fsdp_checkpointing(model)                      
    else:
        model.to(device)

    # Load datasets
    dataset_train = get_preprocessed_custom({}, tokenizer, args.train_path, model_mode=model_mode)
    dataset_val = get_preprocessed_custom({}, tokenizer, args.eval_path, model_mode=model_mode)
    
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length, model_mode=model_mode)
    
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train", model_mode=model_mode)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length, model_mode=model_mode)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val", model_mode=model_mode)
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # Start the training process
    results = train_custom_model(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp else None,
        rank if train_config.enable_fsdp else None,
        wandb_run,
        model_mode=model_mode,
        device=device,
        model_id=args.model_id,
        datasize=args.datasize
    )
    if not train_config.enable_fsdp or rank == 0:
        [print(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v

                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--device', default="cuda:0", type=str)
    parser.add_argument('--model_mode', default="rank", type=str)
    parser.add_argument('--train_path', default="train_data/msmarco_1000.jsonl", type=str)
    parser.add_argument('--eval_path', default="train_data/validation/listwise_msmarco20_test_5_pid.jsonl", type=str)
    parser.add_argument('--model_id', default="llama3_8B", type=str)
    parser.add_argument('--model_type', default="llama", choices=["llama", "mistral"], type=str, help="Specify the model type to use (llama or mistral)")
    parser.add_argument("--datasize", default="1000", type=str)
    args = parser.parse_args()
    main(args)
