import torch
from torch import nn, Tensor, LongTensor
from transformers.trainer_pt_utils import LabelSmoother
import torch.nn.functional as F
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.modeling_outputs import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# ListNet Loss (Listwise loss function)
class ListNetLoss(nn.Module):
    def __init__(self):
        super(ListNetLoss, self).__init__()

    def forward(self, logits, target):
        logits, target = self._pad_or_truncate(logits, target)
        logits_prob = F.softmax(logits, dim=-1)
        target_prob = F.softmax(target.to(torch.float16), dim=-1)
        loss = F.kl_div(logits_prob.log(), target_prob, reduction='batchmean')
        return loss

    def _pad_or_truncate(self, logits, target):
        min_len = min(logits.size(-1), target.size(-1))
        if logits.size(-1) > min_len:
            logits = logits[..., :min_len]
        elif logits.size(-1) < min_len:
            padding = torch.zeros_like(logits[..., :1]).expand(logits.size(0), logits.size(1), min_len - logits.size(-1))
            logits = torch.cat([logits, padding], dim=-1)

        if target.size(-1) > min_len:
            target = target[..., :min_len]
        elif target.size(-1) < min_len:
            padding = torch.zeros_like(target[..., :1]).expand(target.size(0), target.size(1), min_len - target.size(-1))
            target = torch.cat([target, padding], dim=-1)
        
        return logits, target

# Pointwise MSE Loss
class PointwiseMSELoss(nn.Module):
    def __init__(self):
        super(PointwiseMSELoss, self).__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, logits: Tensor, labels: LongTensor):
        # 将logits的维度从1024降低到10
        logits = F.adaptive_avg_pool1d(logits.unsqueeze(0), output_size=labels.size(-1)).squeeze(0)
        labels.to(torch.float16)
        # 计算均方误差损失
        loss = self.loss_fn(logits, labels)
        return loss

@dataclass
class MultiTaskOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    rank_logits: Optional[torch.FloatTensor] = None
    explicit_reason_logits: Optional[torch.FloatTensor] = None
    listwise_reason_logits: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class CustomLlamaForMultiTask(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        self.post_init()

        # Define loss functions: one for listwise ranking and one for pointwise ranking
        self.rank_listwise_loss_function = ListNetLoss()
        self.rank_pointwise_loss_function = PointwiseMSELoss()
        
        # Separate heads for explicit (direct) and listwise reasoning
        self.explicit_reason_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.listwise_reason_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.rank_head = nn.Linear(config.hidden_size, 1)  # Rank head to predict document scores

        # Initialize dynamic weights for multi-task loss
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, 
                input_ids,
                extra_input_ids=None, 
                attention_mask=None,
                labels=None, 
                rank_id_labels=None, 
                explicit_reason_labels=None,  # Explicit reason labels
                listwise_reason_labels=None,  # Listwise reason labels
                output_hidden_states=True, 
                output_attentions=True,  
                **kwargs):
        # print(f"Input dtype: {input_ids.dtype}")
        # print(f"Model dtype: {next(self.parameters()).dtype}")  # 打印模型的第一个参数的 dtype
        # print(f"LoRA weight dtype: {self.explicit_reason_head.weight.dtype}")  # 打印 LoRA 权重的 dtype

        inputs_embeds = self.model.embed_tokens(input_ids)
        print("inputs_embeds dtype:", inputs_embeds.dtype)
        if extra_input_ids is not None:
            # 其他代码
            extra_embeds = self.model.embed_tokens(extra_input_ids).to(torch.float16)
            inputs_embeds = inputs_embeds + extra_embeds

       

        
        if extra_input_ids is not None:
            target_length = input_ids.size(1)
            padding_size = target_length - extra_input_ids.size(1)
            if padding_size > 0:
                padding = torch.full((extra_input_ids.size(0), padding_size), self.config.pad_token_id, dtype=torch.long).to(extra_input_ids.device)
                extra_input_ids = torch.cat([extra_input_ids, padding], dim=1)

            extra_embeds = self.model.embed_tokens(extra_input_ids).to(torch.float16)
            inputs_embeds = inputs_embeds + extra_embeds

            extra_attention_mask = torch.ones_like(extra_input_ids, dtype=torch.long)
            extra_attention_mask[extra_input_ids == self.config.pad_token_id] = 0
            attention_mask = torch.cat([attention_mask, extra_attention_mask], dim=1)
            
        if 'inputs_embeds' in kwargs:
            del kwargs['inputs_embeds']
        # print("inputs_embed.dtype",inputs_embeds.dtype)
        # print("attention_mask",attention_mask.dtype)
        
        # print("label", labels.dtype)
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,  
            labels=labels,
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else None
        # print("hidden_states dtype:", hidden_states.dtype)
        hidden_states = hidden_states.to(torch.float16)  # Ensure hidden_states are in float16
    
        # Generate reasoning logits
        explicit_reason_logits = self.explicit_reason_head(hidden_states)
        listwise_reason_logits = self.listwise_reason_head(hidden_states)

        # Generate ranking logits
        rank_logits = self.rank_head(hidden_states).squeeze(-1)

        # Ensure logits and labels are properly shaped for pointwise loss
        if rank_logits.dim() > 2:
            rank_logits = rank_logits.view(-1, rank_logits.size(-1))
        if rank_id_labels is not None:
            rank_id_labels = rank_id_labels.view(-1).to(torch.float16)
        
        # Calculate multi-task loss
        loss = outputs.loss
        loss = loss.to(torch.float16)
        rank_listwise_loss = None
        rank_pointwise_loss = None
        explicit_reason_loss = None
        listwise_reason_loss = None

        if rank_id_labels is not None:
            rank_listwise_loss = self.rank_listwise_loss_function(rank_logits, rank_id_labels)
            rank_pointwise_loss = self.rank_pointwise_loss_function(rank_logits, rank_id_labels)
        
        if explicit_reason_labels is not None:
            explicit_reason_loss = self.rank_pointwise_loss_function(explicit_reason_logits.view(-1, explicit_reason_logits.size(-1)), explicit_reason_labels.view(-1))

        if listwise_reason_labels is not None:
            listwise_reason_loss = self.rank_listwise_loss_function(listwise_reason_logits, listwise_reason_labels)

        if rank_listwise_loss is not None and rank_pointwise_loss is not None and explicit_reason_loss is not None and listwise_reason_loss is not None:
            loss = (torch.sigmoid(self.alpha) * rank_listwise_loss +
                    torch.sigmoid(self.beta) * rank_pointwise_loss +
                    torch.sigmoid(self.gamma) * (explicit_reason_loss + listwise_reason_loss))
        elif rank_listwise_loss is not None and rank_pointwise_loss is not None:
            loss = torch.sigmoid(self.alpha) * rank_listwise_loss + torch.sigmoid(self.beta) * rank_pointwise_loss
        elif rank_listwise_loss is not None:
            loss = rank_listwise_loss
        elif explicit_reason_loss is not None:
            loss = explicit_reason_loss
        elif listwise_reason_loss is not None:
            loss = listwise_reason_loss
        
        return MultiTaskOutput(
            loss=loss,
            rank_logits=rank_logits,
            explicit_reason_logits=explicit_reason_logits,
            listwise_reason_logits=listwise_reason_logits,
            attentions=outputs.attentions,
        )
