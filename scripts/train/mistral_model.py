from transformers import MistralForCausalLM, MistralConfig

class CustomMistralForMultiTask(MistralForCausalLM):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        self.post_init()

        # Define loss functions: listwise and pointwise ranking
        self.rank_listwise_loss_function = ListNetLoss()
        self.rank_pointwise_loss_function = PointwiseMSELoss()

        # Heads for reasoning
        self.explicit_reason_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.listwise_reason_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.rank_head = nn.Linear(config.hidden_size, 1)

        # Initialize dynamic weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        self.gamma = nn.Parameter(torch.tensor(0.5))

    def forward(self, 
                input_ids,
                extra_input_ids=None,
                attention_mask=None,
                labels=None, 
                rank_id_labels=None, 
                explicit_reason_labels=None, 
                listwise_reason_labels=None,
                output_hidden_states=True, 
                output_attentions=True,  
                **kwargs):
        
        inputs_embeds = self.model.embed_tokens(input_ids)
        if extra_input_ids is not None:
            extra_embeds = self.model.embed_tokens(extra_input_ids).to(torch.float16)
            inputs_embeds = inputs_embeds + extra_embeds
        
        if 'inputs_embeds' in kwargs:
            del kwargs['inputs_embeds']
        
        outputs = super().forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,  
            labels=labels,
            **kwargs
        )

        hidden_states = outputs.hidden_states[-1].to(torch.float16) if outputs.hidden_states else None

        # Generate reasoning logits
        explicit_reason_logits = self.explicit_reason_head(hidden_states)
        listwise_reason_logits = self.listwise_reason_head(hidden_states)

        # Generate ranking logits
        rank_logits = self.rank_head(hidden_states).squeeze(-1)

        # Ensure proper shape for pointwise loss
        if rank_logits.dim() > 2:
            rank_logits = rank_logits.view(-1, rank_logits.size(-1))
        if rank_id_labels is not None:
            rank_id_labels = rank_id_labels.view(-1).to(torch.float16)
        
        # Calculate multi-task loss
        loss = outputs.loss.to(torch.float16)
        rank_listwise_loss = self.rank_listwise_loss_function(rank_logits, rank_id_labels) if rank_id_labels is not None else None
        rank_pointwise_loss = self.rank_pointwise_loss_function(rank_logits, rank_id_labels) if rank_id_labels is not None else None
        explicit_reason_loss = self.rank_pointwise_loss_function(explicit_reason_logits.view(-1, explicit_reason_logits.size(-1)), explicit_reason_labels.view(-1)) if explicit_reason_labels is not None else None
        listwise_reason_loss = self.rank_listwise_loss_function(listwise_reason_logits, listwise_reason_labels) if listwise_reason_labels is not None else None

        if all(loss is not None for loss in [rank_listwise_loss, rank_pointwise_loss, explicit_reason_loss, listwise_reason_loss]):
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
