import torch
from torch import nn
from transformers import AutoModel, AutoConfig


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class CustomModel(nn.Module):
    def __init__(self,
                 llm_model_path,
                 cfg,
                 num_classes):
        super(CustomModel, self).__init__()
        # Path to model flat files
        self.llm_model_path = llm_model_path

        # Custom config information for the model specified in YAML file
        self.cfg = cfg

        # Number of classes / labels
        self.num_classes = num_classes

        # HF AutoConfig
        self.llm_model_config = AutoConfig.from_pretrained(llm_model_path)

        # HF AutoModel
        self.llm_model = AutoModel.from_config(self.llm_model_config)

        # Freeze Layers if Specified
        # https://discuss.huggingface.co/t/how-to-freeze-some-layers-of-bertmodel/917/4
        if cfg.freeze.apply:
            # Use modules to specify order
            modules = [self.llm_model.embeddings,
                       self.llm_model.encoder.layer[:cfg.freeze.num_layers]]
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False

        # Gradient checkpointing
        if self.cfg.gradient_checkpointing:
            self.llm_model.gradient_checkpointing_enable()

        # Mean Pooling [TODO more testing needed here]
        self.pool = MeanPooling()

        # Dense layer for classification and weight initialization
        self.fc = nn.Linear(self.llm_model_config.hidden_size, num_classes)
        self._init_weights(self.fc)


    def _init_weights(self, module):
        "Initialize weights for classification weights for dense layer"
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0,
                                       std=self.llm_model_config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0,
                                       std=self.llm_model_config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, inputs):
        # Outputs from model
        outputs = self.llm_model(**inputs)
        last_hidden_states = outputs[0]
        # Pooling
        if self.cfg.pooling.apply:
            feature = self.pool(last_hidden_states, inputs['attention_mask'])
        else:
            feature = last_hidden_states
        output = self.fc(feature)
        return output
