import torch
import torch.utils.checkpoint
from torch import nn
from typing import Optional
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import MPNetTokenizerFast
from transformers import BatchEncoding
from transformers.models.roformer.modeling_roformer import (
    RoFormerEmbeddings,
    RoFormerModel,
    RoFormerEncoder,
    RoFormerLayer,
    RoFormerAttention,
    RoFormerIntermediate,
    RoFormerOutput,
    RoFormerSelfAttention,
    RoFormerPreTrainedModel
)

from transformers.models.mpnet.modeling_mpnet import MPNetModel

from accelerate.logging import get_logger

logger = get_logger(__name__)

class JRoFormerEmbeddings(RoFormerEmbeddings):
    """Construct the embeddings from word and token_type embeddings."""

    def __init__(self, config):
        super().__init__(config)
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id
        )
        self.token_type_embeddings = self.word_embeddings


class JRoFormerSelfAttention(RoFormerSelfAttention):
    def __init__(self, config):
        super().__init__(config)
        self.query = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.key = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )
        self.value = nn.Linear(
            config.hidden_size, self.all_head_size, bias=config.use_bias
        )


class JRoFormerAttention(RoFormerAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = JRoFormerSelfAttention(config)


class JRoFormerLayer(RoFormerLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = JRoFormerAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(
                    f"{self} should be used as a decoder model if cross attention is added"
                )
            self.crossattention = RoFormerAttention(config)
        self.intermediate = RoFormerIntermediate(config)
        self.output = RoFormerOutput(config)


class JRoFormerEncoder(RoFormerEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layer = nn.ModuleList(
            [JRoFormerLayer(config) for _ in range(config.num_hidden_layers)]
        )

class JRoFormerModel(RoFormerModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = JRoFormerEmbeddings(config)

        if config.embedding_size != config.hidden_size:
            self.embeddings_project = nn.Linear(
                config.embedding_size, config.hidden_size
            )

        self.encoder = JRoFormerEncoder(config)
        self.post_init()

class AsmEncoder(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roformer = JRoFormerModel(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        self.attention_mask = attention_mask
        token_embeddings = outputs[0]
        return token_embeddings


class AsmTokenizer(MPNetTokenizerFast):

    @property
    def pad_token_type_id(self) -> int:
        """
        `int`: Id of the padding token type in the vocabulary.
        """
        return self.pad_token_id

    def tokenize_function(self, function, model_max_length=1024):
            #self.model_max_length = 1024
            total_len = 0
            tokenized_functions = {"token": [], "instr": []}
            for key, value in function.items():
                tokens = self.tokenize(value.replace(',', ''), max_length=20, truncation=True, add_special_tokens=False)
                instr_index = "INSTR" + key
                instructions = [instr_index] * len(tokens)
                tokenized_functions["token"].extend(tokens)
                tokenized_functions["instr"].extend(instructions)
                total_len += len(tokens)
                if total_len > model_max_length:
                    tokenized_functions['token'] = tokenized_functions['token'][:model_max_length]
                    tokenized_functions['instr'] = tokenized_functions['instr'][:model_max_length]
                    break
            return tokenized_functions

    def encode_function(self, function, model_max_length=1024):
        tokenized_functions = self.tokenize_function(function, model_max_length)
        token_ids = self.convert_tokens_to_ids(tokenized_functions["token"])
        instr_ids = self.convert_tokens_to_ids(tokenized_functions["instr"])
        return BatchEncoding({
            "input_ids": token_ids,
            "attention_mask": [1] * len(token_ids),
            "token_type_ids": instr_ids,
        })

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


class TextEncoder(MPNetModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ):
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        token_embeddings = output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        text_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_embedding = F.normalize(text_embedding, p=2, dim=1)
        return text_embedding


class Supervisor(RoFormerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.roformer = JRoFormerModel(config)
        self.projection = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        token_embeddings = outputs[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        asm_embedding = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        asm_embedding = self.projection(asm_embedding)
        asm_embedding = F.normalize(asm_embedding, p=2, dim=1)
        return asm_embedding

class BertRegression(nn.Module):
    def __init__(self, in_dim):
        super(BertRegression, self).__init__() 
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activation = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

    def forward(self, x):
        out = torch.tanh(self.fc1(x))
        out = torch.tanh(self.fc2(out))
        out = self.fc3(out)
        out = self.activation(out)
        return out
    

if __name__ == "__main__":
    import json
    tokenzier = AsmTokenizer.from_pretrained("/home/szh/subject/naming/LLMs/llasm/encoder/llasm/tokenizer")
    for data in load_from_disk("/mnt/data/szh/datasets/llasm/real_world/mirai_datasets"):
        asm = tokenzier.encode_function(json.loads(data["clap"]))
        print(asm)
        