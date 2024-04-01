from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from llasm.model.encoder import *

DEFAULT_ASM_TOKEN = "<asm>"
DEFAULT_ASM_PATCH_TOKEN = "<asm_patch>"
DEFAULT_ASM_START_TOKEN = "<asm_start>"
DEFAULT_ASM_END_TOKEN = "<asm_end>"

class LlasmConfig(LlamaConfig):
    model_type = "llasm"

class LlasmLlamaModel(LlamaModel):
    config_class = LlasmConfig

    def __init__(self, config: LlamaConfig):
        super(LlasmLlamaModel, self).__init__(config)

        if hasattr(config, "mm_encoder"):
            self.encoder = [AsmEncoder.from_pretrained(config.mm_encoder)]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)

    def get_encoder(self):
        encoder = getattr(self, 'encoder', None)
        return encoder
    
    def initialize_asm_module(self, encoder, asm_tokenizer, mm_encoder_select_layer,
                                  pretrain_mm_mlp_adapter=None, fsdp=None):

        self.config.mm_encoder = encoder        
        assemble_processor = AsmTokenizer.from_pretrained(asm_tokenizer)

        if not hasattr(self, 'encoder'):
            encoder = AsmEncoder.from_pretrained(encoder)
        else:
            encoder = self.encoder[0]
        encoder.requires_grad_(False)

        self.encoder = encoder

        encoder_config = encoder.config
        self.config.use_mm_proj = True
        self.config.mm_hidden_size = encoder_config.hidden_size
        self.config.mm_encoder_select_layer = mm_encoder_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(encoder_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            assemble_processor=assemble_processor,
            assemble_token_len=assemble_processor.model_max_length,
            encoder_config=encoder_config
        )   

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        assembles: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        encoder = self.get_encoder()
        if encoder is not None:
            with torch.no_grad():
                if type(assembles) is list:
                    assemble_features = []
                    for assemble in assembles:
                        asm = assemble[0]
                        asm_embeddings = encoder(**asm)
                        if assemble[1] != None:
                            asm = assemble[1]
                            site_embeddings = encoder(**asm)
                        else:
                            site_embeddings = None
                        assemble_features.append((asm_embeddings, site_embeddings))

        
            assemble_features = [(self.mm_projector(assemble_feature[0])[0], self.mm_projector(assemble_feature[1])[0] if assemble_feature[1] != None else assemble_feature[1]) for assemble_feature in assemble_features]

            dummy_assemble_features = torch.zeros(256, self.config.mm_hidden_size, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_assemble_features = self.mm_projector(dummy_assemble_features)
            new_input_embeds = []
            cur_assemble_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds):
                if (cur_input_ids == encoder.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_assemble_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    cur_assemble_idx += 1
                    continue

                cur_assemble_features = assemble_features[cur_assemble_idx]
                if (cur_input_ids == encoder.config.im_start_token).sum() != (cur_input_ids == encoder.config.im_end_token).sum():
                    raise ValueError("The number of assemble start tokens and assemble end tokens should be the same.")
                assemble_start_tokens = torch.where(cur_input_ids == encoder.config.im_start_token)[0]
                for i, assemble_start_token_pos in enumerate(assemble_start_tokens):
                    if assemble_features[cur_assemble_idx][i] == None:
                        continue
                    cur_assemble_features = assemble_features[cur_assemble_idx][i].to(device=cur_input_embeds.device)
                    num_patches = cur_assemble_features.shape[0]
                    if cur_input_ids[assemble_start_token_pos + num_patches + 1] != encoder.config.im_end_token:
                        raise ValueError("The assemble end token should follow the assemble start token.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:assemble_start_token_pos].detach(), cur_input_embeds[assemble_start_token_pos:assemble_start_token_pos+1], cur_assemble_features, cur_input_embeds[assemble_start_token_pos + num_patches + 1:assemble_start_token_pos + num_patches + 2], cur_input_embeds[assemble_start_token_pos + num_patches + 2:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:assemble_start_token_pos+1], cur_assemble_features, cur_input_embeds[assemble_start_token_pos + num_patches + 1:]), dim=0)
                cur_assemble_idx += 1
                new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlasmLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

class LlasmLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlasmConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlasmLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)  
        self.post_init()

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.get_model().get_encoder()

    def get_encoder(self):
        model = self.get_model()
        encoder = model.encoder
        return encoder

    def get_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_rank(self, label):
        labels = []
        self.labels = self.tokenizer.batch_decode(self.tokenizer(label)["input_ids"], skip_special_tokens=True)[1:]
        labels = [x.lower() for x in self.labels]
        self.labels = labels
        self.rank = [0] * (len(self.labels))
        self.predict_list = []
    
    def supervision_loss(self, logits):
        if not hasattr(self, "rank"):
            return
        preds = self.tokenizer.batch_decode(torch.argsort(logits[0][-1], descending=True)[:100], skip_special_tokens=True)
        for i, rank in enumerate(self.rank):
            for j, pred in enumerate(preds):
                pred = pred.lower()
                if pred == self.labels[i]:
                    if 100 - j > rank:
                        self.rank[i] = 100 - j

        self.predict_list.append(0)
        for j, pred in enumerate(preds):
            pred = pred.lower()  
            for i, rank in enumerate(self.rank):
                if pred == self.labels[i]:
                    if 100 - j > self.predict_list[len(self.predict_list) - 1]:
                        self.predict_list[len(self.predict_list) - 1] = 100 - j

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        assembles: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        score: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            assembles=assembles
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        self.supervision_loss(logits)

        loss = loss * score

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "assembles": kwargs.get("assembles", None),
                "score": kwargs.get("score", None),
            }
        )
        return model_inputs

    def initialize_encoder_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        encoder_config = self.get_encoder().config
        encoder_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_ASM_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_ASM_START_TOKEN, DEFAULT_ASM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            encoder_config.im_start_token, encoder_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_ASM_START_TOKEN, DEFAULT_ASM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.get_model().orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        encoder_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_ASM_PATCH_TOKEN])[0]
        self.tokenizer = tokenizer

AutoConfig.register("llasm", LlasmConfig)
AutoModelForCausalLM.register(LlasmConfig, LlasmLlamaForCausalLM)
