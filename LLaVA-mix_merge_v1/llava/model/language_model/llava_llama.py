#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM



from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

# from ..kv_token_merge.modify_llama_2 import H2OLlamaForCausalLM_drop, H2OLlamaAttention_drop, LlamaAttention_drop
from ..kv_token_merge.modify_llama import H2OLlamaAttention_drop, MixMerLlamaAttention_drop, \
                                            PixelPrunMergeAttention, \
                                            PixelPrunMergeDecoderLayer, \
                                            PixelPrunMergeAttentionLeft, \
                                            PixelPrunMergeDecoderLayerLeft, \
                                            PixelPrunMergeLlamaModel, \
                                            WeightedLlamaAttention_drop, \
                                            PivotMergeLlamaAttention_drop, \
                                            TextH2OLlamaAttention_drop, \
                                            TextWeightedLlamaAttention_drop, \
                                            TextPivotLlamaAttention_drop, \
                                            PoolingWindowLlamaAttention_drop, \
                                            AVGMergeLlamaAttention_drop, \
                                            MeanH2OLlamaAttention_drop, \
                                            TextAVGMergeLlamaAttention_drop
                                            # TokenMergeDecoderLayer
                                            # TokenMergeDecoderLayerNext, \
# from ..kv_token_merge.modeling_llama_drop_merge import H2OLlamaAttention_merge
from ..kv_token_merge.v433_modeling_llama import LlamaModel, LlamaForCausalLM

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM

import math
import matplotlib.pyplot as plt
import time

class TimeTrack:
    Prefill_time = 0
    start_prefill_time = 0
    start_decode_time = 0

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)

class PixelPrunMergeLlavaLlamaModel(LlavaMetaModel, PixelPrunMergeLlamaModel):
    config_class = LlavaConfig
    def __init__(self, config):
        super().__init__(config)

KV_DICT = {
    "weighted_merge": MixMerLlamaAttention_drop,
}
## original LLaVA
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config, kv_mode, hh_ratio, recent_ratio):
        super(LlamaForCausalLM, self).__init__(config)
        #################### add here ###########################
        config.hh_size = 200
        config.recent_size = 200
        config.hh_ratio = hh_ratio
        config.recent_ratio = recent_ratio
        # super().__init__(config)
        ########################################################
        # kv-merge
        self.model = LlavaLlamaModel(config)
        self.kv_mode = kv_mode
        num_layers = len(self.model.layers)
        if kv_mode == "weighted_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = WeightedLlamaAttention_drop(config)
        if kv_mode == "pivot_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = PivotMergeLlamaAttention_drop(config)
        if kv_mode == "h2o":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = H2OLlamaAttention_drop(config)
        if kv_mode == "text_prior_h2o":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = TextH2OLlamaAttention_drop(config)
        if kv_mode == "text_prior_weighted_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = TextWeightedLlamaAttention_drop(config)
        if kv_mode == "text_prior_pivot_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = TextPivotLlamaAttention_drop(config)
        if kv_mode == "snapkv":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = PoolingWindowLlamaAttention_drop(config)
        if kv_mode == "avg_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = AVGMergeLlamaAttention_drop(config)
        if kv_mode == "mean_h2o":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = MeanH2OLlamaAttention_drop(config)
        if kv_mode == "text_prior_avg_merge":
            for layer_idx in range(num_layers):
                self.model.layers[layer_idx].self_attn = TextAVGMergeLlamaAttention_drop(config)
        # if kv_mode == "snapkv":
            
        
        # tokenmerge-before-prefill
        # self.model = TokenMergeLlavaLlamaModel(config)
        # self.model.layers[0].self_attn = TokenMergeAttention_drop(config)
        # self.model.layers[0] = TokenMergeDecoderLayer(config)

        # tokenprumerge-in-layer-2
        # self.model = PixelPrunMergeLlavaLlamaModel(config)
        # self.model.layers[21] = PixelPrunMergeDecoderLayer(config)
        # num_layers = len(self.model.layers)
        # for layer_idx in range(22, num_layers):
            # self.model.layers[layer_idx] = PixelPrunMergeDecoderLayerLeft(config)
            
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model
    
    def plain_prune(self, 
                  inputs_embeds, 
                  ):
        recent_len = 150
        seq_len = inputs_embeds.shape[1]
        select_mask = torch.rand(seq_len-recent_len) < 0.5
        recent_mask = torch.ones(recent_len).to(torch.bool)
        full_mask = torch.concat((select_mask, recent_mask), dim=0)
        inputs_embeds = inputs_embeds[:, full_mask, :]
        # attention_mask = torch.ones(1, save_len).to(torch.int64).to(device=inputs_embeds.device)
        # position_ids = torch.tensor(range(0, save_len)).to(torch.int64).unsqueeze(0).to(device=inputs_embeds.device)
        return inputs_embeds
    
    def rotate_half(self, x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q, k, cos, sin, position_ids):
        # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
        cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
        sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed
    
    def hh_prune(self, inputs_embeds):
        recent_len = 150
        bsz, seq_len = inputs_embeds.shape[:2]
        keep_len = int(0.5*(seq_len-recent_len))
        self_attn = self.model.layers[0].self_attn
        num_key_val_heads, head_dim = self_attn.num_key_value_heads, self_attn.head_dim
        
        query_states = self_attn.q_proj(inputs_embeds)
        key_states = self_attn.k_proj(inputs_embeds)
        value_states = self_attn.v_proj(inputs_embeds)
        kv_seq_len = key_states.shape[-2]
        query_states = query_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)

        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        position_ids = torch.tensor(range(seq_len), dtype=torch.int64).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.full((seq_len, seq_len), float(-65504.), dtype=torch.float16, device=query_states.device)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        attn_weight = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
        attn_weight += attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hh_score = torch.mean(attn_weight.sum(0), dim=0).sum(0)
        _, keep_topk = torch.topk(hh_score[:seq_len-recent_len], keep_len, dim=-1)
        keep_topk = keep_topk.sort().values
        keep_recent = position_ids[:,seq_len-recent_len:].squeeze(0)
        keep = torch.cat((keep_topk, keep_recent), dim=0)
        inputs_embeds = inputs_embeds[:,keep,:]
        return inputs_embeds
    def pixel_prune(self, inputs_embeds, img_st_pos, img_end_pos):
        pruned_ratio = 1/4
        img_len = 576
        seqlen = inputs_embeds.shape[1]
        img_keep_mask = torch.rand(img_len) < pruned_ratio
        fore_keep_mask = torch.ones(img_st_pos).to(torch.bool)
        after_keep_mask = torch.ones(seqlen-img_end_pos).to(torch.bool)
        keep_mask = torch.cat((fore_keep_mask, img_keep_mask, after_keep_mask), dim=-1).to(device=inputs_embeds.device)
        inputs_embeds = inputs_embeds[:, keep_mask, :]
        return inputs_embeds
    def pixel_att_prune(self, inputs_embeds, img_st_pos, img_end_pos):
        bsz, seq_len = inputs_embeds.shape[:2]
        self_attn = self.model.layers[0].self_attn
        num_key_val_heads, head_dim = self_attn.num_key_value_heads, self_attn.head_dim
        
        query_states = self_attn.q_proj(inputs_embeds)
        key_states = self_attn.k_proj(inputs_embeds)
        value_states = self_attn.v_proj(inputs_embeds)
        kv_seq_len = key_states.shape[-2]
        query_states = query_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, num_key_val_heads, head_dim).transpose(1, 2)

        cos, sin = self_attn.rotary_emb(value_states, seq_len=kv_seq_len)
        position_ids = torch.tensor(range(seq_len), dtype=torch.int64).unsqueeze(0).to(inputs_embeds.device)
        attention_mask = torch.full((seq_len, seq_len), float(-65504.), dtype=torch.float16, device=query_states.device)
        attention_mask = torch.triu(attention_mask, diagonal=1)
        query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        attn_weight = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self_attn.head_dim)
        attn_weight += attention_mask
        attn_weight = nn.functional.softmax(attn_weight, dim=-1, dtype=torch.float32).to(query_states.dtype)
        hh_score = torch.mean(attn_weight.sum(0), dim=0).sum(0)

        pruned_ratio = 1/4
        img_len = int(pruned_ratio * 576)
        seqlen = inputs_embeds.shape[1]
        _, keep_topk = torch.topk(hh_score[img_st_pos:1+img_end_pos], img_len, dim=-1)
        keep_topk = keep_topk.sort().values
        img_keep_mask = torch.zeros(img_end_pos-img_st_pos).to(torch.bool)
        img_keep_mask[keep_topk] = True
        fore_keep_mask = torch.ones(img_st_pos).to(torch.bool)
        after_keep_mask = torch.ones(seqlen-img_end_pos).to(torch.bool)
        keep_mask = torch.cat((fore_keep_mask, img_keep_mask, after_keep_mask), dim=0).to(inputs_embeds.device)
        return inputs_embeds[:, keep_mask, :]
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        cache_position=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes
            )
        # if not input_ids:
            # inputs_embeds, attention_mask, position_ids = self.plain_prune(inputs_embeds)
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        image_position = torch.where(inputs[-1]==-200)[0]
        # img_end_pos = img_st_pos + torch.tensor(576) #The end is img token
    
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # breakpoint()
        # inputs_embeds = self.plain_prune(inputs_embeds) best
        # inputs_embeds = self.hh_prune(inputs_embeds) second
        # inputs_embeds = self.pixel_att_prune(inputs_embeds, img_st_pos, img_end_pos)
        if self.kv_mode != "origin" and self.kv_mode != "snapkv":
            for layer_idx in range(len(self.model.layers)):
                self.model.layers[layer_idx].self_attn.kv_cache.image_position=image_position
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
