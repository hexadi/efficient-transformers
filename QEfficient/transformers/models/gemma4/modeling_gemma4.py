# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import logging
import copy
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4Config,
    Gemma4ForCausalLM,
    Gemma4ForConditionalGeneration,
    Gemma4RMSNorm,
    Gemma4TextAttention,
    Gemma4TextConfig,
    Gemma4TextDecoderLayer,
    Gemma4TextModel,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)

from QEfficient.customop.rms_norm import CustomRMSNorm
from QEfficient.transformers.cache_utils import QEffSlidingWindowCache
from QEfficient.transformers.modeling_attn_mask_utils import _create_causal_mask

logger = logging.getLogger(__name__)
from QEfficient.utils import constants
from QEfficient.utils._utils import IOInfo
from QEfficient.utils.constants import MIN_MASKED_ATTENTION_VALUE


class QEffGemma4RMSNormFunc(torch.autograd.Function):
    @staticmethod
    def forward(hidden_states: torch.Tensor, weight: torch.Tensor, epsilon: float):
        div_first = hidden_states * torch.rsqrt(torch.tensor(hidden_states.shape[-1], dtype=hidden_states.dtype))
        variance = div_first.pow(2).sum(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
        return weight * hidden_states

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        pass

    @staticmethod
    def symbolic(g: torch.Graph, hidden_states: torch.Value, weight: torch.Value, epsilon: torch.Value) -> torch.Value:
        return g.onnxscript_op(CustomRMSNorm, hidden_states, weight, epsilon_f=epsilon).setTypeAs(hidden_states)


class QEffGemma4CustomRMSNormAIC(nn.Module):
    def forward(self, hidden_states):
        if hasattr(self, "weight"):
            weight = self.weight.to(hidden_states.dtype)
        else:
            weight = torch.ones(hidden_states.shape[-1], dtype=hidden_states.dtype, device=hidden_states.device)
        return QEffGemma4RMSNormFunc.apply(
            hidden_states,
            weight,
            self.eps,
        )


class QEffGemma4TextRotaryEmbedding(nn.Module):
    def __init__(self, config: Gemma4TextConfig, device=None):
        super().__init__()
        self.config = config
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.layer_types = set(config.layer_types)
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters.get(layer_type, {}) or {}
            base = rope_params.get("rope_theta", getattr(config, "rope_theta", 10000.0))
            dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            if layer_type == "full_attention" and rope_params.get("rope_type") == "proportional":
                dim = getattr(config, "global_head_dim", dim)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
            self.register_buffer(f"{layer_type}_inv_freq", inv_freq, persistent=False)
            self._set_cos_sin_cache(layer_type, seq_len=self.max_seq_len_cached, device=device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, layer_type, seq_len, device, dtype):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        t = torch.arange(seq_len, device=device, dtype=torch.int64).type_as(inv_freq)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(f"{layer_type}_cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer(f"{layer_type}_sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids, layer_type=None):
        layer_type = layer_type or "full_attention"
        cos_cached = getattr(self, f"{layer_type}_cos_cached", None)
        sin_cached = getattr(self, f"{layer_type}_sin_cached", None)
        if cos_cached is None:
            raise ValueError(f"Unknown layer_type: {layer_type}")
        cos = cos_cached[position_ids]
        sin = sin_cached[position_ids]
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def qeff_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    else:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim ** -0.5
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:
        attn_weights = torch.where(
            attention_mask.bool(),
            torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=module.config.torch_dtype),
            attn_weights,
        )
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class QEffGemma4TextAttention(Gemma4TextAttention):
    def __init__(self, config: Gemma4TextConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.__qeff_init__()

    def __qeff_init__(self):
        if self.is_kv_shared_layer and not hasattr(self, "k_proj"):
            storage_layer = self._find_storage_layer()
            if storage_layer is not None:
                self.k_proj = copy.deepcopy(storage_layer.k_proj)
                self.k_norm = copy.deepcopy(storage_layer.k_norm)
                if storage_layer.v_proj is not None:
                    self.v_proj = copy.deepcopy(storage_layer.v_proj)
                else:
                    self.v_proj = None
                self.v_norm = copy.deepcopy(storage_layer.v_norm)
            else:
                head_dim = self.head_dim
                num_kv_heads = self.config.num_key_value_heads
                self.k_proj = nn.Linear(self.config.hidden_size, num_kv_heads * head_dim, bias=self.config.attention_bias)
                self.k_norm = Gemma4RMSNorm(dim=head_dim, eps=self.config.rms_norm_eps)
                self.v_proj = nn.Linear(self.config.hidden_size, num_kv_heads * head_dim, bias=self.config.attention_bias)
                self.v_norm = Gemma4RMSNorm(head_dim, eps=self.config.rms_norm_eps, with_scale=False)

    def _find_storage_layer(self):
        return getattr(self, "_qeff_storage_layer", None)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        shared_kv_states: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape)
        query_states = self.q_norm(query_states)

        cos, sin = position_embeddings
        query_states = apply_rotary_pos_emb(query_states, cos, sin, unsqueeze_dim=2)
        query_states = query_states.transpose(1, 2)

        if self.is_kv_shared_layer and shared_kv_states is not None and self.layer_type in shared_kv_states:
            key_states, value_states = shared_kv_states[self.layer_type]
            key_states = key_states.to(query_states.device)
            value_states = value_states.to(query_states.device)
        else:
            key_states = self.k_proj(hidden_states).view(hidden_shape)
            value_states = self.v_proj(hidden_states).view(hidden_shape) if self.v_proj is not None else key_states
            key_states = self.k_norm(key_states)
            key_states = apply_rotary_pos_emb(key_states, cos, sin, unsqueeze_dim=2)
            key_states = key_states.transpose(1, 2)
            value_states = self.v_norm(value_states)
            value_states = value_states.transpose(1, 2)

        if past_key_values is not None and not self.is_kv_shared_layer:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "batch_index": batch_index,
                "position_ids": position_ids,
                "is_sliding": self.is_sliding,
                "sliding_window_pattern": getattr(self.config, "_sliding_window_pattern", 5),
                "sliding_window": past_key_values.sliding_window_len if hasattr(past_key_values, "sliding_window_len") else None,
            }
            if comp_ctx_lengths is not None:
                attention_mask = attention_mask[:, :, :, : comp_ctx_lengths.shape[-1]]
                cache_kwargs["CCL"] = attention_mask.shape[-1]
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.store_full_length_kv and shared_kv_states is not None:
            shared_kv_states[self.layer_type] = (key_states, value_states)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if getattr(self.config, "attn_logit_softcapping", None) is not None:
            attn_weights = attn_weights / self.config.attn_logit_softcapping
            attn_weights = torch.tanh(attn_weights)
            attn_weights = attn_weights * self.config.attn_logit_softcapping

        if attention_mask is not None:
            attn_weights = torch.where(
                attention_mask.bool(),
                torch.tensor(MIN_MASKED_ATTENTION_VALUE, dtype=self.config.torch_dtype),
                attn_weights,
            )

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class QEffGemma4TextDecoderLayer(Gemma4TextDecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        per_layer_input: Optional[torch.Tensor] = None,
        shared_kv_states: Optional[Dict[str, Tuple[torch.Tensor, torch.Tensor]]] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if past_key_value is not None:
            if self.self_attn.is_sliding:
                attention_mask = _create_causal_mask(
                    position_ids=position_ids,
                    target_length=past_key_value.sliding_window_len,
                    sliding_window=past_key_value.sliding_window_len,
                )
            else:
                target_length = past_key_value.get_seq_length()
                for i, lt in enumerate(self.config.layer_types):
                    if "sliding" not in lt and i < len(past_key_value.key_cache):
                        target_length = past_key_value.key_cache[i].shape[-2]
                        break
                attention_mask = _create_causal_mask(
                    position_ids=position_ids,
                    target_length=target_length,
                )
        elif attention_mask is None:
            attention_mask = _create_causal_mask(
                position_ids=position_ids,
                target_length=position_ids.shape[-1],
            )

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            cache_position=cache_position,
            shared_kv_states=shared_kv_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        if self.enable_moe_block:
            hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
            hidden_states_flat = residual.reshape(-1, residual.shape[-1])
            _, top_k_weights, top_k_index = self.router(hidden_states_flat)
            hidden_states_2 = self.pre_feedforward_layernorm_2(hidden_states_flat)
            hidden_states_2 = self.experts(hidden_states_2, top_k_index, top_k_weights)
            hidden_states_2 = hidden_states_2.reshape(residual.shape)
            hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
            hidden_states = hidden_states_1 + hidden_states_2

        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        if self.hidden_size_per_layer_input and per_layer_input is not None:
            residual = hidden_states
            hidden_states = self.per_layer_input_gate(hidden_states)
            hidden_states = self.act_fn(hidden_states)
            hidden_states = hidden_states * per_layer_input
            hidden_states = self.per_layer_projection(hidden_states)
            hidden_states = self.post_per_layer_input_norm(hidden_states)
            hidden_states = residual + hidden_states

        hidden_states *= self.layer_scalar
        return hidden_states


class QEffGemma4TextModel(Gemma4TextModel):
    def __qeff_init__(self):
        self.rotary_emb = QEffGemma4TextRotaryEmbedding(config=self.config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        per_layer_inputs: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is None and inputs_embeds is None:
            raise ValueError("You must specify input_ids or inputs_embeds")

        if input_ids is not None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.hidden_size_per_layer_input:
            if per_layer_inputs is None:
                per_layer_inputs = self.get_per_layer_inputs(input_ids, inputs_embeds)
            per_layer_inputs = self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

        if use_cache and not isinstance(past_key_values, Cache):
            past_key_values = QEffSlidingWindowCache.from_legacy_cache(
                config=self.config, past_key_values=past_key_values
            )

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in self.unique_layer_types:
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        shared_kv_states = kwargs.pop("shared_kv_states", {})
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            per_layer_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            layer_type = self.config.layer_types[i]

            hidden_states = decoder_layer(
                hidden_states,
                per_layer_input=per_layer_input,
                shared_kv_states=shared_kv_states,
                position_embeddings=position_embeddings[layer_type],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                comp_ctx_lengths=comp_ctx_lengths,
                batch_index=batch_index,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values.to_legacy_cache() if use_cache else None

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()


class QEffGemma4ForCausalLM(Gemma4ForCausalLM):
    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4TextDecoderLayer}

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        comp_ctx_lengths: Optional[torch.LongTensor] = None,
        batch_index: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states).float()
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        n_heads = config.num_key_value_heads
        local_d_head = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        global_d_head = getattr(config, "global_head_dim", local_d_head)
        sliding_window = min(getattr(config, "sliding_window", seq_len), seq_len)
        layer_types = getattr(config, "layer_types", ["sliding_attention"] * config.num_hidden_layers)
        past_key_values = []
        for i in range(config.num_hidden_layers):
            is_full = layer_types[i] == "full_attention"
            d_head = global_d_head if is_full else local_d_head
            ctx_len = seq_len if is_full else sliding_window
            cache_shape = [batch_size, n_heads, ctx_len, d_head]
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.config.torch_dtype)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.config.torch_dtype)
            past_key_values.append((new_layer_key_cache, new_layer_value_cache))
        return past_key_values


class QEffGemma4VisionEncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.vision_tower = self.model.vision_tower

    def get_submodules_for_export(self) -> Type[nn.Module]:
        if hasattr(self.vision_tower, "encoder") and hasattr(self.vision_tower.encoder, "layers"):
            return {self.vision_tower.encoder.layers[0].__class__}
        return set()

    def forward(self, pixel_values, pixel_position_ids):
        vision_outputs = self.vision_tower(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
        image_features = self.model.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
        return image_features


class QEffGemma4DecoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.language_model = self.model.language_model
        self.config = self.model.config
        self.lm_head = self.model.lm_head

    def get_submodules_for_export(self) -> Type[nn.Module]:
        return {QEffGemma4TextDecoderLayer}

    def forward(
        self,
        input_ids,
        vision_embeds,
        position_ids,
        image_idx,
        past_key_values,
        comp_ctx_lengths: Optional[List[int]] = None,
        batch_index: Optional[torch.LongTensor] = None,
    ):
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        selected = input_ids == self.model.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = vision_embeds.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            batch_index=batch_index,
            use_cache=True,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits, vision_embeds, image_idx, outputs.past_key_values


class QEffGemma4ForConditionalGeneration(Gemma4ForConditionalGeneration):
    def get_qeff_vision_encoder(self):
        return QEffGemma4VisionEncoderWrapper(self)

    def get_qeff_language_decoder(self):
        return QEffGemma4DecoderWrapper(self)

    def forward(
        self,
        input_ids,
        position_ids,
        pixel_values,
        image_idx,
        past_key_values,
        pixel_position_ids: Optional[torch.LongTensor] = None,
        comp_ctx_lengths: Optional[List[int]] = None,
        **kwargs,
    ):
        vision_outputs = self.model.vision_tower(pixel_values=pixel_values, pixel_position_ids=pixel_position_ids)
        image_features = self.model.embed_vision(inputs_embeds=vision_outputs.last_hidden_state)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        B, N, C = inputs_embeds.shape
        selected = input_ids == self.config.image_token_id
        indices1 = selected.to(torch.int64).cumsum(1) - 1
        indices1 = torch.where(indices1 != -1, indices1 + image_idx, indices1)
        indices0 = torch.arange(selected.unsqueeze(0).shape[0]).view(-1, 1)
        image_features_expanded = image_features.reshape(-1, C).unsqueeze(0)[indices0, indices1]
        image_input_embeds = torch.where(selected.unsqueeze(-1), image_features_expanded, inputs_embeds)
        inputs_embeds = torch.where(input_ids.shape[1] == torch.tensor(1), inputs_embeds, image_input_embeds)

        outputs = self.model.language_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            past_key_values=past_key_values,
            comp_ctx_lengths=comp_ctx_lengths,
            use_cache=True,
        )
        image_idx = (indices1.max() + 1).unsqueeze(0).unsqueeze(0)
        logit_index = position_ids.to(torch.int32).argmax(1, keepdim=True)
        hidden_states = outputs[0][torch.arange(position_ids.shape[0]).view(-1, 1), logit_index]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        return logits, pixel_values, image_idx, outputs.past_key_values

    def get_specializations(
        self,
        batch_size: int,
        prefill_seq_len: int,
        ctx_len: int,
        img_size: int = None,
        comp_ctx_lengths_prefill: Optional[List[int]] = None,
        comp_ctx_lengths_decode: Optional[List[int]] = None,
        kv_offload: bool = False,
        continuous_batching: bool = False,
        kv_cache_batch_size: Optional[int] = None,
        full_batch_size: Optional[int] = None,
        **compiler_options,
    ):
        prefill_seq_len = prefill_seq_len if prefill_seq_len else 32
        ctx_len = ctx_len if ctx_len else constants.INTERN_CTX_LEN
        if img_size is None and hasattr(self.config.vision_config, "image_size"):
            img_size = getattr(self.config.vision_config, "image_size")
        elif img_size is None:
            img_size = 896
            logger.warning("Setting img_size to 896, as it was neither passed nor found in vision_config")
        mm_tokens_per_image = getattr(self.config, "mm_tokens_per_image", 256)

        vision = [
            {
                "batch_size": batch_size,
                "img_size": img_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
            }
        ]

        if comp_ctx_lengths_prefill and comp_ctx_lengths_decode:
            lang = []
            for i in range(0, len(comp_ctx_lengths_prefill)):
                lang_prefill = {
                    "batch_size": 1 if continuous_batching else batch_size,
                    "seq_len": prefill_seq_len,
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_prefill[i],
                    "sliding_window": self.config.text_config.sliding_window,
                    "img_size": img_size,
                    "mm_tokens_per_image": mm_tokens_per_image,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_prefill["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_prefill["batch_size"] = kv_cache_batch_size
                if full_batch_size:
                    lang_prefill["full_batch_exec_size"] = full_batch_size
                lang.append(lang_prefill)

            for i in range(0, len(comp_ctx_lengths_decode)):
                lang_decode = {
                    "batch_size": full_batch_size if continuous_batching else batch_size,
                    "seq_len": "1",
                    "ctx_len": ctx_len,
                    "comp_ctx_lengths": comp_ctx_lengths_decode[i],
                    "sliding_window": self.config.text_config.sliding_window,
                    "img_size": img_size,
                    "mm_tokens_per_image": mm_tokens_per_image,
                    "vision_batch_size": batch_size,
                }
                if continuous_batching:
                    lang_decode["full_batch_size"] = kv_cache_batch_size
                else:
                    lang_decode["batch_size"] = kv_cache_batch_size
                lang.append(lang_decode)
        else:
            lang_prefill = {
                "batch_size": 1 if continuous_batching else batch_size,
                "seq_len": prefill_seq_len,
                "ctx_len": ctx_len,
                "sliding_window": self.config.text_config.sliding_window,
                "img_size": img_size,
                "mm_tokens_per_image": mm_tokens_per_image,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_prefill["full_batch_size"] = kv_cache_batch_size
            else:
                lang_prefill["batch_size"] = kv_cache_batch_size
            if full_batch_size:
                lang_prefill["full_batch_exec_size"] = full_batch_size

            lang_decode = {
                "batch_size": full_batch_size if continuous_batching else batch_size,
                "seq_len": "1",
                "ctx_len": ctx_len,
                "sliding_window": self.config.text_config.sliding_window,
                "img_size": img_size,
                "mm_tokens_per_image": mm_tokens_per_image,
                "vision_batch_size": batch_size,
            }
            if continuous_batching:
                lang_decode["full_batch_size"] = kv_cache_batch_size
            else:
                lang_decode["batch_size"] = kv_cache_batch_size
            lang = [lang_prefill, lang_decode]

        specializations = {}
        if kv_offload:
            specializations["vision"] = vision
            specializations["lang"] = lang
            return specializations, compiler_options
        else:
            return lang, compiler_options

    def get_onnx_dynamic_axes(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        vision_dynamic_axes = {}
        lang_dynamic_axes = {}
        lang_dynamic_axes["input_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["position_ids"] = {0: "batch_size", 1: "seq_len"}
        lang_dynamic_axes["vision_embeds"] = {0: "vision_batch_size", 1: "mm_tokens_per_image"}
        if continuous_batching:
            lang_dynamic_axes["batch_index"] = {0: "batch_size"}
        vision_dynamic_axes["pixel_values"] = {0: "batch_size", 1: "num_patches"}
        if getattr(self.config.vision_config, "use_position_embeddings", True):
            vision_dynamic_axes["pixel_position_ids"] = {0: "batch_size", 1: "num_patches"}

        pkv_dynamic_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "ctx_len"}
        pkv_dynamic_sliding_axes = {0: "full_batch_size" if continuous_batching else "batch_size", 2: "sliding_window"}
        layer_types = getattr(self.config.text_config, "layer_types", ["sliding_attention"] * self.config.text_config.num_hidden_layers)
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                apply_dynamic_axes = pkv_dynamic_axes if layer_types[i] == "full_attention" else pkv_dynamic_sliding_axes
                lang_dynamic_axes[f"past_{kv}.{i}"] = apply_dynamic_axes

        if comp_ctx_lengths is not None:
            lang_dynamic_axes["comp_ctx_lengths"] = {0: "comp_ctx_lengths"}

        dynamic_axes = {}
        if kv_offload:
            dynamic_axes["vision"] = vision_dynamic_axes
            dynamic_axes["lang"] = lang_dynamic_axes
        else:
            dynamic_axes = {**vision_dynamic_axes, **lang_dynamic_axes}
        return dynamic_axes

    def get_output_names(self, kv_offload: bool = False):
        vision_output_names = ["vision_embeds"]
        lang_output_names = ["logits"]
        for i in range(self.config.text_config.num_hidden_layers):
            for kv in ["key", "value"]:
                lang_output_names.append(f"past_{kv}.{i}_RetainedState")

        output_names = {}
        if kv_offload:
            lang_output_names.insert(1, "vision_embeds_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            output_names["vision"] = vision_output_names
            output_names["lang"] = lang_output_names
        else:
            lang_output_names.insert(1, "pixel_values_RetainedState")
            lang_output_names.insert(2, "image_idx_output")
            return lang_output_names
        return output_names

    def get_dummy_pkv_cache(self, config, batch_size, seq_len):
        return QEffGemma4ForCausalLM.get_dummy_pkv_cache(self, config, batch_size, seq_len)

    def get_dummy_inputs(
        self, comp_ctx_lengths: Optional[List[int]] = None, kv_offload: bool = False, continuous_batching: bool = False
    ):
        if vis_cfg := getattr(self.config, "vision_config", None):
            img_size = getattr(vis_cfg, "image_size", 896)
        else:
            img_size = 896
        mm_tokens_per_image = getattr(self.config, "mm_tokens_per_image", 256)

        inputs_shapes = {}
        inputs_shapes["input_ids"] = (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
        inputs_shapes["vision_embeds"] = (1, mm_tokens_per_image, self.config.text_config.hidden_size)
        inputs_shapes["position_ids"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )
        inputs_shapes["pixel_values"] = (
            constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE,
            252,
            self.config.vision_config.hidden_size,
        )
        inputs_shapes["image_idx"] = (1, 1)

        vision_inputs = {}
        lang_inputs = {}
        vision_inputs["pixel_values"] = torch.zeros((inputs_shapes["pixel_values"]), dtype=self.config.torch_dtype)
        if getattr(self.config.vision_config, "use_position_embeddings", True):
            vision_inputs["pixel_position_ids"] = torch.zeros(
                (constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 252, 2), dtype=torch.int64
            )
        lang_inputs["input_ids"] = torch.zeros((inputs_shapes["input_ids"]), dtype=torch.int64)
        lang_inputs["vision_embeds"] = torch.zeros((inputs_shapes["vision_embeds"]), dtype=self.config.torch_dtype)
        lang_inputs["position_ids"] = (
            torch.arange(constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN, dtype=torch.int64)
            .view(1, constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN)
            .repeat(constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE, 1)
        )
        lang_inputs["image_idx"] = torch.zeros((inputs_shapes["image_idx"]), dtype=torch.int64)

        bs = constants.ONNX_EXPORT_EXAMPLE_BATCH_SIZE
        fbs = constants.ONNX_EXPORT_EXAMPLE_FBS

        lang_inputs["past_key_values"] = self.get_dummy_pkv_cache(
            config=self.config.text_config,
            batch_size=fbs if continuous_batching else bs,
            seq_len=constants.ONNX_EXPORT_EXAMPLE_SEQ_LEN,
        )

        if comp_ctx_lengths is not None:
            lang_inputs["comp_ctx_lengths"] = torch.randint(0, 100, (40,), dtype=torch.int8)
        if continuous_batching:
            lang_inputs["batch_index"] = torch.arange(bs).view(bs, 1)

        inputs = {}
        if kv_offload:
            inputs["vision"] = vision_inputs
            inputs["lang"] = lang_inputs
        else:
            lang_inputs.pop("vision_embeds")
            inputs = {**vision_inputs, **lang_inputs}

        return inputs

    def get_inputs_info(self):
        return [
            IOInfo(name="input_ids", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(name="attention_mask", datatype=torch.int64, shape=("batch_size", "seq_len")),
            IOInfo(
                name="pixel_values",
                datatype=self.config.torch_dtype,
                shape=("batch_size", 3, "img_size", "img_size"),
            ),
        ]
