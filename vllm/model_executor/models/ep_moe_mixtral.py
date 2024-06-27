# coding=utf-8
# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Mixtral model."""
from typing import Iterable, List, Optional, Tuple

import torch
from torch import nn
import torch.distributed
from transformers import MixtralConfig

from vllm import _custom_ops as ops
from vllm.attention import Attention, AttentionMetadata
from vllm.config import CacheConfig, LoRAConfig
from vllm.distributed import (get_tensor_model_parallel_world_size,
                              get_expert_parallel_rank,
                              get_expert_parallel_world_size,
                              get_expert_parallel_group)
from vllm.model_executor.layers.fused_moe import fused_moe, fused_moe_ep, fused_topk
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (QKVParallelLinear,
                                               ReplicatedLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig)
from vllm.model_executor.layers.quantization.fp8 import Fp8Config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE, ParallelLMHead, VocabParallelEmbedding)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.sequence import SamplerOutput
import torch.nn.functional as F
from vllm.utils import print_warning_once
import time

# einsum rewrites are on par or more performant
# switch can be bubbled up in future
USE_EINSUM = True

# einsum dimensions: (g)roup, (s)equence, (e)xpert, (m)odel, (c)apacity
# See https://arxiv.org/pdf/2006.16668.pdf for details.
# Copied From https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/moe/sharded_moe.py#L118
def einsum(rule, a, b):
    if USE_EINSUM:
        return torch.einsum(rule, a, b)
    elif rule == 's,se->se':
        return a.reshape(a.shape[0], -1) * b
    elif rule == 'se,sc->sec':
        return a.unsqueeze(2) * b.unsqueeze(1)
    elif rule == 'se,se->s':
        return torch.bmm(a.unsqueeze(1), b.unsqueeze(2)).reshape(-1)
    elif rule == 'sec,sm->ecm':
        s = a.shape[0]
        e = a.shape[1]
        c = a.shape[2]
        m = b.shape[1]
        return torch.matmul(a.reshape(s, -1).t(), b).reshape(e, c, m)
    elif rule == 'sec,ecm->sm':
        return torch.matmul(a.reshape(a.shape[0], -1), b.reshape(-1, b.shape[-1]))
    elif rule == 'ks,ksm->sm':
        k = b.shape[0]
        s = b.shape[1]
        m = b.shape[2]
        # [k, s] -> [s, k] -> [s, 1, k]
        a = a.t().unsqueeze(1)
        # [k,s,m] -> [k, sm] -> [sm, k] -> [s, m, k]
        b = b.reshape(k, -1).t().reshape(s, m, k)
        # bmm([s, 1, k], [s, m, k]^t) -> [s, m, 1]
        return torch.bmm(a, b.transpose(1, 2)).squeeze(2)
    else:
        return torch.einsum(rule, a, b)

def gate_and_alltoall(router_logits, hidden_states, top_k):
    """In this function, we borrowed the implementation of deepspeed MoE inference.
    This is the first all-to-all, which is before Experts computing.
    """
    num_total_experts = router_logits.shape[1]
    hidden_size = hidden_states.shape[1]
    topk_weights, topk_ids = fused_topk(hidden_states, router_logits, top_k, renormalize=True)
    topk_ids = topk_ids.to(torch.int64)
    mask1 = F.one_hot(topk_ids[:, 0], num_classes=num_total_experts)
    mask2 = F.one_hot(topk_ids[:, 1], num_classes=num_total_experts)
    
    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    
    exp_counts = torch.sum(mask1 + mask2, dim=0)
    new_capacity = torch.max(exp_counts)
    torch.distributed.all_reduce(new_capacity, op=torch.distributed.ReduceOp.MAX, group=get_expert_parallel_group())
    capacity = new_capacity
    # print(f"capacity: {capacity}")
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    
    # Calculate combine_weights and dispatch_mask
    gates1_se = einsum("s,se->se", topk_weights[:, 0], mask1.float())
    gates2_se = einsum("s,se->se", topk_weights[:, 1], mask2.float())
    locations1_sc = F.one_hot(locations1_s, capacity).float()
    locations2_sc = F.one_hot(locations2_s, capacity).float()
    combine1_sec = einsum("se,sc->sec", gates1_se, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2_se, locations2_sc)
    combine_weights = combine1_sec + combine2_sec # [s, e, c]
    dispatch_mask = combine_weights.bool()

    # build tensor for comm
    dispatched_hidden_states = einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), hidden_states)
    dispatched_input = torch.ones_like(dispatched_hidden_states) # [e, c, m]
    
    torch.distributed.all_to_all_single(dispatched_input, dispatched_hidden_states, group=get_expert_parallel_group())
    
    return dispatched_input, combine_weights

def gate_and_alltoall_sparse(router_logits, hidden_states, top_k):
    """Sparse implementation of all-to-all.
    This is the first all-to-all, which is before Experts computing.
    """
    num_total_experts = router_logits.shape[1]
    hidden_size = hidden_states.shape[1]
    topk_weights, topk_ids = fused_topk(hidden_states, router_logits, top_k, renormalize=True)
    topk_ids = topk_ids.to(torch.int64)
    mask1 = F.one_hot(topk_ids[:, 0], num_classes=num_total_experts)
    mask2 = F.one_hot(topk_ids[:, 1], num_classes=num_total_experts)
    
    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1
    locations2 = torch.cumsum(mask2, dim=0) - 1
    # Update 2nd's location by accounting for locations of 1st
    locations2 += torch.sum(mask1, dim=0, keepdim=True)
    
    exp_counts = torch.sum(mask1 + mask2, dim=0)
    new_exp_cnt = torch.empty_like(exp_counts)
    torch.distributed.all_to_all_single(new_exp_cnt, exp_counts, group=get_expert_parallel_group())

    
    
    # print(f"capacity: {capacity}")
    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)
    locations2_s = torch.sum(locations2 * mask2, dim=1)
    
    # Calculate combine_weights and dispatch_mask
    gates1_se = einsum("s,se->se", topk_weights[:, 0], mask1.float())
    gates2_se = einsum("s,se->se", topk_weights[:, 1], mask2.float())
    locations1_sc = F.one_hot(locations1_s, capacity).float()
    locations2_sc = F.one_hot(locations2_s, capacity).float()
    combine1_sec = einsum("se,sc->sec", gates1_se, locations1_sc)
    combine2_sec = einsum("se,sc->sec", gates2_se, locations2_sc)
    combine_weights = combine1_sec + combine2_sec # [s, e, c]
    dispatch_mask = combine_weights.bool()

    # build tensor for comm
    dispatched_hidden_states = einsum("sec,sm->ecm", dispatch_mask.type_as(hidden_states), hidden_states)
    dispatched_input = torch.ones_like(dispatched_hidden_states) # [e, c, m]
    
    torch.distributed.all_to_all_single(dispatched_input, dispatched_hidden_states, group=get_expert_parallel_group())
    
    return dispatched_input, combine_weights

class EPMixtralMoE(nn.Module):
    """A expert-parallel MoE implementation for Mixtral that dispatches all expert
    to all ranks.

    Each rank will hold some experts and a fused MoE
    kernel is used for the forward pass.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_total_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        use_padding: Optional[bool] = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        self.use_padding = use_padding
        self.num_local_experts = num_local_experts
        self.num_total_experts = num_total_experts
        # arguments for expert-parallel

        self.ep_rank = get_expert_parallel_rank()
        self.experts = [i for i in range(self.ep_rank * self.num_local_experts, (self.ep_rank + 1) * self.num_local_experts)] # not use at this moment
        
        self.top_k = top_k
        self.hidden_size = hidden_size
        # self.intermediate_size = intermediate_size // self.tp_size
        self.intermediate_size = intermediate_size
        self.quant_config = quant_config

        # FIXME(pcmoritz): Make this more general to support different
        # quantization schemes
        self.use_fp8 = isinstance(quant_config, Fp8Config)

        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype

        # Gate always runs at half / full precision for now.
        self.gate = ReplicatedLinear(self.hidden_size,
                                     self.num_total_experts,
                                     bias=False,
                                     params_dtype=self.params_dtype,
                                     quant_config=None)

        if self.use_fp8 and self.quant_config.is_checkpoint_fp8_serialized:
            params_dtype = torch.float8_e4m3fn

        self.w13_weight = nn.Parameter(
            torch.empty(self.num_local_experts,
                        2 * self.intermediate_size,
                        self.hidden_size,
                        dtype=params_dtype))
        self.w2_weight = nn.Parameter(
            torch.empty(self.num_local_experts,
                        self.hidden_size,
                        self.intermediate_size,
                        dtype=params_dtype))

        set_weight_attrs(self.w13_weight, {
            "weight_loader": self.weight_loader,
        })
        set_weight_attrs(self.w2_weight, {
            "weight_loader": self.weight_loader,
        })

        # Used for fp8.
        self.w13_scale = None
        self.w2_scale = None
        self.a13_scale = None
        self.a2_scale = None

        if self.use_fp8:
            # WEIGHT_SCALE (for fp8)
            self.w13_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                     dtype=torch.float32),
                                          requires_grad=False)
            self.w2_scale = nn.Parameter(torch.ones(self.num_total_experts,
                                                    dtype=torch.float32),
                                         requires_grad=False)

            # If loading fp8 checkpoint, pass the weight loaders.
            # If loading an fp16 checkpoint, do not (we will quantize in
            #   process_weights_after_loading()
            if quant_config.is_checkpoint_fp8_serialized:
                set_weight_attrs(self.w13_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.w2_scale, {
                    "weight_loader": self.weight_loader,
                })

            # ACT_SCALE (for fp8)
            if quant_config.activation_scheme == "static":
                if not quant_config.is_checkpoint_fp8_serialized:
                    raise ValueError(
                        "Found static activation scheme for checkpoint that "
                        "was not serialized fp8.")
                self.a13_scale = nn.Parameter(torch.zeros(
                    self.num_total_experts, dtype=torch.float32),
                                              requires_grad=False)
                self.a2_scale = nn.Parameter(torch.zeros(
                    self.num_total_experts, dtype=torch.float32),
                                             requires_grad=False)

                set_weight_attrs(self.a13_scale, {
                    "weight_loader": self.weight_loader,
                })
                set_weight_attrs(self.a2_scale, {
                    "weight_loader": self.weight_loader,
                })

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor,
                      weight_name: str, expert_id: int):
        raise NotImplementedError
        # tp_rank = get_tensor_model_parallel_rank()
        # param_data = param.data
        # shard_size = self.intermediate_size
        # shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        # if weight_name.endswith("w1.weight"):
        #     param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        # if weight_name.endswith("w3.weight"):
        #     param_data[expert_id,
        #                shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        # if weight_name.endswith("w2.weight"):
        #     param_data[expert_id, :, :] = loaded_weight[:, shard]
        # if "act_scale" in weight_name or "weight_scale" in weight_name:
        #     param_data[expert_id] = loaded_weight

    def process_weights_after_loading(self):
        # Fp8 is the only case where we need to process after loading.
        if not self.use_fp8:
            return

        # If checkpoint is fp16, quantize here.
        if not self.quant_config.is_checkpoint_fp8_serialized:
            w13_weight = torch.empty_like(self.w13_weight.data,
                                          dtype=torch.float8_e4m3fn)
            w2_weight = torch.empty_like(self.w2_weight.data,
                                         dtype=torch.float8_e4m3fn)
            for expert in range(self.num_total_experts):
                w13_weight[expert, :, :], self.w13_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w13_weight.data[expert, :, :])
                w2_weight[expert, :, :], self.w2_scale[
                    expert] = ops.scaled_fp8_quant(
                        self.w2_weight.data[expert, :, :])
            self.w13_weight = nn.Parameter(w13_weight, requires_grad=False)
            self.w2_weight = nn.Parameter(w2_weight, requires_grad=False)

        # If checkpoint is fp8 + static, cleanup act_scales.
        #   Since state_dict has an act_scale per expert but our kernels
        #   are passed one act_scale shared across all experts.
        elif self.quant_config.activation_scheme == "static":
            if self.a13_scale is None or self.a2_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None.")

            if (not all_close_1d(self.a13_scale)
                    or not all_close_1d(self.a2_scale)):
                print_warning_once(
                    "Found act_scales that are not equal for fp8 MoE layer. "
                    "Using the maximum across experts for each layer. ")

            self.a13_scale = nn.Parameter(self.a13_scale.max(),
                                          requires_grad=False)
            self.a2_scale = nn.Parameter(self.a2_scale.max(),
                                         requires_grad=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape # tokens of current batch on this device
        hidden_states = hidden_states.view(-1, self.hidden_size)
        
        # the below version just Simulates the execution of all2all communication and are fully load balanced
        # TODO: reorder hidden_states of tokens based on device_ids and start all-to-all comm
        # can try all-to-all implementation in Tutel or Hetu or FasterMoE; perhaps will need layout transform before all-to-all
        # example: deepspeed moe_inference: dependent on padding and there is a lot of redundant communication
        # and get max_padding_size by all_reduce
        # so perhaps we can use alltoall to get the split_sizes?
        # UPDATE1: A version by padding adapted from Deepspeed
        if self.use_padding:
            router_logits, _ = self.gate(hidden_states) # router_logits: (num_tokens, n_experts)
            hacked_router_logits = torch.rand_like(router_logits, dtype=torch.float)
            dispatched_input, combined_weights = gate_and_alltoall(hacked_router_logits, hidden_states, self.top_k) #[n*e/n, c, m], [s,e,c]
            # print(f"dispatched_input: {dispatched_input.shape} | combined_weights: {combined_weights.shape} \
            #       | w13 and w2: {self.w13_weight.shape} + {self.w2_weight.shape}")
        else:
            router_logits, _ = self.gate(hidden_states) # router_logits: (num_tokens, n_experts)
            hacked_router_logits = torch.rand_like(router_logits, dtype=torch.float)
            dispatched_input, combined_weights = gate_and_alltoall_sparse(hacked_router_logits, hidden_states, self.top_k)
        
        if self.use_padding:
            # UPDATE1: A version by padding adapted from Deepspeed
            num_worker = get_expert_parallel_world_size()
            num_expert = dispatched_input.shape[0] // num_worker
            capcity = dispatched_input.shape[1]
            m = dispatched_input.shape[-1]
            final_hidden_states = fused_moe_ep(dispatched_input,
                                        self.w13_weight,
                                        self.w2_weight,
                                        self.num_total_experts,
                                        self.top_k,
                                        padded=self.use_padding,
                                        inplace=True,
                                        use_fp8=self.use_fp8,
                                        w1_scale=self.w13_scale,
                                        w2_scale=self.w2_scale,
                                        a1_scale=self.a13_scale,
                                        a2_scale=self.a2_scale)
            final_hidden_states = final_hidden_states.view(num_expert, num_worker, capcity, m).transpose_(0, 1).contiguous() # [nw, ne, c, m]

        if self.use_padding:
            # UPDATE1: A version by padding adapted from Deepspeed
            dispatch_output = torch.empty_like(final_hidden_states)
            torch.distributed.all_to_all_single(dispatch_output, final_hidden_states, group=get_expert_parallel_group())
        
        # compute dispatch_output with router_logits
        # UPDATE1: A version by padding adapted from Deepspeed (will cause many redundant comm and comp)
        output = torch.matmul(
            combined_weights.type_as(hidden_states).reshape(combined_weights.shape[0], -1),
            dispatch_output.reshape(-1, dispatch_output.shape[-1])) # [s,e,c]*[e,c,m], s = num_tokens here

        return output.view(num_tokens, hidden_size)


class MixtralAttention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        rope_theta: float = 10000,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # TODO: Tensor-parallel is not enabled at this moment, each worker have all attention heads
        # tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        # assert self.total_num_heads % tp_size == 0
        # self.num_heads = self.total_num_heads // tp_size
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = num_kv_heads
        # if self.total_num_kv_heads >= tp_size:
        #     # Number of KV heads is greater than TP size, so we partition
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert self.total_num_kv_heads % tp_size == 0
        # else:
        #     # Number of KV heads is less than TP size, so we replicate
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert tp_size % self.total_num_kv_heads == 0
        
        # self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.num_kv_heads = self.total_num_kv_heads
        
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta

        # self.qkv_proj = QKVParallelLinear(
        #     hidden_size,
        #     self.head_dim,
        #     self.total_num_heads,
        #     self.total_num_kv_heads,
        #     bias=False,
        #     quant_config=quant_config,
        # )
        # use nn.Linear as an immediate attempt to achieve it
        # TODO: getting by load weights in the future
        output_features = (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
        self.qkv_proj = nn.Linear(in_features=hidden_size, out_features=output_features, 
                                  bias=False, dtype=torch.float16)
        
        # self.o_proj = RowParallelLinear(
        #     self.total_num_heads * self.head_dim,
        #     hidden_size,
        #     bias=False,
        #     quant_config=quant_config,
        # )
        self.o_proj = nn.Linear(in_features=self.total_num_heads * self.head_dim, out_features=hidden_size, 
                                bias=False, dtype=torch.float16)
        
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=int(self.rope_theta),
            is_neox_style=True,
        )
        self.attn = Attention(self.num_heads,
                              self.head_dim,
                              self.scaling,
                              num_kv_heads=self.num_kv_heads,
                              cache_config=cache_config,
                              quant_config=quant_config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output = self.o_proj(attn_output)
        return output


class EPMixtralDecoderLayer(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        # Requires transformers > 4.32.0
        rope_theta = getattr(config, "rope_theta", 10000)
        self.self_attn = MixtralAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            cache_config=cache_config,
            quant_config=quant_config)
        expert_parallel_size = get_expert_parallel_world_size()
        assert config.num_local_experts % expert_parallel_size == 0
        self.block_sparse_moe = EPMixtralMoE(
            num_local_experts=config.num_local_experts // expert_parallel_size,
            num_total_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            use_padding=False, # use_padding=True
            quant_config=quant_config)
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(
                hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states, residual)
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class EPMoEModel(nn.Module):

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size *
                      (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )
        assert get_tensor_model_parallel_world_size() == 1, "only support tp=1 now"
        self.layers = nn.ModuleList([
            EPMixtralDecoderLayer(config,
                                cache_config,
                                quant_config=quant_config)
            for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(positions, hidden_states,
                                            kv_caches[i], attn_metadata,
                                            residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class EPMoEForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
    }

    # LoRA specific attributes
    # TODO: check this for LoRA
    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "embed_tokens",
        "lm_head",
    ]
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }
    embedding_padding_modules = ["lm_head"]

    def __init__(
        self,
        config: MixtralConfig,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        lora_config: Optional[LoRAConfig] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = EPMoEModel(config,
                                  cache_config,
                                  quant_config,
                                  lora_config=lora_config)
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            config.hidden_size,
            org_num_embeddings=config.vocab_size,
            padding_size=DEFAULT_VOCAB_PADDING_SIZE
            # We need bigger padding if using lora for kernel
            # compatibility
            if not lora_config else lora_config.lora_vocab_padding_size,
        )
        self.logits_processor = LogitsProcessor(self.unpadded_vocab_size,
                                                config.vocab_size)
        self.sampler = Sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   attn_metadata)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor,
                       sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head.weight, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
        ]

        expert_params_mapping = [
            # These are the weight scales for the experts
            # (param_name, weight_name, expert_id)
            ("w13_scale" if weight_name in ["w1", "w3"] else "w2_scale",
             f"experts.{expert_id}.{weight_name}.weight_scale", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ] + [
            # These are the weights for the experts
            # (param_name, weight_name, expert_id)
            ("w13_weight" if weight_name in ["w1", "w3"] else "w2_weight",
             f"experts.{expert_id}.{weight_name}.weight", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ] + [
            # These are the activation scales for the experts
            # (param_name, weight_name, expert_id)
            ("a13_scale" if weight_name in ["w1", "w3"] else "a2_scale",
             f"experts.{expert_id}.{weight_name}.act_scale", expert_id)
            for expert_id in range(self.config.num_local_experts)
            for weight_name in ["w1", "w2", "w3"]
        ]

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            for (param_name, weight_name, shard_id) in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for param_name, weight_name, expert_id in expert_params_mapping:
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param,
                                  loaded_weight,
                                  weight_name,
                                  expert_id=expert_id)
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    # Remapping the name of FP8 kv-scale.
                    if name.endswith("kv_scale"):
                        remapped_kv_scale_name = name.replace(
                            ".kv_scale", ".attn.kv_scale")
                        if remapped_kv_scale_name not in params_dict:
                            print_warning_once(
                                "Found kv scale in the checkpoint "
                                f"(e.g. {name}), but not found the expected "
                                f"name in the model "
                                f"(e.g. {remapped_kv_scale_name}). "
                                "kv-scale is not loaded.")
                            continue
                        else:
                            name = remapped_kv_scale_name
                    param = params_dict[name]
                    weight_loader = getattr(param, "weight_loader",
                                            default_weight_loader)
                    weight_loader(param, loaded_weight)


def all_close_1d(x: torch.Tensor) -> bool:
    assert len(x.shape) == 1
    return all(torch.allclose(x[0], x[i]) for i in range(x.shape[0]))
