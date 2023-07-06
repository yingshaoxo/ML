# Copyright (c) 2021 EleutherAI
# This file is based on code by the authors denoted below and has been modified from its original version.
#
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

"""Transformer."""

import math
import torch
import torch.nn.functional as F
import torch.nn as nn

from .norms import get_norm
from .. import mpu
from .fused_softmax import FusedScaleMaskSoftmax
from .activations import get_activation
from .utils import exists, get_fusion_type
from .positional_embeddings import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    AliBi,
)
from .fused_bias_dropout import (
    get_bias_dropout_add,
    bias_dropout_add_fused_train,
    bias_dropout_add_fused_inference,
)
from .utils import configure_sparse_attention

# flags required to enable jit fusion kernels
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
        attention_mask_func: a function that takes `unmasked-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
               masked-attention-scores = attention_mask_func(
                                     unmasked-attention-scores, attention-mask)
"""


class ParallelMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension. At the end, dropout is also
    applied.
    """

    def __init__(
        self, config, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(config)
        self.activation_type = config.hidden_act
        self.bias_gelu_fusion = config.bias_gelu_fusion

        # auto scale so geglu has equal parameters
        ff_mult = 4 * 2 / 3 if self.activation_type == "geglu" else 4
        ff_dim = (
            int(ff_mult * config.hidden_size) * 2
            if self.activation_type == "geglu"
            else ff_mult * config.hidden_size
        )
        self.dense_h_to_4h = mpu.ColumnParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
        )
        ff_dim_in = ff_dim // 2 if self.activation_type == "geglu" else ff_dim
        # Project back to h.
        self.dense_4h_to_h = mpu.RowParallelLinear(
            config=config,
            input_size=ff_dim_in,
            output_size=config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
        )

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if (
            self.activation_type == "gelu" and self.bias_gelu_fusion
        ) or self.activation_type == "geglu":
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            intermediate_parallel = self.activation_func(
                intermediate_parallel + bias_parallel
            )

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias


class ParallelLinear(nn.Module):
    """
    A Parallel Linear Layer transforming the transformer outputs from hidden_size -> vocab_size
    """

    def __init__(
        self,
        config,
        parallel_output=True,
        init_method=nn.init.xavier_normal_,
    ):
        super().__init__()
        parallelism = config.output_layer_parallelism
        if parallelism == "column":
            self.final_linear = mpu.ColumnParallelLinear(
                config=config,
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                init_method=init_method,
                gather_output=not parallel_output,
                skip_bias_add=False,
            )
        else:
            self.final_linear = mpu.RowParallelLinear(
                config=config,
                input_size=config.hidden_size,
                output_size=config.vocab_size,
                bias=False,
                input_is_parallel=False,
                init_method=init_method,
                parallel_output=parallel_output,
                skip_bias_add=False,
            )

    def forward(self, hidden_states):
        return self.final_linear(hidden_states)


class ParallelSelfAttention(nn.Module):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [b, s, h]
    and returns output of the same size.
    """

    def __init__(
        self,
        config,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False,
        parallel_output=False,
    ):
        super().__init__()

        self.fp16 = config.torch_dtype == torch.half
        self.bf16 = config.torch_dtype == torch.bfloat16
        self.attention_mask_func = attention_mask_func
        self.apply_query_key_layer_scaling = config.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = config.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = layer_number
        # Per attention head and per partition values.
        world_size = mpu.get_model_parallel_world_size()
        self.hidden_size_per_partition = mpu.divide(config.hidden_size, world_size)
        self.hidden_size_per_attention_head = mpu.divide(
            config.hidden_size, config.num_attention_heads
        )
        self.num_attention_heads_per_partition = mpu.divide(
            config.num_attention_heads, world_size
        )
        self.pos_emb = config.pos_emb

        # Strided linear layer.
        self.query_key_value = mpu.ColumnParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=3 * config.hidden_size,
            gather_output=False,
            init_method=init_method,
            bias=config.use_bias_in_attn_linear,
        )

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = max(1, self.layer_number)
            self.norm_factor *= coeff

        self.rpe = rpe

        if self.pos_emb == "alibi":
            self.alibi_embed = AliBi(
                config.num_attention_heads,
                config.model_parallel_size,
                mpu.get_model_parallel_rank(),
            )

        # TODO: this arg shouldn't need to be passed in - get from config
        if rotary:
            if config.rotary_pct == 1:
                self.rotary_ndims = None
            else:
                assert config.rotary_pct < 1
                self.rotary_ndims = int(
                    self.hidden_size_per_attention_head * config.rotary_pct
                )
            dim = (
                self.rotary_ndims
                if self.rotary_ndims is not None
                else self.hidden_size_per_attention_head
            )
            self.rotary_emb = RotaryEmbedding(
                dim, base=config.rotary_emb_base, max_position_embeddings=config.max_position_embeddings
            )
        else:
            self.rotary_emb = None

        self.attention_type = config.attention_config[layer_number]
        self.use_flash_attention = self.attention_type == "flash"
        self.sparse = self.attention_type != "global" and not self.use_flash_attention
        if self.sparse:
            self.sparse_attn = configure_sparse_attention(
                config,
                self.attention_type,
                self.num_attention_heads_per_partition,
                mpu=mpu,
            )
        else:
            if self.use_flash_attention:
                from .flash_attention import (
                    flash_attn_unpadded_qkvpacked_func,
                )

                self.flash_attention_function = flash_attn_unpadded_qkvpacked_func
                if self.pos_emb == "alibi":
                    raise ValueError(
                        "Flash attention is currently not compatible with AliBi positional embeddings. Use sinuisoidal, learned, or rotary embeddings instead."
                    )
            self.scale_mask_softmax = FusedScaleMaskSoftmax(
                input_in_fp16=self.fp16,
                input_in_bf16=self.bf16,
                fusion_type=get_fusion_type(config),
                mask_func=self.attention_mask_func,
                softmax_in_fp32=self.attention_softmax_in_fp32,
                scale=coeff,
            )

            # Dropout. Note that for a single iteration, this layer will generate
            # different outputs on different number of parallel partitions but
            # on average it should not be partition dependent.
            self.dropout_p = config.attention_dropout
            self.attention_dropout = nn.Dropout(self.dropout_p)

        # Output.
        self.dense = mpu.RowParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=config.use_bias_in_attn_linear,
        )

    def attention(
        self, query_layer, key_layer, value_layer, layer_past, attention_mask, use_cache=False
    ):
        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(
            output_size[2], output_size[0] * output_size[1], -1
        )
        key_layer = key_layer.view(output_size[3], output_size[0] * output_size[1], -1)

        # preallocating result tensor: [b * np, sq, sk]
        matmul_result = torch.empty(
            output_size[0] * output_size[1],
            output_size[2],
            output_size[3],
            dtype=query_layer.dtype,
            device=torch.cuda.current_device(),
        )

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_result,
            query_layer.transpose(0, 1),  # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0,
            alpha=(1.0 / self.norm_factor),
        )

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ==================================================
        # Update attention mask for inference. [b, np, sq, sk]
        # ==================================================

        if use_cache:
            with torch.no_grad():
                attention_mask = attention_mask[
                    ..., : attention_scores.size(3), : attention_scores.size(3)
                ]

        # ===========================
        # Attention probs and dropout
        # ===========================

        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
            attention_scores += rpe  # [1, np, sq, sk]

        if self.pos_emb == "alibi":
            attention_scores = self.alibi_embed(attention_scores)

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores, attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with mpu.get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (
            value_layer.size(1),
            value_layer.size(2),
            query_layer.size(0),
            value_layer.size(3),
        )

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(
            value_layer.size(0), output_size[0] * output_size[1], -1
        )

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(
            output_size[0] * output_size[1], output_size[2], -1
        )

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)
        return context_layer

    def flash_attention(self, query_layer, key_layer, value_layer):
        # [b, np, sq, sk]
        output_size = (
            query_layer.size(1),
            query_layer.size(2),
            query_layer.size(0),
            key_layer.size(0),
        )
        # [s, b, np, hn] -> [b, s, np, hn] -> [b * s, 1, np, hn]
        query_layer = query_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[2], 1, output_size[1], -1
        )
        key_layer = key_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )
        value_layer = value_layer.transpose(0, 1).reshape(
            output_size[0] * output_size[3], 1, output_size[1], -1
        )
        # Combined q/k/v into [b * s, 3, np, hn].
        qkv = torch.concat([query_layer, key_layer, value_layer], dim=1)

        batch_size = output_size[0]
        seqlen = output_size[2]
        max_s = seqlen
        cu_seqlens = torch.arange(
            0,
            (batch_size + 1) * seqlen,
            step=seqlen,
            dtype=torch.int32,
            device=qkv.device,
        )
        output = self.flash_attention_function(
            qkv,
            cu_seqlens,
            max_s,
            self.dropout_p if self.training else 0.0,
            softmax_scale=None,
            causal=True,
        )
        # [b * sq, np, hn] -> [b, sq, np, hn]
        matmul_result = output.view(
            output_size[0], output_size[2], output.shape[1], output.shape[2]
        )
        # [b, sq, np, hn] -> [b, np, sq, hn]
        matmul_result = matmul_result.transpose(1, 2)

        return matmul_result

    def sparse_attention(self, query_layer, key_layer, value_layer, attention_mask):
        # TODO: sparse attn dropout?
        # TODO: pad to block size
        # shape of q/k/v is [sq, b, np, hn] and needs to be transposed to [b, np, sq, hn]
        query_layer, key_layer, value_layer = map(
            lambda t: t.permute(1, 2, 0, 3).contiguous(),
            (query_layer, key_layer, value_layer),
        )
        # output shape [b, np(heads), sq, hn]
        attn_mask = attention_mask.to(query_layer.dtype) * -10000
        if exists(self.rpe):
            rpe = self.rpe(query_layer.size(0), key_layer.size(0))
        else:
            rpe = None
        return self.sparse_attn(
            query_layer, key_layer, value_layer, attn_mask=attn_mask, rpe=rpe
        )

    def forward(self, hidden_states, attention_mask, position_ids, layer_past=None, use_cache=False):

        # hidden_states: [sq, b, h]

        # =====================
        # Query, Key, and Value
        # =====================

        # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
        mixed_x_layer, _ = self.query_key_value(hidden_states)

        # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
        new_tensor_shape = mixed_x_layer.size()[:-1] + (
            self.num_attention_heads_per_partition,
            3 * self.hidden_size_per_attention_head,
        )
        mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

        # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
        (query_layer, key_layer, value_layer) = mpu.split_tensor_along_last_dim(
            mixed_x_layer, 3
        )

        if exists(self.rotary_emb):
            if exists(self.rotary_ndims):
                # partial rotary
                query_rot, query_pass = (
                    query_layer[..., : self.rotary_ndims],
                    query_layer[..., self.rotary_ndims:],
                )
                key_rot, key_pass = (
                    key_layer[..., : self.rotary_ndims],
                    key_layer[..., self.rotary_ndims:],
                )
            else:
                # full rotary
                query_rot, key_rot = query_layer, key_layer
            apply_rotary_fn = apply_rotary_pos_emb
            seq_len = key_layer.shape[0]
            if exists(layer_past) and layer_past.numel() > 0:
                seq_len += layer_past[0].shape[0]
            cos, sin = self.rotary_emb(value_layer, seq_len=seq_len)
            query_layer, key_layer = apply_rotary_fn(
                query_rot, key_rot, cos, sin, position_ids=position_ids
            )
            if exists(self.rotary_ndims):
                query_layer = torch.cat((query_layer, query_pass), dim=-1)
                key_layer = torch.cat((key_layer, key_pass), dim=-1)

        # ==================================
        # Cache key and value for inference
        # ==================================

        if exists(layer_past) and layer_past.numel() > 0:
            past_key, past_value = layer_past
            key_layer = torch.cat((past_key.type_as(key_layer), key_layer), dim=0)
            value_layer = torch.cat(
                (past_value.type_as(value_layer), value_layer), dim=0
            )

        if use_cache:
            present = torch.stack((key_layer, value_layer))

        if self.use_flash_attention and not use_cache:
            context_layer = self.flash_attention(query_layer, key_layer, value_layer)
        elif not self.sparse:
            context_layer = self.attention(
                query_layer, key_layer, value_layer, layer_past, attention_mask, use_cache
            )
        else:
            context_layer = self.sparse_attention(
                query_layer, key_layer, value_layer, attention_mask
            )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.hidden_size_per_partition,
        )
        context_layer = context_layer.view(*new_context_layer_shape)

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        if use_cache:
            output = [output, present]

        return output, bias


class LLaMAParallelMLP(nn.Module):
    """LLaMA's MLP.

     MLP will take the input with h hidden state, project it to 4*h
     hidden dimension, perform nonlinear transformation, and project the
     state back into h hidden dimension. At the end, dropout is also
     applied.
     """

    def __init__(
        self, config, init_method, output_layer_init_method, parallel_output=False
    ):
        super().__init__()

        self.activation_func = get_activation(config)
        self.activation_type = config.hidden_act
        self.multiple_of = config.llama_mlp_multiple_of

        ff_dim = int(2 * config.hidden_size * 4 / 3)
        ff_dim = self.multiple_of * ((ff_dim + self.multiple_of - 1) // self.multiple_of)
        self.w1 = mpu.ColumnParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w3 = mpu.ColumnParallelLinear(
            config=config,
            input_size=config.hidden_size,
            output_size=ff_dim,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            bias=False,
        )
        self.w2 = mpu.RowParallelLinear(
            config=config,
            input_size=ff_dim,
            output_size=config.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            parallel_output=parallel_output,
            bias=False,
        )

    def forward(self, hidden_states):
        w1_out, _ = self.w1(hidden_states)
        w3_out, _ = self.w3(hidden_states)
        return self.w2(self.activation_func(w1_out) * w3_out)


class ParallelTransformerLayer(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [b, s, h] and returns an
    output of the same size.
    """

    def __init__(
        self,
        config,
        attention_mask_func,
        init_method,
        output_layer_init_method,
        layer_number,
        rpe=None,
        rotary=False
    ):

        super().__init__()
        self.layer_number = layer_number

        norm, eps = get_norm(config)

        # Layernorm on the input data.
        self.input_layernorm = norm(config.hidden_size, eps=eps)

        self.hidden_dropout = config.hidden_dropout
        self.gpt_j_residual = config.gpt_j_residual
        self.gpt_j_tied = config.gpt_j_tied

        if self.gpt_j_residual:
            self.reduce = mpu.mappings.reduce_from_model_parallel_region

        # Self attention.
        self.attention = ParallelSelfAttention(
            config=config,
            attention_mask_func=attention_mask_func,
            init_method=init_method,
            output_layer_init_method=output_layer_init_method,
            layer_number=layer_number,
            rpe=rpe,
            rotary=rotary,
            parallel_output=self.gpt_j_residual,
        )

        # Layernorm on the output of the attention layer.
        # If GPT-J residuals are used, this is surpurfulous but leaving it in
        # leads to cleaner code
        self.post_attention_layernorm = norm(config.hidden_size, eps=eps)

        # MLP
        self.mlp_type = "regular" if config.mlp_type is None else config.mlp_type
        if self.mlp_type == "regular":
            self.mlp = ParallelMLP(
                config=config,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        elif self.mlp_type == "llama":
            self.mlp = LLaMAParallelMLP(
                config=config,
                init_method=init_method,
                output_layer_init_method=output_layer_init_method,
                parallel_output=self.gpt_j_residual,
            )
        else:
            raise KeyError(self.mlp_type)
        self.bias_dropout_fusion = config.bias_dropout_fusion

    def _get_bias_dropout(self):
        if self.bias_dropout_fusion:
            fn = (
                bias_dropout_add_fused_train
                if self.training
                else bias_dropout_add_fused_inference
            )
        else:
            fn = get_bias_dropout_add(self.training)
        return fn

    def forward(self, x, attention_mask, position_ids, layer_past=None, use_cache=False):
        bias_dropout_fn = self._get_bias_dropout()
        # x: [b, s, h]
        if self.gpt_j_residual:
            # pseudocode:
            # x = x + attn(ln(x)) + mlp(ln(x))
            # this means we can avoid doing the allreduce in the attn / mlp outputs
            # to save communication time (we can do a single allreduce after we add mlp / attn outputs).
            # due to a bug, the two layernorms are not tied in GPT-NeoX-20B. This is non-desirable, but
            # we preserve the functionality for backwards compatibility

            residual = x
            # applies the correct normalization depending on if the norms are tied
            if self.gpt_j_tied:
                x = self.input_layernorm(x)
                x1, x2 = x, x
            else:
                x1, x2 = self.input_layernorm(x), self.post_attention_layernorm(x)

            # attention operator
            attention_output, attention_bias = self.attention(
                x1, attention_mask, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
            )
            if use_cache:
                attention_output, presents = attention_output

            with torch.enable_grad():
                attention_output = bias_dropout_fn(
                    attention_output,
                    bias=attention_bias.expand_as(attention_output),
                    residual=None,
                    prob=self.hidden_dropout,
                )

            # mlp operator
            mlp_output, mlp_bias = self.mlp(x2)
            with torch.enable_grad():
                output = bias_dropout_fn(
                    mlp_output,
                    bias=mlp_bias.expand_as(mlp_output),
                    residual=attention_output,
                    prob=self.hidden_dropout,
                )

            # output = (x + attn(ln(x)) + mlp(ln(x))
            output = residual + self.reduce(output)
        else:
            # pseudocode:
            # x = x + attn(ln1(x))
            # x = x + mlp(ln2(x))

            residual = x

            # x = x + attn(ln1(x))
            attention_output, attention_bias = self.attention(
                self.input_layernorm(x), attention_mask, position_ids=position_ids, layer_past=layer_past, use_cache=use_cache
            )
            if use_cache:
                attention_output, presents = attention_output
            with torch.enable_grad():
                if attention_bias is not None:
                    # Use special bias_dropout_fn if we have a bias term from the above attention layer
                    attention_output = bias_dropout_fn(
                        attention_output,
                        bias=attention_bias.expand_as(residual),
                        residual=residual,
                        prob=self.hidden_dropout,
                    )
                else:
                    attention_output = torch.nn.functional.dropout(
                        attention_output, p=self.hidden_dropout, training=self.training
                    ) + residual

            # output = x + mlp(ln2(x))
            mlp_output, mlp_bias = self.mlp(
                self.post_attention_layernorm(attention_output)
            )
            with torch.enable_grad():
                if self.mlp_type == "llama":
                    # No dropout either
                    assert mlp_bias is None
                    output = mlp_output + attention_output
                else:
                    output = bias_dropout_fn(
                        mlp_output,
                        bias=mlp_bias.expand_as(attention_output),
                        residual=attention_output,
                        prob=self.hidden_dropout,
                    )

        return output if not use_cache else (output, presents)


def parallel_lm_logits(input_, word_embeddings_weight, parallel_output, bias=None):
    """LM logits using word embedding weights."""
    # Parallel logits.
    input_parallel = mpu.copy_to_model_parallel_region(input_)

    # Matrix multiply.
    if bias is None:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight)
    else:
        logits_parallel = F.linear(input_parallel, word_embeddings_weight, bias)

    # Gather if needed.
    if parallel_output:
        return logits_parallel

    return mpu.gather_from_model_parallel_region(logits_parallel)
