import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from .quant_linear import QuantLinear
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaConfig
import triton
import triton.language as tl


@triton.jit
def rotate_half_kernel(
        qk_seq_ptr,
        position_ids_ptr,
        qk_seq_stride,
        position_ids_batch_stride,
        seq_len,
        HEAD_DIM: tl.constexpr,
        BLOCK_HEIGHT: tl.constexpr,
        BLOCK_WIDTH: tl.constexpr,
        INV_BASE: tl.constexpr
):
    # qk_seq_ptr: (bsz, seq_len, 2, num_heads, head_dim) -- OK to be discontinuous in 2nd dimension.
    # position ids: (bsz, seq_len) -- must be contiguous in the last dimension.

    HALF_HEAD: tl.constexpr = HEAD_DIM // 2
    STEPS_PER_ROW: tl.constexpr = HALF_HEAD // BLOCK_WIDTH

    batch_seq = tl.program_id(axis=0)
    row_blk_x_col_blk = tl.program_id(axis=1)

    row_blk = row_blk_x_col_blk // STEPS_PER_ROW
    row = row_blk * BLOCK_HEIGHT
    if BLOCK_WIDTH < HALF_HEAD:
        col_blk = row_blk_x_col_blk % STEPS_PER_ROW
        col = col_blk * BLOCK_WIDTH
    else:
        col: tl.constexpr = 0

    # A block will never cross a sequence boundary, which simplifies things a lot.
    batch = batch_seq // seq_len
    seq = batch_seq % seq_len
    position_id = tl.load(position_ids_ptr + batch * position_ids_batch_stride + seq)
    # As sometimes happens, just calculating this on the fly is faster than loading it from memory.
    # Use `tl.libdevice.exp` rather than `tl.exp` -- the latter is less accurate.
    # TODO triton 2.1.0 moved tl.libdevice.exp to tl.math.exp
    freq = tl.libdevice.exp((col + tl.arange(0, BLOCK_WIDTH)).to(tl.float32) * INV_BASE) * position_id

    cos = tl.cos(freq).to(tl.float32)
    sin = tl.sin(freq).to(tl.float32)

    col_offsets: tl.constexpr = tl.arange(0, BLOCK_WIDTH)
    embed_offsets = (row * HEAD_DIM + col) + col_offsets
    x_ptrs = (qk_seq_ptr + batch_seq * qk_seq_stride) + embed_offsets

    for k in range(0, BLOCK_HEIGHT):
        x = tl.load(x_ptrs).to(tl.float32)
        y = tl.load(x_ptrs + HALF_HEAD).to(tl.float32)
        out_x = x * cos - y * sin
        tl.store(x_ptrs, out_x)
        out_y = x * sin + y * cos
        tl.store(x_ptrs + HALF_HEAD, out_y)
        x_ptrs += HEAD_DIM


def triton_rotate_half_(qk, position_ids, config=None):
    batch_size, seq_len, qandk, num_heads, head_dim = qk.shape

    # This default is the fastest for most job sizes, at least on my RTX 4090, and when it's not it's within spitting
    # distance of the best option. There are some odd cases where having a block height of 2 or 4 helps but the
    # difference is within 5%. It makes sense that this configuration is fast from a memory bandwidth and caching
    # perspective.
    config = config or {'BLOCK_HEIGHT': 1, 'BLOCK_WIDTH': min(128, head_dim // 2), 'num_warps': 1}
    config['BLOCK_HEIGHT'] = min(config['BLOCK_HEIGHT'], 2 * num_heads)

    assert qk.stride(3) == head_dim
    assert qk.stride(4) == 1
    assert position_ids.shape == (batch_size, seq_len)
    assert position_ids.stride(1) == 1, 'position_ids must be contiguous in the last dimension'
    assert (2 * num_heads) % config[
        'BLOCK_HEIGHT'] == 0, f'number of rows not evenly divisible by {config["BLOCK_HEIGHT"]}'
    assert (head_dim // 2) % config[
        'BLOCK_WIDTH'] == 0, f'number of columns ({head_dim // 2}) not evenly divisible by {config["BLOCK_WIDTH"]}'

    qk_by_seq = qk.view(batch_size * seq_len, 2 * num_heads * head_dim)
    grid = (qk_by_seq.shape[0], (2 * num_heads // config['BLOCK_HEIGHT']) * (head_dim // 2 // config['BLOCK_WIDTH']))

    # Must be the same as the theta of the frequencies used to train the model.
    BASE = 10000.0

    rotate_half_kernel[grid](
        qk_by_seq,
        position_ids,
        qk_by_seq.stride(0),
        position_ids.stride(0),
        seq_len,
        HEAD_DIM=head_dim,
        BLOCK_HEIGHT=config['BLOCK_HEIGHT'],
        BLOCK_WIDTH=config['BLOCK_WIDTH'],
        INV_BASE=-2.0 * math.log(BASE) / head_dim,
        num_warps=config['num_warps']
    )


def make_quant_attn(model):
    """
    Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
    """
    for name, m in model.named_modules():
        if not isinstance(m, LlamaAttention):
            continue

        q_proj = m.q_proj
        k_proj = m.k_proj
        v_proj = m.v_proj

        qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
        qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
        scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)

        qkv_layer = QuantLinear(q_proj.bits, q_proj.groupsize, q_proj.infeatures,
                                q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures, bias=False)
        qkv_layer.qweight = qweights
        qkv_layer.qzeros = qzeros
        qkv_layer.scales = scales
        qkv_layer.bias = None

        # We're dropping the rotary embedding layer m.rotary_emb here. We don't need it in the triton branch.
        attn = QuantLlamaAttention(m.config, qkv_layer, m.o_proj)

        if '.' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name[len(parent_name) + 1:]
            parent = model.get_submodule(parent_name)
        else:
            parent_name = ''
            parent = model
            child_name = name

        # print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

        setattr(parent, child_name, attn)


class QuantLlamaAttention(nn.Module):
    """
    Modified version of LlamaAttention that fuses the q, k, v projections.
    """
    def __init__(
            self,
            config: LlamaConfig,
            qkv_proj,
            o_proj
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""
        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        qkv_states = qkv_states.view(bsz, q_len, 3, self.num_heads, self.head_dim)

        # This updates the query and key states in-place, saving VRAM.
        triton_rotate_half_(qkv_states[:, :, :2], position_ids)

        query_states, key_states, value_states = torch.split(qkv_states, 1, dim=2)
        del qkv_states

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = q_len
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]

            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        del query_states, key_states, value_states

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value