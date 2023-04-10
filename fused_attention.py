import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from quant_linear import QuantLinear
from transformers.models.llama.modeling_llama import LlamaAttention, apply_rotary_pos_emb


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

		qkv_layer = QuantLinear(4, -1, q_proj.infeatures, q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures)
		qkv_layer.qweight = qweights
		qkv_layer.qzeros = qzeros
		qkv_layer.scales = scales
		qkv_layer.bias = None

		attn = QuantLlamaAttention(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, m.rotary_emb)

		if '.' in name:
			parent_name = name.rsplit('.', 1)[0]
			child_name = name[len(parent_name) + 1:]
			parent = model.get_submodule(parent_name)
		else:
			parent_name = ''
			parent = model
			child_name = name

		#print(f"Replacing {name} with quant_attn; parent: {parent_name}, child's name: {child_name}")

		setattr(parent, child_name, attn)


class QuantLlamaAttention(nn.Module):
	"""
	Modified version of LlamaAttention that fuses the q, k, v projections.
	"""

	def __init__(
		self,
		hidden_size: int,
		num_heads: int,
		qkv_proj,
		o_proj,
		rotary_emb,
	):
		super().__init__()
		self.hidden_size = hidden_size
		self.num_heads = num_heads
		self.head_dim = hidden_size // num_heads

		if (self.head_dim * num_heads) != self.hidden_size:
			raise ValueError(
				f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
				f" and `num_heads`: {num_heads})."
			)
		self.qkv_proj = qkv_proj
		self.o_proj = o_proj
		self.rotary_emb = rotary_emb

	def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
		return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

	def forward(
		self,
		hidden_states: torch.Tensor,
		past_key_value: Optional[Tuple[torch.Tensor]] = None,
		attention_mask: Optional[torch.Tensor] = None,
		output_attentions: bool = False,
		use_cache: bool = False,
	) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
		"""Input shape: Batch x Time x Channel"""

		bsz, q_len, _ = hidden_states.size()

		qkv_states = self.qkv_proj(hidden_states)
		query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)

		query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
		value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

		kv_seq_len = key_states.shape[-2]
		offset = 0
		if past_key_value is not None:
			offset = past_key_value[0].shape[-2]
			kv_seq_len += offset
		cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
		query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, offset=offset)
		# [bsz, nh, t, hd]

		if past_key_value is not None:
			# reuse k, v, self_attention
			key_states = torch.cat([past_key_value[0], key_states], dim=2)
			value_states = torch.cat([past_key_value[1], value_states], dim=2)

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