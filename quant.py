import json
import math
from pathlib import Path
from typing import Optional

import custom_autotune
import torch
import torch.nn as nn
import transformers
import triton
import triton.language as tl
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


def load_quant(checkpoint: str, warmup_autotune: bool = True, device: Optional[str] = 'cuda'):
	"""
	Load a quantized model from a checkpoint.
	"""
	quant_config = json.load(open(Path(checkpoint) / 'quant_config.json'))
	wbits = quant_config['wbits']
	groupsize = quant_config['groupsize']

	# Load the model config
	config = LlamaConfig.from_pretrained(checkpoint)
	def noop(*args, **kwargs):
		pass
	# NOTE: This is a nasty hack, but it speeds up creation of the model by a huge amount.
	old_init = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
	torch.nn.init.kaiming_uniform_ = noop 
	torch.nn.init.uniform_ = noop 
	torch.nn.init.normal_ = noop 

	# Build the model
	# TODO: Is this needed?
	torch.set_default_dtype(torch.half)
	old_init_weights = transformers.modeling_utils._init_weights
	transformers.modeling_utils._init_weights = False
	torch.set_default_dtype(torch.half)
	model = LlamaForCausalLM(config)
	torch.set_default_dtype(torch.float)

	# Restore the original init functions
	(torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_) = old_init
	transformers.modeling_utils._init_weights = old_init_weights

	# Swap out linear layers for quantized ones
	make_quant(model, wbits, groupsize)

	# Load the quantized checkpoint
	print('Loading model ...')
	if (Path(checkpoint) / 'model.safetensors').exists():
		from safetensors.torch import load_file as safe_load
		model.load_state_dict(safe_load(Path(checkpoint) / 'model.safetensors'))
	elif (Path(checkpoint) / 'model.pt').exists():
		model.load_state_dict(torch.load(Path(checkpoint) / 'model.pt'))
	else:
		raise FileNotFoundError(f"Could not find model checkpoint at {checkpoint}; please ensure that the path is correct and contains a `model.pt` or `model.safetensors` file.")
	
	# Go through all the QuantLinear layers and if their bias is all zeros, set it to None
	for name, m in model.named_modules():
		if isinstance(m, QuantLinear):
			if (m.bias == 0).all():
				m.bias = None
	
	# Move the model to the correct device
	if device is not None:
		model = model.to(device)
	
	# Warm up the autotune cache
	if warmup_autotune:
		if device is None:
			raise ValueError("You must specify a device when warmup_autotune is True.")
		
		autotune_warmup(model)
	
	model.seqlen = 2048
	print('Done.')

	return model


def autotune_warmup(model):
	"""
	Pre-tunes the quantized kernel
	"""
	from tqdm import tqdm

	# Find all the QuantLinear layers
	n_values = {}

	for _, m in model.named_modules():
		if not isinstance(m, QuantLinear):
			continue

		k = m.infeatures
		n = m.outfeatures

		n_values[n] = (m.qweight, m.scales, m.qzeros, k)

	print(f'Found {len(n_values)} unique N values.')
	
	print('Warming up autotune cache ...')
	for m in tqdm(range(0, 12)):
		m = 2 ** m   # [1, 2048]
		for n, (qweight, scales, qzeros, k) in n_values.items():
			a = torch.randn(1, m, k, dtype=torch.float16, device='cuda')
			triton_matmul4(a, qweight, scales, qzeros)


def make_quant(model, bits, groupsize):
	"""
	Replace all linear layers in a model with quantized ones.
	Except for the lm_head, which is not quantized.
	"""
	for name, m in model.named_modules():
		if not isinstance(m, torch.nn.Linear):
			continue

		if name == 'lm_head':
			continue

		# Replace the linear layer with a quantized one
		qlayer = QuantLinear(bits, groupsize, m.in_features, m.out_features)
		parent_name = name.rsplit('.', 1)[0]
		parent = model.get_submodule(parent_name)

		#print(f"Replacing {name} with quant; parent: {parent_name}, child's name: {name[len(parent_name) + 1:]}")

		setattr(parent, name[len(parent_name) + 1:], qlayer)


class QuantLinear(nn.Module): 
	def __init__(self, bits: int, groupsize: int, infeatures: int, outfeatures: int):
		super().__init__()

		if bits not in [4]:
			raise NotImplementedError("Only 4 bits are supported.")
		
		if groupsize != -1:
			raise NotImplementedError("Groupsize is not supported, must be -1.")
		
		groupsize = infeatures
		
		self.infeatures = infeatures
		self.outfeatures = outfeatures
		self.bits = bits
		self.groupsize = groupsize

		features_per_int = 32 // bits

		assert outfeatures % features_per_int == 0

		self.register_buffer('qweight', torch.zeros((infeatures // features_per_int, outfeatures), dtype=torch.int32))
		self.register_buffer('qzeros', torch.zeros((math.ceil(infeatures / groupsize), outfeatures // features_per_int), dtype=torch.int32))
		self.register_buffer('scales', torch.zeros((math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16))
		self.register_buffer('bias', torch.zeros(outfeatures, dtype=torch.float16))

	def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
		y = triton_matmul4(x, self.qweight, self.scales, self.qzeros, self.bias)
		return y


# This Triton kernel is adapted from the Triton matmul example
# It unpacks the quantized weights and then performs the matmul like usual
# It operates in FP16 mode
@custom_autotune.autotune(
	configs=[
		# These weren't useful, at least on a 3090
		#triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		#triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		#triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
		#triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),

		triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),

		# These provided a benefit on a 3090
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
	],
	key=['M', 'N'],
	nearest_power_of_two=True,
)
@triton.jit
def matmul4_kernel(
	a_ptr, b_ptr, c_ptr, #debug_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_am, stride_ak,
	stride_bk, stride_bn,
	stride_cm, stride_cn,
	stride_scales, stride_zeros, #stride_dk, stride_dn,
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
):
	"""
	Compute the matrix multiplication C = A x B.
	A is of shape (M, K) float16
	B is of shape (K//8, N) int32
	C is of shape (M, N) float16
	scales is of shape (1, N) float16
	zeros is of shape (1, N//8) int32

	WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
	WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
	"""
	pid = tl.program_id(axis=0)
	num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
	num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
	num_pid_k = tl.cdiv(K, BLOCK_SIZE_K)
	num_pid_in_group = GROUP_SIZE_M * num_pid_n
	group_id = pid // num_pid_in_group
	first_pid_m = group_id * GROUP_SIZE_M
	group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
	pid_m = first_pid_m + (pid % group_size_m)
	pid_n = (pid % num_pid_in_group) // group_size_m

	offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	offs_k = tl.arange(0, BLOCK_SIZE_K)
	a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
	a_mask = (offs_am[:, None] < M)
	# b_ptrs is set up such that it repeats elements along the K axis 8 times
	b_ptrs = b_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
	scales_ptrs = scales_ptr + offs_bn * stride_scales
	# zeros_ptrs is set up such that it repeats elements along the N axis 8 times
	zeros_ptrs = zeros_ptr + (offs_bn // 8) * stride_zeros

	# shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
	shifter = (offs_k % 8) * 4
	zeros_shifter = (offs_bn % 8) * 4

	# Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
	scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
	zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

	# Unpack zeros
	zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
	zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16

	# For debugging
	#offs_dk = 0 + tl.arange(0, BLOCK_SIZE_K)
	#offs_dn = 0 + tl.arange(0, BLOCK_SIZE_N)
	#debug_ptrs = debug_ptr + stride_dk * offs_dk[:, None] + stride_dn * offs_dn[None, :]
	#tl.store(debug_ptrs, b)

	# Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
	# M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
	# So this loop is along the infeatures dimension (K)
	# It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
	accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	for k in range(0, num_pid_k):
		a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
		b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

		# Now we need to unpack b (which is 4-bit values) into 32-bit values
		b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
		b = b * scales[None, :] - zeros[None, :]  # Scale and shift
		#tl.store(debug_ptrs, b)

		accumulator += tl.dot(a, b)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
		#debug_ptrs += BLOCK_SIZE_K * stride_dk
	
	c = accumulator.to(tl.float16)
	
	# Store the result
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul4(a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = A x B + bias.
	Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

	A is of shape (..., K) float16
	qweight is of shape (K//8, N) int32
	scales is of shape (1, N) float16
	qzeros is of shape (1, N//8) int32
	bias is of shape (1, N) float16

	Returns C of shape (..., N) float16
	"""
	assert a.shape[-1] == (qweight.shape[0] * 8)
	assert a.is_contiguous()

	# Flatten a into (-1, K)
	x = a.view(-1, a.shape[-1])

	M, K = x.shape
	N = qweight.shape[1]
	# This is based on the maximum BLOCK_SIZE_K; for our use cases (LLMs) we expect K to be large and a power of two
	assert K % 32 == 0
	# This is based on the maximum BLOCK_SIZE_N; for our use cases (LLMs) we expect N to be large and a power of two
	assert N % 256 == 0, "N must be a multiple of 256"

	c = torch.empty((M, N), device='cuda', dtype=torch.float16)
	#debug = torch.empty((32*32, 128), device='cuda', dtype=torch.float32)
	grid = lambda META: (
		triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	#grid = lambda META: (1,)  # For debugging
	matmul4_kernel[grid](
		x, qweight, c, #debug,
		scales, qzeros,
		M, N, K,
		x.stride(0), x.stride(1),
		qweight.stride(0), qweight.stride(1),
		c.stride(0), c.stride(1),
		scales.stride(1), qzeros.stride(1),
		#debug.stride(0), debug.stride(1),
	)

	# Reshape c
	c = c.view(a.shape[:-1] + (N,))  # (..., N)

	# Add bias
	if bias is not None:
		c = c + bias
	
	return c
	#return debug
