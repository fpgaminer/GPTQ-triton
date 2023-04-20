import functools
import math
from typing import Optional

from . import custom_autotune
import torch
import torch.nn as nn
import triton
import triton.language as tl
from .utils import matmul4_kernel_config_pruner


def make_quant(model, bits, groupsize):
	"""
	Replace all linear layers in a model with quantized ones.
	Except for the lm_head, which is not quantized.
	"""
	for name, m in model.named_modules():
		if not isinstance(m, nn.Linear):
			continue

		if name == 'lm_head':
			continue

		# Replace the linear layer with a quantized one
		qlayer = QuantLinear(bits, groupsize, m.in_features, m.out_features, m.bias is not None)
		parent_name = name.rsplit('.', 1)[0]
		parent = model.get_submodule(parent_name)

		#print(f"Replacing {name} with quant; parent: {parent_name}, child's name: {name[len(parent_name) + 1:]}")

		setattr(parent, name[len(parent_name) + 1:], qlayer)


def autotune_warmup(model):
	# Find all the QuantLinear layers
	modules = (m for m in model.modules() if isinstance(m, QuantLinear))
	kn_values = {(m.infeatures, m.outfeatures): (m.qweight, m.scales, m.qzeros, m.groupsize) for m in modules}

	print(f'QuantLinear Warmup: Found {len(kn_values)} unique KN values.')

	def func(m, k, qweight, scales, qzeros, groupsize):
		a = torch.randn(1, m, k, dtype=torch.float16, device='cuda')
		triton_matmul4(groupsize, a, qweight, scales, qzeros)
	
	return (functools.partial(func, k=k, qweight=qweight, scales=scales, qzeros=qzeros, groupsize=groupsize) for (k, n), (qweight, scales, qzeros, groupsize) in kn_values.items())


class QuantLinear(nn.Module):
	def __init__(self, bits: int, groupsize: int, infeatures: int, outfeatures: int, bias: bool):
		super().__init__()

		if bits not in [4]:
			raise NotImplementedError("Only 4 bits are supported.")
		
		groupsize = infeatures if groupsize == -1 else groupsize
		
		self.infeatures = infeatures
		self.outfeatures = outfeatures
		self.bits = bits
		self.groupsize = groupsize

		features_per_int = 32 // bits

		assert outfeatures % features_per_int == 0, "outfeatures must be a multiple of features_per_int"

		self.register_buffer('qweight', torch.empty((infeatures // features_per_int, outfeatures), dtype=torch.int32))
		self.register_buffer('qzeros', torch.empty((math.ceil(infeatures / groupsize), outfeatures // features_per_int), dtype=torch.int32))
		self.register_buffer('scales', torch.empty((math.ceil(infeatures / groupsize), outfeatures), dtype=torch.float16))
		if bias:
			self.register_buffer('bias', torch.empty(outfeatures, dtype=torch.float16))
		else:
			self.register_parameter('bias', None)

	def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
		y = triton_matmul4(self.groupsize, x, self.qweight, self.scales, self.qzeros, self.bias)
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

		#triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		#triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),

		# These provided a benefit on a 3090
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
		triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),

		# From PyTorch Inductor
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
		#triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),

		#triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),
		#triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=8),

		#triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
	],
	key=['M', 'N', 'K', 'NO_GROUPS'],
	nearest_power_of_two=['M', 'N', 'K'],
	prune_configs_by={
		'early_config_prune': matmul4_kernel_config_pruner,
		'perf_model': None,
		'top_k': None,
	},
)
@triton.jit
def matmul4_kernel(
	a_ptr, b_ptr, c_ptr,
	scales_ptr, zeros_ptr,
	M, N, K,
	stride_am, stride_ak,
	stride_bk, stride_bn,
	stride_cm, stride_cn,
	stride_scales_g, stride_scales_n,
	stride_zeros_g, stride_zeros_n,
	groupsize, NO_GROUPS: tl.constexpr,
	BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
	GROUP_SIZE_M: tl.constexpr,
):
	"""
	Compute the matrix multiplication C = A x B.
	A is of shape (M, K) float16
	B is of shape (K//8, N) int32
	C is of shape (M, N) float16
	scales is of shape (G, N) float16
	zeros is of shape (G, N//8) int32
	groupsize is an int specifying the size of groups for scales and zeros.
	G is K // groupsize.
	Set NO_GROUPS to groupsize == K, in which case G = 1 and the kernel is more efficient.

	WARNING: This kernel assumes that K is a multiple of BLOCK_SIZE_K.
	WARNING: This kernel assumes that N is a multiple of BLOCK_SIZE_N.
	WARNING: This kernel assumes that groupsize is a multiple of BLOCK_SIZE_K.
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
	scales_ptrs = scales_ptr + offs_bn * stride_scales_n   # (BLOCK_SIZE_N,)
	# zeros_ptrs is set up such that it repeats elements along the N axis 8 times
	zeros_ptrs = zeros_ptr + ((offs_bn // 8) * stride_zeros_n)   # (BLOCK_SIZE_N,)

	# shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
	shifter = (offs_k % 8) * 4
	zeros_shifter = (offs_bn % 8) * 4

	# If G == 1, scales and zeros are the same for all K, so we can load them once
	if NO_GROUPS:
		# Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
		scales = tl.load(scales_ptrs)  # (BLOCK_SIZE_N,)
		zeros = tl.load(zeros_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

		# Unpack zeros
		zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
		zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16

	# Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
	# M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
	# So this loop is along the infeatures dimension (K)
	# It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
	accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	for k in range(0, num_pid_k):
		a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
		b = tl.load(b_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

		if not NO_GROUPS:
			g_id = k // (groupsize // BLOCK_SIZE_K)
			ptr = scales_ptrs + g_id * stride_scales_g
			scales = tl.load(ptr)  # (BLOCK_SIZE_N,)
			ptr = zeros_ptrs + g_id * stride_zeros_g   # (BLOCK_SIZE_N,)
			zeros = tl.load(ptr)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

			# Unpack zeros
			zeros = (zeros >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
			zeros = (zeros + 1) * scales  # (BLOCK_SIZE_N,) float16

		# Now we need to unpack b (which is 4-bit values) into 32-bit values
		b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
		b = b * scales[None, :] - zeros[None, :]  # Scale and shift

		accumulator += tl.dot(a, b)
		a_ptrs += BLOCK_SIZE_K * stride_ak
		b_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
	
	c = accumulator.to(tl.float16)
	
	# Store the result
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	tl.store(c_ptrs, accumulator, mask=c_mask)


def triton_matmul4(groupsize: int, a: torch.FloatTensor, qweight: torch.IntTensor, scales: torch.FloatTensor, qzeros: torch.IntTensor, bias: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
	"""
	Compute the matrix multiplication C = A x B + bias.
	Where B is quantized using GPTQ and groupsize = -1 into 4-bit values.

	A is of shape (..., K) float16
	qweight is of shape (K//8, N) int32
	scales is of shape (G, N) float16
	qzeros is of shape (G, N//8) int32
	bias is of shape (1, N) float16

	groupsize is the number of infeatures in each group.
	G = K // groupsize

	Returns C of shape (..., N) float16
	"""
	assert a.shape[-1] == (qweight.shape[0] * 8), "A must be a multiple of 8 in the last dimension"
	assert a.is_contiguous(), "A must be contiguous"

	# Flatten a into (-1, K)
	x = a.view(-1, a.shape[-1])

	M, K = x.shape
	N = qweight.shape[1]
	# This is based on the possible BLOCK_SIZE_Ks
	assert K % 16 == 0 and K % 32 == 0 and K % 64 == 0 and K % 128 == 0, "K must be a multiple of 16, 32, 64, and 128"
	# This is based on the possible BLOCK_SIZE_Ns
	assert N % 16 == 0 and N % 32 == 0 and N % 64 == 0 and N % 128 == 0 and N % 256 == 0, "N must be a multiple of 16, 32, 64, 128, and 256"
	# This is based on the possible BLOCK_SIZE_Ks
	assert groupsize % 32 == 0 and groupsize % 64 == 0 and groupsize % 128 == 0, "groupsize must be a multiple of 32, 64, and 128"

	c = torch.empty((M, N), device='cuda', dtype=torch.float16)

	grid = lambda META: (
		triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
	)
	matmul4_kernel[grid](
		x, qweight, c,
		scales, qzeros,
		M, N, K,
		x.stride(0), x.stride(1),
		qweight.stride(0), qweight.stride(1),
		c.stride(0), c.stride(1),
		scales.stride(0), scales.stride(1),
		qzeros.stride(0), qzeros.stride(1),
		groupsize, groupsize == K,
	)

	# Reshape c
	c = c.view(a.shape[:-1] + (N,))  # (..., N)

	# Add bias
	if bias is not None:
		c = c + bias
	
	return c