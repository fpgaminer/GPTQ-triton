import functools

import torch
import torch.nn as nn
import triton
import triton.language as tl
from transformers.models.llama.modeling_llama import LlamaMLP

from . import custom_autotune
from .utils import matmul4_kernel_config_pruner


def make_fused_mlp(m, parent_name=''):
	"""
	Replace all LlamaMLP modules with QuantLlamaMLP modules, which fuses many of the operations.
	"""
	if isinstance(m, LlamaMLP):
		return QuantLlamaMLP(m.gate_proj, m.down_proj, m.up_proj)

	for name, child in m.named_children():
		child = make_fused_mlp(child, parent_name=f"{parent_name}.{name}")

		if isinstance(child, QuantLlamaMLP):
			setattr(m, name, child)
			#print(f"Replacing {name} with fused_mlp; parent: {parent_name}")
	
	return m


def autotune_warmup(model):
	# Find all the QuantLlamaMLP layers
	modules = (m for m in model.modules() if isinstance(m, QuantLlamaMLP))
	k_values = {m.infeatures: {
		'gate_proj_qweight': m.gate_proj_qweight,
		'gate_proj_scales': m.gate_proj_scales,
		'gate_proj_qzeros': m.gate_proj_qzeros,
		'up_proj_qweight': m.up_proj_qweight,
		'up_proj_scales': m.up_proj_scales,
		'up_proj_qzeros': m.up_proj_qzeros,
		'groupsize': m.groupsize,
	} for m in modules}

	print(f'FusedMLP Warmup: Found {len(k_values)} unique K values.')

	def func(m, k, gate_proj_qweight, gate_proj_scales, gate_proj_qzeros, up_proj_qweight, up_proj_scales, up_proj_qzeros, groupsize):
		a = torch.randn(1, m, k, dtype=torch.float16, device='cuda')
		triton_llama_mlp_4(groupsize, a, gate_proj_qweight, gate_proj_scales, gate_proj_qzeros, up_proj_qweight, up_proj_scales, up_proj_qzeros)
	
	return (functools.partial(func, k=k, **v) for k, v in k_values.items())


class QuantLlamaMLP(nn.Module):
	def __init__(
		self,
		gate_proj,
		down_proj,
		up_proj,
	):
		super().__init__()

		assert gate_proj.groupsize == up_proj.groupsize

		# Only save the quantized weights, not the QuantLinear modules
		# This prevents the QuantLinear autotuning warmup from considering these modules
		self.register_buffer('gate_proj_qweight', gate_proj.qweight)
		self.register_buffer('gate_proj_scales', gate_proj.scales)
		self.register_buffer('gate_proj_qzeros', gate_proj.qzeros)
		self.register_buffer('up_proj_qweight', up_proj.qweight)
		self.register_buffer('up_proj_scales', up_proj.scales)
		self.register_buffer('up_proj_qzeros', up_proj.qzeros)
		self.groupsize = gate_proj.groupsize

		self.infeatures = gate_proj.infeatures
		self.outfeatures = down_proj.outfeatures

		self.down_proj = down_proj

	def forward(self, x):
		return self.down_proj(triton_llama_mlp_4(self.groupsize, x, self.gate_proj_qweight, self.gate_proj_scales, self.gate_proj_qzeros, self.up_proj_qweight, self.up_proj_scales, self.up_proj_qzeros))


# This Triton kernel fuses the gate_proj, up_proj, activation, and multiplication of LlamaMLP
# It operates on quantized weights
@custom_autotune.autotune(
	configs=[
		triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),   # 3090
		triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),   # 3090

		triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),   # 3090
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),   # 3090
		triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),    # 3090

		# This configuration provides a benefit to groupsize=128, but groupsize isn't recommended for fused mlp right now
		#triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),    # 3090
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
def llama_mlp_fused_4_kernel(
	a_ptr, c_ptr,
	b1_ptr, scales1_ptr, zeros1_ptr,
	b2_ptr, scales2_ptr, zeros2_ptr,
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
	Computes: C = silu(A * B1) * (A * B2)
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
	b1_ptrs = b1_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
	b2_ptrs = b2_ptr + ((offs_k[:, None] // 8) * stride_bk + offs_bn[None, :] * stride_bn)   # (BLOCK_SIZE_K, BLOCK_SIZE_N)
	scales1_ptrs = scales1_ptr + offs_bn * stride_scales_n  # (BLOCK_SIZE_N,)
	scales2_ptrs = scales2_ptr + offs_bn * stride_scales_n  # (BLOCK_SIZE_N,)
	# zeros_ptrs is set up such that it repeats elements along the N axis 8 times
	zeros1_ptrs = zeros1_ptr + (offs_bn // 8) * stride_zeros_n  # (BLOCK_SIZE_N,)
	zeros2_ptrs = zeros2_ptr + (offs_bn // 8) * stride_zeros_n  # (BLOCK_SIZE_N,)

	# shifter is used to extract the 4 bits of each element in the 32-bit word from B and zeros
	shifter = (offs_k % 8) * 4
	zeros_shifter = (offs_bn % 8) * 4

	# If G == 1, scales and zeros are the same for all K, so we can load them once
	if NO_GROUPS:
		# Fetch scales and zeros; these are per-outfeature and thus reused in the inner loop
		scales1 = tl.load(scales1_ptrs)  # (BLOCK_SIZE_N,)
		scales2 = tl.load(scales2_ptrs)  # (BLOCK_SIZE_N,)
		zeros1 = tl.load(zeros1_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
		zeros2 = tl.load(zeros2_ptrs)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

		# Unpack zeros
		zeros1 = (zeros1 >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
		zeros1 = (zeros1 + 1) * scales1  # (BLOCK_SIZE_N,) float16
		zeros2 = (zeros2 >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
		zeros2 = (zeros2 + 1) * scales2  # (BLOCK_SIZE_N,) float16

	# Now calculate a block of output of shape (BLOCK_SIZE_M, BLOCK_SIZE_N)
	# M is along the batch dimension, N is along the outfeatures dimension, K is along the infeatures dimension
	# So this loop is along the infeatures dimension (K)
	# It's calculating BLOCK_SIZE_M batches in parallel, and for each batch, BLOCK_SIZE_N outfeatures in parallel
	accumulator1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	accumulator2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
	for k in range(0, num_pid_k):
		a = tl.load(a_ptrs, mask=a_mask, other=0.)   # (BLOCK_SIZE_M, BLOCK_SIZE_K)
		b = tl.load(b1_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated

		if not NO_GROUPS:
			g_id = k // (groupsize // BLOCK_SIZE_K)
			scales1 = tl.load(scales1_ptrs + g_id * stride_scales_g)  # (BLOCK_SIZE_N,)
			scales2 = tl.load(scales2_ptrs + g_id * stride_scales_g)  # (BLOCK_SIZE_N,)
			zeros1 = tl.load(zeros1_ptrs + g_id * stride_zeros_g)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32
			zeros2 = tl.load(zeros2_ptrs + g_id * stride_zeros_g)  # (BLOCK_SIZE_N,), each element is repeated 8 times, int32

			# Unpack zeros
			zeros1 = (zeros1 >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
			zeros2 = (zeros2 >> zeros_shifter) & 0xF  # (BLOCK_SIZE_N,) int32
			zeros1 = (zeros1 + 1) * scales1  # (BLOCK_SIZE_N,) float16
			zeros2 = (zeros2 + 1) * scales2  # (BLOCK_SIZE_N,) float16

		# Now we need to unpack b (which is 4-bit values) into 32-bit values
		b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
		b = b * scales1[None, :] - zeros1[None, :]  # Scale and shift

		accumulator1 += tl.dot(a, b)

		b = tl.load(b2_ptrs)   # (BLOCK_SIZE_K, BLOCK_SIZE_N), but repeated
		b = (b >> shifter[:, None]) & 0xF  # Extract the 4-bit values
		b = b * scales2[None, :] - zeros2[None, :]  # Scale and shift

		accumulator2 += tl.dot(a, b)

		a_ptrs += BLOCK_SIZE_K * stride_ak
		b1_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
		b2_ptrs += (BLOCK_SIZE_K // 8) * stride_bk
	
	# Apply activation to accumulator1
	accumulator1 = silu(accumulator1)

	# Multiply accumulator1 and accumulator2
	c = accumulator1 * accumulator2
	#c = c.to(tl.float16)  # Seems like Triton does this conversion automatically if c_ptrs is float16
	
	# Store the result
	offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
	offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
	c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
	tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def triton_llama_mlp_4(
	groupsize: int,
	a: torch.FloatTensor,
	gate_qweight: torch.IntTensor,
	gate_scales: torch.FloatTensor,
	gate_qzeros: torch.IntTensor,
	up_qweight: torch.IntTensor,
	up_scales: torch.FloatTensor,
	up_qzeros: torch.IntTensor,
) -> torch.FloatTensor:
	"""
	Computes: silu(gate(a)) * up(a)
	Where gate and up are quantized using GPTQ and groupsize = -1 into 4-bit values.

	A is of shape (..., K) float16
	*_qweight is of shape (K//8, N) int32
	*_scales is of shape (G, N) float16
	*_qzeros is of shape (G, N//8) int32

	groupsize is the number of infeatures in each group.
	G = K // groupsize

	Returns C of shape (..., N) float16
	"""
	assert gate_qweight.shape == up_qweight.shape and gate_scales.shape == up_scales.shape and gate_qzeros.shape == up_qzeros.shape, "All weights must have the same shape"
	assert a.shape[-1] == (gate_qweight.shape[0] * 8), "A must be a multiple of 8 in the last dimension"
	assert a.is_contiguous(), "A must be contiguous"

	# Flatten a into (-1, K)
	x = a.view(-1, a.shape[-1])

	M, K = x.shape
	N = gate_qweight.shape[1]
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
	llama_mlp_fused_4_kernel[grid](
		x, c,
		gate_qweight, gate_scales, gate_qzeros,
		up_qweight, up_scales, up_qzeros,
		M, N, K,
		x.stride(0), x.stride(1),
		gate_qweight.stride(0), gate_qweight.stride(1),
		c.stride(0), c.stride(1),
		gate_scales.stride(0), gate_scales.stride(1),
		gate_qzeros.stride(0), gate_qzeros.stride(1),
		groupsize, groupsize == K,
	)

	# Reshape c
	c = c.view(a.shape[:-1] + (N,))  # (..., N)

	return c