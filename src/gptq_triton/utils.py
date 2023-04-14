import math

import triton


def matmul4_kernel_config_pruner(configs, nargs):
	"""
	The main purpose of this function is to shrink BLOCK_SIZE_* when the corresponding dimension is smaller.
	"""
	m = max(2 ** int(math.ceil(math.log2(nargs['M']))), 16)
	n = max(2 ** int(math.ceil(math.log2(nargs['N']))), 16)
	k = max(2 ** int(math.ceil(math.log2(nargs['K']))), 16)

	used = set()
	for config in configs:
		block_size_m = min(m, config.kwargs['BLOCK_SIZE_M'])
		block_size_n = min(n, config.kwargs['BLOCK_SIZE_N'])
		block_size_k = min(k, config.kwargs['BLOCK_SIZE_K'])
		group_size_m = config.kwargs['GROUP_SIZE_M']

		if (block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps) in used:
			continue

		used.add((block_size_m, block_size_n, block_size_k, group_size_m, config.num_stages, config.num_warps))
		yield triton.Config({'BLOCK_SIZE_M': block_size_m, 'BLOCK_SIZE_N': block_size_n, 'BLOCK_SIZE_K': block_size_k, 'GROUP_SIZE_M': group_size_m}, num_stages=config.num_stages, num_warps=config.num_warps)