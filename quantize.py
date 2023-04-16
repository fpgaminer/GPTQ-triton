#!/usr/bin/env python3
# Based on https://github.com/IST-DASLab/gptq
# Quantize a model using the GPTQ algorithm.
import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Optional

import torch
import torch.nn as nn
from datautils import get_dataset
from gptq import GPTQ, Quantizer
from gptq_triton import QuantLinear, quant_linear
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaForCausalLM


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, required=True, help='llama model to load')
parser.add_argument('--dataset', type=str, choices=['wikitext-2', 'ptb', 'ptb-new', 'c4'], required=True, help='Where to extract calibration data from.')
parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration data samples.')
parser.add_argument('--percdamp', type=float, default=.01, help='Percent of the average Hessian diagonal to use for dampening.')
parser.add_argument('--wbits', type=int, required=True, choices=[2, 4, 8], help='#bits to use for quantization.')
parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize to use for quantization; default uses full row.')
parser.add_argument('--save', type=str, required=True, help='Save quantized result to this folder.')
parser.add_argument('--safetensors', action='store_true', help='Whether to save tensors in safetensors format.')
parser.add_argument('--sym', action='store_true', help='Whether to perform symmetric quantization.')
parser.add_argument('--act-order', action='store_true', help='Whether to apply the activation order GPTQ heuristic')
parser.add_argument('--true-sequential', action='store_true', help='Whether to run in true sequential model.')


def main():
	args = parser.parse_args()
	args.save = Path(args.save)

	print('Loading model...')
	model = get_llama(args.model)
	model.eval()
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

	print('Loading data...')
	dataloader = get_dataset(args.dataset, tokenizer, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen)

	print('Quantizing...')
	tick = time.time()
	quantizers = llama_sequential(model, dataloader, device='cuda', wbits=args.wbits, nsamples=args.nsamples, true_sequential=args.true_sequential, sym=args.sym, percdamp=args.percdamp, groupsize=args.groupsize, act_order=args.act_order)
	print(f"Total time: {time.time() - tick:.2f}s")

	print('Packing...')
	llama_pack(model, quantizers, args.wbits, args.groupsize)

	print('Saving...')
	args.save.mkdir(parents=True, exist_ok=True)

	# Save the model
	if args.safetensors:
		from safetensors.torch import save_file as safe_save
		safe_save(model.state_dict(), args.save / 'model.safetensors')
	else:
		torch.save(model.state_dict(), args.save / 'model.pt')
	
	# Write quant_config.json
	with open(args.save / 'quant_config.json', 'w') as f:
		f.write(json.dumps({
			'wbits': args.wbits,
			'groupsize': args.groupsize,
		}))

	# Copy the config
	for file in ['config.json', 'generation_config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.model']:
		shutil.copy(args.model + '/' + file, args.save / file)
	
	print('Done.')


def get_llama(model):
	def skip(*args, **kwargs):
		pass

	# NOTE: This is a nasty hack, but it speeds up model building by a huge amount
	torch.nn.init.kaiming_uniform_ = skip
	torch.nn.init.uniform_ = skip
	torch.nn.init.normal_ = skip

	model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
	model.seqlen = 2048

	return model


@torch.no_grad()
def llama_sequential(model, dataloader, device, wbits: int, nsamples: int, true_sequential: bool, sym: bool, percdamp: float, groupsize: int, act_order: bool):
	# Disable caching
	use_cache = model.config.use_cache
	model.config.use_cache = False

	# Prepare
	layers = model.model.layers
	dtype = next(iter(model.parameters())).dtype
	inps = torch.zeros((nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
	outs = torch.zeros_like(inps)

	# Move the first layer to GPU
	model.model.embed_tokens = model.model.embed_tokens.to(device)
	model.model.norm = model.model.norm.to(device)
	layers[0] = layers[0].to(device)

	# Create a dummy layer that catches the input and attention mask, and then bails
	# This allows us to capture all the inputs to the first layer for the calibration data
	cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
	class Catcher(nn.Module):
		def __init__(self, module):
			super().__init__()
			self.module = module
		
		def forward(self, inp, **kwargs):
			inps[cache['i']] = inp
			cache['i'] += 1
			if cache['attention_mask'] is not None:
				assert torch.all(cache['attention_mask'] == kwargs['attention_mask'])
			cache['attention_mask'] = kwargs['attention_mask']
			if cache['position_ids'] is not None:
				assert torch.all(cache['position_ids'] == kwargs['position_ids'])
			cache['position_ids'] = kwargs['position_ids']
			raise ValueError
	
	layers[0] = Catcher(layers[0])
	for batch in dataloader:
		try:
			model(batch.to(device))
		except ValueError:
			pass
	layers[0] = layers[0].module

	# Move things back to the CPU (but not the first layer, since we'll just move it back to GPU immediately below)
	model.model.embed_tokens = model.model.embed_tokens.cpu()
	model.model.norm = model.model.norm.cpu()
	torch.cuda.empty_cache()

	attention_mask = cache['attention_mask']
	position_ids = cache['position_ids']
	quantizers = {}

	# Layers are quantized in order, and only one layer lives on the GPU at a time to save memory
	# Otherwise quantizing large models would be impossible (NOTE for future readers: are you enjoying your 1TB VRAM?)
	for i, layer in tqdm(enumerate(layers), total=len(layers)):
		layer = layer.to(device)
		full = {name: m for name, m in layer.named_modules() if isinstance(m, nn.Linear)}

		if true_sequential:
			sequential = [
				['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
				['self_attn.o_proj'],
				['mlp.up_proj', 'mlp.gate_proj'],
				['mlp.down_proj']
			]
		else:
			sequential = [list(full.keys())]
		
		# For each subset of linear layers
		for names in sequential:
			subset = {n: full[n] for n in names}
			gptq = {}

			# Prepare a quantizer for each linear layer
			for name in subset:
				gptq[name] = GPTQ(subset[name])
				gptq[name].quantizer = Quantizer()
				gptq[name].quantizer.configure(wbits, perchannel=True, sym=sym, mse=False)
			
			# Feed data to the quantizer, and save outs
			def add_batch(name):
				def tmp(_, inp, out):
					gptq[name].add_batch(inp[0].data, out.data)
				return tmp
			
			handles = []
			for name in subset:
				handles.append(subset[name].register_forward_hook(add_batch(name)))
			for j in range(nsamples):
				outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]  # TODO: Saving outs doesn't seem needed here?
			for h in handles:
				h.remove()

			# With the data collected, quantize the layers
			for name in subset:
				print(i, name)
				scale, zero = gptq[name].fasterquant(percdamp=percdamp, groupsize=groupsize, actorder=act_order)
				quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer, scale, zero)
				gptq[name].free()
		
		# Save outputs of the layer after quantization, so we can feed them into the next layer
		for j in range(nsamples):
			outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

		# Move the layer back to the CPU, and free up memory
		layers[i] = layer.cpu()
		del layer
		del gptq 
		torch.cuda.empty_cache()

		# Swap buffers
		inps, outs = outs, inps

	# Restore settings
	model.config.use_cache = use_cache

	return quantizers


def llama_pack(model, quantizers, wbits: int, groupsize: int):
	# Find all the quantized layers
	layers = {name: m for name, m in model.named_modules() if isinstance(m, nn.Linear)}
	layers = {n: layers[n] for n in quantizers}

	# Replace all applicable instances of Linear with QuantLinear in the model
	quant_linear.make_quant(model, wbits, groupsize)

	for name, m in tqdm(model.named_modules(), total=len(list(model.named_modules()))):
		if not isinstance(m, QuantLinear):
			continue

		quantizer, scale, zero = quantizers[name]
		quantizer, scale, zero = quantizer.cpu(), scale.cpu(), zero.cpu()
		pack_linear(m, layers[name].weight.data, scale, zero, m.bias)


def pack_linear(quant, weights: torch.FloatTensor, scales: torch.FloatTensor, zeros, bias: Optional[torch.FloatTensor]):
	"""
	Packs the quantized weights, scales, and zero points into a QuantLinear layer
	"""
	scales = scales.t().contiguous()
	zeros = zeros.t().contiguous()
	scale_zeros = zeros * scales

	quant.scales = scales.clone().to(torch.float16)

	if quant.bias is not None:
		quant.bias = bias.clone().to(torch.float16)
	
	# Round weights to nearest integer based on scale and zero point
	# Each weight will be one int, but should not exceed quant.bits
	intweight = []
	for idx in range(quant.infeatures):
		g_idx = idx // quant.groupsize
		intweight.append(torch.round((weights[:,idx] + scale_zeros[g_idx]) / scales[g_idx]).to(torch.int32)[:,None])
	intweight = torch.cat(intweight,dim=1)
	intweight = intweight.t().contiguous()

	# Now pack the weights into uint32's
	#qweight = torch.zeros((intweight.shape[0] // 32 * quant.bits, intweight.shape[1]), dtype=torch.int32)
	quant.qweight.zero_()
	i = 0
	row = 0
	while row < quant.qweight.shape[0]:
		if quant.bits in [2,4,8]:
			for j in range(i, i + (32 // quant.bits)):
				quant.qweight[row] |= intweight[j] << (quant.bits * (j - i))
			i += 32 // quant.bits
			row += 1
		else:
			raise NotImplementedError("Only 2,4,8 bits are supported.")
	
	# Subtract 1 from the zero point
	zeros -= 1

	# Pack the zero points into uint32's
	zeros = zeros.to(torch.int32)
	#qzeros = np.zeros((zeros.shape[0], zeros.shape[1] // 256 * (self.bits * 8)), dtype=np.uint32)
	quant.qzeros.zero_()
	i = 0
	col = 0
	while col < quant.qzeros.shape[1]:
		if quant.bits in [2,4,8]:
			for j in range(i, i + (32 // quant.bits)):
				quant.qzeros[:, col] |= zeros[:, j] << (quant.bits * (j - i))
			i += 32 // quant.bits
			col += 1
		else:
			raise NotImplementedError("Only 2,4,8 bits are supported.")
			

if __name__ == '__main__':
	main()