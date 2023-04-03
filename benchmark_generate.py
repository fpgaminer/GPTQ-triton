#!/usr/bin/env python3
"""
Benchmarks the generation speed of a model.  While Benchmark.ipynb provides nice detailed performance data, it measures the kernels in isolation.
This script measures "real world" performance by running the whole model in generation mode.
It tests a grid of prompt lengths and generation lengths, and saves the timing results to `results.json`.
"""
import argparse
import time
import random
import itertools
import json

import torch
from quant import load_quant
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig
import transformers
import original_quant


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to model, either a HuggingFace model or a quantized model')
parser.add_argument('--quant', action='store_true', help='Whether the model is quantized')
parser.add_argument('--cuda', type=str, help='Whether to use the old CUDA kernel and format; this must be set to the path to the CUDA quantized model, and --model must be set to a HF model')


def main():
	args = parser.parse_args()

	if args.cuda:
		model = load_cuda_quant(args.model, args.cuda, 4, -1)
		model.eval()
		model.to('cuda')
	elif not args.quant:
		model = get_llama(args.model)
		model.eval()
		model.to('cuda')
	else:
		model = load_quant(args.model)
		model.eval()
		model.to('cuda')
	
	tokenizer = AutoTokenizer.from_pretrained(args.model)

	prompt_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
	max_lengths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

	lengths = list(itertools.product(prompt_lengths, max_lengths))

	# Shuffle the lengths so that we don't always test in the same order and get caching effects
	random.shuffle(lengths)

	results = {}

	for prompt_length, max_length in lengths:
		print(f'Prompt length: {prompt_length}, max length: {max_length}')

		# Generate a long random string
		# We do this every time to avoid caching effects
		prompt = ''.join(random.choice('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?') for _ in range(2048 * 10))

		# Encode and crop down
		encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False, return_tensors='pt')
		encoded_prompt = encoded_prompt[:, :prompt_length]
		encoded_prompt = encoded_prompt.to('cuda')

		start_time = time.time()
		output_sequences = model.generate(
			input_ids=encoded_prompt,
			max_length=max_length + prompt_length,
			do_sample=True,
			num_return_sequences=1,
			suppress_tokens=[model.generation_config.eos_token_id],  # This prevents the sampler from ending early; it must generate max_length tokens
		)
		end_time = time.time()

		gen_time = end_time - start_time
		speed = max_length / gen_time

		results[(prompt_length, max_length)] = (gen_time, speed)

		print(f'Generation took {end_time - start_time:.2f} seconds')
		print(f'Average generation speed: {max_length / (end_time - start_time):.2f} tokens per second')
		print()
	
	with open('results.json', 'w') as f:
		json.dump({str(k): str(v) for k, v in results.items()}, f, indent=4)


def get_llama(model: str):
	"""
	Load a pretrained Llama model
	"""
	def skip(*args, **kwargs):
		pass
	# NOTE: This is a nasty hack, but it speeds up model building by a huge amount
	old_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
	torch.nn.init.kaiming_uniform_ = skip
	torch.nn.init.uniform_ = skip
	torch.nn.init.normal_ = skip

	model = LlamaForCausalLM.from_pretrained(model, torch_dtype='auto')
	model.seqlen = 2048

	# Restore the old initializers
	torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = old_inits

	return model


def load_cuda_quant(model, checkpoint, wbits, groupsize):
	"""
	Load a quantized model using the old CUDA kernel
	"""
	config = LlamaConfig.from_pretrained(model)
	def noop(*args, **kwargs):
		pass
	original_inits = (torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_)
	torch.nn.init.kaiming_uniform_ = noop 
	torch.nn.init.uniform_ = noop 
	torch.nn.init.normal_ = noop 

	torch.set_default_dtype(torch.half)
	original_init_weights = transformers.modeling_utils._init_weights
	transformers.modeling_utils._init_weights = False
	torch.set_default_dtype(torch.half)
	model = LlamaForCausalLM(config)
	torch.set_default_dtype(torch.float)

	transformers.modeling_utils._init_weights = original_init_weights
	torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = original_inits

	model = model.eval()
	layers = original_quant.find_layers(model)
	for name in ['lm_head']:
		if name in layers:
			del layers[name]
	original_quant.make_quant(model, layers, wbits, groupsize, faster=False)

	del layers

	print('Loading model ...')
	if checkpoint.endswith('.safetensors'):
		from safetensors.torch import load_file as safe_load
		model.load_state_dict(safe_load(checkpoint))
	else:
		model.load_state_dict(torch.load(checkpoint))
	model.seqlen = 2048
	print('Done.')

	return model


if __name__ == '__main__':
	main()