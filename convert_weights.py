#!/usr/bin/env python3
from transformers import LlamaConfig, LlamaForCausalLM
import transformers
import argparse
import torch
from pathlib import Path
import json
import shutil


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Path to a HuggingFace model')
parser.add_argument('--quant', type=str, help='Path to quantized model')
parser.add_argument('--output', type=str, help='Path to a directory where the converted model will be saved')


def main():
	args = parser.parse_args()

	# Load quantized data
	use_safe_tensors = False
	if args.quant.endswith('.safetensors'):
		use_safe_tensors = True
		from safetensors.torch import load_file as safe_load
		state_dict = (safe_load(args.quant))
	else:
		state_dict = torch.load(args.quant, map_location='cpu')

	# Build a reference model
	config = LlamaConfig.from_pretrained(args.model)

	def noop(*args, **kwargs):
			pass
	torch.nn.init.kaiming_uniform_ = noop 
	torch.nn.init.uniform_ = noop 
	torch.nn.init.normal_ = noop 

	torch.set_default_dtype(torch.half)
	transformers.modeling_utils._init_weights = False
	torch.set_default_dtype(torch.half)
	model = LlamaForCausalLM(config)

	# Every Linear layer except lm_head should have been quantized
	# For our triton kernel we need to:
	#  * convert the scales to fp16
	#  * convert the bias to fp16 if it exists
	for name, m in model.named_modules():
		if not isinstance(m, torch.nn.Linear):
			continue

		if name == 'lm_head':
			continue

		# qweight, qzero, scales, bias
		state_dict[name + '.scales'] = state_dict[name + '.scales'].to(torch.float16)

		if state_dict[name + '.bias'] is not None:
			print(f"Converting bias for {name}")
			state_dict[name + '.bias'] = state_dict[name + '.bias'].to(torch.float16)
	
	output_path = Path(args.output)
	output_path.mkdir(parents=True, exist_ok=True)

	# Save the model
	if use_safe_tensors:
		from safetensors.torch import save_file as safe_save
		safe_save(state_dict, output_path / 'model.safetensors')
	else:
		torch.save(state_dict, output_path / 'model.pt')

	# Write quant_config.json
	with open(output_path / 'quant_config.json', 'w') as f:
		f.write(json.dumps({
			'wbits': 4,
			'groupsize': -1,
		}))

	# Copy the config
	for file in ['config.json', 'generation_config.json', 'special_tokens_map.json', 'tokenizer_config.json', 'tokenizer.model']:
		shutil.copy(args.model + '/' + file, output_path / file)
	
	print(f"Converted model saved to {output_path}")


if __name__ == '__main__':
	main()