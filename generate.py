#!/usr/bin/env python3
"""
Example of how to use the quantized model to generate text.
"""
import argparse
import time

import torch
from gptq_triton import load_quant
from transformers import AutoTokenizer, LlamaForCausalLM


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help='Path to model, either a HuggingFace model or a quantized model')
parser.add_argument('--quant', action='store_true', help='Whether the model is quantized')
parser.add_argument('--prompt', type=str, default='The quick brown fox', help='Prompt to use for generation')
parser.add_argument('--max-length', type=int, default=2048, help='Maximum length of generated text')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for generation')
parser.add_argument('--top-k', type=int, default=0, help='Top-k for generation')
parser.add_argument('--top-p', type=float, default=0.0, help='Top-p for generation')
parser.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty for generation')


def main():
	args = parser.parse_args()

	if not args.quant:
		model = get_llama(args.model)
		model.eval()
		model.to('cuda')
	else:
		model = load_quant(args.model)
		model.eval()
		model.to('cuda')
	
	tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

	encoded_prompt = tokenizer.encode(args.prompt, add_special_tokens=False, return_tensors='pt').to('cuda')

	start_time = time.time()
	output_sequences = model.generate(
		input_ids=encoded_prompt,
		max_length=args.max_length + len(encoded_prompt[0]),
		temperature=args.temperature,
		top_k=args.top_k,
		top_p=args.top_p,
		repetition_penalty=args.repetition_penalty,
		do_sample=True,
		num_return_sequences=1,
	)
	end_time = time.time()

	if len(output_sequences.shape) > 2:
		output_sequences.squeeze_()
	
	total_tokens_generated = 0
	
	for generated_sequence in output_sequences:
		generated_sequence = generated_sequence.tolist()
		total_tokens_generated += len(generated_sequence) - len(encoded_prompt[0])

		text = tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)

		total_sequence = (
			args.prompt + text[len(tokenizer.decode(encoded_prompt[0], clean_up_tokenization_spaces=True)):]
		)

		print(total_sequence)
	
	print()
	print(f'Generation took {end_time - start_time:.2f} seconds')
	print(f'Total tokens generated: {total_tokens_generated}')
	print(f'Average generation speed: {total_tokens_generated / (end_time - start_time):.2f} tokens per second')


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


if __name__ == '__main__':
	main()