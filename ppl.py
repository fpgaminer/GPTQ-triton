#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn
from datasets import load_dataset
from quant import load_quant
from tqdm import tqdm
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, help='Path to model, either a HuggingFace model or a quantized model')
parser.add_argument('--quant', action='store_true', help='Whether the model is quantized')
parser.add_argument('--stride', type=int, default=512, help='Stride for calculating perplexity')
parser.add_argument('--context-length', type=int, default=2048, help='Length of context to use')


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
	
	tokenizer = AutoTokenizer.from_pretrained(args.model)
	context_length = model.seqlen if args.context_length is None else args.context_length

	for dataset in ['wikitext-2', 'ptb', 'c4']:
		ppl = calculate_perplexity(model, tokenizer, dataset, max_length=context_length, stride=args.stride)
		print(f"{dataset} perplexity: {ppl}")


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


def get_dataset(dataset_name: str, tokenizer) -> torch.Tensor:
	if dataset_name == "wikitext-2":
		test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
		encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt").input_ids
	elif dataset_name == 'ptb':
		test = load_dataset("ptb_text_only", 'penn_treebank', split="validation")
		encodings = tokenizer("\n\n".join(test["sentence"]), return_tensors="pt").input_ids
	elif dataset_name == 'c4':
		# WARNING: Many of the files in the allenai/c4 repo are marked as "Unsafe" by HuggingFace, possibly containing a virus.  This particular file is not, and I doubt it's an issue, but worth noting.
		test = load_dataset('allenai/c4', 'allenai--c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation')
		encodings = [tokenizer(x, return_tensors="pt").input_ids for x in test['text'][:1000]]
		encodings = torch.cat(encodings, dim=1)
	else:
		raise ValueError(f"Unknown dataset {dataset_name}")

	return encodings


def calculate_perplexity(model, tokenizer, dataset: str, max_length: int, stride: int = 512) -> float:
	encodings = get_dataset(dataset, tokenizer)
	seq_len = encodings.size(1)

	print(f"Sequence length: {seq_len}")
	print(f"Max length: {max_length}")
	print(f"Stride: {stride}")

	nlls = []
	prev_end_loc = 0

	for begin_loc in tqdm(range(0, seq_len - 1, stride)):
		torch.cuda.synchronize()
		end_loc = min(seq_len - 1, begin_loc + max_length)
		trg_len = end_loc - prev_end_loc  # How many tokens we want to predict
		input_ids = encodings[:, begin_loc:end_loc+1].to('cuda')  # +1 for the labels

		with torch.no_grad():
			# Ask the model for logits
			outputs = model(input_ids[:, :-1])
			# We only want the last trg_len logits
			logits = outputs.logits[..., -trg_len:, :].contiguous()
			# The last trg_len tokens are the labels
			labels = input_ids[:, -trg_len:].contiguous()

			# Compute the NLL for this batch using flattened logits and labels
			loss_fct = nn.CrossEntropyLoss()
			loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
		
		nlls.append(loss)

		prev_end_loc = end_loc
		if end_loc == (seq_len - 1):
			break

	ppl = torch.exp(torch.stack(nlls).mean())

	return ppl


if __name__ == '__main__':
	main()
