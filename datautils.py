import random

from datasets import load_dataset


def get_dataset(dataset_name: str, tokenizer, nsamples: int, seed: int, seqlen: int):
	if dataset_name == "wikitext-2":
		return get_wikitext2(nsamples, seed, seqlen, tokenizer)
	elif dataset_name == 'ptb':
		return get_ptb(nsamples, seed, seqlen, tokenizer, jointext='\n\n')
	elif dataset_name == 'ptb-new':
		return get_ptb(nsamples, seed, seqlen, tokenizer, jointext=' ')
	elif dataset_name == 'c4':
		return get_c4(nsamples, seed, seqlen, tokenizer)
	else:
		raise ValueError(f"Unknown dataset {dataset_name}")


def get_wikitext2(nsamples: int, seed: int, seqlen: int, tokenizer, jointext: str = '\n\n'):
	traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')

	trainenc = tokenizer(jointext.join(traindata['text']), return_tensors='pt')

	rng = random.Random(seed)
	trainloader = (rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1) for _ in range(nsamples))
	trainloader = [trainenc.input_ids[:, i:i+seqlen] for i in trainloader]

	return trainloader


def get_ptb(nsamples: int, seed: int, seqlen: int, tokenizer, jointext: str):
	traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')

	trainenc = tokenizer(jointext.join(traindata['sentence']), return_tensors='pt')

	rng = random.Random(seed)
	trainloader = (rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1) for _ in range(nsamples))
	trainloader = [trainenc.input_ids[:, i:i+seqlen] for i in trainloader]

	return trainloader


def get_c4(nsamples: int, seed: int, seqlen: int, tokenizer):
	# WARNING: Many of the files in the allenai/c4 repo are marked as "Unsafe" by HuggingFace, possibly containing a virus.  This particular file is not, and I doubt it's an issue, but worth noting.
	traindata = load_dataset('allenai/c4', 'allenai--c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')

	rng = random.Random(seed)

	trainloader = []
	for _ in range(nsamples):
		while True:
			i = rng.randint(0, len(traindata) - 1)
			trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
			if trainenc.input_ids.shape[1] >= seqlen:
				break
		
		i = rng.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
		inp = trainenc.input_ids[:, i:i + seqlen]
		trainloader.append(inp)

	return trainloader