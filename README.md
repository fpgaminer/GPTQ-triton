# GPTQ-triton

This is my attempt at implementing a Triton kernel for GPTQ inference.  This code is based on the [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) codebase, which is itself based on the [GPTQ](https://github.com/IST-DASLab/gptq) codebase.

Citation:

```
@article{frantar-gptq,
  title={{GPTQ}: Accurate Post-training Compression for Generative Pretrained Transformers}, 
  author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
  year={2022},
  journal={arXiv preprint arXiv:2210.17323}
}
```

## Motivation

As of today (2023-03-27) the CUDA kernels in the aforementioned codebases do not scale well with context length, running up to 10x slower when the context is large versus the equivilent FP16 model.  To solve this I'm implementing the inference kernel in Triton, which should allow for much better scaling.

The implementation is based around the matmul tutorial from the Triton documentation.  The main difference is decoding the quantized weights before performing each sub-block of the matrix multiplication.


## Performance

This benchmark was run on a 3090 using the `Benchmark.ipynb` notebook.

![Triton benchmark graph](TritonBench.png)


## Accuracy

The following results were obtained using the `ppl.py` script with a stride of 512 and a context length of 2048.
For the 4bit CUDA results, a custom version of `ppl.py` was used, as the current script is dedicated to the Triton kernel convensions.


| [LLaMA-7B](https://arxiv.org/abs/2302.13971)       | Bits | group-size | memory(MiB) | it/s | Wikitext2 |  PTB  |  C4  | 
| -------------------------------------------------- | ---- | ---------- | ----------- | ---- | --------- | ----- | ---- |
| FP16                                               |  16  |      -     |    17373    | 1.64 |    5.04   |  7.85 | 6.99 |
| GPTQ CUDA                                          |   4  |     -1     |     8805    | 0.11 |    5.44   |  8.24 |   -  |
| GPTQ Triton                                        |   4  |     -1     |     8611    | 1.52 |    5.44   |  8.24 | 7.48 |


## Usage

### Converting a model

You need a 4-bit quantized model, which you can either download or create yourself using the original GPTQ-for-LLaMa repo.  The Triton kernel is currently only implemented for 4-bits and groupsize -1.  Then the quantized model needs to be converted.  The Triton implementation is slightly different from the CUDA implementation, so a conversion script is provided.

`./convert_weights.py --model <Path to a HF FP16 model> --quant <Path to the quantized pt file> --output <Path to the output folder>`

The conversion script will create a folder and save the converted model, along with configuration files.


### Running PPL

The `ppl.py` script can be used to calculate the perplexity of a model against wikitext2, PTB, and C4.  Example usage:

`./ppl.py --model <Path to an HF FP16 model>`

This will calculate PPL for an original FP16 model.

`./ppl.py --model <Path to the quantized folder> --quant`

This will calculate PPL for a quantized model.

This is useful for verifying correctness of the Triton kernel, comparing it to the CUDA kernel, and comparing it to the original FP16 model.


### Elsewhere

See the `ppl.py` code for how to load and use the model in your own code.