# GPTQ-triton

This is my attempt at implementing a Triton kernel for GPTQ inference.  This code is based on the [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa) codebase, which is itself based on the [GPTQ](https://github.com/IST-DASLab/gptq) codebase.

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

This benchmark was run on a 3090 using the `benchmark_generate.py` script.

![Triton benchmark graph](TritonBench.png)


## Accuracy (PPL)

The following results were obtained using the `ppl.py` script with a stride of 512 and a context length of 2048.
For the 4bit CUDA results, a custom version of `ppl.py` was used, as the current script is dedicated to the Triton kernel convensions.
it/s numbers are from a 3090.


| [LLaMA-7B](https://arxiv.org/abs/2302.13971)       | Bits | group-size | memory(MiB) | it/s | Wikitext2 |  PTB  |  C4  | 
| -------------------------------------------------- | ---- | ---------- | ----------- | ---- | --------- | ----- | ---- |
| FP16                                               |  16  |      -     |    17373    | 1.64 |    5.04   |  7.85 | 6.99 |
| GPTQ CUDA                                          |   4  |     -1     |     8805    | 0.11 |    5.44   |  8.24 |   -  |
| GPTQ Triton                                        |   4  |     -1     |     8099    | 1.63 |    5.44   |  8.24 | 7.48 |


| [LLaMA-13B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | it/s | Wikitext2 |  PTB  |  C4  |
| -------------------------------------------------- | ---- | ---------- | ----------- | ---- | --------- | ----- | ---- |
| FP16                                               |  16  |      -     |    31633    |   -  |    4.52   |  7.19 | 6.66 |
| GPTQ Triton                                        |   4  |     -1     |    13241    | 0.89 |    4.74   |  7.49 | 7.00 |


| [LLaMA-30B](https://arxiv.org/abs/2302.13971)      | Bits | group-size | memory(MiB) | it/s | Wikitext2 |  PTB  |  C4  |
| -------------------------------------------------- | ---- | ---------- | ----------- | ---- | --------- | ----- | ---- |
| FP16                                               |  16  |      -     |    72491    |   -  |    3.61   |  6.50 | 6.07 |


## Requirements

I haven't formalised the requirements yet, but generally nightly `transformers`; GPTQ-for-LLaMa to be able to quantize models and if you want to run comparison tests; triton 2.0; the usual other PyTorch requirements.

**WARNING**: Please use a `transformers` commit _before_ 7dcd870.  There is a 10% performance regression at that commit.


## Converting a model

You need a 4-bit quantized model, which you can either download or create yourself using the original GPTQ-for-LLaMa repo.  The Triton kernel is currently only implemented for 4-bits and groupsize -1.  Then the quantized model needs to be converted.  The Triton implementation is slightly different from the CUDA implementation, so a conversion script is provided.

`./convert_weights.py --model <Path to a HF FP16 model> --quant <Path to the quantized pt file> --output <Path to the output folder>`

The conversion script will create a folder and save the converted model, along with configuration files.


## Files

* `benchmark_generate.py` - A script for benchmarking generation speed at different prompt lengths and generation lengths.

* `Benchmark.ipynb` - A notebook for benchmarking the Triton kernel against the CUDA kernel and FP16.

* `convert_weights.py` - A script for converting a GPTQ-for-LLaMa quantized model to a format compatible with this repo.

* `generate.py` - An example script for generating text from a model.  Example usage: `./generate.py --model <Path to your quantized model> --quant --prompt "Write a story about a duck: Once upon a time there was a duck" --temperature 0.6 --top-p 0.6 --repetition-penalty 1.1`

* `ppl.py` - A script for calculating the perplexity of a model against wikitext2, PTB, and C4.  This is useful for verifying correctness of the Triton kernel, comparing it to the CUDA kernel and the original FP16 model.

* `Verify.ipynb` - A notebook for verifying the correctness of the Triton kernel.