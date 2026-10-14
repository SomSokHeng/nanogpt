# nanogpt

A character-level transformer written from scratch on numpy. No PyTorch, no TensorFlow, no autodiff. Every forward and backward pass is explicit and readable.

One file. The only dependency is numpy.

## Run

```
python3 nanogpt.py
```

Trains a small transformer on the built-in corpus, prints the loss curve, samples text, and writes `run.json`.

```
python3 nanogpt.py --steps 600 --gen 400 --temp 0.7 --topk 40
python3 nanogpt.py --dim 64 --heads 4 --layers 2 --ctx 128
python3 nanogpt.py --gradcheck
```

## What is implemented

Token embedding, learned positional embedding, pre-norm transformer blocks, causal multi-head self-attention, GELU feed-forward with 4x expansion, final layer norm, weight-tied output projection, cross-entropy loss, AdamW with decoupled weight decay, cosine learning rate with warmup, gradient clipping, causal KV caching for generation, temperature and top-k sampling.

## Correctness

The backward pass is verified against central finite differences:

```
$ python3 nanogpt.py --gradcheck
gradcheck worst relative error: 1.412e-06
PASS
```

Every parameter tensor in the model is perturbed and compared. This is checked at 1, 2, and 3 layers and across several head and width combinations. The backward pass is not trusted on inspection alone.

## Speed

Pure numpy is roughly two orders of magnitude slower than a framework implementation. This is a teaching and reference implementation, not something to train a production model with. The value is that every operation is on screen.

## Files

- `nanogpt.py` — the entire implementation and training driver

## License

Apache-2.0
