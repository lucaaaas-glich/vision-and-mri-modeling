# PyTorch Barnsley Fern

This project generates a Barnsley Fern fractal using PyTorch tensors.

## Method

The fractal is generated using an Iterated Function System consisting of four affine transformations. Each point randomly selects a transformation according to a predefined probability.

PyTorch tensors are used to update thousands of walkers in parallel. On my Apple Silicon Mac, the program uses the MPS device when available.

## Run

```bash
pip install -r requirements.txt
python barnsley_fern.py