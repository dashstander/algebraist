# Fast Fourier Transform for Sn

A PyTorch implementation of the [Clausen & Baum (1993)](https://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192969-X/S0025-5718-1993-1192969-X.pdf) Fast Fourier Transform (FFT) for the symmetric group $S_n$.

## Introduction

The Fourier transform is a fundamental tool in signal processing and analysis, typically associated with continuous or discrete time signals. However, the concept can be generalized to other mathematical structures, including finite groups. This project implements the Fast Fourier Transform for the symmetric group Sn, which has applications in various fields including computational group theory, machine learning, and data analysis on permutations.

## Features

- Efficient implementation of the Sn FFT algorithm in PyTorch
- Support for both forward and inverse transforms.
- Utilities for working with permutations and representations of Sn
- Examples and tests demonstrating usage and correctness

## Installation

This is still a very early stages project so it is not yet on PyPI, but the repo is pip installable:


```bash
git clone https://github.com/dashstander/algebraist.git
cd algebraist
pip install -e .
```

## Usage

Here's a basic example of how to use the $S_n$ FFT. For more in-depth examples of how to use this code check the previous versions of it in the [sn-grok](https://github.com/dashstander/sn-grok) repo or read our paper ['Grokking Group Multiplication with Cosets'](https://arxiv.org/abs/2312.06581).

```python
import torch
from algebraist import sn_fft

# Create a function on S5 (represented as a tensor of size 120)
n = 5
fn = torch.randn(120)

# Compute the Fourier transform
ft = sn_fft(fn, n)

# ft is now a dictionary mapping partitions to their Fourier transforms
```

## Contributing

Contributions are welcome! Please feel free to submit a pull Request.


## Acknowledgements

- Michael Clausen and Ulrich Baum for the original algorithm.
- Risi Kondor for the excellent exposition of Clausen & Baum's algorithm in [his thesis](https://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf).

## Contact

If you have any questions or feedback, please open an issue on this repository.