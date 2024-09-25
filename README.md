# Fast Fourier Transform for Sn

A PyTorch implementation of the [Clausen & Baum (1993)](https://www.ams.org/journals/mcom/1993-61-204/S0025-5718-1993-1192969-X/S0025-5718-1993-1192969-X.pdf) Fast Fourier Transform (FFT) for the [symmetric group](https://en.wikipedia.org/wiki/Symmetric_group) $S_n$.

## Introduction

The Fourier transform is a fundamental tool in signal processing and analysis, typically associated with continuous or discrete time signals. The concept can be generalized to other mathematical structures, however, including [finite groups](https://en.wikipedia.org/wiki/Fourier_transform_on_finite_groups). The classical discrete Fourier transform can be thought of as a transform on the [cyclic group](https://en.wikipedia.org/wiki/Cyclic_group) $C_n$, with $n$ the number of time samples in the signal. The frequencies that the Fourier transfrom maps to are the [characters](https://en.wikipedia.org/wiki/Character_theory) of $C_t$.

This project implements the Fast Fourier Transform for the symmetric group $S_n$, which is the group of permutations on $n$ letters. e.g. the permutation $(3, 2, 1, 4, 5)$ acts on an sequence of five letters by permuting the first and third elements: $(3, 2, 1, 4, 5) \circ [a, b, c, d, e] = [c, b, a, d, e]$.

Analyzing data on the symmetric group has applications in many fields including [probability](https://projecteuclid.org/ebooks/institute-of-mathematical-statistics-lecture-notes-monograph-series/group-representations-in-probability-and-statistics/toc/10.1214/lnms/1215467407), machine learning, and data analysis on permutations.

## Features

- Efficient implementation of the $S_n$ FFT algorithm in PyTorch
- Support for both forward and inverse transforms.
- Utilities for working with permutations and representations of $S_n$

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
from algebraist import sn_fft, sn_ifft

# Create a function on S5 (represented as a tensor of size 120)
n = 5
fn = torch.randn(120)

# Compute the Fourier transform
ft = sn_fft(fn, n)

# ft is now a dictionary mapping partitions to their Fourier transforms
for partition, ft_matrix in ft.items():
    # The frequencies of the Sn Fourier transform are the partitions of n
    print(partition) # The partitions of 5 are (5,), (4, 1), (3, 1, 1), (2, 2, 1), (2, 1, 1, 1), and (1, 1, 1, 1, 1)
    print(ft_matrix) # because S_n isn't abelian the output of the Fourier transform for each partition is a matrix

# The Fourier transform is completely invertible
assert fn == sn_ifft(ft, n)
```

## Requirements

So far `algebraist` has been developed with Python 3.11 and PyTorch 2.4 and I cannot promise that it will work with any other versions. Though the only new-ish feature that the library uses often is `torch.vmap`, so any PyTorch version that has merged in `functorch` _should_ work.


## License
This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a pull Request.


## Acknowledgements

- Michael Clausen and Ulrich Baum for the original algorithm.
- Risi Kondor for the excellent exposition of Clausen & Baum's algorithm in [his thesis](https://people.cs.uchicago.edu/~risi/papers/KondorThesis.pdf).

## Contact

If you have any questions or feedback, please open an issue on this repository.