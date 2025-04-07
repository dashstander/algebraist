
import jax
import jax.numpy as jnp

import math
import numpy as np
import pytest
#import torch
import random

import algebraist
from algebraist.tableau import generate_partitions
from algebraist.fourier import (
    slow_sn_ft, slow_sn_ift, sn_fft, sn_ifft, sn_fourier_decomposition, calc_power
)
from algebraist.permutations import Permutation
from algebraist.irreps import SnIrrep


@pytest.fixture
def set_base_case():
    original_base_case = algebraist.fourier.BASE_CASE
    algebraist.fourier.BASE_CASE = 3
    yield
    algebraist.fourier.BASE_CASE = original_base_case

"""
def convolve(f, g, n):
    
    perms = list(Permutation.full_group(n))
    if jnp.ndim(f) == 1:
        f = jnp.expand_dims(f, 0)
        g = jnp.expand_dims(g, 0)
    batch_size = f.shape[0]
    result = jnp.zeros_like(f)
    for b in range(batch_size):
        for i, pi in enumerate(perms):
            for j, sigma in enumerate(perms):
                result[b, i] += f[b, j] * g[b, list(perms).index(pi * sigma.inverse)]
    return result.squeeze()

"""


def convolve(f, g, n):
    """Compute the convolution of f and g on Sn using JAX.
    
    Args:
        f: Function values on the permutation group Sn, shape (batch_size, n!)
        g: Function values on the permutation group Sn, shape (batch_size, n!)
        n: Size of the permutation group (permutations on n elements)
        
    Returns:
        Convolution result with shape matching f
    """
    perms = list(Permutation.full_group(n))
    
    # Handle 1D inputs by adding batch dimension
    if jnp.ndim(f) == 1:
        f = jnp.expand_dims(f, 0)
        g = jnp.expand_dims(g, 0)
    
    
    # Precompute the permutation composition table
    # For each pair (pi, sigma), store the index of pi * sigma.inverse()
    composition_table = jnp.array([
        [list(perms).index(pi * sigma.inverse) for j, sigma in enumerate(perms)]
        for i, pi in enumerate(perms)
    ])
    
    # Define a function to process a single batch item
    def process_batch_item(f_item, g_item):
        # Use jax.vmap to vectorize over all permutations pi
        def process_pi(pi_idx):
            # Use composition table to get correct g indices for current pi
            g_indices = composition_table[pi_idx]
            # Select and multiply corresponding g values
            return jnp.sum(f_item * g_item[g_indices])
        
        # Apply the function to all pi values (rows of the result)
        return jax.vmap(process_pi)(jnp.arange(len(perms)))
    
    # Use vmap to vectorize across the batch dimension
    result = jax.vmap(process_batch_item)(f, g)
    
    return result.squeeze()


def generate_random_function(n, batch_size=None):
    """Generate a random function on Sn."""
    key = jax.random.key(random.getrandbits(31))
    if batch_size is None:
        return jax.random.normal(key, math.factorial(n))
    return jax.random.normal(key, (batch_size, math.factorial(n)))


def generate_random_fourier_transform(n, batch_size=None):
    has_batch = batch_size is not None
    batch_size = batch_size if batch_size else 1
    ft = {}
    key = jax.random.key(random.getrandbits(31))
    keys = jax.random.split(key, len(generate_partitions(n)))
    for i, irrep in enumerate(SnIrrep.generate_all_irreps(n)):
        ft[irrep.partition] = jax.random.normal(keys[i], (batch_size, irrep.dim, irrep.dim))
        if not has_batch:
            ft[irrep.partition] = ft[irrep.partition].squeeze()
    return ft


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_fourier_transform_invertibility(n, batch_size):
    f = generate_random_function(n, batch_size)
    ft = sn_fft(f, n)
    ift = sn_ifft(ft, n)
    f = f.squeeze()
    assert ift.shape == f.shape
    assert jnp.allclose(f, ift, atol=1e-5), f"Fourier transform not invertible for n={n}, batch_size={batch_size}"

@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_fourier_decomposition(n, batch_size):
    f = generate_random_function(n, batch_size)
    ft = sn_fft(f, n)
    decomp = sn_fourier_decomposition(ft, n)

    assert jnp.allclose(f, sum(decomp.values()), atol=1e-5), f"Fourier decomposition failed for n={n}, batch_size={batch_size}"


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_fourier_transform_norm_preservation(n, batch_size):
    f = generate_random_function(n, batch_size)
    ft = sn_fft(f, n)
    power = calc_power(ft, n)
    total_power = sum(p for p in power.values())
    if batch_size is None:
        f = jnp.expand_dims(f, 0)
    assert jnp.allclose(jnp.sum(f**2, axis=1), total_power, atol=1e-5), f"Norm not preserved for n={n}, batch_size={batch_size}"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_convolution_theorem(n):
    f = generate_random_function(n, None)
    g = generate_random_function(n, None)
    trivial_irrep = (n,)
    sign_irrep = tuple([1] * n)
    # Compute convolution in group domain
    conv_group = convolve(g, f, n)

    ft_conv_time = sn_fft(conv_group, n)
    
    # Compute convolution in Fourier domain
    ft_f = sn_fft(f, n)
    ft_g = sn_fft(g, n)
    ft_conv_freq = {}
    for shape in ft_f.keys():
        if shape == trivial_irrep or shape == sign_irrep:
            ft_conv_freq[shape] = ft_f[shape] * ft_g[shape]
        else:
            ft_conv_freq[shape] = ft_f[shape] @ ft_g[shape]
    for shape in ft_f.keys():
        assert jnp.allclose(ft_conv_time[shape], ft_conv_freq[shape], atol=1.e-3),\
            f"Convolution theorem failed for n={n}, partition={shape}, max diff = {(ft_conv_time[shape] - ft_conv_freq[shape]).abs().max()}"
    

@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_action(n):
    f = generate_random_function(n, None)
    ft = sn_fft(f, n)
    permutations = Permutation.full_group(n)
    perm = permutations[np.random.randint(0, math.factorial(n))]
    permutation_action = [(perm.inverse * p).permutation_index() for p in permutations ]
    # Action in group domain
    f_perm = f[jnp.array(permutation_action)]
    ft_perm = sn_fft(f_perm, n)
    
    # Action in Fourier domain
    ft_action = {}
    for shape, matrix in ft.items():
        irrep = SnIrrep(n, shape)
        if irrep.dim == 1:
            continue
        rho = jnp.array(
            irrep.matrix_representations[perm.sigma],
        )
        ft_action[shape] = jnp.matmul(rho, matrix)
    
    for shape in ft_action.keys():
        assert jnp.allclose(ft_perm[shape], ft_action[shape], atol=1e-3), \
            f"Permutation action failed for n={n}, shape={shape}"     


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sn_fft(n):
    f = generate_random_function(n)
    slow_ft = slow_sn_ft(f, n)
    fast_ft = sn_fft(f, n)

    equalities = {}
    for irrep, tensor in fast_ft.items():
        equalities[irrep] = jnp.allclose(slow_ft[irrep], tensor, atol=1.e-3)
    assert all(equalities.values()), equalities


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sn_ifft(n):
    ft = generate_random_fourier_transform(n)
    slow_ift = slow_sn_ift(ft, n)
    fast_ift = sn_ifft(ft, n)

    assert jnp.allclose(slow_ift, fast_ift)


if __name__ == '__main__':
    pytest.main(['-v', '-s'])
