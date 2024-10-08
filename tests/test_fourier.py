
import math
import numpy as np
import pytest
import torch
import algebraist
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


def convolve(f, g, n):
    """Compute the convolution of f and g on Sn."""
    perms = list(Permutation.full_group(n))
    if f.dim() == 1:
        f = f.unsqueeze(0)
        g = g.unsqueeze(0)
    batch_size = f.shape[0]
    result = torch.zeros_like(f)
    for b in range(batch_size):
        for i, pi in enumerate(perms):
            for j, sigma in enumerate(perms):
                result[b, i] += f[b, j] * g[b, list(perms).index(pi * sigma.inverse)]
    return result.squeeze()


def generate_random_function(n, batch_size=None):
    """Generate a random function on Sn."""
    if batch_size is None:
        return torch.randn(math.factorial(n))
    return torch.randn(batch_size, math.factorial(n))


def generate_random_fourier_transform(n, batch_size=None):
    has_batch = batch_size is not None
    batch_size = batch_size if batch_size else 1
    ft = {}
    for irrep in SnIrrep.generate_all_irreps(n):
        ft[irrep.partition] = torch.randn(batch_size, irrep.dim, irrep.dim)
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
    assert torch.allclose(f, ift, atol=1e-5), f"Fourier transform not invertible for n={n}, batch_size={batch_size}"

@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_fourier_decomposition(n, batch_size):
    f = generate_random_function(n, batch_size)
    ft = sn_fft(f, n)
    decomp = sn_fourier_decomposition(ft, n)

    assert torch.allclose(f, sum(decomp.values()), atol=1e-5), f"Fourier decomposition failed for n={n}, batch_size={batch_size}"


@pytest.mark.parametrize("n", [3, 4, 5])
@pytest.mark.parametrize("batch_size", [None, 1, 5])
def test_fourier_transform_norm_preservation(n, batch_size):
    f = generate_random_function(n, batch_size)
    ft = sn_fft(f, n)
    power = calc_power(ft, n)
    total_power = sum(p for p in power.values())
    if batch_size is None:
        f = f.unsqueeze(0)
    assert torch.allclose(torch.sum(f**2, dim=1), total_power, atol=1e-5), f"Norm not preserved for n={n}, batch_size={batch_size}"


@pytest.mark.parametrize("n", [3, 4, 5])
def test_convolution_theorem(n):
    f = generate_random_function(n, None).to(torch.float64)
    g = generate_random_function(n, None).to(torch.float64)
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
        assert torch.allclose(ft_conv_time[shape], ft_conv_freq[shape], atol=1.e-4),\
            f"Convolution theorem failed for n={n}, partition={shape}, max diff = {(ft_conv_time[shape] - ft_conv_freq[shape]).abs().max()}"
    

@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_action(n):
    f = generate_random_function(n, None)
    ft = sn_fft(f, n)
    permutations = Permutation.full_group(n)
    perm = permutations[np.random.randint(0, math.factorial(n))]
    permutation_action = [(perm.inverse * p).permutation_index() for p in permutations ]
    # Action in group domain
    f_perm = f[permutation_action]
    ft_perm = sn_fft(f_perm, n)
    
    # Action in Fourier domain
    ft_action = {}
    for shape, matrix in ft.items():
        irrep = SnIrrep(n, shape)
        if irrep.dim == 1:
            continue
        rho = torch.tensor(
            irrep.matrix_representations[perm.sigma],
            dtype=f.dtype,
            device=matrix.device
        )
        ft_action[shape] = torch.matmul(rho, matrix)
    
    for shape in ft_action.keys():
        assert torch.allclose(ft_perm[shape], ft_action[shape], atol=1e-4), \
            f"Permutation action failed for n={n}, shape={shape}"     


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sn_fft(n):
    f = generate_random_function(n)
    slow_ft = slow_sn_ft(f, n)
    fast_ft = sn_fft(f, n)

    equalities = {}
    for irrep, tensor in fast_ft.items():
        equalities[irrep] = torch.allclose(slow_ft[irrep], tensor, atol=1.e-4)
    assert all(equalities.values()), equalities


@pytest.mark.parametrize("n", [3, 4, 5])
def test_sn_ifft(n):
    ft = generate_random_fourier_transform(n)
    slow_ift = slow_sn_ift(ft, n)
    fast_ift = sn_ifft(ft, n)

    assert torch.allclose(slow_ift, fast_ift)


if __name__ == '__main__':
    pytest.main(['-v', '-s'])
