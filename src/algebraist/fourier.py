from itertools import product
import torch
from .irreps import SnIrrep
from .permutations import Permutation
from .tableau import generate_partitions


def _dot(fx, rho):
    return fx * rho

fft_dot = torch.vmap(_dot, in_dims=(0, 0))


def _fft_sum(fx, rho):
    if rho.dim() == 1:
        return torch.dot(fx, rho)
    else:
        return fft_dot(fx, rho).sum(dim=0)


def _frob_norm(ft_vals):
    dim = ft_vals.dim()
    if dim < 2:
        return ft_vals**2
    else:
        dim = ft_vals.shape[0]
        return dim * torch.trace(ft_vals.T @ ft_vals)


def _ift_trace(ft_vals, inv_rep):
    dim = inv_rep.dim()
    if dim < 2:
        return inv_rep * ft_vals
    else:
        dim = inv_rep.shape[0]
        return dim * torch.trace(inv_rep @ ft_vals)


ift_trace = torch.vmap(_ift_trace, in_dims=(0, None))
fft_sum = torch.vmap(_fft_sum, in_dims=(1, None))
batch_kron = torch.vmap(torch.kron, in_dims=(0, 0))
frob = torch.vmap(_frob_norm, in_dims=0)


def calc_power(ft, group_order):
    return {k: (frob(v) / group_order**2)  for k, v in ft.items()}


def slow_sn_ft_1d(fn_vals, n):
    all_partitions = generate_partitions(n)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for irrep in all_irreps:
        matrices = irrep.matrix_tensor().to(fn_vals.device).to(torch.float32)
        results[irrep.shape] = fft_sum(fn_vals, matrices).squeeze()
    return results


def slow_sn_ft_2d(fn_vals, n):
    all_partitions = generate_partitions(n)
    all_irreps = [SnIrrep(n, p) for p in all_partitions]
    results = {}
    for lirrep, rirrep in product(all_irreps, all_irreps):
        mats1, mats2 = zip(
            *product(
                lirrep.matrix_tensor().to(torch.float32).split(1),
                rirrep.matrix_tensor().to(torch.float32).split(1))
        )
        mats = batch_kron(torch.cat(mats1).squeeze(), torch.cat(mats2).squeeze())
        mats = mats.to(torch.float32)
        mats = mats.to(fn_vals.device)
        results[(lirrep.shape, rirrep.shape)] = fft_sum(fn_vals, mats).squeeze()
    return results


def sn_fourier_basis(ft, G, device='cpu'):
    permutations = G.elements
    group_order = len(permutations)
    all_irreps = G.irreps()
    for k, v in all_irreps.items():
        all_irreps[k] = v.matrix_representations()
    ift_decomps = []
    for perm in permutations:
        fourier_decomp = []
        for part, m in all_irreps.items():
            inv_rep = torch.asarray(m[perm.sigma].T, device=device).squeeze()           
            fourier_decomp.append(ift_trace(ft[part], inv_rep.to(torch.float32)).unsqueeze(0))
        ift_decomps.append(torch.cat(fourier_decomp).unsqueeze(0))
    return torch.cat(ift_decomps) / group_order


def sn_fourier_basis_2d(ft, n, device):
    all_partitions = generate_partitions(n)
    permutations = Permutation.full_group(n)
    group_order = 1.0 * len(permutations)**2
    all_irreps = {p: SnIrrep(n, p).matrix_representations() for p in all_partitions}
    ift_decomps = []
    for perm1, perm2 in product(permutations, permutations):
        fourier_decomp = []
        for part1, part2 in product(all_partitions, all_partitions):
            inverse_mat1 = torch.asarray(all_irreps[part1][perm1.sigma].T, device=device).squeeze().contiguous()   
            inverse_mat2 = torch.asarray(all_irreps[part2][perm2.sigma].T, device=device).squeeze().contiguous()
            inv_mat = torch.kron(inverse_mat1 , inverse_mat2).to(torch.float32)
            trace = ift_trace(ft[(part1, part2)], inv_mat).unsqueeze(0)
            fourier_decomp.append(trace)
        ift_decomps.append(torch.cat(fourier_decomp).unsqueeze(0))
    return torch.cat(ift_decomps) / group_order
