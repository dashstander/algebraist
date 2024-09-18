import math
import torch
from torch.utils._pytree import tree_map
from algebraist.irreps import SnIrrep
from algebraist.permutations import Permutation
from algebraist.tableau import generate_partitions


def get_all_irreps(n):
    return [SnIrrep(n, p) for p in generate_partitions(n)]


def slow_sn_ft(fn_vals, n: int):
    """
    Compute the Fourier transform on Sn.
    
    Args:
    fn_vals (torch.Tensor): Input tensor of shape (batch_size, n!) or (n!,)
    n (int): The order of the symmetric group
    
    Returns:
    dict: A dictionary mapping partitions to their Fourier transforms
    """
    all_irreps = [SnIrrep(n, p) for p in generate_partitions(n)]
    results = {}
    
    if fn_vals.dim() == 1:
        fn_vals = fn_vals.unsqueeze(0)  # Add batch dimension if not present
    for irrep in all_irreps:
        matrices = irrep.matrix_tensor().to(fn_vals.device).to(torch.float32)

        if matrices.dim() == 1:  # One-dimensional representation
            result = torch.einsum('bi,i->b', fn_vals, matrices)
        else:  # Higher-dimensional representation
            result = torch.einsum('bi,ijk->bjk', fn_vals, matrices).squeeze()
        results[irrep.shape] = result
    
    return results


def slow_sn_ift(ft, n: int):
    """
    Compute the inverse Fourier transform on Sn.
    
    Args:
    ft (dict): A dictionary mapping partitions to their Fourier transforms
    n (int): The order of the symmetric group
    
    Returns:
    torch.Tensor: The inverse Fourier transform of shape (batch_size, n!)
    """
    permutations = Permutation.full_group(n)
    group_order = len(permutations)
    irreps = {shape: SnIrrep(n, shape).matrix_tensor() for shape in ft.keys()}
    trivial_irrep = (n,)
    sign_irrep = tuple([1] * n)

    batch_size = ft[(n - 1, 1)].shape[0] if len(ft[(n - 1, 1)].shape) == 3 else None
    if batch_size is None:
        ift = torch.zeros((group_order,), device=ft[(n - 1, 1)].device)
    else:
        ift = torch.zeros((batch_size, group_order), device=ft[(n - 1, 1)].device)
    for shape, irrep_ft in ft.items():
        # Properly the formula says we should multiply by $rho(g^{-1})$, i.e. the transpose here
        inv_rep = irreps[shape].to(irrep_ft.dtype).to(irrep_ft.device)
        if shape == trivial_irrep or shape == sign_irrep:
            ift += torch.einsum('...i,g->...ig', irrep_ft, inv_rep).squeeze()
        else: 
            dim = inv_rep.shape[-1] 
            # But this contracts the tensors in the correct order without the transpose
            ift += dim * torch.einsum('...ij,gij->...g', irrep_ft, inv_rep)
    
    return (ift / group_order)


def slow_sn_fourier_decomposition(ft, n: int):
    """
    Compute the inverse Fourier transform on Sn.
    
    Args:
    ft (dict): A dictionary mapping partitions to their Fourier transforms
    n (int): The order of the symmetric group
    
    Returns:
    torch.Tensor: The inverse Fourier transform of shape (batch_size, n!)
    """
    permutations = Permutation.full_group(n)
    group_order = len(permutations)
    irreps = {shape: SnIrrep(n, shape).matrix_tensor() for shape in ft.keys()}
    trivial_irrep = (n,)
    sign_irrep = tuple([1] * n)

    num_irreps = len(ft.keys())
    batch_size = ft[(n - 1, 1)].shape[0] if len(ft[(n - 1, 1)].shape) == 3 else None

    if batch_size is None:
        ift = torch.zeros((num_irreps, group_order,), device=ft[(n - 1, 1)].device)
    else:
        ift = torch.zeros((num_irreps, batch_size, group_order), device=ft[(n - 1, 1)].device)

    for i, (shape, irrep_ft) in enumerate(ft.items()):
        inv_rep = irreps[shape].to(irrep_ft.dtype).to(irrep_ft.device)
        if shape == trivial_irrep or shape == sign_irrep:  # One-dimensional representation
            ift[i] = torch.einsum('...i,g->...ig', irrep_ft, inv_rep).squeeze()
        else:  # Higher-dimensional representation
            dim = inv_rep.shape[-1] 
            ift[i] += dim * torch.einsum('...ij,gij->...g', irrep_ft, inv_rep)
    if batch_size is not None:
        ift = ift.permute(1, 0, 2)
    return (ift / group_order).squeeze()



def _calc_power(ft, n: int):
    group_order = math.factorial(n)
    power = {}
    trivial_irrep = (n,)
    sign_irrep = tuple([1] * n)
    for tableau, tensor in ft.items():
        if tableau == trivial_irrep or tableau == sign_irrep:  # 1D representation
            power[tableau] = (tensor ** 2).sum(dim=0) / group_order
        else:
            dim = tensor.shape[-1]
            power[tableau] = dim * torch.trace(tensor @ tensor.T) / group_order    
    return power



def calc_power(ft, n: int):
    """
    Calculate the power of the Fourier transform.
    
    Args:
    ft (dict): A dictionary mapping partitions to their Fourier transforms
    group_order (int): The order of the group
    
    Returns:
    dict: A dictionary mapping partitions to their powers
    """
    
    has_batch_dim = len(ft[(n - 1, 1)].shape) == 3

    if has_batch_dim:
        return torch.vmap(_calc_power, in_dims=(0, None))(ft, n)
    else:
        return _calc_power(ft, n)

