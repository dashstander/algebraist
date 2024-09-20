from functools import lru_cache
import math
import torch
from tqdm import tqdm

from algebraist.irreps import SnIrrep
from algebraist.permutations import Permutation
from algebraist.tableau import generate_partitions
from algebraist.utils import generate_all_permutations


BASE_CASE = 4


@lru_cache(maxsize=20)
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


def _fourier_projection(fn_vals: torch.Tensor, irrep: SnIrrep):
    """
    A non-recursive projection onto one of the irreducible representations (irreps) of Sn, for the "base case" of n= 4 or 5 where the 
    number of group elements is small enough that it is easier to rely on the inherent parallelism of PyTorch.
    
    Args:
    fn_vals (torch.Tensor): Input tensor of shape (batch_size, n!) or (n!,)
    irrep (SnIrrep): an irreducible representation of Sn
    
    Returns:
    torch.Tensor: the projection of `fn_vals` onto the irreducible representation given by `irrep`
    """
    if fn_vals.dim() == 1:
        fn_vals = fn_vals.unsqueeze(0)  # Add batch dimension if not present

    matrices = irrep.matrix_tensor(fn_vals.dtype, fn_vals.device)

    if matrices.dim() == 1:  # One-dimensional representation
        result = torch.einsum('bi,i->b', fn_vals, matrices)
    else:  # Higher-dimensional representation
        result = torch.einsum('bi,ijk->bjk', fn_vals, matrices).squeeze()
   
    return result


def sn_minus_1_coset(tensor: torch.Tensor, sn_perms: torch.Tensor, idx: int) -> torch.Tensor:
    """
    We adapt the basis 
    """
    n = sn_perms.shape[1]
    fixed_element = n - 1
    coset_idx = torch.argwhere(sn_perms[:, idx] == fixed_element).squeeze()
    return tensor[..., coset_idx]


def fourier_projection(fn_vals: torch.Tensor, irrep: SnIrrep) -> torch.Tensor:
    """
    Fast projection of a function on Sn (given as a pytorch tensor) onto one of the irreducible representations (irreps) of Sn. If n > 5 then
    this is done recursively, splitting the irreps into irreps of S(n-1).

    Args:
    fn_vals (torch.Tensor): 
    """
    n = irrep.n
    if n <= BASE_CASE or irrep.dim == 1:
        return _fourier_projection(fn_vals, irrep)
    sn_perms = generate_all_permutations(n)
    # Ensure fn_vals is always 2D (batch_dim, n!)
    if fn_vals.dim() == 1:
        fn_vals = fn_vals.unsqueeze(0)
    
    batch_dim = fn_vals.shape[0]
    
    coset_fns = torch.stack([sn_minus_1_coset(fn_vals, sn_perms, i) for i in range(n)])
    # Now coset_fns shape is (n, batch_dim, (n-1)!)
    
    coset_rep_matrices = torch.stack(irrep.coset_rep_matrices(fn_vals.dtype))
    split_irreps = [SnIrrep(n-1, split_shape) for split_shape in irrep.split_partition()]
    
    # Reshape for recursive call: (n * batch_dim, (n-1)!)
    recursive_input = coset_fns.reshape(-1, math.factorial(n-1))
    
    # Recursive call
    sub_fts = [fourier_projection(recursive_input, split_irrep) for split_irrep in split_irreps]
    # Reshape sub_fts to be 3D: (n * batch_dim, split_irrep_dim, split_irrep_dim)
    sub_fts = [sub_ft.reshape(n * batch_dim, split_irrep.dim, split_irrep.dim) for sub_ft, split_irrep in zip(sub_fts, split_irreps)]
    
    # Use vmap to apply block_diag across the combined n * batch_dim
    block_diag_vmap = torch.vmap(lambda *matrices: torch.block_diag(*matrices))
    combined_sub_fts = block_diag_vmap(*sub_fts)
    # combined_sub_fts shape: (n * batch_dim, irrep_dim, irrep_dim)
    
    # Repeat coset_rep_matrices for each item in the batch
    repeated_coset_reps = coset_rep_matrices.repeat(batch_dim, 1, 1)
    
    result = torch.matmul(repeated_coset_reps, combined_sub_fts).reshape(batch_dim, n, irrep.dim, irrep.dim).sum(1)    
    return result  



def sn_fft(fn_vals: torch.Tensor, n: int, verbose=False) -> dict[tuple[int, ...], torch.Tensor]:    
    result = {}
    all_irreps = list(SnIrrep.generate_all_irreps(n))
    if verbose:
        all_irreps = tqdm(all_irreps)
    for irrep in all_irreps:
        result[irrep.shape] = fourier_projection(fn_vals, irrep)
    return result



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

