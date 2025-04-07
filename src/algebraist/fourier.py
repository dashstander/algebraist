# Copyright [2024] [Dashiell Stander]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from functools import partial
import jax
import jax.numpy as jnp
import math

from algebraist.irreps import SnIrrep
from algebraist.permutations import Permutation
from algebraist.tableau import generate_partitions
from algebraist.utils import generate_all_permutations


BASE_CASE = 5


def get_all_irreps(n: int) -> list[SnIrrep]:
    """
    Collects a list of all of the irreducible representations (irreps) of S_n.

    Helper method so we don't end up calling this a ton.

    Args:
    n (int): the number of elements on which the permutations of S_n acts

    Returns:
    list[SnIrrep] all of the irreps of S_n, indexed by the interger partitions of n
    """
    return [SnIrrep(n, p) for p in generate_partitions(n)]


@jax.jit
def lift_from_coset(lifted_fn, coset_fn: jax.Array, sn_perms: jax.Array, idx: int) -> jax.Array:
    """
    Inverse operation of restrict_to_coset. Assigns values from S_{n-1} cosets back to their correct positions in S_n.

    Args:
    tensor (jax.Array): The function on S_{n-1} cosets, shape (n, batch_size, (n-1)!)
    sn_perms (jax.Array): A tensor-version of S_n with shape (n!, n)

    Returns:
    None, operates in place on lifted_fn
    """
    n = sn_perms.shape[1]
    fixed_element = n - 1
    coset_idx = jnp.argwhere(sn_perms[:, idx] == fixed_element).squeeze()
    lifted_fn[:, coset_idx] = coset_fn[idx]
    

@jax.jit
def restrict_to_coset(tensor: jax.Array, sn_perms: jax.Array, idx: int) -> jax.Array:
    """
    Returns the values that a function on S_n takes on of one of the cosets of S_{n-1} < S_n

    There are n cosets of S_{n-1} < S_n. Young's Orthogonal Form (YOR) is specifically adapted to the copy of S_{n-1} where the element n is fixed in the nth position. The _cosets_ of this subgroup correspond to the elements that all have n in a given position.

    Args:
    tensor (torch.Tensor): The function on S_n we are working with, either shape (batch, n!) or (n!,)
    sn_perms (torch.Tensor): A tensor-version of S_n with shape (n!, n), each row is the elements 0..n-1 permuted, and the rows are in lexicographic order
    idx (int): The index of n that defines the coset we are grabbing

    Returns:
    torch.Tensor either of shape (batch, (n-1)!) or ((n-1)!, ), depending on whether or not tensor had a batch dimension
    """
    n = sn_perms.shape[1]
    fixed_element = n - 1
    coset_idx = jnp.argwhere(sn_perms[:, idx] == fixed_element).squeeze()
    return tensor[..., coset_idx]


@partial(jax.jit, static_argnums=1)
def _fourier_projection(fn_vals: jax.Array, irrep: SnIrrep):
    """
    A non-recursive projection onto one of the irreducible representations (irreps) of Sn, for the "base case" of n= 4 or 5 where the 
    number of group elements is small enough that it is easier to rely on the inherent parallelism of PyTorch.
    
    Args:
    fn_vals (torch.Tensor): Input tensor of shape (batch_size, n!) or (n!,)
    irrep (SnIrrep): an irreducible representation of Sn
    
    Returns:
    torch.Tensor: the projection of `fn_vals` onto the irreducible representation given by `irrep`
    """
    
    matrices = irrep.matrix_tensor()

    if jnp.ndim(matrices) == 1:  # One-dimensional representation
        result = jnp.einsum('...i,i->...', fn_vals, matrices)
    else:  # Higher-dimensional representation
        result = jnp.einsum('...i,ijk->...jk', fn_vals, matrices).squeeze()
   
    return result


@partial(jax.jit, static_argnums=1)
def _inverse_fourier_projection(ft: jax.Array, irrep: SnIrrep):
    """
    A non-recursive projection onto one of the irreducible representations (irreps) of Sn, for the "base case" of n= 4 or 5 where the 
    number of group elements is small enough that it is easier to rely on the inherent parallelism of PyTorch.
    
    Args:
    ft (torch.Tensor): Input tensor of shape (batch_size, irrep_dim, irrep_dim) or (irrep_dim, irrep_dim)
    irrep (SnIrrep): an irreducible representation of Sn
    
    Returns:
    jax.Array: the projection of `fn_vals` onto the irreducible representation given by `irrep`
    """
    
    matrices = irrep.matrix_tensor()
    if jnp.ndim(ft) < 2:
        ft = jnp.expand_dims(ft, 0)

    if irrep.dim == 1:
        result = jnp.einsum('...i,g->...ig', ft, matrices).squeeze()
    else: 
        dim = irrep.dim
        # In the normal formula we multiply by the inverse (transpose) of the irreps,
        # but this contracts the tensors in the correct order without the transpose
        result = dim * jnp.einsum('...ij,gij->...g', ft, matrices)
   
    return result


@partial(jax.jit, static_argnums=1)
def inverse_fourier_projection(ft, irrep):
    n = irrep.n
    if n <= BASE_CASE or irrep.dim == 1:
        return _inverse_fourier_projection(ft, irrep) / math.factorial(n)
    
    sn_perms = generate_all_permutations(n)

    # Ensure ft is always 3D (batch_dim, irrep.dim, irrep.dim)
    has_batch = True
    if jnp.ndim(ft) == 2:
        has_batch = False
        ft = jnp.expand_dims(ft, 0)
    
    batch_dim = ft.shape[0]
    
    # Inverse this time
    coset_rep_matrices = jnp.expand_dims(
        jnp.stack([
            mat.T for mat in irrep.coset_rep_matrices()
        ]),
        0
    )
    #assert coset_rep_matrices.shape == (1, n, irrep.dim, irrep.dim), \
    #    f'{coset_rep_matrices.shape} != {(1, n, irrep.dim, irrep.dim)}'

    # equivalent to [coset_rep_inverse @ ft for coset_rep_inverse in cosets]
    # we have now translated the Fourier transform to be amenable to the S_{n-1} basis
    coset_fts = jnp.matmul(coset_rep_matrices, jnp.expand_dims(ft, 1))
    #assert coset_fts.shape == (n, batch_dim, irrep.dim, irrep.dim), \
    #    f'{coset_fts.shape} != {(n, batch_dim, irrep.dim, irrep.dim)}'

    split_irreps = [SnIrrep(n-1, sub_partition) for sub_partition in irrep.split_partition()]


    # We have a big (irrep.dim, irrep.dim) matrix as the result of the forward FFT
    # With respect to the S_{n-1} irreps it has a block structure, here we pull those blocks out
    # We scale by (irrep.dim / (sub_irrep.dim * n)) to keep things working nicely with the recursion.
    # Non - recursively we scale by irrep.dim, we divide here to cancel that out. Basically at the very top level
    # we only want the top or "main" irrep dim to contribute
    sub_ft_blocks = [
        (irrep.dim / (sub_irrep.dim * n)) * coset_fts[..., rows, cols] 
        for (rows, cols), sub_irrep in zip(irrep.get_block_indices(), split_irreps)
    ]

    # recursive call here
    sub_ifts = [
        jax.vmap(inverse_fourier_projection, in_axes=(0, None))(block, sub_irrep)
        for block, sub_irrep in zip(sub_ft_blocks, split_irreps)
    ]
    # there are n elements in sub_ifts, each is an ift on 
    # assert all([ift.shape == (batch_dim, math.factorial(n-1)) for ift in sub_ifts])

    fn_vals = jax.zeros((batch_dim, math.factorial(n)), dtype=ft.dtype, device=ft.device)
    
    # reshapes from 
    for i, coset_ift in enumerate(sub_ifts):
        # operates in place on fn_vals
        lift_from_coset(fn_vals, coset_ift, sn_perms, i)
    
    if not has_batch:
        fn_vals = fn_vals.squeeze()
    
    return fn_vals / math.factorial(n)
    

@partial(jax.jit, static_argnums=1)
def fourier_projection(fn_vals: jax.Array, irrep: SnIrrep) -> jax.Array:
    """
    Fast projection of a function on S_n (given as a pytorch tensor) onto one of the irreducible representations (irreps) of S_n. If n > 5 then
    this is done recursively, splitting the irreps into irreps of S_{n-1}.

    An important invariant of this function is that fn_vals is either 1D of shape (n!,) or 2D of shape (batch, n!).

    This function is, however, called recursively and throughout computation `fn_vals` will not always be 1 or 2D depending on the depth of recursion and whether or not it started with a batch dimension.

    Given fn_vals on S_n, we will reshape it to have shape (n, (n-1)!), where each ((n-1)!,) is thought of as its own function on S_{n-1}. This is passed down to `fourier_projection` **as if `n` is the batch dimension**.

    To get around this we: (1) Use vmap across the given batch dimension of fn_vals (2) Always return the tensor to have the same batch dimension (or none) as fn_vals
    
    Args:
    fn_vals (torch.Tensor): A tensor of shape (batch_size, n!) for an integer n
    irrep (SnIrrep): An irreducible representation of Sn, given by an integer partition of n

    Returns:
    torch.Tensor the projection of `fn_vals` onto `irrep` with shape (batch_size, irrep.dim, irrep.dim)
    """
    n = irrep.n
    if n <= BASE_CASE or irrep.dim == 1:
        return _fourier_projection(fn_vals, irrep)
    sn_perms = generate_all_permutations(n)
    # Ensure fn_vals is always 2D (batch_dim, n!)
    has_batch = True
    if jnp.ndim(fn_vals) == 1:
        has_batch = False
        fn_vals = jnp.expand_dims(fn_vals, 0)

    coset_fns = jnp.stack([restrict_to_coset(fn_vals, sn_perms, i) for i in range(n)]).permute(1, 0, 2)
    # Now coset_fns shape is (batch_dim, n, (n-1)!)
    # assert coset_fns.shape == (fn_vals.shape[0], n, math.factorial(n-1)), coset_fns.shape
    
    coset_rep_matrices = jnp.expand_dims(
        jax.stack(irrep.coset_rep_matrices(fn_vals.dtype, fn_vals.device)),
        0
    )
    # assert coset_rep_matrices.shape == (1, n, irrep.dim, irrep.dim), coset_rep_matrices.shape
    split_irreps = [SnIrrep(n-1, split_shape) for split_shape in irrep.split_partition()]
    
    # Recursive call, use vmap to apply across the batch dimension
    sub_fts = [
        jax.vmap(fourier_projection, in_axes=(0, None))(coset_fns, split_irrep) 
        for split_irrep in split_irreps
    ]
    
    # Use vmap to apply block_diag across the combined n * batch_dim
    block_diag_vmap = jax.vmap(jax.vmap(jax.block_diag))
    combined_sub_fts = block_diag_vmap(*sub_fts)
    # combined_sub_fts shape: (n * batch_dim, irrep_dim, irrep_dim)
    # assert combined_sub_fts.shape == (fn_vals.shape[0], n, irrep.dim, irrep.dim)
    
    result = jax.matmul(coset_rep_matrices, combined_sub_fts).sum(1)

    if not has_batch:
        result = result.squeeze(0) 
    
    return result


def sn_fft(fn_vals: jax.Array, n: int) -> dict[tuple[int, ...], jax.Array]:    
    result = {}
    all_irreps = list(SnIrrep.generate_all_irreps(n))
    for irrep in all_irreps:
        result[irrep.partition] = fourier_projection(fn_vals, irrep)
    return result


def sn_fourier_decomposition(ft, n):
    return {
        irrep.partition: inverse_fourier_projection(ft[irrep.partition], irrep)
        for irrep in SnIrrep.generate_all_irreps(n)
    }
 

def sn_ifft(ft, n):
    return sum(sn_fourier_decomposition(ft, n).values())


def slow_sn_ft(fn_vals: jax.Array, n: int):
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
    
    if jnp.ndim(fn_vals) == 1:
        fn_vals = jnp.expand_dims(fn_vals, 0)  # Add batch dimension if not present

    for irrep in all_irreps:
        matrices = irrep.matrix_tensor()

        if jnp.ndim(matrices) == 1:  # One-dimensional representation
            result = jnp.einsum('bi,i->b', fn_vals, matrices)
        else:  # Higher-dimensional representation
            result = jnp.einsum('bi,ijk->bjk', fn_vals, matrices).squeeze()
        results[irrep.partition] = result
    
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
    irreps = {shape: SnIrrep(n, shape) for shape in ft.keys()}
    
    batch_size = ft[(n - 1, 1)].shape[0] if len(ft[(n - 1, 1)].shape) == 3 else None
    if batch_size is None:
        ift = jnp.zeros((group_order,), device=ft[(n - 1, 1)].device)
    else:
        ift = jnp.zeros((batch_size, group_order), device=ft[(n - 1, 1)].device)
    for shape, irrep_ft in ft.items():
        # Properly the formula says we should multiply by $rho(g^{-1})$, i.e. the transpose here
        #inv_rep = irreps[shape].to(irrep_ft.dtype).to(irrep_ft.device)
        ift += _inverse_fourier_projection(irrep_ft, irreps[shape])
    
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
    irreps = {shape: SnIrrep(n, shape) for shape in ft.keys()}
    num_irreps = len(ft.keys())
    batch_size = ft[(n - 1, 1)].shape[0] if len(ft[(n - 1, 1)].shape) == 3 else None

    if batch_size is None:
        ift = jnp.zeros((num_irreps, group_order,), device=ft[(n - 1, 1)].device)
    else:
        ift = jnp.zeros((num_irreps, batch_size, group_order), device=ft[(n - 1, 1)].device)

    for i, (shape, irrep_ft) in enumerate(ft.items()):
        #inv_rep = irreps[shape].to(irrep_ft.dtype).to(irrep_ft.device)
        ift[i] = _inverse_fourier_projection(irrep_ft, irreps[shape])
        
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
            power[tableau] = (tensor ** 2) / group_order
        else:
            dim = tensor.shape[-1]
            power[tableau] = dim * jnp.trace(tensor @ tensor.T) / group_order    
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
        return jax.vmap(_calc_power, in_axes=(0, None))(ft, n)
    else:
        return _calc_power(ft, n)

