from copy import deepcopy
from functools import cached_property, reduce
from itertools import combinations, pairwise
import numpy as np
from numpy.typing import ArrayLike
import torch
from typing import Iterator, Self

from algebraist.permutations import Permutation
from algebraist.tableau import enumerate_standard_tableau, generate_partitions, hook_length, YoungTableau
from algebraist.utils import adj_trans_decomp, cycle_to_one_line, trans_to_one_line


def contiguous_cycle(n: int, i: int):
    """ Generates a permutation (in cycle notation) of the form (i, i+1, ..., n)
    """
    if i == n - 1:
        return tuple(range(n))
    else:
        cycle_rep = [tuple(range(i, n))] + [(j,) for j in reversed(range(i))]
        return cycle_to_one_line([cyc for cyc in cycle_rep if len(cyc) > 0])


class SnIrrep:

    def __init__(self, n: int, partition: tuple[int, ...]):
        self.n = n
        self.shape = partition
        self.dim = hook_length(self.shape)
        self.permutations = Permutation.full_group(n)

    @staticmethod
    def generate_all_irreps(n: int) -> Iterator[Self]:
        for partition in generate_partitions(n):
            yield SnIrrep(n, partition)

    def __eq__(self, other) -> bool:
        return self.shape == other.shape
    
    def __hash__(self) -> int:
        return hash(str(self.shape))
    
    def __repr__(self) -> str:
        return f'S{self.n} Irrep: {self.shape}'
    
    @cached_property
    def basis(self) -> list[YoungTableau]:
        return sorted(enumerate_standard_tableau(self.shape))

    def split_partition(self) -> list[tuple[int, ...]]:
        new_partitions = []
        k = len(self.shape)
        for i in range(k - 1):
            # check if valid subrepresentation
            if self.shape[i] > self.shape[i+1]:
                # if so, copy, modify, and append to list
                partition = list(deepcopy(self.shape))
                partition[i] -= 1
                new_partitions.append(tuple(partition))
        # the last subrep
        partition = list(deepcopy(self.shape))
        if partition[-1] > 1:
            partition[-1] -= 1
        else:
        # removing last element of partition if it’s a 1
            del partition[-1]
        new_partitions.append(tuple(partition))
        return sorted(new_partitions)

    def adjacent_transpositions(self) -> list[tuple[int, int]]:
        return pairwise(range(self.n))
    
    def non_adjacent_transpositions(self) -> list[tuple[int, int]]:
        return [(i, j) for i, j in combinations(range(self.n), 2) if i + 1 != j]

    def adj_transposition_matrix(self, a: int, b: int) -> ArrayLike:
        perm = Permutation.transposition(self.n, a, b)
        irrep = np.zeros((self.dim, self.dim))
        def fn(i, j):
            tableau = self.basis[i]
            if i == j:
                d = tableau.transposition_dist(a, b)
                return 1. / d
            else:
                new_tab = perm * tableau
                if new_tab == self.basis[j]:
                    d = tableau.transposition_dist(a, b)**2
                    return np.sqrt(1 - (1. / d))
                else:
                    return 0.
        for x in range(self.dim):
            for y in range(self.dim):
                irrep[x, y] = fn(x, y)
        return irrep
    
    def generate_transposition_matrices(self) -> dict[tuple[int], ArrayLike]:
        matrices = {
            (i, j): self.adj_transposition_matrix(i, j) for i, j in self.adjacent_transpositions()
        }
        for i, j in self.non_adjacent_transpositions():
            decomp = [matrices[pair] for pair in adj_trans_decomp(i, j)]
            matrices[(i, j)] = reduce(lambda x, y: x @ y, decomp)
        return matrices

    @cached_property
    def matrix_representations(self) -> dict[tuple[int], ArrayLike]:
        transpo_matrices = self.generate_transposition_matrices()
        matrices = {
            trans_to_one_line(*k, self.n): v for k, v in transpo_matrices.items()
        }
        matrices[tuple(range(self.n))] = np.eye(self.dim)
        for perm in self.permutations:
            if perm.sigma in matrices:
                continue
            else:
                cycle_mats = [
                    transpo_matrices[t] for t in perm.transposition_decomposition()
                ]
                perm_rep = reduce(lambda x, y: x @ y, cycle_mats)
                matrices[perm.sigma] = perm_rep
                if perm.inverse.sigma not in matrices:
                    matrices[perm.inverse.sigma] = perm_rep.T
        self._matrices = matrices
        return matrices

    def matrix_tensor(self, dtype=torch.float64, device=torch.device('cpu')) -> torch.Tensor:
        tensors = [
            torch.from_numpy(self.matrix_representations[perm.sigma]).unsqueeze(0).to(dtype)
            for perm in self.permutations
        ]
        return torch.concatenate(tensors, dim=0).squeeze().to(device)

    def coset_rep_matrices(self, dtype=torch.float64) -> list[torch.Tensor]:
        coset_reps = [Permutation(contiguous_cycle(self.n, i)).sigma for i in range(self.n)]
        return  [torch.from_numpy(self.matrix_representations[rep]).to(dtype) for rep in coset_reps]
    
    def alternating_matrix_tensor(self, dtype=torch.float64, device=torch.device('cpu')):
        tensors = [
            torch.asarray(self.matrix_representations[perm.sigma]).unsqueeze(0).to(dtype)
            for perm in self.permutations if perm.parity == 0
        ]
        return torch.concatenate(tensors, dim=0).squeeze().to(device)
