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


from copy import deepcopy
from functools import cached_property, reduce
from itertools import combinations, pairwise
import numpy as np
from numpy.typing import ArrayLike
import torch
from typing import Iterator, Self

from algebraist.permutations import Permutation
from algebraist.tableau import enumerate_standard_tableau, generate_partitions, hook_length, youngs_lattice_covering_relation, YoungTableau
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
        self.partition = partition
        self.dim = hook_length(self.partition)
        self.permutations = Permutation.full_group(n)

    @staticmethod
    def generate_all_irreps(n: int) -> Iterator[Self]:
        for partition in generate_partitions(n):
            yield SnIrrep(n, partition)

    def __eq__(self, other) -> bool:
        return self.partition == other.shape
    
    def __hash__(self) -> int:
        return hash(str(self.partition))
    
    def __repr__(self) -> str:
        return f'S{self.n} Irrep: {self.partition}'
    
    @cached_property
    def basis(self) -> list[YoungTableau]:
        return sorted(enumerate_standard_tableau(self.partition))

    def split_partition(self) -> list[tuple[int, ...]]:
        """A list of the partitions directly underneath the partition that defines this irrep, in terms of Young's lattice.
        These partitions directly beneath self.partition define the irreducible representations of S_{n-1} that this irrep "splits" into when we restrict to S_{n-1}. This relationship form the core of the "fast" part of the FFT.
        """
        return sorted(youngs_lattice_covering_relation(self.partition))
    
    def get_block_indices(self):
        """When restricted to S_{n-1} this irrep has a block-diagonal form--one block for each of the split irreps of S_{n-1}. This helper method gets the indices of those blocks.
        """
        curr_row, curr_col = 0, 0
        block_idx = []
        for split_irrep in self.split_partition():
            dim = SnIrrep(self.n - 1, split_irrep).dim
            next_row = curr_row + dim
            next_col = curr_col + dim
            block_idx.append((slice(curr_row, next_row ), slice(curr_col, next_col)))
            curr_row = next_row 
            curr_col = next_col
        
        return block_idx

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

    def coset_rep_matrices(self, dtype=torch.float64, device=torch.device('cpu')) -> list[torch.Tensor]:
        coset_reps = [Permutation(contiguous_cycle(self.n, i)).sigma for i in range(self.n)]
        return  [
            torch.from_numpy(self.matrix_representations[rep]).to(dtype).to(device)
            for rep in coset_reps
        ]
    
    def alternating_matrix_tensor(self, dtype=torch.float64, device=torch.device('cpu')):
        tensors = [
            torch.asarray(self.matrix_representations[perm.sigma]).unsqueeze(0).to(dtype)
            for perm in self.permutations if perm.parity == 0
        ]
        return torch.concatenate(tensors, dim=0).squeeze().to(device)
