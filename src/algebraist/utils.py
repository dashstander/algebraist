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
from functools import lru_cache
from itertools import pairwise, permutations
import torch
from typing import Sequence


def adj_trans_decomp(i: int, j: int) -> Sequence[tuple[int, ...]]:
    """
    Given two integers, i < j, gives a sequences of adjacent transpositions that when composed transpose i and j
    Args:
        i (int): first element of transposition
        j (int): second element of transposition
    Returns:
        list[tuple[int]] a sequence of two element tuples, i.e. [(i + 1, i + 2), (i + 2, i + 3), ...]
    """
    center = [(i, i + 1)]
    i_to_j = list(range(i+1, j+1))
    adj_to_j = list(pairwise(i_to_j))
    return list(reversed(adj_to_j)) + center + adj_to_j



def cycle_to_one_line(cycle_rep: list[tuple[int, ...]]) -> tuple[int, ...]:
    """
    Given a permutation in cycle representation where (i, j, k) means that i -> j, j -> k, and k -> i, this returns the permutation in one-line notation.
    Args:
        cycle_rep (list[tuple[int]]): the cycle representation of a permutation where fixed points are **not** omitted. 
    Returns:
        tuple[int] a permutation of n elements given as a tuple of the integers 0...n-1
    """
    n = sum([len(c) for c in cycle_rep])
    sigma = [-1] * n
    for cycle in cycle_rep:
        first = cycle[0]
        if len(cycle) == 1:
            sigma[first] = first
        else:
            for val1, val2 in pairwise(cycle):
                sigma[val2] = val1
                lastval  = val2
            sigma[first] = lastval
    return tuple(sigma)



@lru_cache(maxsize=20)
def generate_all_permutations(n: int) -> torch.Tensor:
    return torch.tensor(list(permutations(range(n))), dtype=torch.int64)


def trans_to_one_line(i: int, j: int, n: int) -> tuple[int, ...]:
    """
    Helper function to create a permutation that is the transposition of two elements.
    Required that i < j < n
    Args:
        i (int): smaller element of transposition
        j (int): larger element of transposition
        n (int): total number of elements
    Returns:
        tuple[int] the permutation with n elements is the same as (0, ..., n-1) except i and j are swapped
    """
    assert i < j and j < n
    sigma = list(range(n))
    sigma[i] = j
    sigma[j] = i
    return tuple(sigma)


def youngs_lattice_covering_relation(partition: tuple[int, ...]) -> list[tuple[int, ...]]:
    children = []
    k = len(partition)
    for i in range(k - 1):
        # check if valid subrepresentation
        if partition > partition[i+1]:
            # if so, copy, modify, and append to list
            child = list(deepcopy(partition))
            child[i] -= 1
            children.append(tuple(child))
    # the last subrep
    child = list(deepcopy(partition))
    if partition[-1] > 1:
        child[-1] -= 1
    else:
    # removing last element of partition if it’s a 1
        del child[-1]
    children.append(tuple(child))
    return sorted(children)
