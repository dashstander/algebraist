from functools import reduce
from hypothesis import given, strategies as st
import math
import numpy as np
import pytest

from algebraist.irreps import SnIrrep
from algebraist.permutations import Permutation
from algebraist.tableau import generate_partitions


@st.composite
def permutation_list_strategy(draw, n, min_length=1, max_length=5):
    length = draw(st.integers(min_value=min_length, max_value=max_length))
    return [draw(st.sampled_from(Permutation.full_group(n))) for _ in range(length)]


@st.composite
def sn_with_permutations(draw):
    n = draw(st.integers(3, 5))
    permutations = draw(permutation_list_strategy(n))
    return n, permutations


def test_snirrep_initialization():
    irrep = SnIrrep(3, (2, 1))
    assert irrep.n == 3
    assert irrep.shape == (2, 1)
    assert len(irrep.basis) == 2  # There are two standard Young tableaux for (2,1)


def test_trivial_rep():
    triv = SnIrrep(4, (4,))
    assert triv.n == 4
    assert triv.dim == 1
    for perm in Permutation.full_group(4):
        assert np.allclose(triv.matrix_representations()[perm.sigma], [[1]])


def test_alternating_rep():
    alt = SnIrrep(4, (1, 1, 1, 1))
    assert alt.n == 4
    assert alt.dim == 1
    for perm in Permutation.full_group(4):
        expected = 1 if perm.parity == 0 else -1
        assert np.allclose(alt.matrix_representations()[perm.sigma], [[expected]])


@given(st.integers(2, 5))
def test_adjacent_transposition_matrices(n):
    irrep = SnIrrep(n, (n-1, 1))
    matrices = irrep.generate_transposition_matrices()
    for (i, j), matrix in matrices.items():
        assert matrix.shape == (n-1, n-1)
        assert np.allclose(matrix @ matrix, np.eye(n-1))  # Transpositions are involutions


def test_matrix_representations():
    irrep = SnIrrep(3, (2, 1))
    reps = irrep.matrix_representations()
    assert len(reps) == 6  # There are 6 permutations in S3
    for _, matrix in reps.items():
        assert matrix.shape == (2, 2)  # The (2,1) irrep of S3 is 2-dimensional


@given(st.integers(3, 5))
def test_matrix_tensor(n):
    irrep = SnIrrep(n, (n-1, 1))
    tensor = irrep.matrix_tensor()
    assert tensor.shape == (len(Permutation.full_group(n)), n-1, n-1)


@given(st.integers(3, 5))
def test_orthogonality_relations(n):
    partitions = generate_partitions(n)
    irreps = [SnIrrep(n, p) for p in partitions]
    
    # First orthogonality relation
    for irrep1 in irreps:
        for irrep2 in irreps:
            if irrep1.shape != irrep2.shape:
                continue
            reps1 = irrep1.matrix_representations()
            reps2 = irrep2.matrix_representations()
            sum_matrix = sum(reps1[perm.sigma] @ np.conj(reps2[perm.sigma].T) for perm in Permutation.full_group(n))
            expected = np.eye(sum_matrix.shape[0]) if irrep1.shape == irrep2.shape else np.zeros_like(sum_matrix)
            assert np.allclose(sum_matrix / len(reps1), expected)

    # Second orthogonality relation (sum of squares of dimensions equals n!)
    dimensions = [irrep.dim for irrep in irreps]
    assert sum(d**2 for d in dimensions) == math.factorial(n), f'Failing dimensions: {dimensions}'


@given(st.integers(3, 5))
def test_all_partitions(n):
    partitions = generate_partitions(n)
    for partition in partitions:
        irrep = SnIrrep(n, partition)
        assert irrep.n == n
        assert irrep.shape == partition
        assert irrep.dim == len(irrep.basis)
        reps = irrep.matrix_representations()
        assert len(reps) == math.factorial(n)
        for _, matrix in reps.items():
            assert matrix.shape == (irrep.dim, irrep.dim)


@given( sn_with_permutations())
def test_representation_homomorphism(n_and_permutations):
    n, permutations = n_and_permutations
    partitions = generate_partitions(n)
    
    for partition in partitions:
        irrep = SnIrrep(n, partition)
        reps = irrep.matrix_representations()
        
        composed_perm = reduce(lambda x, y: x * y, permutations)
        rep_composed = reps[composed_perm.sigma]
        
        if irrep.dim == 1:  # For trivial and sign representations
            rep_product = reduce(lambda x, y: x * y, [reps[perm.sigma][0][0] for perm in permutations])
            assert np.isclose(rep_composed[0][0], rep_product), \
                f"Homomorphism property failed for partition {partition} and permutations {permutations}"
        else:
            rep_product = reduce(lambda x, y: x @ y, [reps[perm.sigma] for perm in permutations])
            assert np.allclose(rep_composed, rep_product), \
                f"Homomorphism property failed for partition {partition} and permutations {permutations}"


if __name__ == '__main__':
    pytest.main()