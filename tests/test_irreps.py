import pytest
from hypothesis import given, strategies as st
from algebraist.irreps import SnIrrep, TrivialRep, AlternatingRep, make_irrep
from algebraist.permutations import Permutation
import numpy as np

def test_snirrep_initialization():
    irrep = SnIrrep(3, (2, 1))
    assert irrep.n == 3
    assert irrep.shape == (2, 1)
    assert len(irrep.basis) == 2  # There are two standard Young tableaux for (2,1)

def test_trivial_rep():
    triv = TrivialRep(4)
    assert triv.n == 4
    for perm in Permutation.full_group(4):
        assert triv.matrix_representations()[perm.sigma] == 1

def test_alternating_rep():
    alt = AlternatingRep(4)
    assert alt.n == 4
    for perm in Permutation.full_group(4):
        assert alt.matrix_representations()[perm.sigma] in [-1, 1]

def test_make_irrep():
    assert isinstance(make_irrep((3,)), TrivialRep)
    assert isinstance(make_irrep((1, 1, 1)), AlternatingRep)
    assert isinstance(make_irrep((2, 1)), SnIrrep)

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
    for perm, matrix in reps.items():
        assert matrix.shape == (2, 2)  # The (2,1) irrep of S3 is 2-dimensional

@given(st.integers(2, 5))
def test_matrix_tensor(n):
    irrep = SnIrrep(n, (n-1, 1))
    tensor = irrep.matrix_tensor()
    assert tensor.shape == (len(Permutation.full_group(n)), n-1, n-1)

@given(st.integers(2, 5))
def test_orthogonality_relations(n):
    partitions = [(n,), tuple([1]*n)] + [(n-1, 1)] if n > 2 else [(n,), tuple([1]*n)]
    irreps = [make_irrep(p) for p in partitions]
    
    # First orthogonality relation
    for irrep1 in irreps:
        for irrep2 in irreps:
            if irrep1.shape != irrep2.shape:
                continue
            reps1 = irrep1.matrix_representations()
            reps2 = irrep2.matrix_representations()
            sum_matrix = sum(reps1[perm] @ np.conj(reps2[perm].T) for perm in reps1)
            expected = np.eye(sum_matrix.shape[0]) if irrep1.shape == irrep2.shape else np.zeros_like(sum_matrix)
            assert np.allclose(sum_matrix / len(reps1), expected)

    # Second orthogonality relation (sum of squares of dimensions equals n!)
    dimensions = [len(irrep.basis) for irrep in irreps]
    assert sum(d**2 for d in dimensions) == np.math.factorial(n)

if __name__ == '__main__':
    pytest.main(['-v', '-s'])