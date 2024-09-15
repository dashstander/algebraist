from collections import Counter
from hypothesis import given, strategies as st
from math import factorial
import pytest
import random
from algebraist.tableau import (
    YoungTableau, generate_partitions, enumerate_standard_tableau, conjugate_partition, hook_length
)

@st.composite
def partition_strategy(draw, max_n=10):
    n = draw(st.integers(min_value=1, max_value=max_n))
    k = draw(st.integers(min_value=1, max_value=n))
    
    # Assign each element to a random bin
    bin_assignments = draw(st.lists(st.integers(min_value=0, max_value=k-1), min_size=n, max_size=n))
    
    # Count occurrences in each bin
    counts = Counter(bin_assignments)
    
    # Convert to partition and sort in decreasing order
    partition = sorted(counts.values(), reverse=True)
    
    return tuple(partition)

@st.composite
def young_tableau_strategy(draw):
    partition = draw(partition_strategy())
    values = list(range(1, sum(partition) + 1))
    random.shuffle(values)
    tableau = [values[sum(partition[:i]):sum(partition[:i+1])] for i in range(len(partition))]
    return YoungTableau(tableau)


def test_young_tableau_init():
    yt = YoungTableau([[0, 1], [2]])
    assert yt.shape == (2, 1)
    assert yt.n == 3


def test_young_tableau_equality():
    yt1 = YoungTableau([[0, 1], [2]])
    yt2 = YoungTableau([[0, 1], [2]])
    yt3 = YoungTableau([[0, 2], [1]])
    assert yt1 == yt2
    assert yt1 != yt3


def test_young_tableau_indexing():
    yt = YoungTableau([[0, 1], [2]])
    assert yt[0, 1] == 1
    assert yt[1, 0] == 2


def test_young_tableau_index():
    yt = YoungTableau([[0, 1], [2]])
    assert yt.index(1) == (0, 1)
    assert yt.index(2) == (1, 0)


def test_young_tableau_transposition_dist():
    yt = YoungTableau([[0, 1], [2]])
    assert yt.transposition_dist(0, 1) == 1
    assert yt.transposition_dist(0, 2) == -1


def test_hook_length_known_values():
    assert hook_length((1,)) == 1
    assert hook_length((2,)) == 1
    assert hook_length((1, 1)) == 1
    assert hook_length((3,)) == 1
    assert hook_length((2, 1)) == 2
    assert hook_length((1, 1, 1)) == 1
    assert hook_length((4,)) == 1
    assert hook_length((3, 1)) == 3
    assert hook_length((2, 2)) == 2
    assert hook_length((2, 1, 1)) == 3
    assert hook_length((1, 1, 1, 1)) == 1
    assert hook_length((5, 4, 1)) == 288



def test_generate_partitions():
    assert set(generate_partitions(4)) == {(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)}


def test_enumerate_standard_tableau():
    tableaux = enumerate_standard_tableau((2, 1))
    assert len(tableaux) == 2
    assert YoungTableau([[0, 1], [2]]) in tableaux
    assert YoungTableau([[0, 2], [1]]) in tableaux


def test_conjugate_partition():
    assert conjugate_partition((3, 2, 1)) == (3, 2, 1)
    assert conjugate_partition((4, 2)) == (2, 2, 1, 1)


@given(partition=partition_strategy())
def test_partition_sum(partition):
    assert sum(partition) == sum(conjugate_partition(partition))


@given(tableau=young_tableau_strategy())
def test_tableau_shape_matches_values(tableau):
    assert sum(tableau.shape) == tableau.n
    assert all(len(row) == tableau.shape[i] for i, row in enumerate(tableau.values))


@given(tableau=young_tableau_strategy())
def test_tableau_contains_all_numbers(tableau):
    all_numbers = set(range(1, tableau.n + 1))
    tableau_numbers = set(num for row in tableau.values for num in row)
    assert all_numbers == tableau_numbers


@given(partition=partition_strategy())
def test_conjugate_involution(partition):
    assert conjugate_partition(conjugate_partition(partition)) == partition


@given(n=st.integers(1, 10))
def test_generate_partitions_sum(n):
    partitions = generate_partitions(n)
    assert all(sum(p) == n for p in partitions)


@given(partition=partition_strategy())
def test_standard_tableau_count(partition):
    tableaux = enumerate_standard_tableau(partition)
    # This is not an efficient way to calculate the number of standard tableaux,
    # but it works for small partitions. For larger ones, you'd use the hook length formula.
    expected_count = len(set(enumerate_standard_tableau(partition)))
    assert len(tableaux) == expected_count


@given(partition=partition_strategy())
def test_hook_length_matches_enumeration(partition):
    hook_count = hook_length(partition)
    enumeration_count = len(enumerate_standard_tableau(partition))
    assert hook_count == enumeration_count

@given(partition=partition_strategy())
def test_hook_length_positive(partition):
    assert hook_length(partition) > 0

@given(n=st.integers(1, 10))
def test_hook_length_sum_factorial(n):
    partitions = generate_partitions(n)
    print(partitions)
    total = sum(hook_length(p)**2 for p in partitions)
    expected = factorial(n)
    print(f"n: {n}, Total from hook lengths: {total}, n!: {expected}")  # Debug print
    assert total == expected, f"Sum of hook lengths ({total}) does not equal {n}! ({expected}) for n={n}"



if __name__ == '__main__':
    pytest.main()