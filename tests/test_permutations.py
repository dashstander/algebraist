from hypothesis import given, strategies as st
import math
import pytest

from algebraist.permutations import Permutation

# Helper strategy to generate valid permutations
@st.composite
def permutation_strategy(draw, max_n=10):
    n = draw(st.integers(min_value=1, max_value=max_n))
    return Permutation(draw(st.permutations(range(n))))


def test_init():
    p = Permutation([2, 0, 1])
    assert p.sigma == (2, 0, 1)
    assert p.n == 3


def test_full_group():
    group = Permutation.full_group(3)
    assert len(group) == 6
    assert Permutation([0, 1, 2]) in group
    assert Permutation([2, 1, 0]) in group


def test_identity():
    id3 = Permutation.identity(3)
    assert id3.sigma == (0, 1, 2)
    assert id3.is_identity()


def test_transposition():
    t = Permutation.transposition(4, 1, 3)
    assert t.sigma == (0, 3, 2, 1)


def test_multiplication():
    p1 = Permutation([1, 2, 0])
    p2 = Permutation([2, 0, 1])
    p3 = p1 * p2
    assert p3.sigma == (0, 1, 2)

@pytest.mark.parametrize("perm, power, expected", [
    (Permutation([1, 2, 0]), 2, (2, 0, 1)),
    (Permutation([1, 2, 0]), 3, (0, 1, 2)),
    (Permutation([1, 0, 2]), 2, (0, 1, 2)),
])
def test_power(perm, power, expected):
    assert (perm ** power).sigma == expected


def test_inverse():
    p = Permutation([2, 0, 1])
    inv = p.inverse
    assert (p * inv).sigma == (0, 1, 2)


def test_cycle_rep():
    p = Permutation([1, 2, 3, 0])
    assert p.cycle_rep == [(0, 3, 2, 1)]


@pytest.mark.parametrize("perm, expected_parity", [
    (Permutation([1, 0, 3, 2]), 0),
    (Permutation([1, 2, 3, 0]), 1),
])
def test_parity(perm, expected_parity):
    assert perm.parity == expected_parity


def test_order():
    p = Permutation([1, 2, 0])
    assert p.order == 3


def test_transposition_decomposition():
    p = Permutation([1, 2, 0, 3])
    assert p.transposition_decomposition() == [(0, 2), (1, 2)]



@pytest.mark.parametrize("n", [3, 4, 5])
def test_permutation_index(n):
    indices = [p.permutation_index() for p in Permutation.full_group(n)]
    assert indices == list(range(math.factorial(n)))


@pytest.mark.parametrize("perm, expected_class", [
    (Permutation([1, 0, 2, 3]), (2, 1, 1)),
    (Permutation([1, 2, 3, 0]), (4,)),
    (Permutation([1, 0, 3, 2]), (2, 2)),
])
def test_conjugacy_class(perm, expected_class):
    assert perm.conjugacy_class == expected_class


def test_adjacent_transposition_decomposition():
    p = Permutation([2, 0, 1, 3])
    expected = [[(0, 1)], [(1, 2)]]
    assert p.adjacent_transposition_decomposition() == expected


def test_fft_coset_rep_and_new_value():
    p = Permutation([2, 3, 1, 0])
    expected_rep = Permutation((0, 3, 2, 1))
    expected_s3_element = (2, 0, 1)
    rep, new_value = p.fft_coset_rep_and_new_value()
    assert new_value.sigma == expected_s3_element
    assert rep == expected_rep


# Property: parity of product is product of parities
@given(perm1=permutation_strategy(), perm2=permutation_strategy())
def test_parity_product_property(perm1, perm2):
    if perm1.n != perm2.n:
        return  # Skip if permutations are of different sizes
    assert (perm1 * perm2).parity == (perm1.parity + perm2.parity) % 2


# Property: inverse of inverse is the original permutation
@given(perm=permutation_strategy())
def test_double_inverse_property(perm):
    assert perm == perm.inverse.inverse

# Property: order of permutation divides group order (n!)
@given(perm=permutation_strategy())
def test_order_divides_group_order(perm):
    from math import factorial
    assert factorial(perm.n) % perm.order == 0

# Property: conjugacy class partition sums to n
@given(perm=permutation_strategy())
def test_conjugacy_class_sum(perm):
    assert sum(perm.conjugacy_class) == perm.n

# Property: multiplication is associative
@given(perm1=permutation_strategy(), perm2=permutation_strategy(), perm3=permutation_strategy())
def test_multiplication_associativity(perm1, perm2, perm3):
    if perm1.n != perm2.n or perm2.n != perm3.n:
        return  # Skip if permutations are of different sizes
    assert (perm1 * perm2) * perm3 == perm1 * (perm2 * perm3)

# Property: identity permutation is neutral element
@given(perm=permutation_strategy())
def test_identity_neutral(perm):
    identity = Permutation.identity(perm.n)
    assert perm * identity == perm
    assert identity * perm == perm

# Property: permutation to the power of its order is identity
@given(perm=permutation_strategy())
def test_power_order_is_identity(perm):
    assert (perm ** perm.order).is_identity()
