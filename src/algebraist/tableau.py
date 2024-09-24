from collections import deque
from functools import cache, total_ordering, reduce
from itertools import chain
from math import factorial
from operator import mul
from typing import Iterator, Sequence


def check_partition_shape(partition_shape: tuple[int, ...]) -> bool:
    """
    Checks that a given shape defines a correct partition, in particular that is in decreasing order.
    Args:
        partition_shape (tuple[int, ...]): the partition of n
    Returns:
        bool True if valid, False otherwise
    """
    for i in range(len(partition_shape) - 1):
        j = i + 1
        if partition_shape[j] > partition_shape[i]:
            return False
    return True


def youngs_lattice_covering_relation(partition: tuple[int, ...]) -> list[tuple[int, ...]]:
    """
    Takes a partition and returns a list of the partitions directly below it in Young's lattice.

    Args:
        partititon (tuple[int, ...]): an integer partition of n
    
    Returns:
        list[tuple[int, ...]] all of the partitions directly beneath partition
    """
    children = []
    for i in range(len(partition)):
        if partition[i] > (partition[i+1] if i+1 < len(partition) else 0):
            child = list(partition)
            child[i] -= 1
            if child[-1] == 0:
                child.pop()
            children.append(tuple(child))
    return children


def youngs_lattice_down(top_partition: tuple[int, ...]) -> dict[tuple[int, ...], list[tuple[int, ...]]]:
    """
    Generate the sub-lattice of Young's lattice from top_partition to (1,)
    """
    lattice = {}
    queue = deque([top_partition])
    while queue:
        partition = queue.popleft()
        children = youngs_lattice_covering_relation(partition)
        lattice[partition] = children
        for child in children:
            if child not in lattice:
                queue.append(child)
    return lattice


def generate_standard_young_tableaux(shape: tuple[int, ...]) -> list[list[list[int]]]:
    """
    A standard Young Tableau is a filling of a tableau lambda with the values 0...n-1 where the values are increasing from left to right in each row and top to bottom in each column. This function generates all of them for a given shape.

    The standard Young Tableau for a given partition lambda are one-to-one with chains in the sub-lattice of Young's lattice from [(1,), lambda]. Here we generate the sub-lattice and then generate each chain.
    """
    n = sum(shape) - 1
    lattice = youngs_lattice_down(shape)

    def backtrack(partition: tuple[int], value: int) -> Iterator[list[list[int]]]:
        if partition == (1,):
            yield [[0]]
        
        for child_partition in lattice[partition]:
            child_tableaux = backtrack(child_partition, value - 1)
            for tableau in child_tableaux:
                new_tableau = [row[:] for row in tableau]
                for i, (parent_count, child_count) in enumerate(zip(partition, child_partition + (0,))):
                    if parent_count > child_count:
                        if i >= len(new_tableau):
                            new_tableau.append([])
                        new_tableau[i].append(value)
                        yield new_tableau
                        break

    return backtrack(shape, n)


@cache
def generate_partitions(n: int) -> Sequence[tuple[int, ...]]:
    if n <= 5:
        return {
            0: [],
            1: [(1,)],
            2: [(2,), (1, 1)],
            3: [(3,), (2, 1), (1, 1, 1)],
            4: [(4,), (3, 1), (2, 2), (2, 1, 1), (1, 1, 1, 1)],
            5: [(5,), (4, 1), (3, 2), (3, 1, 1), (2, 2, 1), (2, 1, 1, 1), (1, 1, 1, 1, 1)],
            6: [(6,), (5, 1), (4, 2), (4, 1, 1), (3, 3), (3, 2, 1), (3, 1, 1, 1), (2, 1, 1, 1, 1), (1, 1, 1, 1, 1, 1)]
        }[n]
    
    partitions = [(n,)]
    for k in range(1, n):
        for p in generate_partitions(n - k):
            new_partition = tuple(sorted((k, *p), reverse=True))
            if new_partition not in partitions:
                partitions.append(new_partition)
    
    return partitions


def conjugate_partition(partition: tuple[int, ...]) -> tuple[int, ...]:
    n = sum(partition)
    conj_part = []
    for i in range(n):
        reverse = [j for j in partition if j > i]
        if reverse:
            conj_part.append(len(reverse))
    return tuple(conj_part)
    

@total_ordering
class YoungTableau:

    def __init__(self, values: list[list[int]]):
        self.values = values
        self.shape = tuple([len(row) for row in values])
        self.n = sum(self.shape)

    def __repr__(self) -> str:
        strrep = []
        for row in self.values:
            strrep.append('|' + '|'.join([str(v) for v in row]) + '|' )
        return '\n'.join(strrep)

    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, key):
        i, j = key
        return self.values[i][j]
    
    def __setitem__(self, key, value):
        i, j = key
        self.values[i][j] = value

    def __eq__(self, other) -> bool:
        if not isinstance(other, YoungTableau):
            other = YoungTableau(other)
        if (self.n != other.n) or (self.shape != other.shape):
            return False
        for row1, row2 in zip(self.values, other.values):
            if row1 != row2:
                return False
        return True

    def __lt__(self, other) -> bool:
        """Define a custom ordering for Young tableaux consistent with restrictions.

        !!!!! A SHOCKINGLY IMPORTANT METHOD !!!!!
        This is __not__ the lexicographic order on tableaux. It is an order that is 
        designed to be consistent with restrictions to smaller tableaux.
        
        This ordering is crucial because:
        1. It defines the basis of the matrix irreps.
        2. We specifically want the elements of S_{n-1} < S_n (the elements of S_n 
        that fix n) to have a block-diagonal structure.
        3. We want those blocks to be identical to the irreps of S_{n-1} as its own group.
        
        This approach saves us a lot of complication when doing the recursion in the FFT.
        """
        if self.n != other.n:
            return self.n < other.n
        
        if self.n == 1:
            return False  # All tableaux of size 1 are equal
        self_n_index = self.index(self.n - 1)
        other_n_index = other.index(self.n - 1)
                                  
        if self_n_index == other_n_index:
            # Compare restricted shapes
            return self.restrict() <  other.restrict()
        else:
            return self_n_index < other_n_index
        
    def restrict(self):
        return YoungTableau([[v for v in row if v != self.n - 1] for row in self.values])
    
    def restricted_shape(self) -> tuple[int, ...]:
        """Return the shape of the tableau after removing the largest number."""
        return tuple([tuple([v for v in row if v != self.n - 1]) for row in self.values])

    def index(self, val: int) -> tuple[int, ...]:
        for r, row in enumerate(self.values):
            if val in row:
                c = row.index(val)
                return r, c
        raise ValueError(f'{val} could not be found in tableau.')
    
    def transposition_dist(self, x: int, y: int) -> int:
        #assert y == x + 1
        row_x, col_x = self.index(x)
        row_y, col_y = self.index(y)
        row_dist = row_x - row_y
        col_dist = col_y - col_x
        return row_dist + col_dist
    
    def __hash__(self) -> int:
        return hash(str(self.values))


def enumerate_standard_tableau(partition_shape: tuple[int, ...]) -> list[YoungTableau]:
    if not check_partition_shape(partition_shape):
        raise ValueError(f'Shape {partition_shape} is not a valid partition.')
    
    return sorted([
        YoungTableau(tableau) 
        for tableau in generate_standard_young_tableaux(partition_shape)
    ])


def hook_length(partition: tuple[int, ...]) -> int:
    """
    Calculate the number of standard Young tableaux for a given partition
    using the hook length formula.
    
    Args:
    partition (tuple): A tuple representing the partition (in descending order)
    
    Returns:
    int: The number of standard Young tableaux for the given partition
    """
    n = sum(partition)

    if n == 1:
        return 1
    
    # Create the Young diagram
    diagram = [[0 for _ in range(partition[i])] for i in range(len(partition))]
    
    # Calculate hook lengths
    for i in range(len(partition)):
        for j in range(partition[i]):
            # Hook length is (number of boxes below + number of boxes to the right )
            diagram[i][j] = partition[i] - j + sum(1 for row in partition[i+1:] if row > j)
        
    # Calculate the product of hook lengths
    hook_product = reduce(mul, chain.from_iterable(diagram))

    # Apply the formula
    return factorial(n) // hook_product
