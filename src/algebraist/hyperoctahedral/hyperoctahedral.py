
from algebraist.permutations import Permutation



class HyperOctahedralElement:

    def __init__(self, sign_vector, permutation: Permutation):
        assert len(sign_vector) == len(permutation)
        self.n = len(sign_vector)
        self.signs = sign_vector
        self.permutation = permutation

    def __mul__(self, other):
        new_perm = self.permutation * other.permutation
        new_signs = self.signs * (self.permutation * other.signs)
    