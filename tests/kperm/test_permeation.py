import numpy as np
from kperm.permeation import _count_perm_cross


def test__count_perm_cross_simulated_input_1():
    occ = [
        [[100], [0], [1], [], [9], [101]],
        [[0], [], [1], [], [9], [101]],
        [[0], [1], [9], [], [], [101]],
        [[100], [0], [1], [], [9], [101]],
        [[100], [0], [1], [], [9], [101]],
    ]
    output = _count_perm_cross(occ)

    assert np.array_equal(output[0], np.array([0, 0, 1, -1, 0]))
    assert output[1] == [(2, 9)]
    assert output[2] == [(3, 9)]
