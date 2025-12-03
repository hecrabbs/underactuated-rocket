import random

import numpy as np
from numpy.testing import assert_array_equal

from underactuated_rocket.helpers import cross_2d, new_col_vec


class TestHelpers:
    def test_new_col_vec(self):
        for _ in range(10):
            x = random.random()
            y = random.random()
            assert_array_equal(np.array([[x],[y]]), new_col_vec(x,y))

    def test_cross_2d(self):
        a = new_col_vec(1,0)
        b = new_col_vec(0,1)
        assert cross_2d(a,b) == 1
        assert cross_2d(a,a) == 0
        assert cross_2d(a,-a) == 0

        for _ in range(10):
            a = new_col_vec(random.random(), random.random())
            b = new_col_vec(random.random(), random.random())
            assert cross_2d(a,b) == -cross_2d(b,a)
