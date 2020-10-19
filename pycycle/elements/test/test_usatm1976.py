import numpy as np
import unittest
import os

from openmdao.api import Problem
from openmdao.utils.assert_utils import assert_check_partials

from pycycle.elements.US1976 import USatm1976Comp

class TestCase1976(unittest.TestCase):

    def test_derivs(self):

        p = Problem()
        p.model.add_subsystem('std_1976', USatm1976Comp())

        p.setup(force_alloc_complex=True)
        p.final_setup()

        data = p.check_partials(out_stream=None)

        assert_check_partials(data, atol=1e-4, rtol=1e-1)


if __name__ == "__main__":
    unittest.main()

