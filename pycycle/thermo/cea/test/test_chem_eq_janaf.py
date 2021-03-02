import unittest
import numpy as np

from openmdao.api import Problem, Group

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea.chem_eq import ChemEq
from pycycle.thermo.cea import species_data
from pycycle import constants


class ChemEqTestCase(unittest.TestCase):

    def setUp(self):
        self.thermo = species_data.Properties(species_data.janaf, init_elements=constants.AIR_ELEMENTS)
        p = self.p = Problem(model=Group())
        p.model.suppress_solver_output = True
        p.model.set_input_defaults('P', 1.034210, units="bar")

    def test_set_total_tp(self):
        p = self.p
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo), promotes=["*"])
        p.model.set_input_defaults('T', 1500., units='degK')
        p.setup(check=False)
        p.run_model()

        check_val = np.array([3.23319236e-04, 1.00000000e-10, 1.10138429e-05, 1.00000000e-10,
                              1.72853915e-08, 6.76015824e-09, 1.00000000e-10, 2.69578737e-02,
                              4.80653071e-09, 7.23197634e-03])

        tol = 6e-4

        print(p['n'])
        print(check_val)
        assert_near_equal(p['n'], check_val, tol)


if __name__ == "__main__":

    unittest.main()