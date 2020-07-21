import unittest
import numpy as np

from openmdao.api import Problem, Group, IndepVarComp

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.chem_eq import ChemEq
from pycycle.cea import species_data


class ChemEqTestCase(unittest.TestCase):

    def setUp(self):
        self.thermo = species_data.Thermo(species_data.janaf)
        p = self.p = Problem(model=Group())
        p.model.suppress_solver_output = True
        indeps = p.model.add_subsystem('pressure', IndepVarComp(), promotes=['*'])
        indeps.add_output('P', 1.034210, units="bar")

    def test_set_total_tp(self):
        p = self.p
        p.model.add_subsystem('temp', IndepVarComp('T', 1500., units='degK'), promotes=['*'])
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="T"), promotes=["*"])
        p.setup(check=False)
        p.run_model()

        check_val = np.array([3.23319258e-04, 1.16619251e-10, 1.10131075e-05, 1.00000000e-10,
                            4.19205954e-05, 2.27520440e-07, 1.00000000e-10, 2.69368126e-02,
                            6.33239011e-08, 7.21077455e-03])

        tol = 6e-4

        assert_near_equal(p['n'], check_val, tol)

    def test_set_total_hp(self):
        p = self.p
        p.model.add_subsystem('enthalpy', IndepVarComp('h', -24.3, units='cal/g'), promotes=['*'])
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="h"), promotes=["*"])
        p.setup(check=False)
        p.run_model()
        check_val = np.array([3.23319258e-04, 1.00000000e-10, 1.10131241e-05, 1.00000000e-10,
                            1.00000000e-10, 1.00000000e-10, 1.00000000e-10, 2.69578866e-02,
                            1.00000000e-10, 7.23199382e-03])

        tol = 6e-4

        assert_near_equal(p['n'], check_val, tol)

    def test_set_total_sp(self):
        p = self.p
        p.model.add_subsystem(
            'entropy',
            IndepVarComp(
                'S',
                1.58645,
                units='cal/(g*degK)'),
            promotes=['*'])
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="S"), promotes=["*"])
        p.setup(check=False)
        p.run_model()

        check_val = np.array([3.23319258e-04, 1.00000000e-10, 1.10131241e-05, 1.00000000e-10,
                            1.00000000e-10, 1.00000000e-10, 1.00000000e-10, 2.69578866e-02,
                            1.00000000e-10, 7.23199382e-03])

        tol = 6e-4
        assert_near_equal(p['n'], check_val, tol)


if __name__ == "__main__":

    unittest.main()