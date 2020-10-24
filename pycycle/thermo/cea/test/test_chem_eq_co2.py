import unittest

from openmdao.api import Problem, Group

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea.chem_eq import ChemEq
from pycycle.thermo.cea import species_data
from pycycle import constants


class ChemEqTestCase(unittest.TestCase):

    def setUp(self):
        self.thermo = species_data.Properties(species_data.co2_co_o2, init_elements=constants.CO2_CO_O2_ELEMENTS)
        p = self.p = Problem(model=Group())
        p.model.suppress_solver_output = True
        p.model.set_input_defaults('P', 1.034210, units="bar")

    def test_set_total_tp(self):
        p = self.p
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="T"), promotes=["*"])
        p.model.set_input_defaults('T', 1500., units='degK')
        p.setup(check=False)
        p.run_model()

        tol = 6e-4

        assert_near_equal(p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)

    def test_set_total_hp(self):
        p = self.p
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="h"), promotes=["*"])
        p.model.set_input_defaults('h', -1801.3, units='cal/g')
        p.setup(check=False)
        p.run_model()
        tol = 6e-4

        assert_near_equal(p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)

    def test_set_total_sp(self):
        p = self.p
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="S"), promotes=["*"])
        p.model.set_input_defaults('S', 1.58645, units='cal/(g*degK)')
        p.setup(check=False)
        p.run_model()

        tol = 6e-4
        assert_near_equal(p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)


if __name__ == "__main__":

    unittest.main()
