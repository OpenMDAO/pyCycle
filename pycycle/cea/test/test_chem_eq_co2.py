import unittest

from openmdao.api import Problem, Group, IndepVarComp

from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.chem_eq import ChemEq
from pycycle.cea import species_data


class ChemEqTestCase(unittest.TestCase):

    def setUp(self):
        self.thermo = species_data.Thermo(species_data.co2_co_o2)
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

        tol = 6e-4

        assert_rel_error(self, p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)

    def test_set_total_hp(self):
        p = self.p
        p.model.add_subsystem('enthalpy', IndepVarComp('h', -1801.3, units='cal/g'), promotes=['*'])
        p.model.add_subsystem('ceq', ChemEq(thermo=self.thermo, mode="h"), promotes=["*"])
        p.setup(check=False)
        p.run_model()
        tol = 6e-4

        assert_rel_error(self, p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)

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

        tol = 6e-4
        assert_rel_error(self, p['n'], [8.15344263e-06, 2.27139552e-02, 4.07672148e-06], tol)


if __name__ == "__main__":

    unittest.main()
