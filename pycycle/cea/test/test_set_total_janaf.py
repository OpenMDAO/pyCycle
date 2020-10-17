import unittest
from openmdao.api import Problem

from pycycle.cea import species_data
from pycycle.cea.set_total import SetTotal

from openmdao.utils.assert_utils import assert_near_equal
from pycycle import constants


class _TestJanafThermo(unittest.TestCase):

    def test_std_day(self):

        thermo = species_data.Properties(species_data.janaf, constants.AIR_MIX)

        top = Problem()
        top.model = SetTotal(thermo_data=species_data.janaf, mode="T")
        top.model.set_input_defaults('b0', thermo.b0)
        top.model.set_input_defaults('T', 287.778, units='degK')
        top.model.set_input_defaults('P', 1.02069, units='bar')
        top.setup(check=False)

        top.run_model()

        assert_near_equal(top['gamma'], 1.40023310084, 1e-4)

    def test_mid_temp(self):

        thermo = species_data.Properties(species_data.janaf, constants.AIR_MIX)

        top = Problem()
        top.model = SetTotal(thermo_data=species_data.janaf, mode="T")
        top.model.set_input_defaults('b0', thermo.b0)
        top.model.set_input_defaults('T', 1500, units='degK')
        top.model.set_input_defaults('P', 1.02069, units='bar')

        top.setup(check=False)

        top.run_model()

        assert_near_equal(top['gamma'], 1.30444205736, 1e-4)  # 1.30444
        assert_near_equal(top['flow:S'], 2.05758694175, 1e-4)  # NPSS 2.05717


class TestSetTotalEquivilence(unittest.TestCase):

    def setUp(self):

        thermo = species_data.Properties(species_data.janaf, constants.AIR_MIX)
        
        self.tp_set = Problem(SetTotal(thermo_data=species_data.janaf, mode='T'))
        self.hp_set = Problem(SetTotal(thermo_data=species_data.janaf, mode='h'))
        self.sp_set = Problem(SetTotal(thermo_data=species_data.janaf, mode='S'))

        self.tp_set.model.set_input_defaults('b0', thermo.b0)
        self.hp_set.model.set_input_defaults('b0', thermo.b0)
        self.sp_set.model.set_input_defaults('b0', thermo.b0)

        self.tp_set.model.set_input_defaults('T', 518., units="degR")
        self.tp_set.model.set_input_defaults('P', 14.7, units="psi")
        self.tp_set.setup(check=False)

        self.hp_set.model.set_input_defaults('P', 14.7, units="psi")
        self.hp_set.model.set_input_defaults('h', 1., units="Btu/lbm")
        self.hp_set.setup(check=False)

        self.sp_set.model.set_input_defaults('P', 14.7, units="psi")
        self.sp_set.model.set_input_defaults('S', 1., units="Btu/(lbm*degR)")  # 'cal/(g*degK)'
        self.sp_set.setup(check=False)

    def check_tp(self, T, P):

        tp_set = self.tp_set
        hp_set = self.hp_set
        sp_set = self.sp_set

        tp_set['T'] = T
        tp_set['P'] = P

        print("TP")
        tp_set.run_model()

        hp_set['h'] = tp_set['flow:h']
        hp_set['P'] = tp_set['flow:P']

        print("hP")
        hp_set.run_model()

        assert_near_equal(hp_set['flow:T'], tp_set['T'], 1e-4)

        sp_set['S'] = tp_set['flow:S']
        sp_set['P'] = tp_set['flow:P']

        # sp_set.solver.iprint=1
        print("SP")
        sp_set.run_model()
        assert_near_equal(sp_set['flow:T'], tp_set['T'], 1e-4)

    def test_set_total_equivilence(self):

        self.check_tp(518., 14.7)


if __name__ == "__main__":

    unittest.main()
