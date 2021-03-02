import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.turbine import Turbine
from pycycle.elements.combustor import Combustor
from pycycle.elements.flow_start import FlowStart
from pycycle.maps.lpt2269 import LPT2269

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/turbineOD1.csv",
                      delimiter=",", skiprows=1)

header = [
    'turb.PRdes',
    'turb.effDes',
    'shaft.Nmech',
    'burn.FAR',
    'burn.Fl_I.W',
    'burn.Fl_I.Pt',
    'burn.Fl_I.Tt',
    'burn.Fl_I.ht',
    'burn.Fl_I.s',
    'burn.Fl_I.MN',
    'burn.Fl_I.V',
    'burn.Fl_I.A',
    'burn.Fl_I.Ps',
    'burn.Fl_I.Ts',
    'burn.Fl_I.hs',
    'turb.Fl_I.W',
    'turb.Fl_I.Pt',
    'turb.Fl_I.Tt',
    'turb.Fl_I.ht',
    'turb.Fl_I.s',
    'turb.Fl_I.MN',
    'turb.Fl_I.V',
    'turb.Fl_I.A',
    'turb.Fl_I.Ps',
    'turb.Fl_I.Ts',
    'turb.Fl_I.hs',
    'turb.Fl_O.W',
    'turb.Fl_O.Pt',
    'turb.Fl_O.Tt',
    'turb.Fl_O.ht',
    'turb.Fl_O.s',
    'turb.Fl_O.MN',
    'turb.Fl_O.V',
    'turb.Fl_O.A',
    'turb.Fl_O.Ps',
    'turb.Fl_O.Ts',
    'turb.Fl_O.hs',
    'turb.PR',
    'turb.eff',
    'turb.Np',
    'turb.Wp',
    'turb.pwr',
    'turb.PRmap',
    'turb.effMap',
    'turb.NpMap',
    'turb.WpMap',
    'turb.s_WpDes',
    'turb.s_PRdes',
    'turb.s_effDes',
    'turb.s_NpDes']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TurbineODTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf
        cycle.options['design'] = False

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('burner', Combustor(fuel_type="JP-7"))
        cycle.add_subsystem('turbine', Turbine( map_data=LPT2269))

        cycle.set_input_defaults('burner.Fl_I:FAR', .01, units=None)
        cycle.set_input_defaults('turbine.Nmech', 1000., units='rpm'),
        cycle.set_input_defaults('flow_start.P', 17., units='psi'),
        cycle.set_input_defaults('flow_start.T', 500.0, units='degR'),
        cycle.set_input_defaults('flow_start.W', 0., units='lbm/s'),
        cycle.set_input_defaults('turbine.area', 150., units='inch**2')

        cycle.pyc_connect_flow("flow_start.Fl_O", "burner.Fl_I")
        cycle.pyc_connect_flow("burner.Fl_O", "turbine.Fl_I")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):

        # 6 cases to check against
        for i, data in enumerate(ref_data):
            # input turbine variables
            self.prob['turbine.s_Wp'] = data[h_map['turb.s_WpDes']]
            self.prob['turbine.s_eff'] = data[h_map['turb.s_effDes']]
            self.prob['turbine.s_PR'] = data[h_map['turb.s_PRdes']]
            self.prob['turbine.s_Np'] = data[h_map['turb.s_NpDes']]

            self.prob['turbine.map.NpMap']= data[h_map['turb.NpMap']]
            self.prob['turbine.map.PRmap']= data[h_map['turb.PRmap']]

            # input flowstation variables
            self.prob['flow_start.P'] = data[h_map['burn.Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['burn.Fl_I.Tt']]
            self.prob['flow_start.W'] = data[h_map['burn.Fl_I.W']]
            self.prob['turbine.PR'] = data[h_map['turb.PR']]

            # input shaft variable
            self.prob['turbine.Nmech'] = data[h_map['shaft.Nmech']]

            # input burner variable
            self.prob['burner.Fl_I:FAR'] = data[h_map['burn.FAR']]

            self.prob['turbine.area'] = data[h_map['turb.Fl_O.A']]

            self.prob.run_model()

            print('---- Test Case', i, ' ----')

            print("corrParams --")
            print("Wp", self.prob['turbine.Wp'][0], data[h_map['turb.Wp']])
            print("Np", self.prob['turbine.Np'][0], data[h_map['turb.Np']])

            print("flowConv---")
            print("PR ", self.prob['turbine.PR'][0], data[h_map['turb.PR']])

            print("mapInputs---")
            print("NpMap", self.prob['turbine.map.readMap.NpMap'][0], data[h_map['turb.NpMap']])
            print("PRmap", self.prob['turbine.map.readMap.PRmap'][0], data[h_map['turb.PRmap']])

            print("readMap --")
            print(
                "effMap",
                self.prob['turbine.map.scaledOutput.effMap'][0],
                data[
                    h_map['turb.effMap']])
            print(
                "WpMap",
                self.prob['turbine.map.scaledOutput.WpMap'][0],
                data[
                    h_map['turb.WpMap']])

            print("Scaled output --")
            print("eff", self.prob['turbine.eff'][0], data[h_map['turb.eff']])

            tol = 1.0e-3

            print()
            npss = data[h_map['burn.Fl_I.Pt']]
            pyc = self.prob['flow_start.Fl_O:tot:P'][0]
            print('Pt in:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['burn.Fl_I.s']]
            pyc = self.prob['flow_start.Fl_O:tot:S'][0]
            print('S in:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_O.W']]
            pyc = self.prob['turbine.Fl_O:stat:W'][0]
            print('W in:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_O.ht']] - data[h_map['turb.Fl_I.ht']]
            pyc = self.prob['turbine.Fl_O:tot:h'][0] - self.prob['burner.Fl_O:tot:h'][0]
            print('delta h:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_I.s']]
            pyc = self.prob['burner.Fl_O:tot:S'][0]
            print('S in:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_O.s']]
            pyc = self.prob['turbine.Fl_O:tot:S'][0]
            print('S out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.pwr']]
            pyc = self.prob['turbine.power'][0]
            print('Power:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_O.Pt']]
            pyc = self.prob['turbine.Fl_O:tot:P'][0]
            print('Pt out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            # these fail downstream of combustor
            npss = data[h_map['turb.Fl_O.Ps']]
            pyc = self.prob['turbine.Fl_O:stat:P'][0]
            print('Ps out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)

            npss = data[h_map['turb.Fl_O.Ts']]
            pyc = self.prob['turbine.Fl_O:stat:T'][0]
            print('Ts out:', npss, pyc)
            assert_near_equal(pyc, npss, tol)
            print("")

            print()


if __name__ == "__main__":
    unittest.main()
