import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf
from pycycle.elements.flight_conditions import FlightConditions
from pycycle.constants import AIR_MIX

from pycycle.elements.test.util import check_element_partials


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/ambient.csv",
                      delimiter=",", skiprows=1)

header = ['alt', 'MN', 'dTs', 'Pt', 'Ps', 'Tt', 'Ts']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class FlightConditionsTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
        des_vars.add_output('MN', 0.0)
        des_vars.add_output('alt', 0.0, units="ft")
        des_vars.add_output('dTs', 0.0, units='degR')

        self.prob.model.add_subsystem('fc', FlightConditions())

        self.prob.model.connect('MN', 'fc.MN')
        self.prob.model.connect('alt', 'fc.alt')
        self.prob.model.connect('dTs', 'fc.dTs')

        self.prob.setup(check=False)
        self.prob.set_solver_print(level=-1)

    def test_case1(self):

        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['alt'] = data[h_map['alt']]
            self.prob['MN'] = data[h_map['MN']]
            self.prob['dTs'] = data[h_map['dTs']]

            if self.prob['MN'] < 1e-10:
                self.prob['MN'] += 1e-6

            self.prob.run_model()

            # check outputs
            Pt = data[h_map['Pt']]
            Pt_c = self.prob['fc.Fl_O:tot:P']

            Ps = data[h_map['Ps']]
            Ps_c = self.prob['fc.Fl_O:stat:P']

            Tt = data[h_map['Tt']]
            Tt_c = self.prob['fc.Fl_O:tot:T']

            Ts = data[h_map['Ts']]
            Ts_c = self.prob['fc.Fl_O:stat:T']

            tol = 1e-4
            assert_near_equal(Pt_c, Pt, tol)
            assert_near_equal(Ps_c, Ps, tol)
            assert_near_equal(Tt_c, Tt, tol)
            assert_near_equal(Ps_c, Ps, tol)

            check_element_partials(self, self.prob, depth=3)

if __name__ == "__main__":
    unittest.main()
