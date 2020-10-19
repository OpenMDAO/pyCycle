""" Tests the Nozzle component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.flow_start import FlowStart
from pycycle.elements.nozzle import Nozzle
from pycycle.constants import AIR_MIX

from pycycle.elements.test.util import check_element_partials

fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/nozzle.csv", delimiter=",", skiprows=1)

header = ['Cfg', 'PsExh', 'Fl_I.W', 'Fl_I.MN', 'Fl_I.s', 'Fl_I.Pt', 'Fl_I.Tt', 'Fl_I.ht',
          'Fl_I.rhot', 'Fl_I.gamt', 'Fl_O.MN', 'Fl_O.s', 'Fl_O.Pt', 'Fl_O.Tt', 'Fl_O.ht',
          'Fl_O.rhot', 'Fl_O.gamt', 'Fl_O.Ps', 'Fg', 'Vactual', 'Ath', 'AR', 'PR']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class NozzleTestCase(unittest.TestCase):

    def test_case1(self):

        self.prob = Problem()
        self.prob.model = Group()

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf,
                                                              elements=AIR_MIX))
        self.prob.model.add_subsystem('nozzle', Nozzle(elements=AIR_MIX, lossCoef='Cfg', internal_solver=True))

        self.prob.model.set_input_defaults('nozzle.Ps_exhaust', 10.0, units='lbf/inch**2')
        self.prob.model.set_input_defaults('flow_start.MN', 0.0)
        self.prob.model.set_input_defaults('flow_start.T', 500.0, units='degR')
        self.prob.model.set_input_defaults('flow_start.P', 17.0, units='psi')
        self.prob.model.set_input_defaults('flow_start.W', 100.0, units='lbm/s')

        fl_src = "flow_start.Fl_O"
        fl_target = "nozzle.Fl_I"
        for v_name in ('h', 'T', 'P', 'S', 'rho', 'gamma', 'Cp', 'Cv', 'n'):
            self.prob.model.connect('%s:tot:%s' % (
                fl_src, v_name), '%s:tot:%s' % (fl_target, v_name))

        # no prefix
        for v_name in ('W', ): 
            self.prob.model.connect(
                '%s:stat:%s' %
                (fl_src, v_name), '%s:stat:%s' %
                (fl_target, v_name))

        self.prob.setup(check=False)

        # 4 cases to check against
        for i, data in enumerate(ref_data):

            self.prob['nozzle.Cfg'] = data[h_map['Cfg']]
            self.prob['nozzle.Ps_exhaust'] = data[h_map['PsExh']]
            # input flowstation

            self.prob['flow_start.P'] = data[h_map['Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['Fl_I.Tt']]
            self.prob['flow_start.W'] = data[h_map['Fl_I.W']]
            self.prob['flow_start.MN'] = data[h_map['Fl_I.MN']]

            self.prob.run_model()

            # check outputs
            Fg, V, PR = data[h_map['Fg']], data[
                h_map['Vactual']], data[h_map['PR']]
            MN = data[h_map['Fl_O.MN']]
            Ath = data[h_map['Ath']]
            Pt = data[h_map['Fl_O.Pt']]
            MN_computed = self.prob['nozzle.Fl_O:stat:MN']
            Fg_computed = self.prob['nozzle.Fg']
            V_computed = self.prob['nozzle.Fl_O:stat:V']
            PR_computed = self.prob['nozzle.PR']
            Ath_computed = self.prob['nozzle.Fl_O:stat:area']
            Pt_computed = self.prob['nozzle.Fl_O:tot:P']

            # Used for all
            tol = 5.e-3

            assert_near_equal(MN_computed, MN, tol)

            assert_near_equal(Fg_computed, Fg, tol)
            assert_near_equal(V_computed, V, tol)
            assert_near_equal(Pt_computed, Pt, tol)

            assert_near_equal(PR_computed, PR, tol)
            assert_near_equal(Ath_computed, Ath, tol)

            check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
