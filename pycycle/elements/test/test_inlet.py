""" Tests the inlet component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.mp_cycle import Cycle
from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.inlet import Inlet, MilSpecRecovery
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_JETA_TAB_SPEC, TAB_AIR_FUEL_COMPOSITION


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/inlet.csv", delimiter=",", skiprows=1)

header = [
    'eRamBase',
    'Fl_I.W',
    'Fl_I.V',
    'Fl_I.MN',
    'Fl_I.s',
    'Fl_I.Pt',
    'Fl_I.Tt',
    'Fl_I.ht',
    'Fl_I.rhot',
    'Fl_I.gamt',
    'Fl_O.MN',
    'Fl_O.s',
    'Fl_O.Pt',
    'Fl_O.Tt',
    'Fl_O.ht',
    'Fl_O.rhot',
    'Fl_O.gamt',
    'Fram',
    'Fl_O.Ps',
    'Fl_O.Ts',
    'Fl_O.hs',
    'Fl_O.rhos',
    'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))

class MilSpecTestCase(unittest.TestCase):
    
    def test(self):
        
        prob = Problem()
        
        prob.model.add_subsystem('mil_spec', MilSpecRecovery(), promotes=['*'])
        
        prob.setup(check=False, force_alloc_complex=True)
        
        prob.set_val('MN', 0.8)
        prob.set_val('ram_recovery_base', 0.9)
        
        prob.run_model()
        
        tol = 1e-4
        assert_near_equal(prob['ram_recovery'], 0.9, tol)
        
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)
        
        prob.set_val('MN', 2.0)
        
        prob.run_model()
        assert_near_equal(prob['ram_recovery'], 0.8325, tol)
        
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)        
        
        prob.set_val('MN', 6.0)
        
        prob.run_model()
        assert_near_equal(prob['ram_recovery'], 0.35858, tol)
        
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)
        

class InletTestCase(unittest.TestCase):

    def setUp(self):

        self.prob = Problem()
        cycle = self.prob.model = Cycle()
        cycle.options['thermo_method'] = 'CEA'
        cycle.options['thermo_data'] = janaf

        # cycle.options['thermo_method'] = 'TABULAR'
        # cycle.options['thermo_data'] = AIR_JETA_TAB_SPEC

        cycle.set_input_defaults('flow_start.P', 17, units='psi')
        cycle.set_input_defaults('flow_start.T', 500.0, units='degR')
        cycle.set_input_defaults('inlet.MN', 0.5)
        cycle.set_input_defaults('inlet.Fl_I:stat:V', 1., units='ft/s')
        cycle.set_input_defaults('flow_start.W', 1., units='lbm/s')

        cycle.add_subsystem('flow_start', FlowStart())
        cycle.add_subsystem('inlet', Inlet())

        # total and static
        fl_src = "flow_start.Fl_O"
        fl_target = "inlet.Fl_I"
        cycle.pyc_connect_flow("flow_start.Fl_O", "inlet.Fl_I")

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        # 4 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['inlet.ram_recovery'] = data[h_map['eRamBase']]

            # input flowstation
            self.prob['flow_start.P'] = data[h_map['Fl_I.Pt']]
            self.prob['flow_start.T'] = data[h_map['Fl_I.Tt']]
            self.prob['inlet.MN'] = data[h_map['Fl_O.MN']]
            self.prob['flow_start.MN'] = data[h_map['Fl_I.MN']]
            self.prob['flow_start.W'] = data[h_map['Fl_I.W']]
            self.prob['inlet.Fl_I:stat:V'] = data[h_map['Fl_I.V']]

            self.prob.run_model()

            # check outputs

            pt, ht, Fram, ps, ts = data[h_map['Fl_O.Pt']], data[h_map['Fl_O.ht']], data[
                h_map['Fram']], data[h_map['Fl_O.Ps']], data[h_map['Fl_O.Ts']]
            pt_computed = self.prob['inlet.Fl_O:tot:P']
            ht_computed = self.prob['inlet.Fl_O:tot:h']
            Fram_computed = self.prob['inlet.F_ram']
            ps_computed = self.prob['inlet.Fl_O:stat:P']
            ts_computed = self.prob['inlet.Fl_O:stat:T']

            tol = 1e-4
            assert_near_equal(pt_computed, pt, tol)
            assert_near_equal(ht_computed, ht, tol)
            assert_near_equal(Fram_computed, Fram, tol)
            assert_near_equal(ps_computed, ps, tol)
            assert_near_equal(ts_computed, ts, tol)

            partial_data = self.prob.check_partials(out_stream=None, method='cs', includes=['inlet.*'], excludes=['*.base_thermo.*'])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)

if __name__ == "__main__":
    unittest.main()
