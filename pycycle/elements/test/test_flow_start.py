import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea import species_data
from pycycle.elements.flow_start import FlowStart, SetWAR
from pycycle.constants import AIR_MIX, WET_AIR_MIX


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/flowstart.csv",
                      delimiter=",", skiprows=1)

header = [
    'W',
    'MN',
    'V',
    'A',
    's',
    'Pt',
    'Tt',
    'ht',
    'rhot',
    'gamt',
    'Ps',
    'Ts',
    'hs',
    'rhos',
    'gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class FlowStartTestCase(unittest.TestCase):
    
    def test_case1(self):

        self.prob = Problem()
        self.prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        self.prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        self.prob.model.set_input_defaults('fl_start.MN', 0.5)
        self.prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')

        self.prob.model.add_subsystem('fl_start', FlowStart(thermo_data=species_data.janaf, elements=AIR_MIX))

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['fl_start.P'] = data[h_map['Pt']]
            self.prob['fl_start.T'] = data[h_map['Tt']]
            self.prob['fl_start.W'] = data[h_map['W']]
            self.prob['fl_start.MN'] = data[h_map['MN']]

            self.prob.run_model()

            # check outputs
            tol = 1.0e-3

            if data[h_map[
                    'MN']] >= 2.:  # The Mach 2.0 case is at a ridiculously low temperature, so accuracy is questionable
                tol = 5e-2

            print(
                'Case: ', data[
                    h_map['Pt']], data[
                    h_map['Tt']], data[
                    h_map['W']], data[
                    h_map['MN']])
            npss = data[h_map['Pt']]
            pyc = self.prob['fl_start.Fl_O:tot:P']
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['Tt']]
            pyc = self.prob['fl_start.Fl_O:tot:T']
            rel_err = abs(npss - pyc) / npss
            print('Tt:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['W']]
            pyc = self.prob['fl_start.Fl_O:stat:W']
            rel_err = abs(npss - pyc) / npss
            print('W:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['ht']]
            pyc = self.prob['fl_start.Fl_O:tot:h']
            rel_err = abs(npss - pyc) / npss
            print('ht:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['s']]
            pyc = self.prob['fl_start.Fl_O:tot:S']
            rel_err = abs(npss - pyc) / npss
            print('S:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['rhot']]
            pyc = self.prob['fl_start.Fl_O:tot:rho']
            rel_err = abs(npss - pyc) / npss
            print('rhot:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['gamt']]
            pyc = self.prob['fl_start.Fl_O:tot:gamma']
            rel_err = abs(npss - pyc) / npss
            print('gamt:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['MN']]
            pyc = self.prob['fl_start.Fl_O:stat:MN']
            rel_err = abs(npss - pyc) / npss
            print('MN:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['Ps']]
            pyc = self.prob['fl_start.Fl_O:stat:P']
            rel_err = abs(npss - pyc) / npss
            print('Ps:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['Ts']]
            pyc = self.prob['fl_start.Fl_O:stat:T']
            rel_err = abs(npss - pyc) / npss
            print('Ts:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['hs']]
            pyc = self.prob['fl_start.Fl_O:stat:h']
            rel_err = abs(npss - pyc) / npss
            print('hs:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['rhos']]
            pyc = self.prob['fl_start.Fl_O:stat:rho']
            rel_err = abs(npss - pyc) / npss
            print('rhos:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['gams']]
            pyc = self.prob['fl_start.Fl_O:stat:gamma']
            rel_err = abs(npss - pyc) / npss
            print('gams:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['V']]
            pyc = self.prob['fl_start.Fl_O:stat:V']
            rel_err = abs(npss - pyc) / npss
            print('V:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            npss = data[h_map['A']]
            pyc = self.prob['fl_start.Fl_O:stat:area']
            rel_err = abs(npss - pyc) / npss
            print('A:', npss, pyc, rel_err)
            assert_near_equal(pyc, npss, tol)
            print()

    def test_case2(self):

        with self.assertRaises(ValueError) as cm:

            p = Problem()
            p.model = FlowStart(elements=AIR_MIX, use_WAR=True, thermo_data=species_data.janaf)
            p.model.set_input_defaults('WAR', .01)
            p.setup()

        self.assertEqual(str(cm.exception), "The provided elements to FlightConditions do not contain H2O. In order to specify a nonzero WAR the elements must contain H2O.")


        with self.assertRaises(ValueError) as cm:

            prob = Problem()
            prob.model = FlowStart(elements=WET_AIR_MIX, use_WAR=False, thermo_data=species_data.janaf)
            prob.setup()

        self.assertEqual(str(cm.exception), "In order to provide elements containing H2O, a nonzero water to air ratio (WAR) must be specified. Please set the option use_WAR to True.")

class WARTestCase(unittest.TestCase):

    def test_war_vals(self):

        prob = Problem()

        prob.model.add_subsystem('war', SetWAR(thermo_data=species_data.wet_air, elements=WET_AIR_MIX), promotes=['*'])
        
        prob.model.set_input_defaults('WAR', .0001)

        prob.setup(force_alloc_complex=True)

        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['b0'][0], 3.23286926e-04, tol)
        assert_near_equal(prob['b0'][1], 1.10121227e-05, tol)
        assert_near_equal(prob['b0'][2], 1.11016903e-05, tol)
        assert_near_equal(prob['b0'][3], 5.39103820e-02, tol)
        assert_near_equal(prob['b0'][4], 1.44901169e-02, tol)

    def test_war_errors(self):

        with self.assertRaises(ValueError) as cm:

            prob = Problem()
            prob.model.add_subsystem('war', SetWAR(thermo_data=species_data.wet_air, elements=WET_AIR_MIX), promotes=['*'])
            prob.model.set_input_defaults('WAR', 0.0)
            
            prob.setup()
            prob.run_model()

        self.assertEqual(str(cm.exception), "'war' <class SetWAR>: Error calling compute(), You have turned on the use_WAR option in FlightConditions but you have set WAR to be zero.")


        with self.assertRaises(ValueError) as cm:

            p = Problem()
            p.model.add_subsystem('war', SetWAR(thermo_data=species_data.wet_air, elements=WET_AIR_MIX), promotes=['*'])
            p.model.set_input_defaults('WAR', 1.0)
            
            p.setup()
            p.run_model()

        self.assertEqual(str(cm.exception), "'war' <class SetWAR>: Error calling compute(), Cannot specify WAR to have a value of 1. This is a physical impossibility and creates a singularity.")

if __name__ == "__main__":
    unittest.main()
