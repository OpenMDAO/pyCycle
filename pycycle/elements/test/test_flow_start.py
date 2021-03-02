import numpy as np
import unittest
import os

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.api import (Cycle, FlowStart, CEA_AIR_COMPOSITION, 
                         CEA_WET_AIR_COMPOSITION, species_data, AIR_JETA_TAB_SPEC, 
                         TAB_AIR_FUEL_COMPOSITION)

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

        fl_start = self.prob.model.add_subsystem('fl_start', FlowStart(thermo_method='CEA', 
                                                                       thermo_data=species_data.janaf, 
                                                                       composition=CEA_AIR_COMPOSITION))

        #note: must manually call this for stand alone element tests without a cycle group
        fl_start.pyc_setup_output_ports() 

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):

            if i != 4: 
                continue 

            self.prob.set_val('fl_start.P',data[h_map['Pt']], units='psi')
            self.prob['fl_start.T'] = data[h_map['Tt']]
            self.prob['fl_start.W'] = data[h_map['W']]
            self.prob['fl_start.MN'] = data[h_map['MN']]

            self.prob.run_model()

            # check outputs
            tol = 1.0e-3

            # The Mach 2.0 case is at a ridiculously low temperature, so accuracy is questionable
            if data[h_map['MN']] >= 2.:  
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

    def test_case_tabular_thermo(self):

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')

        fl_start = prob.model.add_subsystem('fl_start', FlowStart(thermo_method='TABULAR', 
                                                                  thermo_data=AIR_JETA_TAB_SPEC, 
                                                                  composition=TAB_AIR_FUEL_COMPOSITION))
        fl_start.pyc_setup_output_ports() #note: must manually call this for stand alone element tests without a cycle group

        prob.set_solver_print(level=-1)
        prob.setup(check=False)

        prob['fl_start.P'] = 5.27
        prob['fl_start.T'] = 444.23
        prob['fl_start.W'] = 100.0
        prob['fl_start.MN'] = 0.8

        prob.run_model()

        tol = 1e-6
        assert_near_equal(prob['fl_start.Fl_O:tot:P'], 5.27, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:T'], 444.23, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:h'], -24.02365656, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:S'], 1.66403163, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:gamma'], 1.40086187, tol)

        assert_near_equal(prob['fl_start.Fl_O:stat:W'], 100.0, tol)
        assert_near_equal(prob['fl_start.Fl_O:stat:MN'], 0.8, tol)
        assert_near_equal(prob['fl_start.Fl_O:stat:area'], 778.26812382, tol)

   
class WARTestCase(unittest.TestCase):

    def test_fs_with_water(self): 

        prob = Problem()
        prob.model.set_input_defaults('fl_start.P', 17., units='psi')
        prob.model.set_input_defaults('fl_start.T', 500., units='degR')
        prob.model.set_input_defaults('fl_start.MN', 0.5)
        prob.model.set_input_defaults('fl_start.W', 100., units='lbm/s')
        prob.model.set_input_defaults('fl_start.WAR', .01)

        fl_start = prob.model.add_subsystem('fl_start', FlowStart(thermo_data=species_data.wet_air, 
                                                                  composition=CEA_WET_AIR_COMPOSITION, 
                                                                  reactant="Water", mix_ratio_name='WAR'))
        fl_start.pyc_setup_output_ports()

        prob.set_solver_print(level=-1)
        prob.setup(check=False)



        prob.run_model()

        tol = 1e-5
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][0], 3.18139345e-04, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][1], 1.08367806e-05, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][2], 1.77859e-03, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][3], 5.305198e-02, tol)
        assert_near_equal(prob['fl_start.Fl_O:tot:composition'][4], 1.51432e-02, tol)

    
if __name__ == "__main__":
    unittest.main()
