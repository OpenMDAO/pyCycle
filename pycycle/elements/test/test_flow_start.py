import numpy as np
import unittest
import os

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.species_data import janaf
from pycycle.elements.flow_start import FlowStart
from pycycle.constants import AIR_MIX


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

    def setUp(self):

        self.prob = Problem()

        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp())
        des_vars.add_output('P', 17., units='psi')
        des_vars.add_output('T', 500., units='degR')
        des_vars.add_output('W', 100., units='lbm/s')
        des_vars.add_output('MN', 0.5)

        self.prob.model.add_subsystem('fl_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.connect('des_vars.P', 'fl_start.P')
        self.prob.model.connect('des_vars.T', 'fl_start.T')
        self.prob.model.connect('des_vars.W', 'fl_start.W')
        self.prob.model.connect('des_vars.MN', 'fl_start.MN')

        self.prob.set_solver_print(level=-1)
        self.prob.setup(check=False)

    def test_case1(self):
        np.seterr(divide='raise')
        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['des_vars.P'] = data[h_map['Pt']]
            self.prob['des_vars.T'] = data[h_map['Tt']]
            self.prob['des_vars.W'] = data[h_map['W']]
            self.prob['des_vars.MN'] = data[h_map['MN']]

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
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['Tt']]
            pyc = self.prob['fl_start.Fl_O:tot:T']
            rel_err = abs(npss - pyc) / npss
            print('Tt:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['W']]
            pyc = self.prob['fl_start.Fl_O:stat:W']
            rel_err = abs(npss - pyc) / npss
            print('W:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['ht']]
            pyc = self.prob['fl_start.Fl_O:tot:h']
            rel_err = abs(npss - pyc) / npss
            print('ht:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['s']]
            pyc = self.prob['fl_start.Fl_O:tot:S']
            rel_err = abs(npss - pyc) / npss
            print('S:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['rhot']]
            pyc = self.prob['fl_start.Fl_O:tot:rho']
            rel_err = abs(npss - pyc) / npss
            print('rhot:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['gamt']]
            pyc = self.prob['fl_start.Fl_O:tot:gamma']
            rel_err = abs(npss - pyc) / npss
            print('gamt:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['MN']]
            pyc = self.prob['fl_start.Fl_O:stat:MN']
            rel_err = abs(npss - pyc) / npss
            print('MN:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['Ps']]
            pyc = self.prob['fl_start.Fl_O:stat:P']
            rel_err = abs(npss - pyc) / npss
            print('Ps:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['Ts']]
            pyc = self.prob['fl_start.Fl_O:stat:T']
            rel_err = abs(npss - pyc) / npss
            print('Ts:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['hs']]
            pyc = self.prob['fl_start.Fl_O:stat:h']
            rel_err = abs(npss - pyc) / npss
            print('hs:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['rhos']]
            pyc = self.prob['fl_start.Fl_O:stat:rho']
            rel_err = abs(npss - pyc) / npss
            print('rhos:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['gams']]
            pyc = self.prob['fl_start.Fl_O:stat:gamma']
            rel_err = abs(npss - pyc) / npss
            print('gams:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['V']]
            pyc = self.prob['fl_start.Fl_O:stat:V']
            rel_err = abs(npss - pyc) / npss
            print('V:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            npss = data[h_map['A']]
            pyc = self.prob['fl_start.Fl_O:stat:area']
            rel_err = abs(npss - pyc) / npss
            print('A:', npss, pyc, rel_err)
            assert_rel_error(self, pyc, npss, tol)
            print()

if __name__ == "__main__":
    unittest.main()
