""" Some additional tests for CV-CD nozzles. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea.species_data import janaf, Thermo
from pycycle.connect_flow import connect_flow
from pycycle.constants import AIR_MIX
from pycycle.elements.flow_start import FlowStart
from pycycle.elements.nozzle import Nozzle

from pycycle.elements.test.util import check_element_partials


class NozzleTestCase(unittest.TestCase):

    def setup_helper(self, NozzType, LossType):

        thermo = Thermo(janaf, AIR_MIX)

        self.prob = Problem()
        self.prob.model = Group()

        # Remaining des_vars will be removed when set_input_defaults is fixed
        des_vars = self.prob.model.add_subsystem('des_vars', IndepVarComp(), promotes=['*'])
        des_vars.add_output('W', 0.0, units='lbm/s')

        self.prob.model.add_subsystem('flow_start', FlowStart(thermo_data=janaf, elements=AIR_MIX))
        self.prob.model.add_subsystem('nozzle', Nozzle(nozzType=NozzType, lossCoef=LossType,
                                                       thermo_data=janaf, elements=AIR_MIX,
                                                       internal_solver=True))


        self.prob.model.set_input_defaults('flow_start.P', 17.0, units='psi')
        self.prob.model.set_input_defaults('flow_start.T', 500.0, units='degR')
        self.prob.model.set_input_defaults('flow_start.MN', 0.2)
        self.prob.model.set_input_defaults('nozzle.Ps_exhaust', 17.0, units='psi')

        connect_flow(self.prob.model, "flow_start.Fl_O", "nozzle.Fl_I")

        self.prob.model.connect("W", "flow_start.W")

        if LossType == 'Cv':
            self.prob.model.set_input_defaults('nozzle.Cv', 0.99)
        elif LossType == 'Cfg':
            self.prob.model.set_input_defaults('nozzle.Cfg', 0.99)

        self.prob.set_solver_print(level=2)
        self.prob.setup(check=False)

        header = [
            'Cfg',
            'Cv',
            'PsExh',
            'Fl_I.W',
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
            'Fl_O.Ps',
            'Fl_Th.MN',
            'Fl_Th.s',
            'Fl_Th.Pt',
            'Fl_Th.Tt',
            'Fl_Th.ht',
            'Fl_Th.rhot',
            'Fl_Th.gamt',
            'Fl_Th.Ps',
            'Fl_Th.Aphy',
            'Fg',
            'FgIdeal',
            'Vactual',
            'AthCold',
            'AR',
            'PR']

        self.h_map = dict(((v_name, i) for i, v_name in enumerate(header)))
        self.fpath = os.path.dirname(os.path.realpath(__file__))

    def err_helper(self, name, npss, pyc):
        tol = 5e-4
        rel_err = abs(npss - pyc) / npss
        print(name, npss, pyc, rel_err)
        assert_near_equal(npss, pyc, tol)

    def run_helper(self, LossType):
        for i, data in enumerate(self.ref_data):

            self.prob['flow_start.T'] = data[self.h_map['Fl_I.Tt']]
            self.prob['flow_start.P'] = data[self.h_map['Fl_I.Pt']]
            self.prob['W'] = data[self.h_map['Fl_I.W']]
            self.prob['flow_start.MN'] = data[self.h_map['Fl_I.MN']]
            self.prob['nozzle.Ps_exhaust'] = data[self.h_map['PsExh']]

            if LossType == 'Cv':
                self.prob['nozzle.Cv'] = data[self.h_map['Cv']]
            elif LossType == 'Cfg':
                self.prob['nozzle.Cfg'] = data[self.h_map['Cfg']]

            self.prob.run_model()

            print()
            self.err_helper('Fg', data[self.h_map['Fg']], self.prob['nozzle.Fg'])
            self.err_helper('FgIdeal', data[self.h_map['FgIdeal']],
                            self.prob['nozzle.perf_calcs.Fg_ideal'])
            self.err_helper('AthCold', data[self.h_map['AthCold']],
                            self.prob['nozzle.Throat:stat:area'])
            self.err_helper('PR', data[self.h_map['PR']], self.prob['nozzle.PR'])
            self.err_helper('Fl_Th.MN',
                            data[self.h_map['Fl_Th.MN']],
                            self.prob['nozzle.Throat:stat:MN'])
            self.err_helper('Fl_O.MN', data[self.h_map['Fl_O.MN']],
                            self.prob['nozzle.Fl_O:stat:MN'])

    def test_CVnozzle(self):

        print('Testing CV Nozzle')
        self.setup_helper(NozzType='CV', LossType='Cv')
        self.ref_data = np.loadtxt(self.fpath + "/reg_data/nozzleCV.csv", delimiter=",", skiprows=1)
        self.run_helper(LossType='Cv')

    def test_CDnozzle(self):

        print('Testing CD Nozzle')
        self.setup_helper(NozzType='CD', LossType='Cv')
        self.ref_data = np.loadtxt(self.fpath + "/reg_data/nozzleCD.csv", delimiter=",", skiprows=1)
        self.run_helper(LossType='Cv')
        check_element_partials(self, self.prob)

    def test_CD_CVnozzle(self):

        print('Testing CD_CV Nozzle')
        self.setup_helper(NozzType='CD_CV', LossType='Cv')
        self.ref_data = np.loadtxt(
            self.fpath +
            "/reg_data/nozzleCD_CV.csv",
            delimiter=",",
            skiprows=1)
        self.run_helper(LossType='Cv')
        check_element_partials(self, self.prob)

    def test_CVnozzle_Cfg(self):

        print('Testing CV Nozzle with Cfg')
        self.setup_helper(NozzType='CV', LossType='Cfg')
        self.ref_data = np.loadtxt(
            self.fpath +
            "/reg_data/nozzleCV__Cfg.csv",
            delimiter=",",
            skiprows=1)
        self.run_helper(LossType='Cfg')

    def test_CDnozzle_Cfg(self):

        print('Testing CD Nozzle with Cfg')
        self.setup_helper(NozzType='CD', LossType='Cfg')
        self.ref_data = np.loadtxt(
            self.fpath +
            "/reg_data/nozzleCD__Cfg.csv",
            delimiter=",",
            skiprows=1)
        self.run_helper(LossType='Cfg')

    def test_CD_CVnozzle_Cfg(self):

        print('Testing CD_CV Nozzle with Cfg')
        self.setup_helper(NozzType='CD_CV', LossType='Cfg')
        self.ref_data = np.loadtxt(
            self.fpath +
            "/reg_data/nozzleCD_CV__Cfg.csv",
            delimiter=",",
            skiprows=1)
        self.run_helper(LossType='Cfg')
        check_element_partials(self, self.prob)

if __name__ == "__main__":
    unittest.main()
