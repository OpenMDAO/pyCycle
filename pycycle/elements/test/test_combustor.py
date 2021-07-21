""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea import species_data
from pycycle.elements.combustor import Combustor
from pycycle.elements.flow_start import FlowStart
from pycycle.mp_cycle import Cycle


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/combustorJP7.csv",
                      delimiter=",", skiprows=1)

header = ['Fl_I.W', 'Fl_I.Pt', 'Fl_I.Tt', 'Fl_I.ht', 'Fl_I.s', 'Fl_I.MN', 'FAR', 'eff', 'Fl_O.MN',
          'Fl_O.Pt', 'Fl_O.Tt', 'Fl_O.ht', 'Fl_O.s', 'Wfuel', 'Fl_O.Ps', 'Fl_O.Ts', 'Fl_O.hs',
          'Fl_O.rhos', 'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class BurnerTestCase(unittest.TestCase):

    def test_case1(self):

        prob = Problem()
        model = prob.model = Cycle()
        model.options['thermo_method'] = 'CEA'
        model.options['thermo_data'] = species_data.janaf

        model.add_subsystem('ivc', IndepVarComp('in_composition', [3.23319235e-04, 1.10132233e-05, 
                                                     5.39157698e-02, 1.44860137e-02]))

        model.add_subsystem('flow_start', FlowStart())
        model.add_subsystem('combustor', Combustor())

        model.pyc_connect_flow('flow_start.Fl_O', 'combustor.Fl_I')
        
        # model.set_input_defaults('Fl_I:tot:P', 100.0, units='lbf/inch**2')
        # model.set_input_defaults('Fl_I:tot:h', 100.0, units='Btu/lbm')
        # model.set_input_defaults('Fl_I:stat:W', 100.0, units='lbm/s')
        model.set_input_defaults('combustor.Fl_I:FAR', 0.0)
        model.set_input_defaults('combustor.MN', 0.5)


        # needed because composition is sized by connection
        # model.connect('ivc.in_composition', ['Fl_I:tot:composition', 'Fl_I:stat:composition', ])


        prob.set_solver_print(level=2)
        prob.setup(check=False, force_alloc_complex=True)

        # 6 cases to check against
        for i, data in enumerate(ref_data):

            # input flowstation
            # prob['Fl_I:tot:P'] = data[h_map['Fl_I.Pt']]
            # prob['Fl_I:tot:h'] = data[h_map['Fl_I.ht']]
            # prob['Fl_I:stat:W'] = data[h_map['Fl_I.W']]
            # prob['Fl_I:FAR'] = data[h_map['FAR']]
            prob.set_val('flow_start.P', data[h_map['Fl_I.Pt']], units='psi')
            prob.set_val('flow_start.T', data[h_map['Fl_I.Tt']], units='degR')
            prob.set_val('flow_start.W', data[h_map['Fl_I.W']], units='lbm/s')
            prob['combustor.Fl_I:FAR'] = data[h_map['FAR']]
            prob['combustor.MN'] = data[h_map['Fl_O.MN']]

            prob.run_model()

            # prob.model.combustor.mix_fuel.list_inputs(print_arrays=True)
            # prob.model.combustor.mix_fuel.list_outputs(print_arrays=True)
            # print(prob['combustor.Fl_I:tot:composition'])
            # print(prob['combustor.Fl_I:tot:n'])
            print(prob['combustor.Fl_I:tot:h'])
            print(prob['combustor.Fl_I:tot:P'])
            # exit()

            # check outputs
            tol = 1.0e-2

            npss = data[h_map['Fl_O.Pt']]
            pyc = prob['combustor.Fl_O:tot:P']
            rel_err = abs(npss - pyc) / npss
            print('Pt out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Tt']]
            pyc = prob['combustor.Fl_O:tot:T']
            rel_err = abs(npss - pyc) / npss
            print('Tt out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.ht']]
            pyc = prob['combustor.Fl_O:tot:h']
            rel_err = abs(npss - pyc) / npss
            print('ht out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Ps']]
            pyc = prob['combustor.Fl_O:stat:P']
            rel_err = abs(npss - pyc) / npss
            print('Ps out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Ts']]
            pyc = prob['combustor.Fl_O:stat:T']
            rel_err = abs(npss - pyc) / npss
            print('Ts out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Wfuel']]
            pyc = prob['combustor.Fl_I:stat:W'] * (prob['combustor.Fl_I:FAR'])
            rel_err = abs(npss - pyc) / npss
            print('Wfuel:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            print('')

            partial_data = prob.check_partials(out_stream=None, method='cs', 
                                               includes=['combustor.*',], excludes=['*.base_thermo.*', '*.mix_fuel.*'])
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
