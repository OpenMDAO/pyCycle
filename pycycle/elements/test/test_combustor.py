""" Tests the duct component. """

import unittest
import os

import numpy as np

from openmdao.api import Problem, Group
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.constants import AIR_FUEL_ELEMENTS, AIR_ELEMENTS
from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea import species_data
from pycycle.elements.combustor import Combustor, MixFuel

from pycycle.elements.test.util import check_element_partials


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/combustorJP7.csv",
                      delimiter=",", skiprows=1)

header = ['Fl_I.W', 'Fl_I.Pt', 'Fl_I.Tt', 'Fl_I.ht', 'Fl_I.s', 'Fl_I.MN', 'FAR', 'eff', 'Fl_O.MN',
          'Fl_O.Pt', 'Fl_O.Tt', 'Fl_O.ht', 'Fl_O.s', 'Wfuel', 'Fl_O.Ps', 'Fl_O.Ts', 'Fl_O.hs',
          'Fl_O.rhos', 'Fl_O.gams']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class BurnerTestCase(unittest.TestCase):


    def test_mix_fuel(self): 

        thermo_spec = species_data.janaf

        air_thermo = species_data.Properties(thermo_spec, init_elements=AIR_ELEMENTS)

        p = Problem()

        fuel_type = 'JP-7'
        p.model = MixFuel(inflow_thermo_data=thermo_spec, thermo_data=thermo_spec,
                          inflow_elements=AIR_ELEMENTS, fuel_type=fuel_type)


        p.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:FAR'] = 0.02673
        p['Fl_I:tot:h'] = 181.381769
        p['fuel_Tt'] = 518.
        p['Fl_I:tot:b0'] = air_thermo.b0

        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
        assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
        assert_near_equal(p['b0_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

        data = p.check_partials(out_stream=None, method='cs')
        assert_check_partials(data, atol=1.e-6, rtol=1.e-6)


    def test_case1(self):

        prob = Problem()
        model = prob.model 

        n_init = np.array([3.23319258e-04, 1.00000000e-10, 1.10131241e-05, 1.00000000e-10,
                           1.63212420e-10, 6.18813039e-09, 1.00000000e-10, 2.69578835e-02,
                           1.00000000e-10, 7.23198770e-03])

        model.add_subsystem('combustor', Combustor(), promotes=["*"])
        model.set_input_defaults('Fl_I:tot:P', 100.0, units='lbf/inch**2')
        model.set_input_defaults('Fl_I:tot:h', 100.0, units='Btu/lbm')
        model.set_input_defaults('Fl_I:stat:W', 100.0, units='lbm/s')
        model.set_input_defaults('Fl_I:FAR', 0.0)
        model.set_input_defaults('MN', 0.5)
        model.set_input_defaults('Fl_I:tot:b0', val=[3.23319235e-04, 1.10132233e-05, 
                                                     5.39157698e-02, 1.44860137e-02])

        prob.set_solver_print(level=2)
        prob.setup(check=False)

        # 6 cases to check against
        for i, data in enumerate(ref_data):

            # input flowstation
            prob['Fl_I:tot:P'] = data[h_map['Fl_I.Pt']]
            prob['Fl_I:tot:h'] = data[h_map['Fl_I.ht']]
            prob['Fl_I:stat:W'] = data[h_map['Fl_I.W']]
            prob['Fl_I:FAR'] = data[h_map['FAR']]
            prob['MN'] = data[h_map['Fl_O.MN']]

            prob.run_model()

            prob.model.combustor.mix_fuel.list_inputs(print_arrays=True)
            prob.model.combustor.mix_fuel.list_outputs(print_arrays=True)
            # print(prob['Fl_I:tot:b0'])
            # print(prob['Fl_I:tot:n'])
            # print(prob['Fl_I:tot:h'])
            # print(prob['Fl_I:tot:P'])
            # exit()

            # check outputs
            tol = 1.0e-2

            npss = data[h_map['Fl_O.Pt']]
            pyc = prob['Fl_O:tot:P']
            rel_err = abs(npss - pyc) / npss
            print('Pt out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Tt']]
            pyc = prob['Fl_O:tot:T']
            rel_err = abs(npss - pyc) / npss
            print('Tt out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.ht']]
            pyc = prob['Fl_O:tot:h']
            rel_err = abs(npss - pyc) / npss
            print('ht out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Ps']]
            pyc = prob['Fl_O:stat:P']
            rel_err = abs(npss - pyc) / npss
            print('Ps out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Fl_O.Ts']]
            pyc = prob['Fl_O:stat:T']
            rel_err = abs(npss - pyc) / npss
            print('Ts out:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            npss = data[h_map['Wfuel']]
            pyc = prob['Fl_I:stat:W'] * (prob['Fl_I:FAR'])
            rel_err = abs(npss - pyc) / npss
            print('Wfuel:', npss, pyc, rel_err)
            self.assertLessEqual(rel_err, tol)

            print('')

            check_element_partials(self, prob, tol=1e-4)

if __name__ == "__main__":
    unittest.main()
