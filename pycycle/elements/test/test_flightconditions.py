import numpy as np
import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.thermo.cea.species_data import janaf
from pycycle.elements.flight_conditions import FlightConditions
from pycycle.mp_cycle import Cycle


fpath = os.path.dirname(os.path.realpath(__file__))
ref_data = np.loadtxt(fpath + "/reg_data/ambient.csv",
                      delimiter=",", skiprows=1)

header = ['alt', 'MN', 'dTs', 'Pt', 'Ps', 'Tt', 'Ts']

h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class FlightConditionsTestCase(unittest.TestCase):

    def setUp(self):
        self.prob = om.Problem()

        self.prob.model = Cycle()
        self.prob.model.options['thermo_method'] = 'CEA'
        self.prob.model.options['thermo_data'] = janaf

        self.prob.model.set_input_defaults('fc.MN', 0.0)
        self.prob.model.set_input_defaults('fc.alt', 0.0, units="ft")
        self.prob.model.set_input_defaults('fc.dTs', 0.0, units='degR')

        fc = self.prob.model.add_subsystem('fc', FlightConditions())

        self.prob.setup(check=False, force_alloc_complex=True)
        self.prob.set_solver_print(level=-1)

    def test_case1(self):

        # 6 cases to check against
        for i, data in enumerate(ref_data):
            self.prob['fc.alt'] = data[h_map['alt']]
            self.prob['fc.MN'] = data[h_map['MN']]
            self.prob['fc.dTs'] = data[h_map['dTs']]

            if self.prob['fc.MN'] < 1e-10:
                self.prob['fc.MN'] += 1e-6

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

            # NOTE: CHeck partials not needed, since no new components are here
            # partial_data = self.prob.check_partials(out_stream=None, method='cs', 
            #                                         includes=['fc.*'], excludes=['*.base_thermo.*', 'fc.ambient.readAtmTable', '*.exit_static*'])
            # assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


class ExternalAtmosphereTestCase(unittest.TestCase):

    def test_matches_builtin_atmosphere(self):
        """An external ambient state reproduces the built-in atmosphere, including a hot day"""
        prob = om.Problem()
        prob.model = Cycle()
        prob.model.options['thermo_method'] = 'CEA'
        prob.model.options['thermo_data'] = janaf

        prob.model.add_subsystem('fc', FlightConditions())
        prob.model.add_subsystem('fc_ext', FlightConditions(use_external_atmosphere=True))

        prob.model.connect('fc.ambient.Ts', 'fc_ext.ambient_static_temperature')
        prob.model.connect('fc.ambient.Ps', 'fc_ext.ambient_static_pressure')
        prob.model.connect('fc.ambient.rhos', 'fc_ext.ambient_density')

        prob.setup(check=False, force_alloc_complex=True)
        prob.set_solver_print(level=-1)

        prob.set_val('fc.W', 100., units='lbm/s')
        prob.set_val('fc_ext.W', 100., units='lbm/s')

        # standard day at cruise, and a +27 degR hot day takeoff
        for alt, MN, dTs in [(35000., 0.8, 0.), (0., 0.25, 27.)]:
            prob.set_val('fc.alt', alt, units='ft')
            prob.set_val('fc.dTs', dTs, units='degR')
            prob.set_val('fc.MN', MN)
            prob.set_val('fc_ext.MN', MN)

            prob.run_model()

            for var in ['tot:P', 'tot:T', 'stat:P', 'stat:T', 'stat:W', 'stat:V']:
                assert_near_equal(prob[f'fc_ext.Fl_O:{var}'], prob[f'fc.Fl_O:{var}'], 1e-8)

        assert_near_equal(prob.get_val('fc_ext.Fl_O:stat:T', units='degR'), 518.67 + 27., 1e-6)

        partial_data = prob.check_partials(out_stream=None, method='cs', includes=['fc_ext.ambient'])
        assert_check_partials(partial_data, atol=1e-10, rtol=1e-10)

    def test_no_altitude_inputs(self):
        """The external atmosphere replaces the alt and dTs inputs"""
        prob = om.Problem()
        prob.model = Cycle()
        prob.model.options['thermo_method'] = 'CEA'
        prob.model.options['thermo_data'] = janaf
        prob.model.add_subsystem('fc', FlightConditions(use_external_atmosphere=True))
        prob.setup(check=False)
        prob.final_setup()

        inputs = [meta['prom_name'] for _, meta in
                  prob.model.list_inputs(out_stream=None, prom_name=True)]
        self.assertIn('fc.ambient_static_temperature', inputs)
        self.assertIn('fc.ambient_static_pressure', inputs)
        self.assertIn('fc.ambient_density', inputs)
        self.assertNotIn('fc.alt', inputs)
        self.assertNotIn('fc.dTs', inputs)


if __name__ == "__main__":
    unittest.main()
