import numpy as np
import unittest
import os

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.species_data import janaf
from pycycle.elements.cfd_start import CFDStart
from pycycle.elements.flight_conditions import FlightConditions
from pycycle.constants import AIR_MIX


class FlowStartTestCase(unittest.TestCase):

    def test_case_basic(self):

        p = Problem()

        params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
        params.add_output('Ps', units='Pa', val=22845.15677648)
        params.add_output('V', units='m/s', val=158.83851913)
        params.add_output('area', units='m**2', val=0.87451328)
        params.add_output('W', units='kg/s', val=50.2454107)

        p.model.add_subsystem('cfd_start', CFDStart(), promotes_inputs=['Ps', 'V', 'area', 'W'])
        p.set_solver_print(level=-1)
        # p.set_solver_print(level=2, depth=1)
        p.setup(check=False)

        p.run_model()

        tol = 1e-4
        assert_rel_error(self, p['cfd_start.balance.P'], 4.02374906, tol)
        assert_rel_error(self, p['cfd_start.balance.MN'], 0.53395824, tol)
        assert_rel_error(self, p['cfd_start.balance.T'], 418.68258747, tol)

    def test_case_sanity_check(self):

        p = Problem()

        params = p.model.add_subsystem('params', IndepVarComp(), promotes=['*'])
        params.add_output('MN', val=0.8)
        params.add_output('alt', val=35000., units='ft')
        params.add_output('W', val=15., units='lbm/s')

        p.model.add_subsystem('fc', FlightConditions(), promotes_inputs=['MN', 'alt', ('fs.W', 'W')])

        p.model.add_subsystem('cfd_start', CFDStart()) #, promotes_inputs=['Ps', 'V', 'area', 'W'])
        p.model.connect('fc.Fl_O:stat:P', 'cfd_start.Ps')
        p.model.connect('fc.Fl_O:stat:area', 'cfd_start.area')
        p.model.connect('fc.Fl_O:stat:V', 'cfd_start.V')
        p.model.connect('fc.Fl_O:stat:W', 'cfd_start.W')
        p.set_solver_print(level=-1)
        # p.set_solver_print(level=2, depth=1)
        p.setup(check=False)
        p.run_model()

        tol = 1e-6
        assert_rel_error(self, p['cfd_start.Fl_O:tot:P'], p['fc.Fl_O:tot:P'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:T'], p['fc.Fl_O:tot:T'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:h'], p['fc.Fl_O:tot:h'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:S'], p['fc.Fl_O:tot:S'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:gamma'], p['fc.Fl_O:tot:gamma'], tol)


        p['MN']=1.2
        p['alt'] = 50000.
        p['W'] = 125.

        p.run_model()

        tol = 1e-6
        assert_rel_error(self, p['cfd_start.Fl_O:tot:P'], p['fc.Fl_O:tot:P'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:T'], p['fc.Fl_O:tot:T'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:h'], p['fc.Fl_O:tot:h'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:S'], p['fc.Fl_O:tot:S'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:gamma'], p['fc.Fl_O:tot:gamma'], tol)


        p['MN']= 0.25
        p['alt'] = 100.
        p['W'] = 12.

        # needs some initial guesses or it won't converge
        p['cfd_start.balance.P'] = 14
        p['cfd_start.balance.T'] = 300
        p['cfd_start.balance.MN'] = .2

        p.run_model()

        tol = 5e-3
        assert_rel_error(self, p['cfd_start.Fl_O:tot:P'], p['fc.Fl_O:tot:P'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:T'], p['fc.Fl_O:tot:T'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:h'], p['fc.Fl_O:tot:h'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:S'], p['fc.Fl_O:tot:S'], tol)
        assert_rel_error(self, p['cfd_start.Fl_O:tot:gamma'], p['fc.Fl_O:tot:gamma'], tol)




if __name__ == "__main__":
    unittest.main()
