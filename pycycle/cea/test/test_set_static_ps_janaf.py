from __future__ import print_function

import os.path
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error
from openmdao.utils.units import convert_units as cu

from pycycle.cea.species_data import janaf
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic

fpath = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(fpath, 'NPSS_Static_CEA_Data.csv')
ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
          'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TestSetStaticPs(unittest.TestCase):

    def test_case_Ps(self):

        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('T', val=518., units='degR')
        indeps.add_output('P', val=14.7, units='psi')
        indeps.add_output('Ps', val=13.0, units='psi')
        indeps.add_output('W', val=1., units='lbm/s')

        p.model.add_subsystem('set_total_TP', SetTotal(thermo_data=janaf))
        p.model.add_subsystem('set_static_Ps', SetStatic(mode='Ps', thermo_data=janaf))

        p.model.connect('set_total_TP.flow:S', 'set_static_Ps.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_Ps.ht')
        p.model.connect('T', 'set_total_TP.T')
        p.model.connect('P', 'set_total_TP.P')
        p.model.connect('Ps', 'set_static_Ps.Ps')
        p.model.connect('W', 'set_static_Ps.W')

        p.setup(check=False)
        p.set_solver_print(level=-1)

        # from openmdao.api import view_model
        # view_model(p)
        # exit()
        # 4 cases to check against
        for i, data in enumerate(ref_data):

            p['T'] = data[h_map['Tt']]
            p['P'] = data[h_map['Pt']]

            p['Ps'] = data[h_map['Ps']]
            p['W'] = data[h_map['W']]

            p.run_model()

            # check outputs
            npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
            Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
                [data[h_map[v_name]] for v_name in npss_vars])

            Ps_computed = p['set_static_Ps.flow:P']
            Ts_computed = p['set_static_Ps.flow:T']
            hs_computed = p['set_static_Ps.flow:h']
            rhos_computed = p['set_static_Ps.flow:rho']
            gams_computed = p['set_static_Ps.flow:gamma']
            V_computed = p['set_static_Ps.flow:V']
            A_computed = p['set_static_Ps.flow:area']
            MN_computed = p['set_static_Ps.flow:MN']

            if MN >= .05:
                tol = 3e-4
            else:
                tol = .2
                # MN values off for low MN cases don't match well, but NPSS doesn't solve well down there
            #
            # print(p['T'], p['P'])
            # print("Ps", Ps_computed, Ps)
            # print("Ts", Ts_computed, Ts)
            # print("hs", hs_computed, hs)
            # print("gamma", gams_computed, gams)
            # print("V", V_computed, V)
            # print("A", A_computed, A)
            # print("MN", MN_computed, MN)
            # print("rhos", rhos_computed, rhos)
            # print()

            assert_rel_error(self, MN_computed, MN, tol)
            assert_rel_error(self, gams_computed, gams, tol)
            assert_rel_error(self, Ps_computed, Ps, tol)
            assert_rel_error(self, Ts_computed, Ts, tol)
            assert_rel_error(self, hs_computed, hs, tol)
            assert_rel_error(self, rhos_computed, rhos, tol)
            assert_rel_error(self, V_computed, V, tol)
            assert_rel_error(self, A_computed, A, tol)

        p.check_partials(includes=['set_static_Ps.statics.ps_calc'], compact_print=True)


if __name__ == "__main__":
    np.seterr(all='raise')
    unittest.main()
