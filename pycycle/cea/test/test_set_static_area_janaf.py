"""
Test for the SetStaticArea component.
"""
import os.path
import unittest

import numpy as np

from openmdao.api import Problem, IndepVarComp
from openmdao.utils.assert_utils import assert_rel_error

from pycycle.cea.species_data import janaf
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic

fpath = os.path.dirname(os.path.realpath(__file__))
data_path = os.path.join(fpath, 'NPSS_Static_CEA_Data.csv')
ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
          'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


class TestSetStaticArea(unittest.TestCase):

    def test_case_Area(self):

        p = Problem()

        indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
        indeps.add_output('T', val=518., units='degR')
        indeps.add_output('P', val=14.7, units='psi')
        indeps.add_output('W', val=1.5, units='lbm/s')
        indeps.add_output('area', val=np.inf, units='inch**2')

        p.model.add_subsystem('set_total_TP', SetTotal(thermo_data=janaf))
        p.model.add_subsystem('set_static_A', SetStatic(mode='area', thermo_data=janaf))

        p.model.connect('T', 'set_total_TP.T')
        p.model.connect('P', ['set_total_TP.P', 'set_static_A.guess:Pt'])

        p.model.connect('set_total_TP.flow:S', 'set_static_A.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_A.ht')
        # p.model.connect('P', 'set_static_A.P')
        p.model.connect('W', 'set_static_A.W')
        p.model.connect('area', 'set_static_A.area')
        p.model.connect('set_total_TP.flow:gamma', 'set_static_A.guess:gamt')
        #p.model.connect('set_total_TP.flow:n', 'set_static_A.n_guess')

        p.set_solver_print(level=-1)
        p.setup(check=False)

        # from openmdao.api import view_model
        # view_model(p)
        # exit(0)

        # 4 cases to check against
        for i, data in enumerate(ref_data):

            p['T'] = data[h_map['Tt']]
            p['P'] = data[h_map['Pt']]

            p['area'] = data[h_map['A']]
            # p['set_static_A.Ps'] = data[h_map['Ps']]
            p['W'] = data[h_map['W']]

            if i is 5:  # supersonic case
                p['set_static_A.guess:MN'] = 3.

            # print("###################################")
            # print(p['T'], p['P'], p['area'], p['W'])
            # print("###################################")
            p.run_model()

            # check outputs
            npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
            Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
                [data[h_map[v_name]] for v_name in npss_vars])

            Ps_computed = p['set_static_A.flow:P']
            Ts_computed = p['set_static_A.flow:T']
            hs_computed = p['set_static_A.flow:h']
            rhos_computed = p['set_static_A.flow:rho']
            gams_computed = p['set_static_A.flow:gamma']
            V_computed = p['set_static_A.flow:V']
            A_computed = p['area']
            MN_computed = p['set_static_A.MN']

            # I think these area already converted in the file: Ken
            # V_SI = cu(V, 'ft/s', 'm/s')
            # A_SI = cu(A, 'inch**2', 'm**2')

            # print(p['T'], p['P'])
            # print("Ps", Ps_computed, Ps)
            # print("Ts", Ts_computed, Ts)
            # print("gamma", gams_computed, gams)
            # print("V", V_computed, V)
            # print("A", A_computed, A)
            # print("MN", MN_computed, MN)
            # print("rhos", rhos_computed, rhos)
            # print()
            tol = 1.0e-4
            assert_rel_error(self, gams_computed, gams, tol)
            assert_rel_error(self, MN_computed, MN, tol)
            assert_rel_error(self, Ps_computed, Ps, tol)
            assert_rel_error(self, Ts_computed, Ts, tol)
            assert_rel_error(self, hs_computed, hs, tol)
            assert_rel_error(self, rhos_computed, rhos, tol)
            assert_rel_error(self, gams_computed, gams, tol)
            assert_rel_error(self, V_computed, V, tol)
            assert_rel_error(self, A_computed, A, tol)


if __name__ == "__main__":
    import scipy

    np.seterr(all='raise')
    scipy.seterr(all='raise')
    unittest.main()
