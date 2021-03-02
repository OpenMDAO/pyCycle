import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea import species_data
from pycycle import constants


class SetTotalSimpleTestCase(unittest.TestCase):

    def test_set_total_TP(self):
        p = om.Problem()
        p.model.add_subsystem('thermo', Thermo(mode='total_TP', 
                                               method='CEA', 
                                               thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                                              'spec': species_data.co2_co_o2 }), 
                              promotes=['*'])


        p.setup(check=False)
        p.set_solver_print(level=-1)
        p.final_setup()

        p.set_val('T', 4000, units='degK')
        p.set_val('P', 1.034210, units='bar')
        
        p.run_model()

        assert_near_equal(p['gamma'], 1.19054697, 1e-4)

        p.set_val('T', 1500, units='degK')
        p.set_val('P', 1.034210, units='bar')
        p.run_model()

        assert_near_equal(p['gamma'], 1.16379233, 1e-4)


    def test_set_total_hP(self):

        p = om.Problem()
        p.model = Thermo(mode='total_hP', 
                         method = 'CEA', 
                         thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                      'spec': species_data.co2_co_o2 }) 

        p.setup()
        # TODO: Investigate this weirdness.... this case won't work if you thermo_TP fully solve itself
        p.set_solver_print(level=-1)

        p.set_val('h', 340, units='cal/g')
        p.set_val('P', 1.034210, units='bar')

        p.run_model()

        assert_near_equal(p['gamma'], 1.19039688581, 1e-4)

        # 1500K
        p['h'] = -1801.35537381
        p.run_model()

        assert_near_equal(p['gamma'], 1.16379012007, 1e-4)

    def test_set_total_SP(self):

        #

        p = om.Problem()
        p.model = Thermo(mode='total_SP', 
                         method = 'CEA', 
                         thermo_kwargs={'composition': constants.CEA_CO2_CO_O2_COMPOSITION, 
                                        'spec': species_data.co2_co_o2 }) 
       
        r = p.model


        p.setup(check=False)

        # NOTE: This case is very touchy and requires weird solver settings
        p.model.nonlinear_solver.options['solve_subsystems'] = True
        p.model.base_thermo.chem_eq.nonlinear_solver.options['maxiter'] = 1

        p.model.base_thermo.chem_eq.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS()
        p.model.base_thermo.chem_eq.nonlinear_solver.linesearch.options['maxiter'] = 2
        
        p.set_solver_print(level=2)
        p.final_setup()

        # p.model.nonlinear_solver.options['maxiter'] = 0

        p.set_val('S', 2.35711010759, units="Btu/(lbm*degR)")
        p.set_val('P', 1.034210, units="bar")

        p.run_model()

        assert_near_equal(p['gamma'], 1.19054696779, 1e-4)

        # 1500K
        p['T'] = 4000. 

        p['S'] = 1.5852424435
        p.run_model()

        assert_near_equal(p['gamma'], 1.16396871, 1e-4)


class TestSetTotalJanaf(unittest.TestCase): 

    def test_set_total_equivalence(self): 

        p_TP = om.Problem()
        p_TP.model = Thermo(mode='total_TP', 
                            method='CEA', 
                            thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p_TP.setup()
        p_TP.set_solver_print(level=-1)

        p_hP = om.Problem()
        p_hP.model = Thermo(mode='total_hP', 
                            method='CEA', 
                            thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p_hP.setup()
        p_hP.set_solver_print(level=-1)

        p_SP = om.Problem()
        p_SP.model = Thermo(mode='total_SP', 
                            method='CEA', 
                            thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p_SP.setup()
        p_SP.set_solver_print(level=-1)


        def check(T, P): 

            p_TP.set_val('T', T, units='degR')
            p_TP.set_val('P', P, units='psi')

            # print('TP check')
            p_TP.run_model()
            h_from_TP = p_TP.get_val('flow:h', units='cal/g')
            S_from_TP = p_TP.get_val('flow:S', units='cal/(g*degK)')


            p_hP.set_val('h', h_from_TP, units='cal/g')
            p_hP.set_val('P', P, units='psi')

            # print('hp check')
            p_hP.run_model()

            assert_near_equal(p_hP['flow:T'], p_TP['flow:T'], 1e-4)

            p_SP.set_val('S', S_from_TP, units='cal/(g*degK)')
            p_SP.set_val('P', P, units='psi')

            # print('SP check')
            p_SP.run_model()

            assert_near_equal(p_SP['flow:T'], p_TP['flow:T'], 1e-4)

        check(518., 14.7)
        check(3000., 30.)
        check(1500., 80.)


class TestStaticJanaf(unittest.TestCase): 

    def setUp(self): 

        fpath = os.path.dirname(os.path.realpath(__file__))
        data_path = os.path.join(fpath, 'NPSS_Static_CEA_Data.csv')
        self.ref_data = np.loadtxt(data_path, delimiter=",", skiprows=1)

        header = ['W', 'MN', 'V', 'A', 's', 'Pt', 'Tt', 'ht', 'rhot',
                  'gamt', 'Ps', 'Ts', 'hs', 'rhos', 'gams']
        self.h_map = dict(((v_name, i) for i, v_name in enumerate(header)))


    def check_vals(self, group_name, npss_data):


        # check outputs
        npss_vars = ('Ps', 'Ts', 'MN', 'hs', 'rhos', 'gams', 'V', 'A', 's', 'ht')
        Ps, Ts, MN, hs, rhos, gams, V, A, S, ht = tuple(
            [npss_data[self.h_map[v_name]] for v_name in npss_vars])

        Ps_computed = self.p[f'{group_name}.flow:P']
        Ts_computed = self.p[f'{group_name}.flow:T']
        hs_computed = self.p[f'{group_name}.flow:h']
        rhos_computed = self.p[f'{group_name}.flow:rho']
        gams_computed = self.p[f'{group_name}.flow:gamma']
        V_computed = self.p[f'{group_name}.flow:V']
        A_computed = self.p[f'{group_name}.flow:area']
        MN_computed = self.p[f'{group_name}.flow:MN']

        if MN >= .05:
            tol = 3e-4
        else:
            tol = .2
            # MN values off for low MN cases don't match well, but NPSS doesn't solve well down there

        assert_near_equal(MN_computed, MN, tol)
        assert_near_equal(gams_computed, gams, tol)
        assert_near_equal(Ps_computed, Ps, tol)
        assert_near_equal(Ts_computed, Ts, tol)
        assert_near_equal(hs_computed, hs, tol)
        assert_near_equal(rhos_computed, rhos, tol)
        assert_near_equal(V_computed, V, tol)
        assert_near_equal(A_computed, A, tol)


    def test_case_Ps(self):

        p = self.p = om.Problem()
        total_TP =  Thermo(mode='total_TP', 
                           method='CEA', 
                           thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p.model.add_subsystem('set_total_TP', total_TP)

        static_Ps =  Thermo(mode='static_Ps', 
                            method='CEA', 
                            thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p.model.add_subsystem('set_static_Ps', static_Ps)
        p.model.connect('set_total_TP.flow:S', 'set_static_Ps.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_Ps.ht')
       

        p.setup()
        p.set_solver_print(level=0)
        p.final_setup()

        h_map = self.h_map
        # 6 cases to check against
        for i, data in enumerate(self.ref_data):

            if i == 4: # low mach number that case that seems to diagree because NPSS doesn't converge as tightly
                continue 
            # print(i, data[h_map['Tt']], data[h_map['Pt']], data[h_map['Ps']], data[h_map['W']], ';', data[h_map['MN']])

            p.set_val('set_total_TP.T', data[h_map['Tt']], units='degR')
            p.set_val('set_total_TP.P', data[h_map['Pt']], units='psi')

            p.set_val('set_static_Ps.Ps', data[h_map['Ps']], units='psi')
            p.set_val('set_static_Ps.W', data[h_map['W']], units='lbm/s')

            p.run_model()
            
            self.check_vals('set_static_Ps', data)


    def test_case_area(self):

        p = self.p = om.Problem()
        total_TP =  Thermo(mode='total_TP', 
                           method='CEA', 
                           thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf })  
        p.model.add_subsystem('set_total_TP', total_TP)

        static_A =  Thermo(mode='static_A', 
                           method='CEA', 
                           thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                          'spec': species_data.janaf }) 
        p.model.add_subsystem('set_static_A', static_A)
        
        p.model.connect('set_total_TP.flow:S', 'set_static_A.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_A.ht')
        p.model.connect('set_total_TP.flow:gamma', 'set_static_A.guess:gamt')
        p.model.connect('set_total_TP.flow:P', 'set_static_A.guess:Pt')

        p.setup()
        # om.n2(p)
        p.set_solver_print(level=-1)
        p.final_setup()

        # p.model.set_static_A.list_inputs(prom_name=True)

        h_map = self.h_map
        # 4 cases to check against
        for i, data in enumerate(self.ref_data):

            p.set_val('set_total_TP.T', data[h_map['Tt']], units='degR')
            p.set_val('set_total_TP.P', data[h_map['Pt']], units='psi')

            p.set_val('set_static_A.area', data[h_map['A']], units='inch**2')
            p.set_val('set_static_A.W', data[h_map['W']], units='lbm/s')

            # print(i, data[h_map['Tt']], data[h_map['Pt']], data[h_map['A']], data[h_map['W']])

            if i == 5:  # supersonic case
                p['set_static_A.guess:MN'] = 3.
            else: # subsonic case
                p['set_static_A.guess:MN'] = 0.5125

            p.run_model()
            
            self.check_vals('set_static_A', data)


    def test_case_MN(self): 

        p = self.p = om.Problem()
        total_TP = Thermo(mode='total_TP', 
                          method='CEA', 
                          thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p.model.add_subsystem('set_total_TP', total_TP)

        static_MN = Thermo(mode='static_MN', 
                           method='CEA', 
                           thermo_kwargs={'composition': constants.CEA_AIR_COMPOSITION, 
                                           'spec': species_data.janaf }) 
        p.model.add_subsystem('set_static_MN', static_MN)

        p.model.connect('set_total_TP.flow:S', 'set_static_MN.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_MN.ht')
        p.model.connect('set_total_TP.flow:gamma', 'set_static_MN.guess:gamt')
        p.model.connect('set_total_TP.flow:P', 'set_static_MN.guess:Pt')

        p.set_solver_print(level=-1)
        p.setup(check=False)

        h_map = self.h_map

        # 4 cases to check against
        for i, data in enumerate(self.ref_data):

            # print(i, data[h_map['Tt']], data[h_map['Pt']], data[h_map['MN']], data[h_map['W']])

            p.set_val('set_total_TP.T', data[h_map['Tt']], units='degR')
            p.set_val('set_total_TP.P', data[h_map['Pt']], units='psi')

            p.set_val('set_static_MN.MN', data[h_map['MN']])
            p.set_val('set_static_MN.W', data[h_map['W']], units='lbm/s')

            p.run_model()
            
            self.check_vals('set_static_MN', data)


if __name__ == "__main__": 
    unittest.main()