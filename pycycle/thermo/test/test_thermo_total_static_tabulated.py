import unittest
import os

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea import species_data
from pycycle import constants


class SetTotalSimpleTestCase(unittest.TestCase):
    """Sanity check that compares TP, hP, SP sets manually""" 

    def test_set_total_TP(self):
        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))

        p.model.add_subsystem('thermo', Thermo(mode='total_TP', 
                                               method='TABULAR', 
                                               thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                                              'spec': constants.AIR_JETA_TAB_SPEC }), 
                              promotes=['*'])

        p.setup(check=False)
        p.set_solver_print(level=-1)
        p.final_setup()

        p.set_val('T', 2000, units='degK')
        p.set_val('P', 1.034210, units='bar')
        p.run_model()
        assert_near_equal(p['gamma'], 1.27532298, 1e-4)

        p.set_val('T', 1500, units='degK')
        p.set_val('P', 1.034210, units='bar')
        p.run_model()
        assert_near_equal(p['gamma'], 1.30444708, 1e-4)

    def test_set_total_hP(self):

        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))
        p.model.add_subsystem('thermo', Thermo(mode='total_hP', 
                                               method='TABULAR', 
                                               thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                                              'spec': constants.AIR_JETA_TAB_SPEC }), 
                              promotes=['*'])

        p.setup()
        # TODO: Investigate this weirdness.... this case won't work if you thermo_TP fully solve itself
        p.set_solver_print(level=-1)

        p.set_val('h', 1976117.04700962, units='J/kg')
        p.set_val('P', 1.034210, units='bar')

        p.run_model()

        assert_near_equal(p['gamma'], 1.27532298, 1e-4)
        assert_near_equal(p['T'], 2000, 1e-4)

        # 1500K
        p.set_val('h', 1337396.89979978, units='J/kg')
        p.run_model()

        assert_near_equal(p['gamma'], 1.30444708, 1e-4)
        assert_near_equal(p['T'], 1500, 1e-4)

    def test_set_total_SP(self):

        #

        p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))
        p.model.add_subsystem('thermo', Thermo(mode='total_SP', 
                                               method='TABULAR', 
                                               thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                                              'spec': constants.AIR_JETA_TAB_SPEC }), 
                              promotes=['*'])
       
        r = p.model

        p.setup(check=False)

        p.set_solver_print(level=-2)

        p.set_val('S', 8982.03057206, units="J/kg/degK")
        p.set_val('P', 1.034210, units="bar")

        p.run_model()

        assert_near_equal(p['gamma'], 1.27532298, 5e-4)
        assert_near_equal(p['T'], 2000, 5e-3)

        # 1500K

        p.set_val('S', 8615.116554906986, units="J/kg/degK")
        p.run_model()

        assert_near_equal(p['gamma'], 1.30444708, 5e-4)
        assert_near_equal(p['T'], 1500, 5e-3)


class TestSetTotalTabular(unittest.TestCase): 
    """
    Run TP and then compare hp and SP 
    outputs to make sure they get the same value at a 
    range of temp and pressure
    """ 

    def test_set_total_equivalence(self): 

        p_TP = om.Problem()
        ivc = p_TP.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))

        p_TP.model.add_subsystem('tp', Thermo(mode='total_TP', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }), promotes=['*']) 

        p_TP.setup()
        p_TP.set_solver_print(level=-1)

        p_hP = om.Problem()
        ivc = p_hP.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))

        p_hP.model.add_subsystem('hp', Thermo(mode='total_hP', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }), promotes=['*']) 
        p_hP.setup()
        p_hP.set_solver_print(level=-1)

        p_SP = om.Problem()
        ivc = p_SP.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))

        p_SP.model.add_subsystem('sp', Thermo(mode='total_SP', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }), promotes=['*']) 
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


class TestStaticTabular(unittest.TestCase): 


    def test_statics(self):

        p = self.p = om.Problem()
        ivc = p.model.add_subsystem('ivc', om.IndepVarComp(), promotes=['*'])
        ivc.add_output('composition', val=np.zeros(1))
        total_TP =  Thermo(mode='total_TP', 
                           method='TABULAR', 
                           thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }) 
        p.model.add_subsystem('set_total_TP', total_TP, promotes=['composition'])

        static_Ps =  Thermo(mode='static_Ps', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }) 
        p.model.add_subsystem('set_static_Ps', static_Ps, promotes_inputs=['composition'])
        p.model.connect('set_total_TP.flow:S', 'set_static_Ps.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_Ps.ht')
       
        static_MN =  Thermo(mode='static_MN', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }) 
        p.model.add_subsystem('set_static_MN', static_MN, promotes_inputs=['composition'])
        p.model.connect('set_total_TP.flow:S', 'set_static_MN.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_MN.ht')
        p.model.connect('set_static_Ps.flow:MN', 'set_static_MN.MN')
        
        static_area =  Thermo(mode='static_A', 
                            method='TABULAR', 
                            thermo_kwargs={'composition': constants.TAB_AIR_FUEL_COMPOSITION, 
                                           'spec': constants.AIR_JETA_TAB_SPEC }) 
        p.model.add_subsystem('set_static_area', static_area, promotes_inputs=['composition'])
        p.model.connect('set_total_TP.flow:S', 'set_static_area.S')
        p.model.connect('set_total_TP.flow:h', 'set_static_area.ht')
        p.model.connect('set_static_Ps.flow:area', 'set_static_area.area')
        p.model.connect('set_total_TP.flow:P',['set_static_MN.guess:Pt','set_static_area.guess:Pt'])

        p.setup(force_alloc_complex=True)
        p.set_solver_print(level=2)
        p.final_setup()

        p.set_val('set_total_TP.T', 1500, units='degR')
        p.set_val('set_total_TP.P', 45, units='psi')

        p.set_val('set_static_Ps.Ps', 30, units='psi')
        p.set_val('set_static_Ps.W', 1, units='lbm/s')
        p.set_val('set_static_MN.W', 1, units='lbm/s')
        p.set_val('set_static_area.W', 1, units='lbm/s')

        p.run_model()
        
        assert_near_equal(p['set_static_Ps.flow:P'], 30, 1e-4)

        assert_near_equal(p['set_static_MN.flow:P'], 30, 1e-4)
        assert_near_equal(p['set_static_MN.flow:MN'], p['set_static_Ps.flow:MN'], 1e-4)

        assert_near_equal(p['set_static_area.flow:P'], 30, 1e-4)
        assert_near_equal(p['set_static_area.flow:MN'], p['set_static_Ps.flow:MN'], 1e-4)



if __name__ == "__main__": 
    unittest.main()