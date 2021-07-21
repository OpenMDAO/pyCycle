import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.thermo.cea import species_data
from pycycle.constants import CEA_AIR_COMPOSITION, CEA_AIR_FUEL_COMPOSITION

from pycycle.thermo.cea.thermo_add import ThermoAdd

class ThermoAddTestCase(unittest.TestCase):
    
    def test_mix_1fuel(self): 

        thermo_spec = species_data.janaf

        air_thermo = species_data.Properties(thermo_spec, init_elements=CEA_AIR_COMPOSITION)

        p = om.Problem()

        fuel = 'JP-7'
        p.model = ThermoAdd(spec=thermo_spec,
                            inflow_composition=CEA_AIR_COMPOSITION, mix_mode='reactant', 
                            mix_composition=fuel, mix_names='fuel')



        p.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = air_thermo.b0

        p['fuel:ratio'] = 0.02673


        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
        assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
        assert_near_equal(p['fuel:W'], p['Fl_I:stat:W']*p['fuel:ratio'], tolerance=tol)
        assert_near_equal(p['composition_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

        # data = p.check_partials(out_stream=None, method='cs')
        data = p.check_partials(method='fd')
        assert_check_partials(data, atol=2.e-4, rtol=2.e-4) # can't get very accurate checks with FD

    def test_mix_2fuel(self): 

        thermo_spec = species_data.janaf

        air_thermo = species_data.Properties(thermo_spec, init_elements=CEA_AIR_COMPOSITION)

        p = om.Problem()

        fuel = 'JP-7'
        p.model = ThermoAdd(spec=thermo_spec,
                            inflow_composition=CEA_AIR_COMPOSITION, mix_mode='reactant', 
                            mix_composition=[fuel, fuel], mix_names=['fuel1','fuel2'])


        p.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        p['Fl_I:stat:W'] = 38.8
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = air_thermo.b0

        # half the ratio from the 1 fuel test
        ratio = 0.02673/2.
        p['fuel1:ratio'] = ratio
        p['fuel2:ratio'] = ratio


        p.run_model()

        tol = 5e-7
        assert_near_equal(p['mass_avg_h'], 176.65965638, tolerance=tol)
        assert_near_equal(p['Wout'], 39.837124, tolerance=tol)
        assert_near_equal(p['fuel1:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)
        assert_near_equal(p['fuel2:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)
        assert_near_equal(p['composition_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)

        data = p.check_partials(out_stream=None, method='fd')
        # data = p.check_partials(method='cs')
        assert_check_partials(data, atol=2.e-4, rtol=2.e-4) # can't get very accurate checks with FD

    def test_mix_1flow(self): 

        thermo_spec = species_data.janaf 

        p = om.Problem()

        p.model = ThermoAdd(spec=thermo_spec,
                            inflow_composition=CEA_AIR_FUEL_COMPOSITION, mix_mode='flow', 
                            mix_composition=CEA_AIR_COMPOSITION, mix_names='mix')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 62.15
        p['Fl_I:tot:composition'] = [0.000313780313538, 0.0021127831122, 0.004208814234964, 0.052325087161902, 0.014058631311261]
        p['Fl_I:tot:h'] = 10. 

        p['mix:W'] = 4.44635
        p['mix:composition'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]
        p['mix:h'] = 5
        p.run_model()

        tol = 5e-7
        assert_near_equal(p['Wout'], 62.15+4.44635, tolerance=tol)
        assert_near_equal(p['composition_out'], np.array([0.00031442, 0.00197246, 0.00392781, 0.05243129, 0.01408717]), tolerance=tol)
        assert_near_equal(p['mass_avg_h'], (62.15*10+4.44635*5)/(62.15+4.44635), tol)

    def test_mix_2flow(self): 

        thermo_spec = species_data.janaf 

        p = om.Problem()

        p.model = ThermoAdd(spec=thermo_spec,
                            inflow_composition=CEA_AIR_FUEL_COMPOSITION, mix_mode='flow', 
                            mix_composition=[CEA_AIR_COMPOSITION, CEA_AIR_COMPOSITION], mix_names=['mix1', 'mix2'])

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 62.15
        # p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = [0.000313780313538, 0.0021127831122, 0.004208814234964, 0.052325087161902, 0.014058631311261]

        p['mix1:W'] = 4.44635/2
        p['mix1:composition'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]

        p['mix2:W'] = 4.44635/2
        p['mix2:composition'] = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]

        p.run_model()

        tol = 5e-7
        assert_near_equal(p['Wout'], 62.15+4.44635, tolerance=tol)
        # assert_near_equal(p['composition_out'], np.array([0.0003149, 0.00186566, 0.00371394, 0.05251212, 0.01410888]), tolerance=tol)
        assert_near_equal(p['composition_out'], np.array([0.00031442, 0.00197246, 0.00392781, 0.05243129, 0.01408717]), tolerance=tol)


    def test_war_vals(self):
        """
        verifies that the ThermoAdd component gives the right answers when adding water to dry air
        """

        prob = om.Problem()

        thermo_spec = species_data.wet_air

        air_thermo = species_data.Properties(thermo_spec, init_elements=CEA_AIR_COMPOSITION)

        prob.model.add_subsystem('war', ThermoAdd(spec=thermo_spec,
                                                  inflow_composition=CEA_AIR_COMPOSITION, mix_composition='Water'), 
                                 promotes=['*'])
        


        prob.setup(force_alloc_complex=True)

        # p['Fl_I:stat:P'] = 158.428
        prob['Fl_I:stat:W'] = 38.8
        prob['mix:ratio'] = .0001 # WAR
        prob['Fl_I:tot:h'] = 181.381769
        prob['Fl_I:tot:composition'] = air_thermo.b0

        prob.run_model()

        tol = 1e-5

        assert_near_equal(prob['composition_out'][0], 3.23286926e-04, tol)
        assert_near_equal(prob['composition_out'][1], 1.10121227e-05, tol)
        assert_near_equal(prob['composition_out'][2], 1.11005769e-05, tol)
        assert_near_equal(prob['composition_out'][3], 5.39103820e-02, tol)
        assert_near_equal(prob['composition_out'][4], 1.44901169e-02, tol)




if __name__ == "__main__": 

    unittest.main()