import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials

from pycycle.thermo.tabular.thermo_add import ThermoAdd


class ThermoAddTestCase(unittest.TestCase): 

    def test_mix_1fuel(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names='fuel')

        p.setup(force_alloc_complex=True)


        p['Fl_I:stat:W'] = 1.
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = [0.01]
        p['fuel:ratio'] = 0.02673
        p['fuel:h'] = 0.0

        p.run_model()

        tol = 1e-6
        # for tabular thermo, adding fuel doesn't change the enthalpy
        W_air_in = p['Fl_I:stat:W']/(1+p['Fl_I:tot:composition'])
        W_fuel_in = W_air_in * p['Fl_I:tot:composition']

        W_fuel_mix = W_air_in * p['fuel:ratio']
        W_out = W_air_in+W_fuel_in+W_fuel_mix
        assert_near_equal(p['Wout'], p['Fl_I:stat:W']+W_fuel_mix, tolerance=tol)

        mass_avg_h=(p['Fl_I:tot:h']*p['Fl_I:stat:W'])/p['Wout']
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tolerance=tol)

        assert_near_equal(p['fuel:W'], W_fuel_mix, tolerance=tol)
        assert_near_equal(p['composition_out'], (W_fuel_in+W_fuel_mix)/W_air_in, tolerance=tol)

    def test_mix_2fuel(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='reactant', mix_composition='FAR', mix_names=['fuel1', 'fuel2'])

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 1.
        p['Fl_I:tot:h'] = 181.381769
        p['Fl_I:tot:composition'] = [0]


        # half the ratio from the 1 fuel test
        ratio = 0.02673/2.
        p['fuel1:ratio'] = ratio
        p['fuel2:ratio'] = ratio
        p['fuel1:h'] = 10.0
        p['fuel2:h'] = 10.0

        p.run_model()

        tol = 1e-6
        # for tabular thermo, adding fuel doesn't change the enthalpy
        assert_near_equal(p['Wout'], 1.02673, tolerance=tol)

        mass_avg_h = (p['Fl_I:tot:h']*p['Fl_I:stat:W']+2*p['fuel1:W']*10.0)/p['Wout']
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tolerance=tol)

        assert_near_equal(p['fuel1:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)
        assert_near_equal(p['fuel1:W'], p['Fl_I:stat:W']*ratio, tolerance=tol)


    def test_mix_1flow(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='flow', mix_composition='FAR', mix_names='mix')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 1.
        p['Fl_I:tot:composition'] = [.01]
        p['Fl_I:tot:h'] = [1]

        p['mix:W'] = 2.
        p['mix:composition'] = [0.1]
        p['mix:h'] = 10.

        p.setup(force_alloc_complex=True)
        p.run_model()


        far_in = p['Fl_I:tot:composition']
        W_air_in =  p['Fl_I:stat:W']/(1+far_in)
        W_fuel_in = W_air_in*far_in

        far_mix = p['mix:composition']
        W_air_mix =  p['mix:W']/(1+far_mix)
        W_fuel_mix = W_air_mix*far_mix

        FAR_out = (W_fuel_in+W_fuel_mix)/(W_air_in+W_air_mix)

        tol = 1e-6
        assert_near_equal(p['composition_out'], FAR_out, tol)
        assert_near_equal(p['Wout'], p['Fl_I:stat:W']+p['mix:W'], tol)

        mass_avg_h = (1+2*10)/3.0
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tol)


    def test_mix_2flow(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='flow', mix_composition='FAR', mix_names=['mix1', 'mix2'])

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 10.
        p['Fl_I:tot:composition'] = .01
        p['Fl_I:tot:h'] = 2.

        p['mix1:W'] = 1.
        p['mix1:composition'] = 0.1
        p['mix1:h'] = 10.

        p['mix2:W'] = 2.
        p['mix2:composition'] = 0.1
        p['mix2:h'] = 20.

        p.setup(force_alloc_complex=True)
        p.run_model()


        far_in = p['Fl_I:tot:composition']
        W_air_in =  p['Fl_I:stat:W']/(1+far_in)
        W_fuel_in = W_air_in*far_in

        far_mix1 = p['mix1:composition']
        W_air_mix1 =  p['mix1:W']/(1+far_mix1)
        W_fuel_mix1 = W_air_mix1*far_mix1

        far_mix2 = p['mix2:composition']
        W_air_mix2 =  p['mix2:W']/(1+far_mix2)
        W_fuel_mix2 = W_air_mix2*far_mix2

        FAR_out = (W_fuel_in+W_fuel_mix1+W_fuel_mix2)/(W_air_in+W_air_mix1+W_air_mix2)

        tol = 1e-6
        assert_near_equal(p['composition_out'], FAR_out, tol)
        assert_near_equal(p['Wout'], p['Fl_I:stat:W']+p['mix1:W']+p['mix2:W'], tol)

        mass_avg_h = (10*2.+1*10.+2*20.)/13
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tol)


    def test_mix_1flow2compo(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='flow', inflow_composition={'FAR':0., 'WAR':0.}, 
                            mix_names='mix1')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 1
        p['Fl_I:tot:composition'] = [.01, 0]
        p['Fl_I:tot:h'] = 2.

        p['mix1:W'] = 1.
        p['mix1:composition'] = [0., .01]
        p['mix1:h'] = 2.


        p.setup(force_alloc_complex=True)
        p.run_model()

        tol = 1e-6
        assert_near_equal(p['composition_out'], [.005, .005], tol)
        assert_near_equal(p['Wout'], 2., tol)

        assert_near_equal(p['mass_avg_h'], 2., tol)


    def test_mix_1fuel2compo(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='reactant', inflow_composition={'FAR':0., 'WAR':0.}, 
                            mix_composition='FAR', mix_names='fuel1')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 1
        p['Fl_I:tot:composition'] = [.01, 0]
        p['Fl_I:tot:h'] = 2.

        p['fuel1:ratio'] = .01
        p['fuel1:h'] = 2.

        p.setup(force_alloc_complex=True)
        p.run_model()

        W_air_in = p['Fl_I:stat:W']/(1+np.sum(p['Fl_I:tot:composition']))
        W_fuel_in = W_air_in * p['Fl_I:tot:composition'][0]

        W_fuel_mix = W_air_in * p['fuel1:ratio']
        W_out = p['Fl_I:stat:W']+W_fuel_mix

        tol = 1e-6
        assert_near_equal(p['Wout'], p['Fl_I:stat:W']+W_fuel_mix, tolerance=tol)

        mass_avg_h=(p['Fl_I:tot:h']*p['Fl_I:stat:W'] + W_fuel_mix*2.)/p['Wout']
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tolerance=tol)

        assert_near_equal(p['fuel1:W'], W_fuel_mix, tolerance=tol)
        assert_near_equal(p['composition_out'][0], (W_fuel_in+W_fuel_mix)/W_air_in, tolerance=tol)
        assert_near_equal(p['composition_out'][1], 0, tolerance=tol)


    def test_mix_1water2compo(self): 

        p = om.Problem()
        p.model = ThermoAdd(mix_mode='reactant', inflow_composition={'FAR':0., 'WAR':0.}, 
                            mix_composition='WAR', mix_names='water1')

        p.setup(force_alloc_complex=True)

        p['Fl_I:stat:W'] = 1
        p['Fl_I:tot:composition'] = [.01, .01]
        p['Fl_I:tot:h'] = 2.

        p['water1:ratio'] = .01
        p['water1:h'] = 2.

        p.setup(force_alloc_complex=True)
        p.run_model()

        W_air_in = p['Fl_I:stat:W']/(1+np.sum(p['Fl_I:tot:composition']))
        W_fuel_in = W_air_in * p['Fl_I:tot:composition'][0]
        W_water_in = W_air_in * p['Fl_I:tot:composition'][1]

        W_water_mix = W_air_in * p['water1:ratio']
        W_out = p['Fl_I:stat:W']+W_water_mix

        tol = 1e-6
        assert_near_equal(p['Wout'], p['Fl_I:stat:W']+W_water_mix, tolerance=tol)

        mass_avg_h=(p['Fl_I:tot:h']*p['Fl_I:stat:W'] + W_water_mix*2.)/p['Wout']
        assert_near_equal(p['mass_avg_h'], mass_avg_h, tolerance=tol)

        assert_near_equal(p['water1:W'], W_water_mix, tolerance=tol)
        assert_near_equal(p['composition_out'][0], W_fuel_in/W_air_in, tolerance=tol)
        assert_near_equal(p['composition_out'][1], (W_water_in+W_water_mix)/W_air_in, tolerance=tol)

if __name__ == "__main__": 

    unittest.main()