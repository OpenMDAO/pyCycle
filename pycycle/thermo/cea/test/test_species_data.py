import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.thermo.cea import species_data
from pycycle.constants import CO2_CO_O2_ELEMENTS, CO2_CO_O2_MIX, AIR_ELEMENTS, AIR_MIX


class SpeciesDataTestCase(unittest.TestCase):

    def test_errors(self):

        product_elements = {'O2':1}

        with self.assertRaises(ValueError) as cm:

            thermo = species_data.Properties(thermo_data_module=species_data.co2_co_o2, init_elements=product_elements)

        self.assertEqual(str(cm.exception), "The provided element `O2` is a product in your provided thermo data, but is not an element.")

        bad_elements = {'H':1}

        with self.assertRaises(ValueError) as cm:

            thermo = species_data.Properties(thermo_data_module=species_data.co2_co_o2, init_elements=bad_elements)

            self.assertEqual(str(cm.exception), "The provided element `H` is not used in any products in your thermo data.")

        with self.assertRaises(ValueError) as cm:

            thermo = species_data.Properties(thermo_data_module=species_data.co2_co_o2)

        self.assertEqual(str(cm.exception), 'You have not provided `init_elements`. In order to set thermodynamic data it must be provided.')

    
    def test_values(self):
        thermo2 = species_data.Properties(thermo_data_module=species_data.janaf, init_elements=AIR_ELEMENTS)
        thermo3 = species_data.Properties(thermo_data_module=species_data.co2_co_o2, init_elements=CO2_CO_O2_ELEMENTS)

        T2 = np.ones(thermo2.num_prod)*800
        T3 = np.ones(thermo3.num_prod)*800
        H02 = thermo2.H0(T2)
        H03 = thermo3.H0(T3)
        H0_expected = np.array([1.56828125, -14.33638055, -55.73109232, 72.63079725, 16.05970705,
        8.50490177, 15.48013356, 2.2620009, 39.06512544, 2.38109781])
        H0_expected3 = np.array([-14.33638055, -55.73109232,   2.38109781])

        S02 = thermo2.S0(T2)
        S03 = thermo3.S0(T3)
        S0_expected = np.array([21.09120423, 27.33539665, 30.96900291, 20.90543864, 28.99563162, 34.04699324,
        37.65697408, 26.58210226, 21.90362596, 28.37546079])
        S0_expected3 = np.array([27.33539665, 30.96900291, 28.37546079])

        Cp02 = thermo2.Cp0(T2)
        Cp03 = thermo3.Cp0(T3)
        Cp0_expected = np.array([2.5, 3.83668584, 6.18585395, 2.5, 3.94173049, 6.06074564,
        8.81078156, 3.78063693, 2.52375035, 4.05857378])
        Cp0_expected3 = np.array([3.83668584, 6.18585395, 4.05857378])

        HJ2 = thermo2.H0_applyJ(T2, 1.)
        HJ3 = thermo3.H0_applyJ(T3, 1.)
        HJ_expected = np.array([0.00116465, 0.02271633, 0.07739618, -0.0876635, -0.01514747, -0.0030552,
        -0.00833669, 0.0018983, -0.04567672, 0.00209684])
        HJ_expected3 = np.array([0.02271633, 0.07739618, 0.00209684])

        SJ2 = thermo2.S0_applyJ(T2, 1)
        SJ3 = thermo3.S0_applyJ(T3, 1)
        SJ_expected = np.array([0.003125, 0.00479586, 0.00773232, 0.003125, 0.00492716, 0.00757593,
        0.01101348, 0.0047258, 0.00315469, 0.00507322])
        SJ_expected3 = np.array([0.00479586, 0.00773232, 0.00507322])

        CpJ2 = thermo2.Cp0_applyJ(T2, 1)
        CpJ3 = thermo3.Cp0_applyJ(T3, 1)
        CpJ_expected = np.array([0.0, 8.49157682e-04, 2.05623736e-03, 0.0,
        8.39005783e-04, 1.91861539e-03, 2.54742879e-03, 8.12550383e-04, -5.62484525e-05, 8.19626699e-04])
        CpJ_expected3 = np.array([8.49157682e-04, 2.05623736e-03, 8.19626699e-04])

        b02 = thermo2.b0
        b03 = thermo3.b0
        b0_expected = np.array([3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02])
        b0_expected3 = np.array([0.02272211, 0.04544422])

        tol = 1e-4

        assert_near_equal(H02, H0_expected, tol)
        assert_near_equal(S02, S0_expected, tol)
        assert_near_equal(Cp02, Cp0_expected, tol)

        assert_near_equal(HJ2, HJ_expected, tol)
        assert_near_equal(SJ2, SJ_expected, tol)
        assert_near_equal(CpJ2, CpJ_expected, tol)
        assert_near_equal(b02, b0_expected, tol)

        assert_near_equal(H03, H0_expected3, tol)
        assert_near_equal(S03, S0_expected3, tol)
        assert_near_equal(Cp03, Cp0_expected3, tol)

        assert_near_equal(HJ3, HJ_expected3, tol)
        assert_near_equal(SJ3, SJ_expected3, tol)
        assert_near_equal(CpJ3, CpJ_expected3, tol)
        assert_near_equal(b03, b0_expected3, tol)

    def test_element_filter(self):

        elements1_provided = {'C':1, 'O':1}
        products1_expected = ['CO', 'CO2', 'O2']
        thermo1 = species_data.Properties(thermo_data_module=species_data.co2_co_o2, init_elements=elements1_provided)
        elements1_expected = {'C', 'O'}
        products1 = thermo1.products
        elements1 = thermo1.elements

        elements2_provided = {'Ar':1, 'C':1, 'N':1, 'O':1}
        products2_expected = ['Ar', 'CO', 'CO2', 'N', 'NO', 'NO2', 'NO3', 'N2', 'O', 'O2']
        thermo2 = species_data.Properties(thermo_data_module=species_data.janaf, init_elements=elements2_provided)
        elements2_expected = {'Ar', 'C', 'N', 'O'}
        products2 = thermo2.products
        elements2 = thermo2.elements

        elements3_provided = {'Ar':1, 'C':1, 'H':1, 'N':1}
        products3_expected = ['Ar', 'CH4', 'C2H4', 'H', 'H2', 'N', 'NH3', 'N2']
        thermo3 = species_data.Properties(thermo_data_module=species_data.janaf, init_elements=elements3_provided)
        elements3_expected = {'Ar', 'C', 'H', 'N'}
        products3 = thermo3.products
        elements3 = thermo3.elements

        self.assertEqual(products1, products1_expected)
        self.assertEqual(set(elements1), elements1_expected)

        self.assertEqual(products2, products2_expected)
        self.assertEqual(set(elements2), elements2_expected)

        self.assertEqual(products3, products3_expected)
        self.assertEqual(set(elements3), elements3_expected)



if __name__ == "__main__":

    import numpy as np
    import scipy as sp

    np.seterr(all='raise')

    unittest.main()

