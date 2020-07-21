import unittest

import numpy as np

import openmdao.api as om

from openmdao.utils.assert_utils import assert_near_equal

from pycycle.cea import species_data


class SpeciesDataTestCase(unittest.TestCase):

    def test_bad_elements(self):

        bad_elements = ['O2']

        with self.assertRaises(ValueError) as cm:

            thermo = species_data.Thermo(thermo_data_module=species_data.co2_co_o2, elements=bad_elements)

        self.assertEqual(str(cm.exception), "The provided element `O2` is not a valid element in the provided thermo data.")


    def test_get_elements(self):

        thermo_data = species_data.janaf
        reactants = {'O':1, 'H':1, 'CO2':1, 'N':1, 'Ar':1}
        expected_elements = ['Ar', 'C', 'H', 'N', 'O']
        expected_proportions = [1, 1, 1, 1, 3]

        thermo = species_data.Thermo(thermo_data_module=thermo_data, init_reacts=reactants)
        elements, proportions = thermo.get_elements(thermo_data, reactants)

        reactants2 = {'O2': 20.78, 'H2O':1.0, 'CO2':.01, 'Ar':0.01}
        expected_elements2 = ['Ar', 'C', 'H', 'O']
        expected_proportions2 = [.01, .01, 2.0, 42.58]

        thermo2 = species_data.Thermo(thermo_data_module=thermo_data, init_reacts=reactants2)
        elements2, proportions2 = thermo2.get_elements(thermo_data, reactants2)

        self.assertEqual(elements, expected_elements)
        assert_near_equal(proportions, expected_proportions, 1e-4)

        self.assertEqual(elements2, expected_elements2)
        assert_near_equal(proportions2, expected_proportions2, 1e-4)

    def test_values(self):
        thermo = species_data.Thermo(thermo_data_module=species_data.janaf)

        T = np.ones(len(thermo.products))*800
        H0 = thermo.H0(T)
        H0_expected = np.array([1.56828125, -14.33638055, -55.73109232, 72.63079725, 16.05970705,
        8.50490177, 15.48013356, 2.2620009, 39.06512544, 2.38109781])

        S0 = thermo.S0(T)
        S0_expected = np.array([21.09120423, 27.33539665, 30.96900291, 20.90543864, 28.99563162, 34.04699324,
        37.65697408, 26.58210226, 21.90362596, 28.37546079])

        Cp0 = thermo.Cp0(T)
        Cp0_expected = np.array([2.5, 3.83668584, 6.18585395, 2.5, 3.94173049, 6.06074564,
        8.81078156, 3.78063693, 2.52375035, 4.05857378])

        HJ = thermo.H0_applyJ(T, 1.)
        HJ_expected = np.array([0.00116465, 0.02271633, 0.07739618, -0.0876635, -0.01514747, -0.0030552,
        -0.00833669, 0.0018983, -0.04567672, 0.00209684])

        SJ = thermo.S0_applyJ(T, 1)
        SJ_expected = np.array([0.003125, 0.00479586, 0.00773232, 0.003125, 0.00492716, 0.00757593,
        0.01101348, 0.0047258, 0.00315469, 0.00507322])

        CpJ = thermo.Cp0_applyJ(T, 1)
        CpJ_expected = np.array([0.0, 8.49157682e-04, 2.05623736e-03, 0.0,
        8.39005783e-04, 1.91861539e-03, 2.54742879e-03, 8.12550383e-04, -5.62484525e-05, 8.19626699e-04])
        n = thermo.init_prod_amounts
        n_expected = np.array([3.23319258e-04, 0.0, 1.10132241e-05, 0.0,
        0.0, 0.0, 0.0, 2.69578868e-02, 0.0, 7.23199412e-03])

        tol = 1e-4

        assert_near_equal(H0, H0_expected, tol)
        assert_near_equal(S0, S0_expected, tol)
        assert_near_equal(Cp0, Cp0_expected, tol)

        assert_near_equal(HJ, HJ_expected, tol)
        assert_near_equal(SJ, SJ_expected, tol)
        assert_near_equal(CpJ, CpJ_expected, tol)
        assert_near_equal(n, n_expected, tol)

    def test_element_filter(self):

        elements1_provided = {'C', 'O'}
        products1_expected = ['CO', 'CO2', 'O2']
        thermo1 = species_data.Thermo(thermo_data_module=species_data.co2_co_o2, elements=elements1_provided)
        elements1_expected = ['C', 'O']
        products1 = thermo1.products
        elements1 = thermo1.elements

        elements2_provided = {'Ar', 'C', 'N', 'O'}
        products2_expected = ['Ar', 'CO', 'CO2', 'N', 'NO', 'NO2', 'NO3', 'N2', 'O', 'O2']
        thermo2 = species_data.Thermo(thermo_data_module=species_data.janaf, elements=elements2_provided)
        elements2_expected = ['Ar', 'C', 'N', 'O']
        products2 = thermo2.products
        elements2 = thermo2.elements

        elements3_provided = {'Ar', 'C', 'H', 'N'}
        products3_expected = ['Ar', 'CH4', 'C2H4', 'H', 'H2', 'N', 'NH3', 'N2']
        thermo3 = species_data.Thermo(thermo_data_module=species_data.janaf, elements=elements3_provided)
        elements3_expected = ['Ar', 'C', 'H', 'N']
        products3 = thermo3.products
        elements3 = thermo3.elements

        self.assertEqual(products1, products1_expected)
        self.assertEqual(elements1, elements1_expected)

        self.assertEqual(products2, products2_expected)
        self.assertEqual(elements2, elements2_expected)

        self.assertEqual(products3, products3_expected)
        self.assertEqual(elements3, elements3_expected)



if __name__ == "__main__":

    import numpy as np
    import scipy as sp

    np.seterr(all='raise')

    unittest.main()

