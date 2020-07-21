import numpy as np
from scipy import interpolate

from collections import OrderedDict

from pycycle.cea.thermo_data import co2_co_o2
from pycycle.cea.thermo_data import janaf
from pycycle.cea.thermo_data import wet_air

#from ad.admath import log
from numpy import log
class Thermo(object):
    """Compute H, S, Cp given a species and temperature"""
    def __init__(self, thermo_data_module, init_reacts=None, elements=None):

        self.a = None
        self.a_T = None
        self.element_wt = None
        self.aij = None
        self.products = None
        self.temp_ranges = None
        self.valid_temp_range = None
        self.wt_mole = None # array of mole weights
        self.init_prod_amounts = None # concentrations (sum to 1)
        self.thermo_data_module = thermo_data_module
        self.init_reacts = init_reacts
        self.temp_base = None # array of lowest end of lowest temperature range

        if elements is not None:
            valid_elements = set()
            for compound in thermo_data_module.products.keys():
                valid_elements.update(thermo_data_module.products[compound]['elements'].keys())
            
            for element in elements:
                if element not in valid_elements:
                    raise ValueError(f'The provided element `{element}` is not a valid element in the provided thermo data.')

        else:
            if init_reacts is not None:
                elements = set(self.get_elements(thermo_data_module, init_reacts)[0])
            else:
                elements = set(thermo_data_module.default_elements)

        self.elements = sorted(elements)

        self.set_data(thermo_data_module=thermo_data_module, elements=elements)

    def H0(self, Tt): # standard-state molar enthalpy for species j at temp T
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return (-a_T[0]/Tt**2 + a_T[1]/Tt*log(Tt) + a_T[2] + a_T[3]*Tt/2. + a_T[4]*Tt**2/3. + a_T[5]*Tt**3/4. + a_T[6]*Tt**4/5.+a_T[7]/Tt)

    def S0(self, Tt): # standard-state molar entropy for species j at temp T
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return (-a_T[0]/(2*Tt**2) - a_T[1]/Tt + a_T[2]*log(Tt) + a_T[3]*Tt + a_T[4]*Tt**2/2. + a_T[5]*Tt**3/3. + a_T[6]*Tt**4/4.+a_T[8])

    def Cp0(self, Tt): #molar heat capacity at constant pressure for
                    #standard state for species or reactant j, J/(kg-mole)_j(K)
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return a_T[0]/Tt**2 + a_T[1]/Tt + a_T[2] + a_T[3]*Tt + a_T[4]*Tt**2 + a_T[5]*Tt**3 + a_T[6]*Tt**4

    def H0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return vec*(2*a_T[0]/Tt**3 + a_T[1]*(1-log(Tt))/Tt**2 + a_T[3]/2. + 2*a_T[4]/3.*Tt + 3*a_T[5]/4.*Tt**2 + 4*a_T[6]/5.*Tt**3 - a_T[7]/Tt**2)

    def S0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return vec*(a_T[0]/(Tt**3) + a_T[1]/Tt**2 + a_T[2]/Tt + a_T[3] + a_T[4]*Tt + a_T[5]*Tt**2 + 4*a_T[6]/4.*Tt**3)

    def Cp0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt < self.valid_temp_range[0] or Tt > self.valid_temp_range[1]: # runs if temperature is outside range of current coefficients
            self.build_coeff_table(Tt)
        a_T = self.a_T
        return vec*(-2*a_T[0]/Tt**3 - a_T[1]/Tt**2 + a_T[3] + 2.*a_T[4]*Tt + 3.*a_T[5]*Tt**2 + 4.*a_T[6]*Tt**3)

    def set_data(self, thermo_data_module, init_reacts=None, elements=None):
        """computes the relevant quantities, given the recatant data"""

        self.thermo_data = thermo_data_module
        self.prod_data = thermo_data_module.products

        # elements = set()

        if init_reacts is None:
            init_reacts = thermo_data_module.init_prod_amounts

        #expand the init_reacts out to include all products, not just those provided
        full_init_react = OrderedDict()
        for p in self.prod_data:
            # initial amounts need to be given in mass ratios
            if p in init_reacts:
                full_init_react[p] = init_reacts[p] * self.prod_data[p]['wt']
            else:
                full_init_react[p] = 0.

        init_reacts = full_init_react

        # for name in init_reacts: # figure out which elements are present
        #     if init_reacts[name] > 0:
        #         elements.update(self.prod_data[name]['elements'])

        init_prod_amounts = [init_reacts[name] for name in init_reacts
                             if elements.issuperset(self.prod_data[name]['elements'])]

        # self.elements = sorted(elements)
        self.init_prod_amounts = np.array(init_prod_amounts)
        self.init_prod_amounts = self.init_prod_amounts/np.sum(self.init_prod_amounts) # normalize to 1

        self.products = [name for name, prod_data in self.prod_data.items()
                         if elements.issuperset(prod_data['elements'])]

        self.num_element = len(self.elements)
        self.num_prod = len(self.products)

        self.a = np.zeros((self.num_prod, 10))
        self.a_T = self.a.T

        element_wt = []
        aij = []

        for e in self.elements:
            element_wt.append(thermo_data_module.element_wts[e])

            row = [self.prod_data[r]['elements'].get(e,0) for r in self.products]
            aij.append(row)

        self.wt_mole = np.empty(self.num_prod)
        for i,r in enumerate(self.products):
            self.wt_mole[i] = self.prod_data[r]['wt']

        self.init_prod_amounts /= self.wt_mole

        self.element_wt = np.array(element_wt)
        self.aij = np.array(aij)

        # pre-computed constants used in calculations
        aij_prod = np.empty((self.num_element,self.num_element, self.num_prod))
        for i in range(self.num_element):
            for j in range(self.num_element):
                aij_prod[i][j] = self.aij[i]*self.aij[j]

        self.aij_prod = aij_prod

        self.aij_prod_deriv = np.zeros((self.num_element**2,self.num_prod))
        for k in range(self.num_element**2):
            for l in range(self.num_prod):
                i = k//self.num_element
                j = np.mod(k,self.num_element)
                self.aij_prod_deriv[k][l] = self.aij_prod[i][j][l]

        self.build_coeff_table(999) # just pick arbitrary default temperature so there is something there right away

    def get_elements(self, thermo_data_module, reactants):

        elements = set()
        products = thermo_data_module.products

        for name in reactants: # figure out which elements are present
            if reactants[name] > 0:
                elements.update(products[name]['elements'])

        elements = sorted(elements)

        aij = []

        for e in elements:

            row = [products[react]['elements'].get(e,0) for react in reactants]
            aij.append(row)

        b_values = np.zeros((len(elements)))

        for i, element in enumerate(elements):
            for j, reactant in enumerate(reactants):
                b_values[i] += aij[i][j]*reactants[reactant]

        return(elements, b_values) #returns a list of the included elements, along with the sum of their atomic ratios


    def build_coeff_table(self, Tt):
        """Build the temperature specific coeff array and find the highest-low value and
        the lowest-high value of temperatures from all the reactants to give the
        valid range for the data fits."""

        if self.temp_base is None:
            self.temp_base = np.zeros(self.num_prod)

        max_low, min_high = -1e50, 1e50
        for i,p in enumerate(self.products):
            tr = self.prod_data[p]['ranges']

            if self.temp_base[i] == 0:
                self.temp_base[i] = tr[0]

            j = int(np.searchsorted(tr, Tt))
            if j == 0: # dont run off the start
                j = 1
            elif j == len(tr): # don't run off the end
                j -= 1

            # find valid range
            low, high = tr[j-1], tr[j]
            max_low = max(low, max_low)
            min_high = min(high, min_high)

            # built a data
            data = self.prod_data[p]['coeffs'][j-1]

            # have to slice because some rows are 9 long and others 10
            self.a[i][:len(data)] = data

        self.valid_temp_range = (max_low, min_high)

if __name__ == "__main__":

    # thermo = Thermo(janaf, janaf.init_prod_amounts)

    # T = np.ones(len(thermo.products))*800
    # H0 = thermo.H0(T)
    # S0 = thermo.S0(T)
    # Cp0 = thermo.Cp0(T)

    # HJ = thermo.H0_applyJ(T, 1.)
    # SJ = thermo.S0_applyJ(T, 1)
    # CpJ = thermo.Cp0_applyJ(T, 1)
    # n = thermo.init_prod_amounts
    # print('\nT', T)
    # print('\nH0', H0)
    # print('\nS0', S0)
    # print('\nCp0', Cp0)
    # print('\nHJ', HJ)
    # print('\nSJ', SJ)
    # print('\nCpJ', CpJ)
    # print('\nn', n, '\n')
    # print(thermo.elements)
    
    # elements1 = {'O'}
    # thermo1 = Thermo(thermo_data_module=co2_co_o2, elements=elements1)
    # print(thermo1.products)

    thermo = Thermo(co2_co_o2)

    T = np.ones(len(thermo.products))*800
    H0 = thermo.H0(T)
    S0 = thermo.S0(T)
    Cp0 = thermo.Cp0(T)

    HJ = thermo.H0_applyJ(T, 1.)
    SJ = thermo.S0_applyJ(T, 1)
    CpJ = thermo.Cp0_applyJ(T, 1)
    n = thermo.init_prod_amounts
    
    print('\nT', T)
    print('\nH0', H0)
    print('\nS0', S0)
    print('\nCp0', Cp0)
    print('\nHJ', HJ)
    print('\nSJ', SJ)
    print('\nCpJ', CpJ)
    print('\nn', n, '\n')
    print(thermo.elements)

    T_actual = np.array([800., 800., 800.])
    H0_actual = np.array([-14.33638055, -55.73109232, 2.38109781])
    S0_actual = np.array([27.33539665, 30.96900291, 28.37546079])
    Cp0_actual = np.array([3.83668584, 6.18585395, 4.05857378])
    HJ_actual = np.array([0.02271633, 0.07739618, 0.00209684])
    SJ_actual = np.array([0.00479586, 0.00773232, 0.00507322])
    CpJ_actual = np.array([0.00084916, 0.00205624, 0.00081963])
    n_actual = np.array([0., 0.02272211, 0.])

    print('T diff', T - T_actual)
    print('H0 diff', H0 - H0_actual)
    print('S0 diff', S0 - S0_actual)
    print('Cp0 diff', Cp0 - Cp0_actual)
    print('HJ diff', HJ - HJ_actual)
    print('SJ diff', SJ - SJ_actual)
    print('CpJ diff', CpJ - CpJ_actual)
    print('n diff', n - n_actual)