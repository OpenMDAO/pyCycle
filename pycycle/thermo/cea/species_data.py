from collections import OrderedDict

import numpy as np
from scipy import interpolate

from pycycle import constants

from pycycle.thermo.cea.thermo_data import co2_co_o2
from pycycle.thermo.cea.thermo_data import janaf
from pycycle.thermo.cea.thermo_data import wet_air

#from ad.admath import log
from numpy import log
class Properties(object):
    """Compute H, S, Cp given a species and temperature"""
    
    def __init__(self, thermo_data_module, init_elements=None):

        self.a = None
        self.a_T = None
        self.b0 = None
        self.element_wt = None
        self.aij = None
        self.products = None
        self.elements = None
        self.temp_ranges = None
        self.valid_temp_range = None
        self.wt_mole = None # array of mole weights
        self.thermo_data_module = thermo_data_module
        self.prod_data = self.thermo_data_module.products
        self.init_elements = init_elements
        self.temp_base = None # array of lowest end of lowest temperature range

        if init_elements is not None :

            elem_set = set(init_elements.keys())
            self.elements = sorted(elem_set)


            valid_elements = set()

            for compound in self.prod_data.keys():
                valid_elements.update(self.prod_data[compound]['elements'].keys())
                
            for element in self.elements:
                if element not in valid_elements:
                    if element in self.prod_data.keys():
                        raise ValueError(f'The provided element `{element}` is a product in your provided thermo data, but is not an element.')
                    else:
                        raise ValueError(f'The provided element `{element}` is not used in any products in your thermo data.')

            self.products = [name for name, prod_data in self.prod_data.items()
                         if elem_set.issuperset(prod_data['elements'])]
        
        else:
                raise ValueError('You have not provided `init_elements`. In order to set thermodynamic data it must be provided.')

        ### setting up object attributes ###
        element_list = sorted(self.elements)

        self.num_element = len(element_list)
        self.num_prod = len(self.products)

        self.a = np.zeros((self.num_prod, 10))
        self.a_T = self.a.T

        element_wt = []
        aij = []

        for e in element_list:
            element_wt.append(self.thermo_data_module.element_wts[e])

            row = [self.prod_data[r]['elements'].get(e,0) for r in self.products]
            aij.append(row)

        self.element_wt = np.array(element_wt)
        self.aij = np.array(aij)

        self.wt_mole = np.empty(self.num_prod)
        for i,r in enumerate(self.products):
            self.wt_mole[i] = self.prod_data[r]['wt']

        #### pre-computed constants used in calculations ###
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

        #### Computing b0 values ###
        self.b0 = np.zeros(self.num_element)
        for i,e in enumerate(element_list): 
            self.b0[i] = self.init_elements[e]

        self.b0 = self.b0*self.element_wt
        self.b0 = self.b0/np.sum(self.b0)
        self.b0 = self.b0/self.element_wt

        self.build_coeff_table(999) # just pick arbitrary default temperature so there is something there right away
        

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

