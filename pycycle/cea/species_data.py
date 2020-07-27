import numpy as np
from scipy import interpolate

from collections import OrderedDict

from pycycle.cea.thermo_data import co2_co_o2
from pycycle.cea.thermo_data import janaf
from pycycle.cea.thermo_data import wet_air
from pycycle import constants

#from ad.admath import log
from numpy import log
class Thermo(object):
    """Compute H, S, Cp given a species and temperature"""
    def __init__(self, thermo_data_module, init_reacts=None, init_elements=None):

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
        # self.init_prod_amounts = None # concentrations (sum to 1)
        self.thermo_data_module = thermo_data_module
        self.prod_data = self.thermo_data_module.products
        self.init_reacts = init_reacts
        self.init_elements = init_elements
        self.temp_base = None # array of lowest end of lowest temperature range

        if init_elements is not None and init_reacts is None:

            self.elements = set(init_elements.keys())

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
                         if self.elements.issuperset(prod_data['elements'])]
        
        elif init_elements is None and init_reacts is not None:
            self.get_elements() #sets self.elements

            self.products = [name for name, prod_data in self.prod_data.items()
                         if self.elements.issuperset(prod_data['elements'])]
        
        elif init_elements is None and init_reacts is None:
                raise ValueError('You have not provided elements or initial reactants (init_reacts). In order to set thermodynamic data, one of the two must be provided.')
        
        else:
            raise ValueError('You have provided both elements and initial reactants (init_reacts). In order to set thermodynamic data, you must only provide one or the other.')       
        
        self.set_data(init_reacts)

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

    def set_data(self, init_reacts):
        """computes the relevant quantities, given the recatant data"""

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

        if init_reacts is not None:

            # These two methods return slightly different b0 values. Possibly could be due to numerical differences between 
            # the molar weight of the products and the molar weight of the elements. In method 1, the molar weight of the products is used, and 
            # in method 2 the molar weight of the elements is used. These weights are multiplied by their respective b0 or concentration values,
            # and then those multiplied results are summed together in order to normalize the total weight. Because of these mathematic operations,
            # some amount of numerical error could be introduced. That may be why test_mixer.py is failing (it has a very tight tolerance).
            # This hypothesis is supported by the fact that the more elements/products you include, the greater the deviation is between b0 calculated using methods 1 and 2.
            # Method number two is the desired method, and method number one is the method which causes all tests to pass. 


############## METHOD NUMBER ONE ##################################################
            # uncomment this section of code and comment method 2 to make all tests pass


            # # expand the init_reacts out to include all products, not just those provided
            # full_init_react = OrderedDict()
            # for p in self.prod_data:
            #     # initial amounts need to be given in mass ratios
            #     if p in init_reacts:
            #         full_init_react[p] = init_reacts[p] * self.prod_data[p]['wt']
            #     else:
            #         full_init_react[p] = 0.

            # init_reacts = full_init_react

            # init_prod_amounts = [init_reacts[name] for name in init_reacts
            #                      if self.elements.issuperset(self.prod_data[name]['elements'])]

            # init_prod_amounts = np.array(init_prod_amounts)
            # init_prod_amounts = init_prod_amounts/np.sum(init_prod_amounts) # normalize to 1kg of matter

            # init_prod_amounts /= self.wt_mole

            # self.b0 = np.sum(self.aij*init_prod_amounts, axis=1) #moles of each element per kg of mixture 

################ END METHOD NUMBER ONE ############################################################################




################ METHOD NUMBER TWO ###############################################################################
            # uncomment this section of code and comment method 1 to simplify code (one of the mixer tests will fail)

            self.b0 = self.get_b0()
            self.b0 = self.b0*self.element_wt
            self.b0 = self.b0/np.sum(self.b0)
            self.b0 = self.b0/self.element_wt  #moles of each element per kg of mixture 

################## END METHOD NUMBER TWO #############################################################

        else:

            self.b0 = list(self.init_elements.values())

            self.b0 = self.b0*self.element_wt
            self.b0 = self.b0/np.sum(self.b0)
            self.b0 = self.b0/self.element_wt

        self.build_coeff_table(999) # just pick arbitrary default temperature so there is something there right away

    def get_elements(self):#note, reactants is assumed to be a dictionary

        self.elements = set()

        for name in self.init_reacts: # figure out which elements are present
            if self.init_reacts[name] > 0:
                self.elements.update(self.prod_data[name]['elements'])

        return

    def get_b0(self):#note, reactants is assumed to be a dictionary

        element_list = sorted(self.elements)

        aij = []

        for e in element_list:

            row = [self.prod_data[react]['elements'].get(e,0) for react in self.init_reacts]
            aij.append(row)

        b_values = np.zeros((len(element_list))) #moles of each element based on provided reactant abundances

        for i, element in enumerate(element_list):
            for j, reactant in enumerate(self.init_reacts):
                b_values[i] += aij[i][j]*self.init_reacts[reactant]

        return(b_values)


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

    # thermo = Thermo(co2_co_o2, init_reacts=constants.co2_co_o2_init_prod_amounts)
    thermo = Thermo(janaf, init_reacts=constants.janaf_init_prod_amounts)
    # thermo = Thermo(janaf, init_reacts=constants.AIR_FUEL_MIX)

    T = np.ones(len(thermo.products))*800
    H0 = thermo.H0(T)
    S0 = thermo.S0(T)
    Cp0 = thermo.Cp0(T)

    HJ = thermo.H0_applyJ(T, 1.)
    SJ = thermo.S0_applyJ(T, 1)
    CpJ = thermo.Cp0_applyJ(T, 1)
    b0 = thermo.b0
    print('\nT', T)
    print('\nH0', H0)
    print('\nS0', S0)
    print('\nCp0', Cp0)
    print('\nHJ', HJ)
    print('\nSJ', SJ)
    print('\nCpJ', CpJ)
    print('\nb0', b0, '\n')
