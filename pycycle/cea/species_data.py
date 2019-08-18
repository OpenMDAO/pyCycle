import numpy as np
from scipy import interpolate
from six.moves import range

from collections import OrderedDict

from pycycle.cea.thermo_data import co2_co_o2
from pycycle.cea.thermo_data import janaf

#from ad.admath import log
from numpy import log
class Thermo(object):
    """Compute H, S, Cp given a species and temperature"""
    def __init__(self, thermo_data_module, init_reacts=None):

        self.a = None
        self.a_T = None
        self.element_wt = None
        self.aij = None
        self.products = None
        self.elements = None
        self.temp_ranges = None
        self.valid_temp_range = None
        self.wt_mole = None # array of mole weights
        self.init_prod_amounts = None # concentrations (sum to 1)
        self.thermo_data_module = thermo_data_module
        self.init_reacts = init_reacts
        self.set_data(thermo_data_module, init_reacts)
        self._lastT = None

    def H0(self, Tt): # standard-state molar enthalpy for species j at temp T
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return (-a_T[0]/Tt**2 + a_T[1]/Tt*log(Tt) + a_T[2] + a_T[3]*Tt/2. + a_T[4]*Tt**2/3. + a_T[5]*Tt**3/4. + a_T[6]*Tt**4/5.+a_T[7]/Tt)

    def S0(self, Tt): # standard-state molar entropy for species j at temp T
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return (-a_T[0]/(2*Tt**2) - a_T[1]/Tt + a_T[2]*log(Tt) + a_T[3]*Tt + a_T[4]*Tt**2/2. + a_T[5]*Tt**3/3. + a_T[6]*Tt**4/4.+a_T[8])

    def Cp0(self, Tt): #molar heat capacity at constant pressure for
                    #standard state for species or reactant j, J/(kg-mole)_j(K)
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return a_T[0]/Tt**2 + a_T[1]/Tt + a_T[2] + a_T[3]*Tt + a_T[4]*Tt**2 + a_T[5]*Tt**3 + a_T[6]*Tt**4

    def H0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return vec*(2*a_T[0]/Tt**3 + a_T[1]*(1-log(Tt))/Tt**2 + a_T[3]/2. + 2*a_T[4]/3.*Tt + 3*a_T[5]/4.*Tt**2 + 4*a_T[6]/5.*Tt**3 - a_T[7]/Tt**2)

    def S0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return vec*(a_T[0]/(Tt**3) + a_T[1]/Tt**2 + a_T[2]/Tt + a_T[3] + a_T[4]*Tt + a_T[5]*Tt**2 + 4*a_T[6]/4.*Tt**3)

    def Cp0_applyJ(self, Tt, vec):
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        a_T = self.a_T
        return vec*(-2*a_T[0]/Tt**3 - a_T[1]/Tt**2 + a_T[3] + 2.*a_T[4]*Tt + 3.*a_T[5]*Tt**2 + 4.*a_T[6]*Tt**3)

    def set_data(self, thermo_data_module, init_reacts=None):
        """computes the relevant quantities, given the recatant data"""

        self._lastT = None
        self.thermo_data = thermo_data_module
        self.prod_data = thermo_data_module.products

        elements = set()

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

        for name in init_reacts: # figure out which elements are present
            if init_reacts[name] > 0:
                elements.update(self.prod_data[name]['elements'])

        init_prod_amounts = [init_reacts[name] for name in init_reacts
                             if elements.issuperset(self.prod_data[name]['elements'])]

        self.elements = sorted(elements)
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

        self.build_coeff_table(1000) # just pick arbitrary default temperature so there is something there right away

    def build_coeff_table(self, Tt):
        """Build the temperature specific coeff array and find the highest-low value and
        the lowest-high value of temperatures from all the reactants to give the
        valid range for the data fits."""

        max_low, min_high = -1e50, 1e50
        for i,p in enumerate(self.products):
            tr = self.prod_data[p]['ranges']

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


class ThermoSpline(object):
    """Drop-in replacement for Thermo using a spline fit to ensure smooth transitions of derivatives for all temperatures"""
    def __init__(self, thermo_data_module, init_reacts=None):

        self.a = None
        self.a_T = None
        self.element_wt = None
        self.aij = None
        self.products = None
        self.elements = None
        self.temp_ranges = None
        self.valid_temp_range = None
        self.wt_mole = None # array of mole weights
        self.init_prod_amounts = None # concentrations (sum to 1)
        self.S_spline = None # B-spline fit
        self.dS_spline = None # B-spline fit
        self.H_spline = None # B-spline fit
        self.dH_spline = None # B-spline fit
        self.Cp_spline = None # B-spline fit
        self.dCp_spline = None # B-spline fit
        self.Tt_fit = np.array([200.,300.,400.,700.,1100.,1500.,2000.,2500.,4000.,5000.,7000.,8500.,10000.,12500.,15000.,17500.,19000.,20000.])
        self.Tt_test = np.arange(200.,20000.,50.)
        self._lastT = None

        self.set_data(thermo_data_module, init_reacts)


    def H0(self, Tt):
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        H_fit, dH_fit = self.calc_H(self.Tt_fit,self.a)
        H_params = interpolate.splrep(self.Tt_fit,H_fit)
        H_predict = interpolate.splev(Tt,H_params)
        return float(H_predict)

    def S0(self, Tt): # standard-state molar entropy for species j at temp T
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        S_fit, dS_fit = self.calc_S(self.Tt_fit,self.a)
        S_params = interpolate.splrep(self.Tt_fit,S_fit)
        S_predict = interpolate.splev(Tt,S_params)
        return S_predict

    def Cp0(self, Tt): #molar heat capacity at constant pressure for
                    #standard state for species or reactant j, J/(kg-mole)_j(K)
        Tt = Tt[0]
        if Tt != self._lastT:
            self.build_coeff_table(Tt)
            self._lastT = Tt
        Cp_fit, dCp_fit = self.calc_Cp(self.Tt_fit,self.a)
        Cp_params = interpolate.splrep(self.Tt_fit,Cp_fit)
        Cp_predict = interpolate.splev(Tt,Cp_params)
        return Cp_predict

    def H0_applyJ(self, Tt, vec):
        H_fit, dH_fit = self.calc_H(self.Tt_fit,self.a)
        dH_params = interpolate.splrep(self.Tt_fit,dH_fit)
        dH_predict = interpolate.splev(Tt,dH_params)
        return dH_predict

    def S0_applyJ(self, Tt, vec):
        S_fit, dS_fit = self.calc_S(self.Tt_fit,self.a)
        dS_params = interpolate.splrep(self.Tt_fit,dS_fit)
        dS_predict = interpolate.splev(Tt,dS_params)
        return dS_predict

    def Cp0_applyJ(self, Tt, vec):
        Cp_fit, dCp_fit = self.calc_Cp(self.Tt_fit,self.a)
        dCp_params = interpolate.splrep(self.Tt_fit,dCp_fit)
        dCp_predict = interpolate.splev(Tt,dCp_params)
        return dCp_predict

    def calc_S(self,Tt, inputs):

        Tt_low = Tt[Tt<=1000]
        Tt_mid = Tt[Tt[Tt<=6000]>1000]
        Tt_high = Tt[Tt>6000]

        a = inputs[0]
        S_low = (-a[0]/(2*Tt_low**2) - a[1]/Tt_low + a[2]*log(Tt_low) + a[3]*Tt_low + a[4]*Tt_low**2/2. + a[5]*Tt_low**3/3. + a[6]*(Tt_low**4)/5. +a[8])
        dS_low = a[0]/(Tt_low**3)+a[1]/(Tt_low**2)+a[2]/Tt_low+a[3]+a[4]*Tt_low+a[5]*Tt_low**2+0.8*a[6]*Tt_low**3

        a = inputs[1]
        S_mid = (-a[0]/(2*Tt_mid**2) - a[1]/Tt_mid + a[2]*log(Tt_mid) + a[3]*Tt_mid + a[4]*Tt_mid**2/2. + a[5]*Tt_mid**3/3. + a[6]*(Tt_mid**4)/5. +a[8])
        dS_mid = a[0]/(Tt_mid**3)+a[1]/(Tt_mid**2)+a[2]/Tt_mid+a[3]+a[4]*Tt_mid+a[5]*Tt_mid**2+0.8*a[6]*Tt_mid**3

        a = inputs[2]
        S_high = (-a[0]/(2*Tt_high**2) - a[1]/Tt_high + a[2]*log(Tt_high) + a[3]*Tt_high + a[4]*Tt_high**2/2. + a[5]*Tt_high**3/3. + a[6]*(Tt_high**4)/5. +a[8])
        dS_high = a[0]/(Tt_high**3)+a[1]/(Tt_high**2)+a[2]/Tt_high+a[3]+a[4]*Tt_high+a[5]*Tt_high**2+0.8*a[6]*Tt_high**3

        S = np.concatenate((S_low,S_mid,S_high),0)
        dS = np.concatenate((dS_low,dS_mid,dS_high),0)
        return S, dS

    def calc_H(self,Tt, inputs):

        Tt_low = Tt[Tt<=1000]
        Tt_mid = Tt[Tt[Tt<=6000]>1000]
        Tt_high = Tt[Tt>6000]

        a = inputs[0]
        H_low = (-a[0]/Tt_low**2 + a[1]/Tt_low*log(Tt_low) + a[2] + a[3]*Tt_low/2. + a[4]*Tt_low**2/3. + a[5]*Tt_low**3/4. + a[6]*Tt_low**4/5.+a[7]/Tt_low)
        dH_low = (2*a[0]/Tt_low**3 + a[1]*(1-log(Tt_low))/Tt_low**2 + a[3]/2. + 2*a[4]/3.*Tt_low + 3*a[5]/4.*Tt_low**2 + 4*a[6]/5.*Tt_low**3 - a[7]/Tt_low**2)

        a = inputs[1]
        H_mid = (-a[0]/Tt_mid**2 + a[1]/Tt_mid*log(Tt_mid) + a[2] + a[3]*Tt_mid/2. + a[4]*Tt_mid**2/3. + a[5]*Tt_mid**3/4. + a[6]*Tt_mid**4/5.+a[7]/Tt_mid)
        dH_mid = (2*a[0]/Tt_mid**3 + a[1]*(1-log(Tt_mid))/Tt_mid**2 + a[3]/2. + 2*a[4]/3.*Tt_mid + 3*a[5]/4.*Tt_mid**2 + 4*a[6]/5.*Tt_mid**3 - a[7]/Tt_mid**2)

        a = inputs[2]
        H_high = (-a[0]/Tt_high**2 + a[1]/Tt_high*log(Tt_high) + a[2] + a[3]*Tt_high/2. + a[4]*Tt_high**2/3. + a[5]*Tt_high**3/4. + a[6]*Tt_high**4/5.+a[7]/Tt_high)
        dH_high = (2*a[0]/Tt_high**3 + a[1]*(1-log(Tt_high))/Tt_high**2 + a[3]/2. + 2*a[4]/3.*Tt_high + 3*a[5]/4.*Tt_high**2 + 4*a[6]/5.*Tt_high**3 - a[7]/Tt_high**2)

        H = np.concatenate((H_low,H_mid,H_high),0)
        dH = np.concatenate((dH_low,dH_mid,dH_high),0)
        return H, dH

    def calc_Cp(self,Tt, inputs):

        Tt_low = Tt[Tt<=1000]
        Tt_mid = Tt[Tt[Tt<=6000]>1000]
        Tt_high = Tt[Tt>6000]

        a = inputs[0]
        Cp_low = a[0]/Tt_low**2 + a[1]/Tt_low + a[2] + a[3]*Tt_low + a[4]*Tt_low**2 + a[5]*Tt_low**3 + a[6]*Tt_low**4
        dCp_low = (-2*a[0]/Tt_low**3 - a[1]/Tt_low**2 + a[3] + 2.*a[4]*Tt_low + 3.*a[5]*Tt_low**2 + 4.*a[6]*Tt_low**3)

        a = inputs[1]
        Cp_mid = a[0]/Tt_mid**2 + a[1]/Tt_mid + a[2] + a[3]*Tt_mid + a[4]*Tt_mid**2 + a[5]*Tt_mid**3 + a[6]*Tt_mid**4
        dCp_mid = (-2*a[0]/Tt_mid**3 - a[1]/Tt_mid**2 + a[3] + 2.*a[4]*Tt_mid + 3.*a[5]*Tt_mid**2 + 4.*a[6]*Tt_mid**3)

        a = inputs[2]
        Cp_high = a[0]/Tt_high**2 + a[1]/Tt_high + a[2] + a[3]*Tt_high + a[4]*Tt_high**2 + a[5]*Tt_high**3 + a[6]*Tt_high**4
        dCp_high = (-2*a[0]/Tt_high**3 - a[1]/Tt_high**2 + a[3] + 2.*a[4]*Tt_high + 3.*a[5]*Tt_high**2 + 4.*a[6]*Tt_high**3)

        Cp = np.concatenate((Cp_low,Cp_mid,Cp_high),0)
        dCp = np.concatenate((dCp_low,dCp_mid,dCp_high),0)
        return Cp, dCp

    def set_data(self, thermo_data_module, init_reacts=None):
        """computes the relevant quantities, given the recatant data"""

        self._lastT = None
        self.thermo_data = thermo_data_module
        self.prod_data = thermo_data_module.products

        elements = set()

        if init_reacts is None:
            init_reacts = thermo_data_module.init_prod_amounts

        #expand the init_reacts out to include all products, not just those provided
        full_init_react = OrderedDict()
        for p in self.prod_data.keys():
            r_amount = init_reacts.get(p,0) * self.prod_data[p]['wt'] # initial amounts need to be given in mass ratios
            full_init_react[p] = r_amount

        init_reacts = full_init_react

        for name, amount in init_reacts.items(): # figure out which elements are present
            if amount > 0:
                elements = elements.union(elements, self.prod_data[name]['elements'].keys())

        init_prod_amounts = []
        for name, amount in init_reacts.items(): # only include products with all elements present
            prod_elements = set(self.prod_data[name]['elements'].keys())
            if prod_elements.issubset(elements):
                init_prod_amounts.append(init_reacts[name])

        self.elements = list(elements)
        self.init_prod_amounts = np.array(init_prod_amounts)
        self.init_prod_amounts = self.init_prod_amounts/np.sum(self.init_prod_amounts) # normalize to 1

        reduced_products = []
        for name, prod_data in self.prod_data.items():
            prod_elements = set(prod_data['elements'].keys())

            if prod_elements.issubset(elements):
                reduced_products.append(name)
        self.products = reduced_products

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

        self.init_prod_amounts = self.init_prod_amounts / self.wt_mole

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

        self.build_coeff_table(1000) # just pick arbitrary default temperature so there is something there right away

    def build_coeff_table(self, Tt):
        """Build the temperature specific coeff array and find the highest-low value and
        the lowest-high value of temperatures from all the reactants to give the
        valid range for the data fits."""

        max_low, min_high = -1e50, 1e50
        for i,p in enumerate(self.products):
            tr = self.prod_data[p]['ranges']

            j = np.searchsorted(tr, Tt)
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
            self.a[i][:len(data)] = data # have to slice because some rows are 9 long and others 10g

        self.valid_temp_range = (max_low, min_high)
