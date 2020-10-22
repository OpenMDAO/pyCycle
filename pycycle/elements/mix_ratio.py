import numpy as np

import openmdao.api as om

from pycycle.constants import AIR_ELEMENTS
from pycycle.thermo.cea.species_data import Properties, janaf

class MixRatio(om.ExplicitComponent):
    """
    MixRatio calculates a new b0 given inflow, a reactant to add, and a mix ratio.
    """

    def initialize(self):
        self.options.declare('inflow_thermo_data', default=None,
                             desc='Thermodynamic data set for incoming flow. This only needs to be set if different thermo data is used for incoming flow and outgoing flow.', recordable=False)
        self.options.declare('thermo_data', default=janaf,
                             desc='Thermodynamic data set for flow. This is used for incoming and outgoing flow unless inflow_thermo_data is set, in which case it is used only for outgoing flow.', recordable=False)
        self.options.declare('inflow_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the flow')
        self.options.declare('mix_reactant', default="JP-7",
                             desc='Type of fuel.')

    def setup(self):
        thermo_data = self.options['thermo_data']
        if self.options['inflow_thermo_data'] is not None:
            # Set the inflow thermodynamic data package if it is different from the outflow one
            inflow_thermo_data = self.options['inflow_thermo_data']

        else:
            # Set the inflow thermodynamic data package if it is the same as the outflow one
            inflow_thermo_data = thermo_data

        inflow_elements = self.options['inflow_elements']
        mix_reactant = self.options['mix_reactant']

        self.mixed_elements = inflow_elements.copy()
        self.mixed_elements.update(thermo_data.reactants[mix_reactant]) #adds the fuel elements to the mix outflow

        inflow_thermo = Properties(inflow_thermo_data, init_elements=inflow_elements)
        self.inflow_elements = inflow_thermo.elements
        self.inflow_wt_mole = inflow_thermo.wt_mole
        self.num_inflow_elements = len(self.inflow_elements)

        air_fuel_thermo = Properties(thermo_data, init_elements=self.mixed_elements)
        self.air_fuel_prods = air_fuel_thermo.products
        self.air_fuel_elements = air_fuel_thermo.elements
        self.air_fuel_wt_mole = air_fuel_thermo.element_wt
        self.aij = air_fuel_thermo.aij

        self.num_prod = n_prods = len(self.air_fuel_prods)
        self.num_elements = n_elements = len(self.air_fuel_elements)

        self.init_air_amounts = np.zeros(n_elements)
        self.init_fuel_amounts = np.zeros(n_elements)
        self.init_fuel_amounts_base = np.zeros(n_elements)

        # inputs
        self.add_input('Fl_I:stat:W', val=0.0, desc='weight flow', units='lbm/s')
        self.add_input('mix_ratio', val=0.0, desc='reactant to air mass ratio')
        self.add_input('Fl_I:tot:h', val=0.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('Fl_I:tot:b0', val=inflow_thermo.b0, desc='incoming flow composition')
        self.add_input('reactant_Tt', val=518., units='degR', desc="fuel temperature")

        # outputs
        self.add_output('mass_avg_h', shape=1, units='Btu/lbm',
                        desc="mass flow rate averaged specific enthalpy")
        self.add_output('Wout', shape=1, units="lbm/s", desc="total massflow out")
        self.add_output('Wfuel', shape=1, units="lbm/s", desc="total fuel massflow out")
        self.add_output('b0_out', val=air_fuel_thermo.b0)

        for i, e in enumerate(self.air_fuel_elements): 
            self.init_fuel_amounts_base[i] = thermo_data.reactants[mix_reactant].get(e, 0) * thermo_data.element_wts[e]

        self.init_fuel_amounts_base = self.init_fuel_amounts_base/sum(self.init_fuel_amounts_base)
       

        # create a mapping between the composition indices of the inflow and outflow arrays
        # which is basically a permutation matrix of ones resize the input to the output

        self.in_out_flow_idx_map = np.zeros((n_elements, len(self.inflow_elements)))
        for i,e in enumerate(self.inflow_elements): 
            j = self.air_fuel_elements.index(e)
            self.in_out_flow_idx_map[j,i] = 1.

        self.declare_partials('mass_avg_h', ['mix_ratio', 'Fl_I:tot:h'])
        self.declare_partials('Wout', ['Fl_I:stat:W', 'mix_ratio'])
        self.declare_partials('Wfuel', ['Fl_I:stat:W', 'mix_ratio'])
        self.declare_partials('b0_out', ['mix_ratio', 'Fl_I:tot:b0'])


    def compute(self, inputs, outputs):
        FAR = inputs['mix_ratio']
        W = inputs['Fl_I:stat:W']
        Fl_I_tot_b0 = inputs['Fl_I:tot:b0']

        if inputs._under_complex_step:
            self.init_air_amounts = self.init_air_amounts.astype(np.complex)
        else:
            self.init_air_amounts = self.init_air_amounts.real

        # copy the incoming flow into a correctly sized array for the outflow composition
        self.init_air_amounts = self.in_out_flow_idx_map.dot(Fl_I_tot_b0)

        self.init_air_amounts *= self.air_fuel_wt_mole
        self.init_air_amounts /= np.sum(self.init_air_amounts)
        self.init_air_amounts *= W  # convert to kg and scale with mass flow


        # compute the amount of fuel-flow rate in terms of the incoming mass-flow rate
        init_fuel_amounts = self.init_fuel_amounts_base * W * FAR


        init_stuff = (self.init_air_amounts + init_fuel_amounts)
        init_stuff /= np.sum(init_stuff)

        outputs['b0_out'] = init_stuff/self.air_fuel_wt_mole

        self.fuel_ht = 0  # makes ht happy

        outputs['mass_avg_h'] = (inputs['Fl_I:tot:h']+FAR*self.fuel_ht)/(1+FAR)

        outputs['Wout'] = W * (1+FAR)

        outputs['Wfuel'] = W * FAR

    def compute_partials(self, inputs, J):
        FAR = inputs['mix_ratio']
        W = inputs['Fl_I:stat:W']
        ht = inputs['Fl_I:tot:h']
        Fl_I_tot_b0 = inputs['Fl_I:tot:b0']

        # AssertionError: 4.2991138611171866e-05 not less than or equal to 1e-05 : DESIGN.burner.mix_fuel: init_prod_amounts  w.r.t Fl_I:tot:n
        J['mass_avg_h', 'mix_ratio'] = -ht/(1+FAR)**2 + self.fuel_ht/(1+FAR)**2  # - self.fuel_ht*FAR/(1+FAR)**2
        J['mass_avg_h', 'Fl_I:tot:h'] = 1.0/(1.0 + FAR)

        J['Wout', 'Fl_I:stat:W'] = (1.0 + FAR)
        J['Wout', 'mix_ratio'] = W

        J['Wfuel', 'Fl_I:stat:W'] = FAR
        J['Wfuel', 'mix_ratio'] = W

        # for i, j in enumerate(self.in_out_flow_idx_map):
        #     self.init_air_amounts[j] = Fl_I_tot_b0[i]
        self.init_air_amounts = self.in_out_flow_idx_map.dot(Fl_I_tot_b0)

        self.init_air_amounts *= self.air_fuel_wt_mole
        # iam => init_air_amounts
        sum_iam = np.sum(self.init_air_amounts)
        d_iam0__db0 = self.in_out_flow_idx_map.dot(np.ones(4))*self.air_fuel_wt_mole

        term1 = -self.init_air_amounts/sum_iam**2
        d_iam1__db0 = np.einsum('i,j->ij', term1, d_iam0__db0).dot(self.in_out_flow_idx_map)
        d_iam0__db0 = np.einsum('i,ij->ij', d_iam0__db0, self.in_out_flow_idx_map)
        
        self.init_air_amounts /= sum_iam

        self.init_air_amounts *= W  # convert to kg and scale with mass flow
        d_iam__db0 = (d_iam0__db0 + d_iam1__db0)*W

        init_fuel_amounts = self.init_fuel_amounts_base * W * FAR

        init_stuff = (self.init_air_amounts + init_fuel_amounts)
        sum_is = np.sum(init_stuff)
   
        dinit_fuel__dFAR = self.init_fuel_amounts_base * W # check
        J['b0_out', 'mix_ratio'] = (-(self.init_air_amounts + init_fuel_amounts)/sum_is**2 *np.sum(dinit_fuel__dFAR)
                                   + dinit_fuel__dFAR/sum_is) / self.air_fuel_wt_mole

        for i in range(self.num_inflow_elements): 
            # print('bar', d_iam__db0[:,0]/sum_is - init_stuff[0]/sum_is**2 * np.sum(d_iam__db0[:,0]))
            J['b0_out', 'Fl_I:tot:b0'][:,i] = (d_iam__db0[:,i]/sum_is - init_stuff[i]/sum_is**2 * np.sum(d_iam__db0[:,i]))/self.air_fuel_wt_mole
