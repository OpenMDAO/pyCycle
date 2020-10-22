""" Class definition for Combustor."""

import numpy as np

import openmdao.api as om

from pycycle.constants import AIR_FUEL_ELEMENTS, AIR_ELEMENTS

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea.species_data import Properties, janaf

from pycycle.elements.duct import PressureLoss
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough


class MixFuel(om.ExplicitComponent):
    """
    MixFuel calculates fuel and air mixture.
    """

    def initialize(self):
        self.options.declare('inflow_thermo_data', default=None,
                             desc='Thermodynamic data set for incoming flow. This only needs to be set if different thermo data is used for incoming flow and outgoing flow.', recordable=False)
        self.options.declare('thermo_data', default=janaf,
                             desc='Thermodynamic data set for flow. This is used for incoming and outgoing flow unless inflow_thermo_data is set, in which case it is used only for outgoing flow.', recordable=False)
        self.options.declare('inflow_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the flow')
        self.options.declare('fuel_type', default="JP-7",
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
        fuel_type = self.options['fuel_type']

        self.mixed_elements = inflow_elements.copy()
        self.mixed_elements.update(thermo_data.reactants[fuel_type]) #adds the fuel elements to the mix outflow

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
        self.add_input('Fl_I:FAR', val=0.0, desc='Fuel to air ratio')
        self.add_input('Fl_I:tot:h', val=0.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('Fl_I:tot:b0', val=inflow_thermo.b0, desc='incoming flow composition')
        self.add_input('fuel_Tt', val=518., units='degR', desc="fuel temperature")

        # outputs
        self.add_output('mass_avg_h', shape=1, units='Btu/lbm',
                        desc="mass flow rate averaged specific enthalpy")
        self.add_output('Wout', shape=1, units="lbm/s", desc="total massflow out")
        self.add_output('Wfuel', shape=1, units="lbm/s", desc="total fuel massflow out")
        self.add_output('b0_out', val=air_fuel_thermo.b0)

        for i, e in enumerate(self.air_fuel_elements): 
            self.init_fuel_amounts_base[i] = thermo_data.reactants[fuel_type].get(e, 0) * thermo_data.element_wts[e]

        self.init_fuel_amounts_base = self.init_fuel_amounts_base/sum(self.init_fuel_amounts_base)
       

        # create a mapping between the composition indices of the inflow and outflow arrays
        # which is basically a permutation matrix of ones resize the input to the output

        self.in_out_flow_idx_map = np.zeros((n_elements, len(self.inflow_elements)))
        for i,e in enumerate(self.inflow_elements): 
            j = self.air_fuel_elements.index(e)
            self.in_out_flow_idx_map[j,i] = 1.

        self.declare_partials('mass_avg_h', ['Fl_I:FAR', 'Fl_I:tot:h'])
        self.declare_partials('Wout', ['Fl_I:stat:W', 'Fl_I:FAR'])
        self.declare_partials('Wfuel', ['Fl_I:stat:W', 'Fl_I:FAR'])
        self.declare_partials('b0_out', ['Fl_I:FAR', 'Fl_I:tot:b0'])


    def compute(self, inputs, outputs):
        FAR = inputs['Fl_I:FAR']
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
        FAR = inputs['Fl_I:FAR']
        W = inputs['Fl_I:stat:W']
        ht = inputs['Fl_I:tot:h']
        Fl_I_tot_b0 = inputs['Fl_I:tot:b0']

        # AssertionError: 4.2991138611171866e-05 not less than or equal to 1e-05 : DESIGN.burner.mix_fuel: init_prod_amounts  w.r.t Fl_I:tot:n
        J['mass_avg_h', 'Fl_I:FAR'] = -ht/(1+FAR)**2 + self.fuel_ht/(1+FAR)**2  # - self.fuel_ht*FAR/(1+FAR)**2
        J['mass_avg_h', 'Fl_I:tot:h'] = 1.0/(1.0 + FAR)

        J['Wout', 'Fl_I:stat:W'] = (1.0 + FAR)
        J['Wout', 'Fl_I:FAR'] = W

        J['Wfuel', 'Fl_I:stat:W'] = FAR
        J['Wfuel', 'Fl_I:FAR'] = W

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
        J['b0_out', 'Fl_I:FAR'] = (-(self.init_air_amounts + init_fuel_amounts)/sum_is**2 *np.sum(dinit_fuel__dFAR)
                                   + dinit_fuel__dFAR/sum_is) / self.air_fuel_wt_mole

        for i in range(self.num_inflow_elements): 
            # print('bar', d_iam__db0[:,0]/sum_is - init_stuff[0]/sum_is**2 * np.sum(d_iam__db0[:,0]))
            J['b0_out', 'Fl_I:tot:b0'][:,i] = (d_iam__db0[:,i]/sum_is - init_stuff[i]/sum_is**2 * np.sum(d_iam__db0[:,i]))/self.air_fuel_wt_mole


class Combustor(om.Group):
    """
    A combustor that adds a fuel to an incoming flow mixture and burns it

    --------------
    Flow Stations
    --------------
    Fl_I
    Fl_O

    -------------
    Design
    -------------
        inputs
        --------
        Fl_I:FAR
        dPqP
        MN

        outputs
        --------
        Wfuel


    -------------
    Off-Design
    -------------
        inputs
        --------
        Fl_I:FAR
        dPqP
        area

        outputs
        --------
        Wfuel

    """

    def initialize(self):
        self.options.declare('inflow_thermo_data', default=None,
                             desc='Thermodynamic data set for incoming flow. This only needs to be set if different thermo data is used for incoming flow and outgoing flow.', recordable=False)
        self.options.declare('thermo_data', default=janaf,
                             desc='Thermodynamic data set for flow. This is used for incoming and outgoing flow unless inflow_thermo_data is set, in which case it is used only for outgoing flow.', recordable=False)
        self.options.declare('inflow_elements', default=AIR_ELEMENTS,
                             desc='set of elements present in the air flow')
        self.options.declare('air_fuel_elements', default=AIR_FUEL_ELEMENTS,
                             desc='set of elements present in the fuel')
        self.options.declare('design', default=True,
                             desc='Switch between on-design and off-design calculation.')
        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('fuel_type', default="JP-7",
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
        air_fuel_elements = self.options['air_fuel_elements']
        design = self.options['design']
        statics = self.options['statics']
        fuel_type = self.options['fuel_type']

        air_fuel_thermo = Properties(thermo_data, init_elements=air_fuel_elements)
        self.air_fuel_prods = air_fuel_thermo.products

        air_thermo = Properties(inflow_thermo_data, init_elements=inflow_elements)
        self.air_prods = air_thermo.products

        self.num_air_fuel_prod = len(self.air_fuel_prods)
        self.num_air_prod = air_thermo.num_prod
        num_air_element = air_thermo.num_element

        # Create combustor flow station
        in_flow = FlowIn(fl_name='Fl_I', num_prods=self.num_air_prod, num_elements=num_air_element)
        self.add_subsystem('in_flow', in_flow, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Perform combustor engineering calculations
        self.add_subsystem('mix_fuel',
                           MixFuel(inflow_thermo_data=inflow_thermo_data, thermo_data=thermo_data,
                                    inflow_elements=inflow_elements, fuel_type=fuel_type),
                           promotes=['Fl_I:stat:W','Fl_I:FAR', 'Fl_I:tot:b0', 'Fl_I:tot:h', 'Wfuel', 'Wout'])

        # Pressure loss
        prom_in = [('Pt_in', 'Fl_I:tot:P'),'dPqP']
        self.add_subsystem('p_loss', PressureLoss(), promotes_inputs=prom_in)

        # Calculate vitiated flow station properties
        vit_flow = Thermo(mode='total_hP', fl_name='Fl_O:tot', 
                          method='CEA', 
                          thermo_kwargs={'elements':air_fuel_elements, 
                                         'spec':thermo_data})
        self.add_subsystem('vitiated_flow', vit_flow, promotes_outputs=['Fl_O:*'])
        self.connect("mix_fuel.mass_avg_h", "vitiated_flow.h")
        self.connect("mix_fuel.b0_out", "vitiated_flow.b0")
        self.connect("p_loss.Pt_out","vitiated_flow.P")

        if statics:
            if design:
                # Calculate static properties.

                out_stat = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                                  method='CEA', 
                                  thermo_kwargs={'elements':air_fuel_elements, 
                                                 'spec':thermo_data})
                prom_in = ['MN']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect("mix_fuel.b0_out", "out_stat.b0")
                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')
                self.connect('Wout','out_stat.W')

            else:
                # Calculate static properties.
                out_stat = Thermo(mode='static_A', fl_name='Fl_O:stat', 
                                  method='CEA', 
                                  thermo_kwargs={'elements':air_fuel_elements, 
                                                 'spec':thermo_data})
                prom_in = ['area']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect("mix_fuel.b0_out", "out_stat.b0")

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')
                self.connect('Wout','out_stat.W')

        else:
            self.add_subsystem('W_passthru', PassThrough('Wout', 'Fl_O:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])

        self.add_subsystem('FAR_pass_thru', PassThrough('Fl_I:FAR', 'Fl_O:FAR', 0.0),
                           promotes=['*'])
        self.set_input_defaults('Fl_I:tot:n', units=None)


if __name__ == "__main__":

    p = om.Problem()
    p.model = om.Group()
    p.model.add_subsystem('comp', MixFuel(), promotes=['*'])

    p.model.add_subsystem('d1', om.IndepVarComp('Fl_I:stat:W', val=1.0, units='lbm/s', desc='weight flow'),
                          promotes=['*'])
    p.model.add_subsystem('d2', om.IndepVarComp('Fl_I:FAR', val=0.2, desc='Fuel to air ratio'), promotes=['*'])
    p.model.add_subsystem('d3', om.IndepVarComp('Fl_I:tot:h', val=1.0, units='Btu/lbm', desc='total enthalpy'),
                          promotes=['*'])
    p.model.add_subsystem('d4', om.IndepVarComp('fuel_Tt', val=518.0, units='degR', desc='fuel temperature'),
                          promotes=['*'])

    p.setup(check=False, force_alloc_complex=True)
    p.run_model()

    p.check_partials(compact_print=True, method='cs')
