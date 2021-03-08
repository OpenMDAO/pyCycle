import numpy as np

import openmdao.api as om

from pycycle.constants import CEA_AIR_COMPOSITION
from pycycle.thermo.cea.species_data import Properties, janaf


class ThermoAdd(om.ExplicitComponent):
    """
    ThermoAdd calculates a new composition given inflow, a reactant to add, and a mix ratio.
    """

    def initialize(self):
        self.options.declare('spec', default=janaf,
                             desc=('Thermodynamic data set for flow.'), 
                             recordable=False)
        self.options.declare('inflow_composition', default=None,
                             desc='composition present in the flow')

        self.options.declare('mix_mode', values=['reactant', 'flow'], default='reactant')

        self.options.declare('mix_composition', default="JP-7",
                             desc='Type of fuel.', types=(dict, str, list, tuple))
        self.options.declare('mix_names', default='mix', types=(str, list, tuple))


    def output_port_data(self): 
        """
        Computes the thermo data for the mixed properties according to whatever options are configured
        """

        spec = self.options['spec']

        inflow_composition = self.options['inflow_composition']
        if inflow_composition is None: 
            inflow_composition = CEA_AIR_COMPOSITION
            
        mix_mode = self.options['mix_mode']

        mix_composition = self.options['mix_composition']
        if isinstance(mix_composition, (str, dict)): # cast it to tuple
            mix_composition = (mix_composition,)

        self.mix_composition = mix_composition

        mixed_flow_elements = inflow_composition.copy()
        if mix_mode == "reactant": # get the elements from the reactant dict in the spec
            for reactant in mix_composition: 
                mixed_flow_elements.update(spec.reactants[reactant]) #adds the fuel elements to the mix outflow
        else: # flow mode 
            for flow_elements in mix_composition: 
                mixed_flow_elements.update(flow_elements)

        self.mixed_elements = mixed_flow_elements

        return self.mixed_elements

    def setup(self):

        spec = self.options['spec']
        mix_mode = self.options['mix_mode']
        mix_names = self.options['mix_names']
        if isinstance(mix_names, str): # cast it to tuple 
            mix_names = (mix_names,)    
        self.mix_names = mix_names

        inflow_composition = self.options['inflow_composition']

        self.output_port_data()

        inflow_thermo = Properties(spec, init_elements=inflow_composition)
        self.inflow_composition = inflow_thermo.elements
        self.inflow_wt_mole = inflow_thermo.element_wt
        self.num_inflow_composition = len(self.inflow_composition)

        mixed_thermo = Properties(spec, init_elements=self.mixed_elements)
        self.mixed_elements = mixed_thermo.elements
        self.mixed_wt_mole = mixed_thermo.element_wt
        self.num_mixed_elements = len(self.mixed_elements)


        self.init_fuel_amounts_1kg = {}

        if mix_mode == 'reactant': 
            for reactant in self.mix_composition: 
                self.init_fuel_amounts_1kg[reactant] = np.zeros(mixed_thermo.num_element)
                ifa_1kg = self.init_fuel_amounts_1kg[reactant]
                for i, e in enumerate(self.mixed_elements): 
                    ifa_1kg[i] = spec.reactants[reactant].get(e, 0) * spec.element_wts[e]

                ifa_1kg[:] = ifa_1kg/sum(ifa_1kg) # make it 1 kg of fuel

        else: # flow 
            mix_b0 = {}
            self.mix_wt_mole = {}
            self.mix_out_flow_idx_maps = {}
            for name, elements in zip(mix_names, self.mix_composition): 
                thermo = Properties(spec, init_elements=elements)
                mix_b0[name] = thermo.b0
                self.mix_wt_mole[name] = thermo.element_wt

                # mapping matrix to convert mix to outflow
                self.mix_out_flow_idx_maps[name] = mix_map = np.zeros((mixed_thermo.num_element, thermo.num_element))
                for i,e in enumerate(thermo.elements): 
                    j = self.mixed_elements.index(e)
                    mix_map[j,i] = 1.


        # inputs
        self.add_input('Fl_I:stat:W', val=0.0, desc='weight flow', units='lbm/s')
        self.add_input('Fl_I:tot:h', val=0.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('Fl_I:tot:composition', val=inflow_thermo.b0, desc='incoming flow composition')
        
        for name in mix_names: 
            self.add_input(f'{name}:h', val=0.0, units='Btu/lbm', desc="reactant enthalpy")

            if mix_mode == 'reactant': 
                self.add_input(f'{name}:ratio', val=0.0, desc='reactant to air mass ratio')
                self.add_output(f'{name}:W', shape=1, units="lbm/s", desc="mix input massflow")

            else: 
                self.add_input(f'{name}:composition', val=mix_b0[name], desc='mix flow composition' )
                self.add_input(f'{name}:W', shape=1, units="lbm/s", desc="mix input massflow")


        # outputs
        self.add_output('mass_avg_h', shape=1, units='Btu/lbm',
                        desc="mass flow rate averaged specific enthalpy")
        self.add_output('Wout', shape=1, units="lbm/s", desc="total massflow out")
        self.add_output('composition_out', val=mixed_thermo.b0)

       
        # create a mapping between the composition indices of the inflow and outflow arrays
        # which is basically a permutation matrix of ones resize the input to the output

        self.in_out_flow_idx_map = np.zeros((mixed_thermo.num_element, inflow_thermo.num_element))
        for i,e in enumerate(self.inflow_composition): 
            j = self.mixed_elements.index(e)
            self.in_out_flow_idx_map[j,i] = 1.

        self.declare_partials('*', '*', method='cs')
        # NOTE: Due to some complexity from python vectorization, 
        # computing partials manually is tricky and will not be much faster than pure CS

    def compute(self, inputs, outputs):
        W = inputs['Fl_I:stat:W']
        Fl_I_tot_b0 = inputs['Fl_I:tot:composition']

        # copy the incoming flow into a correctly sized array for the outflow composition
        b0_out = self.in_out_flow_idx_map.dot(Fl_I_tot_b0)


        b0_out *= self.mixed_wt_mole # convert to mass units
        sum_b0_out = np.sum(b0_out)
        b0_out /= sum_b0_out # scale to 1 kg
        b0_out *= W  # scale to full mass flow

        
        mass_avg_h = inputs['Fl_I:tot:h'] * W
        W_out = W.copy()

        if self.options['mix_mode'] == 'reactant': 
            for name, reactant in zip(self.mix_names, self.mix_composition): 
                ratio = inputs[f'{name}:ratio']
                # compute the amount of fuel-flow rate in terms of the incoming mass-flow rate
                outputs[f'{name}:W'] = W_mix = W*ratio
                b0_out += self.init_fuel_amounts_1kg[reactant]*W_mix

                mass_avg_h += inputs[f'{name}:h'] * W_mix
                W_out += W_mix

        else: # inflow mixing
            for name in self.mix_names: 
                W_mix = inputs[f'{name}:W']
                mix_stuff = inputs[f'{name}:composition'].copy()
                mix_stuff *= self.mix_wt_mole[name]
                mix_stuff /= np.sum(mix_stuff) # normalize to 1kg 
                mix_stuff *= W_mix# scale to actual mass flow of that mix stream
                
                b0_out += self.mix_out_flow_idx_maps[name].dot(mix_stuff)
                W_out += W_mix

                mass_avg_h += inputs[f'{name}:h'] * W_mix

        b0_out /= np.sum(b0_out) # scale back to 1 kg
        outputs['composition_out'] = b0_out/self.mixed_wt_mole

        mass_avg_h /= W_out
        outputs['mass_avg_h'] = mass_avg_h
        outputs['Wout'] = W_out

