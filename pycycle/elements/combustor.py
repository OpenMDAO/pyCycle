""" Class definition for Combustor."""

import numpy as np

import openmdao.api as om

from pycycle.constants import AIR_FUEL_ELEMENTS, AIR_ELEMENTS

from pycycle.thermo.thermo import Thermo
from pycycle.thermo.cea.thermo_add import ThermoAdd

from pycycle.thermo.cea.species_data import Properties, janaf

from pycycle.elements.duct import PressureLoss

from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough


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

        num_air_element = len(inflow_elements)

        # Create combustor flow station
        in_flow = FlowIn(fl_name='Fl_I')
        self.add_subsystem('in_flow', in_flow, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Perform combustor engineering calculations
        self.add_subsystem('mix_fuel',
                           ThermoAdd(inflow_thermo_data=inflow_thermo_data, mix_thermo_data=thermo_data,
                                    inflow_elements=inflow_elements, mix_elements=fuel_type),
                           promotes=['Fl_I:stat:W', ('mix:ratio', 'Fl_I:FAR'), 'Fl_I:tot:composition', 'Fl_I:tot:h', ('mix:W','Wfuel'), 'Wout'])

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
        self.connect("mix_fuel.composition_out", "vitiated_flow.composition")
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

                self.connect("mix_fuel.composition_out", "out_stat.composition")
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
                self.connect("mix_fuel.composition_out", "out_stat.composition")

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')
                self.connect('Wout','out_stat.W')

        else:
            self.add_subsystem('W_passthru', PassThrough('Wout', 'Fl_O:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])


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
