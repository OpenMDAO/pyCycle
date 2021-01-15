""" Class definition for Combustor."""

import numpy as np

import openmdao.api as om

from pycycle.thermo.thermo import Thermo, ThermoAdd

from pycycle.thermo.cea.species_data import Properties, janaf

from pycycle.elements.duct import PressureLoss

from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element



class Combustor(Element):
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

        self.options.declare('statics', default=True,
                             desc='If True, calculate static properties.')
        self.options.declare('fuel_type', default="JP-7",
                             desc='Type of fuel.')

        super().initialize()

    def pyc_setup_output_ports(self): 

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        fuel_type = self.options['fuel_type']


        self.thermo_add_comp = ThermoAdd(method=thermo_method, mix_mode='reactant',
                                         thermo_kwargs={'spec':thermo_data,
                                                        'inflow_composition':self.Fl_I_data['Fl_I'], 
                                                        'mix_composition':fuel_type})
        
        # self.Fl_O_data['Fl_O'] = self.thermo_add_comp.output_port_data()
        self.copy_flow(self.thermo_add_comp, 'Fl_O')

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']

        inflow_composition = self.Fl_I_data['Fl_I']
        air_fuel_composition = self.Fl_O_data['Fl_O']
        design = self.options['design']
        statics = self.options['statics']


        # Create combustor flow station
        in_flow = FlowIn(fl_name='Fl_I')
        self.add_subsystem('in_flow', in_flow, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        self.add_subsystem('mix_fuel', self.thermo_add_comp,
                           promotes=['Fl_I:stat:W', ('mix:ratio', 'Fl_I:FAR'), 'Fl_I:tot:composition', 'Fl_I:tot:h', ('mix:W','Wfuel'), 'Wout'])

        # Pressure loss
        prom_in = [('Pt_in', 'Fl_I:tot:P'),'dPqP']
        self.add_subsystem('p_loss', PressureLoss(), promotes_inputs=prom_in)

        # Calculate vitiated flow station properties
        vit_flow = Thermo(mode='total_hP', fl_name='Fl_O:tot', 
                          method=thermo_method, 
                          thermo_kwargs={'composition':air_fuel_composition, 
                                         'spec':thermo_data})
        self.add_subsystem('vitiated_flow', vit_flow, promotes_outputs=['Fl_O:*'])
        self.connect("mix_fuel.mass_avg_h", "vitiated_flow.h")
        self.connect("mix_fuel.composition_out", "vitiated_flow.composition")
        self.connect("p_loss.Pt_out","vitiated_flow.P")

        if statics:
            if design:
                # Calculate static properties.

                out_stat = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':air_fuel_composition, 
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
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':air_fuel_composition, 
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


        super().setup()


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
