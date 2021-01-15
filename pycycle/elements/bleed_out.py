""" Class definition for a BleedOut."""

import numpy as np
from collections.abc import Iterable

import openmdao.api as om 

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element

class BleedCalcs(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('bleed_names', types=Iterable, desc='list of names for the bleed ports')

    def setup(self):
        self.add_input('W_in', val=30.0, units='lbm/s', desc='entrance mass flow')
        self.add_output('W_out', shape=1, units='lbm/s', desc='exit mass flow', res_ref=1e2)

        # bleed inputs and outputs
        for BN in self.options['bleed_names']:
            self.add_input(BN+':frac_W', val=0.0, desc='bleed mass flow fraction (W_bld/W_in)')
            self.add_output(BN+':stat:W', shape=1, units='lbm/s', desc='bleed mass flow', res_ref=1e2)

            self.declare_partials(BN+':stat:W', ['W_in', BN+':frac_W'])

        self.declare_partials('W_out', ['W_in', '*:frac_W'])

    def compute(self, inputs, outputs):

        # calculate flow and power without bleed flows
        outputs['W_out'] = inputs['W_in']

        # calculate bleed specific outputs and modify exit flow and power
        for BN in self.options['bleed_names']:
            outputs[BN+':stat:W'] = inputs['W_in'] * inputs[BN+':frac_W']
            outputs['W_out'] -= outputs[BN+':stat:W']

    def compute_partials(self, inputs, J):

        # Jacobian elements without bleed flows
        J['W_out','W_in'] = 1.0

        for BN in self.options['bleed_names']:
            J['W_out','W_in'] -= inputs[BN+':frac_W']
            J['W_out',BN+':frac_W'] = -inputs['W_in']

            J[BN+':stat:W','W_in'] = inputs[BN+':frac_W']
            J[BN+':stat:W',BN+':frac_W'] = inputs['W_in']


class BleedOut(Element):
    """
    bleed extration from the incomming flow

    --------------
    Flow Stations
    --------------
    Fl_I -> primary input flow
    Fl_O -> primary output flow
    Fl_{bleed_name} -> bleed output flows
        one for each name in `bleed_names` option

    -------------
    Design
    -------------
        inputs
        --------
        {bleed_name}:frac_W
            fraction of incoming flow to bleed off to FL_{bleed_name}
        MN

    -------------
    Off-Design
    -------------
        inputs
        --------
        area
    """

    def initialize(self):
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('bleed_names', types=(list,tuple), desc='list of names for the bleed ports',
                              default=[])
        
        self.default_des_od_conns = [
            # (design src, off-design target)
            ('Fl_O:stat:area', 'area')
        ]

        super().initialize()

    def pyc_setup_output_ports(self): 
        self.copy_flow('Fl_I', 'Fl_O')

        for b_name in self.options['bleed_names']: 
            self.copy_flow('Fl_I', b_name)

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        statics = self.options['statics']
        design = self.options['design']
        bleeds = self.options['bleed_names']
        composition = self.Fl_I_data['Fl_I']

        # Create inlet flowstation
        flow_in = FlowIn(fl_name='Fl_I')
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Bleed flow calculations
        blds = BleedCalcs(bleed_names=bleeds)
        bld_port_globs = [f'{bn}:*' for bn in bleeds]
        self.add_subsystem('bld_calcs', blds,
                           promotes_inputs=[('W_in', 'Fl_I:stat:W'), '*:frac_W'],
                           promotes_outputs=['W_out']+bld_port_globs)

        bleed_names = []
        for BN in bleeds:

            bleed_names.append(BN+'_flow')
            bleed_flow = Thermo(mode='total_TP', fl_name=BN+":tot", 
                                method=thermo_method, 
                                thermo_kwargs={'composition':composition, 
                                               'spec':thermo_data})
            self.add_subsystem(BN+'_flow', bleed_flow,
                               promotes_inputs=[('composition', 'Fl_I:tot:composition'),('T','Fl_I:tot:T'),('P','Fl_I:tot:P')],
                               promotes_outputs=['{}:tot:*'.format(BN)])

        # Total Calc
        real_flow = Thermo(mode='total_TP', fl_name="Fl_O:tot", 
                           method=thermo_method, 
                           thermo_kwargs={'composition':composition, 
                                          'spec':thermo_data})
        prom_in = [('composition', 'Fl_I:tot:composition'),('T','Fl_I:tot:T'),('P','Fl_I:tot:P')]
        self.add_subsystem('real_flow', real_flow, promotes_inputs=prom_in,
                           promotes_outputs=['Fl_O:*'])

        if statics:
            if design:
            #   Calculate static properties
                out_stat = Thermo(mode='static_MN', fl_name="Fl_O:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':composition, 
                                                 'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           'MN']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')
                self.connect('W_out', 'out_stat.W')

            else:
                # Calculate static properties
                out_stat = Thermo(mode='static_A', fl_name="Fl_O:stat", 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':composition, 
                                                 'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           'area']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')
                self.connect('W_out', 'out_stat.W')
        else:
            self.add_subsystem('W_passthru', PassThrough('W_out', 'Fl_O:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])

        super().setup()

if __name__ == "__main__":

    p = om.Problem()

    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    des_vars.add_output('Fl_I:stat:W', 60.0, units='lbm/s')
    des_vars.add_output('test1:frac_W', 0.05, units=None)
    des_vars.add_output('test2:frac_W', 0.05, units=None)
    des_vars.add_output('Fl_I:tot:T', 518.67, units='degR')
    des_vars.add_output('Fl_I:tot:P', 14.696, units='psi')
    des_vars.add_output('MN', 0.25)

    p.model.add_subsystem('bleed', BleedOut(design=True, statics=True, bleed_names=['test1','test2']), promotes=['*'])

    p.setup(check=False)
    p.run_model()

    print('W',p['Fl_I:stat:W'],p['Fl_O:stat:W'],p['test1:stat:W'],p['test2:stat:W'])
    print('T',p['Fl_I:tot:T'],p['Fl_O:tot:T'],p['test1:tot:T'],p['test2:tot:T'])
    print('P',p['Fl_I:tot:P'],p['Fl_O:tot:P'],p['test1:tot:P'],p['test2:tot:P'])
    # p.check_partials()

