import numpy as np

import openmdao.api as om

from pycycle.flow_in import FlowIn
from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element


class BPRcalc(om.ExplicitComponent):
    """Calculates flow split"""

    def setup(self):

        self.add_input('W_in', 1.0, desc='total weight flow in', units='lbm/s')
        self.add_input('BPR', 1.5, desc='ratio of mass flow in Fl_O2 to Fl_O1')

        self.add_output('W1', 0.44, desc='weight flow for Fl_O1', units='lbm/s')
        self.add_output('W2', 0.56, desc='Weight flow for Fl_O2', units='lbm/s')

        self.declare_partials('*', '*')
 

    def compute(self, inputs, outputs):
        BPR = inputs['BPR']
        outputs['W1'] = inputs['W_in']/(BPR+1)
        outputs['W2'] = inputs['W_in'] - outputs['W1']

    def compute_partials(self, inputs, J):

        BPR = inputs['BPR']
        W = inputs['W_in']
        J['W1','BPR'] = -W/((BPR+1)**2)
        J['W1','W_in'] = 1.0/(BPR+1)
        J['W2','BPR'] = W/((BPR + 1)**2)
        J['W2','W_in'] = 1.0 - 1.0/(BPR+1)#BPR / (BPR + 1.0)


class Splitter(Element):
    """
    Splits a single incomming flow into two outgoing flows

    --------------
    Flow Stations
    --------------
    Fl_I
    Fl_O1
    Fl_O2

    -------------
    Design
    -------------
        inputs
        --------
        BPR
        MN1
        MN2

        outputs
        --------

    -------------
    Off-Design
    -------------
        inputs
        --------
        BPR
        area1
        area2

        outputs
        --------

    """

    def initialize(self):
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')

        self.default_des_od_conns = [
            ('Fl_O1:stat:area', 'area1'),
            ('Fl_O2:stat:area', 'area2')
        ]

        super().initialize()

    def pyc_setup_output_ports(self): 
        
        self.copy_flow('Fl_I', 'Fl_O1')
        self.copy_flow('Fl_I', 'Fl_O2')


    def setup(self):

        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        statics = self.options['statics']
        design = self.options['design']

        composition = self.Fl_I_data['Fl_I']

        # Create inlet flowstation
        flow_in = FlowIn(fl_name='Fl_I')
        self.add_subsystem('flow_in', flow_in, promotes_inputs=('Fl_I:*',))

        # Split the flows
        self.add_subsystem('split_calc', BPRcalc(), promotes_inputs=('BPR', ('W_in', 'Fl_I:stat:W')))

        # Set Fl_out1 totals based on T, P
        real_flow1 = Thermo(mode='total_TP', fl_name='Fl_O1:tot', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':composition, 
                                          'spec':thermo_data})
        self.add_subsystem('real_flow1', real_flow1,
                           promotes_inputs=(('composition', 'Fl_I:tot:composition'),
                                            ('P', 'Fl_I:tot:P'),
                                            ('T', 'Fl_I:tot:T')),
                           promotes_outputs=('Fl_O1:tot:*', ))

        # Set Fl_out2 totals based on T, P
        real_flow2 = Thermo(mode='total_TP', fl_name='Fl_O2:tot', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':composition, 
                                          'spec':thermo_data})
        self.add_subsystem('real_flow2', real_flow2, promotes_inputs=(('composition', 'Fl_I:tot:composition'),
                                            ('P', 'Fl_I:tot:P'),
                                            ('T', 'Fl_I:tot:T')),
                           promotes_outputs=('Fl_O2:tot:*', ))

        if statics:
            if design:
            #   Calculate static properties
                out1_stat = Thermo(mode='static_MN', fl_name='Fl_O1:stat', 
                                   method=thermo_method, 
                                   thermo_kwargs={'composition':composition, 
                                                  'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           ('MN','MN1')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out1_stat', out1_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O1:tot:S', 'out1_stat.S')
                self.connect('Fl_O1:tot:h', 'out1_stat.ht')
                self.connect('Fl_O1:tot:P', 'out1_stat.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out1_stat.guess:gamt')
                self.connect('split_calc.W1', 'out1_stat.W')

                out2_stat = Thermo(mode='static_MN', fl_name='Fl_O2:stat', 
                                   method=thermo_method, 
                                   thermo_kwargs={'composition':composition, 
                                                  'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           ('MN','MN2')]
                prom_out = ['Fl_O2:stat:*']
                self.add_subsystem('out2_stat', out2_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O2:tot:S', 'out2_stat.S')
                self.connect('Fl_O2:tot:h', 'out2_stat.ht')
                self.connect('Fl_O2:tot:P', 'out2_stat.guess:Pt')
                self.connect('Fl_O2:tot:gamma', 'out2_stat.guess:gamt')
                self.connect('split_calc.W2', 'out2_stat.W')

            else:
                # Calculate static properties
                out1_stat = Thermo(mode='static_A', fl_name='Fl_O1:stat', 
                                   method=thermo_method, 
                                   thermo_kwargs={'composition':composition, 
                                                  'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           ('area','area1')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out1_stat', out1_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O1:tot:S', 'out1_stat.S')
                self.connect('Fl_O1:tot:h', 'out1_stat.ht')
                self.connect('Fl_O1:tot:P', 'out1_stat.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out1_stat.guess:gamt')
                self.connect('split_calc.W1', 'out1_stat.W')

                out2_stat = Thermo(mode='static_A', fl_name='Fl_O2:stat', 
                                   method=thermo_method, 
                                   thermo_kwargs={'composition':composition, 
                                                  'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
                           ('area','area2')]
                prom_out = ['Fl_O2:stat:*']
                self.add_subsystem('out2_stat', out2_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O2:tot:S', 'out2_stat.S')
                self.connect('Fl_O2:tot:h', 'out2_stat.ht')
                self.connect('Fl_O2:tot:P', 'out2_stat.guess:Pt')
                self.connect('Fl_O2:tot:gamma', 'out2_stat.guess:gamt')
                self.connect('split_calc.W2', 'out2_stat.W')

        else:
            self.add_subsystem('W1_passthru', PassThrough('split_calc_W1', 'Fl_O1:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])
            self.add_subsystem('W2_passthru', PassThrough('split_calc_W2', 'Fl_O2:stat:W', 1.0, units= "lbm/s"),
                               promotes=['*'])
            self.connect('split_calc.W1', 'split_calc_W1')
            self.connect('split_calc.W2', 'split_calc_W2')

        super().setup()

if __name__ == "__main__":

    p = om.Problem()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    des_vars.add_output('W_in', 1.0, units='lbm/s')
    des_vars.add_output('BPR', 1.5, units=None)

    p.model.add_subsystem('comp', BPRcalc(), promotes=['*'])


    p.setup()
    p.run_model()
    p.check_partials()