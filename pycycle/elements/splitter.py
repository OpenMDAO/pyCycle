import numpy as np

import openmdao.api as om

from pycycle.constants import AIR_MIX
from pycycle.flow_in import FlowIn
from pycycle.cea import species_data
from pycycle.cea.set_total import SetTotal
from pycycle.cea.set_static import SetStatic
from pycycle.passthrough import PassThrough


class BPRcalc(om.ExplicitComponent):
    """Calculates flow split"""

    def setup(self):

        self.add_input('W_in', 1.0, desc='total weight flow in', units='lbm/s')
        self.add_input('BPR', 1.5, desc='ratio of mass flow in Fl_O2 to Fl_O1')

        self.add_output('W1', 0.44, desc='weight flow for Fl_O1', units='lbm/s')
        self.add_output('W2', 0.56, desc='Weight flow for Fl_O2', units='lbm/s')

        self.declare_partials('*', '*')

        self.default_des_od_conns = [
            # (design src, off-design target)
            ('Fl_O1:stat:area', 'area1'), 
            ('Fl_O2:stat:area', 'area2'), 
        ]    



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


class Splitter(om.Group):
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
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('elements', default=AIR_MIX,
                              desc='set of elements present in the flow')
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

    def setup(self):

        thermo_data = self.options['thermo_data']
        elements = self.options['elements']
        statics = self.options['statics']
        design = self.options['design']

        num_prod = species_data.Thermo(thermo_data, init_reacts=elements).num_prod

        # Create inlet flowstation
        flow_in = FlowIn(fl_name='Fl_I', num_prods=num_prod)
        self.add_subsystem('flow_in', flow_in, promotes_inputs=('Fl_I:*',))

        # Split the flows
        self.add_subsystem('split_calc', BPRcalc(), promotes_inputs=('BPR', ('W_in', 'Fl_I:stat:W')))

        # Set Fl_out1 totals based on T, P
        real_flow1 = SetTotal(thermo_data=thermo_data, mode='T',
                             init_reacts=elements, fl_name="Fl_O1:tot")
        self.add_subsystem('real_flow1', real_flow1,
                           promotes_inputs=(('init_prod_amounts', 'Fl_I:tot:n'),
                                            ('P', 'Fl_I:tot:P'),
                                            ('T', 'Fl_I:tot:T')),
                           promotes_outputs=('Fl_O1:tot:*', ))

        # Set Fl_out2 totals based on T, P
        real_flow2 = SetTotal(thermo_data=thermo_data, mode='T',
                             init_reacts=elements, fl_name="Fl_O2:tot")
        self.add_subsystem('real_flow2', real_flow2, promotes_inputs=(('init_prod_amounts', 'Fl_I:tot:n'),
                                            ('P', 'Fl_I:tot:P'),
                                            ('T', 'Fl_I:tot:T')),
                           promotes_outputs=('Fl_O2:tot:*', ))

        if statics:
            if design:
            #   Calculate static properties
                out1_stat = SetStatic(mode="MN", thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O1:stat")
                prom_in = [('init_prod_amounts', 'Fl_I:tot:n'),
                           ('MN','MN1')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out1_stat', out1_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O1:tot:S', 'out1_stat.S')
                self.connect('Fl_O1:tot:h', 'out1_stat.ht')
                self.connect('Fl_O1:tot:P', 'out1_stat.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out1_stat.guess:gamt')
                self.connect('split_calc.W1', 'out1_stat.W')

                out2_stat = SetStatic(mode="MN", thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O2:stat")
                prom_in = [('init_prod_amounts', 'Fl_I:tot:n'),
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
                out1_stat = SetStatic(mode="area", thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O1:stat")
                prom_in = [('init_prod_amounts', 'Fl_I:tot:n'),
                           ('area','area1')]
                prom_out = ['Fl_O1:stat:*']
                self.add_subsystem('out1_stat', out1_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)
                self.connect('Fl_O1:tot:S', 'out1_stat.S')
                self.connect('Fl_O1:tot:h', 'out1_stat.ht')
                self.connect('Fl_O1:tot:P', 'out1_stat.guess:Pt')
                self.connect('Fl_O1:tot:gamma', 'out1_stat.guess:gamt')
                self.connect('split_calc.W1', 'out1_stat.W')

                out2_stat = SetStatic(mode="area", thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O2:stat")
                prom_in = [('init_prod_amounts', 'Fl_I:tot:n'),
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


if __name__ == "__main__":

    p = om.Problem()
    des_vars = p.model.add_subsystem('des_vars', om.IndepVarComp(), promotes=['*'])
    des_vars.add_output('W_in', 1.0, units='lbm/s')
    des_vars.add_output('BPR', 1.5, units=None)

    p.model.add_subsystem('comp', BPRcalc(), promotes=['*'])


    p.setup()
    p.run_model()
    p.check_partials()