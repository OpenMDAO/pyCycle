""" Class definition for Inlet."""

import openmdao.api as om

from pycycle.cea import species_data
from pycycle.cea.set_static import SetStatic
from pycycle.cea.set_total import SetTotal
from pycycle.constants import AIR_FUEL_MIX, AIR_MIX, g_c
from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough

#from pycycle.elements.test.util import regression_generator

class Calcs(om.ExplicitComponent):
    """
    Performs inlet engineering calculations.
    """

    def setup(self):
        # inputs
        self.add_input('Pt_in', val=5.0, units='lbf/inch**2', desc='Entrance total pressure')
        self.add_input('ram_recovery', val=1.0, desc='Ram recovery')
        self.add_input('V_in', val=0.0, units='ft/s', desc='Entrance velocity')
        self.add_input('W_in', val=100.0, units='lbm/s', desc='Entrance flow rate')

        # outputs
        self.add_output('Pt_out', val=14.696, units='lbf/inch**2', desc='Exit total pressure')
        self.add_output('F_ram', val=1.0, units='lbf', desc='Ram drag')

        self.declare_partials('Pt_out', ['Pt_in', 'ram_recovery'])
        self.declare_partials('F_ram', ['V_in', 'W_in'])

    def compute(self, inputs, outputs):
        outputs['Pt_out'] = inputs['Pt_in'] * inputs['ram_recovery']
        outputs['F_ram'] = inputs['W_in'] * inputs['V_in'] / g_c

    def compute_partials(self, inputs, J):
        J['Pt_out', 'Pt_in'] = inputs['ram_recovery']
        J['Pt_out', 'ram_recovery'] = inputs['Pt_in']
        J['F_ram', 'V_in'] = inputs['W_in'] / g_c
        J['F_ram', 'W_in'] = inputs['V_in'] / g_c


class Inlet(om.Group):
    """
    Calculates ram-drag and exit flow conditions for an Inlet with
    specified MN (design) or area (off-design)

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
        ram_recovery
        MN

        outputs
        --------
        F_ram

    -------------
    Off-Design
    -------------
        inputs
        --------
        area

        outputs
        --------
        F_ram
    """

    def initialize(self):
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set')
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

        gas_thermo = species_data.Thermo(thermo_data, init_reacts=elements)
        gas_prods = gas_thermo.products
        num_prod = len(gas_prods)

        # Create inlet flow station
        flow_in = FlowIn(fl_name='Fl_I', num_prods=num_prod)
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Perform inlet engineering calculations
        self.add_subsystem('calcs_inlet', Calcs(),
                           promotes_inputs=['ram_recovery', ('Pt_in', 'Fl_I:tot:P'),
                                            ('V_in', 'Fl_I:stat:V'), ('W_in', 'Fl_I:stat:W')],
                           promotes_outputs=['F_ram'])

        # Calculate real flow station properties
        real_flow = SetTotal(thermo_data=thermo_data, mode="T", init_reacts=elements, fl_name="Fl_O:tot")

        self.add_subsystem('real_flow', real_flow,
                           promotes_inputs=[('T', 'Fl_I:tot:T'), ('init_prod_amounts', 'Fl_I:tot:n')],
                           promotes_outputs=['Fl_O:*'])
        self.connect("calcs_inlet.Pt_out", "real_flow.P")

        self.add_subsystem('FAR_passthru', PassThrough('Fl_I:FAR', 'Fl_O:FAR', 0.0), promotes=['*'])

        if statics:
            if design:
                #   Calculate static properties
                self.add_subsystem('out_stat', SetStatic(mode="MN", thermo_data=thermo_data, init_reacts=elements, fl_name="Fl_O:stat"),
                                   promotes_inputs=[('init_prod_amounts', 'Fl_I:tot:n'), ('W', 'Fl_I:stat:W'), 'MN'],
                                   promotes_outputs=['Fl_O:stat:*'])

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            else:
                # Calculate static properties
                out_stat = SetStatic(mode="area", thermo_data=thermo_data, init_reacts=elements,
                                         fl_name="Fl_O:stat")
                prom_in = [('init_prod_amounts', 'Fl_I:tot:n'),
                           ('W', 'Fl_I:stat:W'),
                           'area']
                prom_out = ['Fl_O:stat:*']
                self.add_subsystem('out_stat', out_stat, promotes_inputs=prom_in,
                                   promotes_outputs=prom_out)

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

        else:
            self.add_subsystem('W_passthru', PassThrough('Fl_I:stat:W', 'Fl_O:stat:W', 0.0, units= "lbm/s"),
                               promotes=['*'])


if __name__ == "__main__":

    p = om.Problem()
    p.model = Inlet()

    p.setup()

    #view_model(p)
    p.run_model()

    # generates regression testing setup
    #regression_generator(p)
