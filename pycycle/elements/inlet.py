""" Class definition for Inlet."""

import openmdao.api as om

from pycycle.constants import AIR_ELEMENTS, g_c

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo

from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element

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


class Inlet(Element):
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
        self.options.declare('thermo_method', default='CEA', values=('CEA',),
                              desc='Method for computing thermodynamic properties')
        self.options.declare('thermo_data', default=species_data.janaf,
                              desc='thermodynamic data set', recordable=False)
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')
        self.options.declare('design', default=True,
                              desc='Switch between on-design and off-design calculation.')

        self.default_des_od_conns = [
            # (design src, off-design target)
            ('Fl_O:stat:area', 'area'),
        ]

    def pyc_setup_output_ports(self): 
        
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        statics = self.options['statics']
        design = self.options['design']

        # elements = self.options['elements']
        elements = self.Fl_I_data['Fl_I']

        num_element = len(elements)

        # Create inlet flow station
        flow_in = FlowIn(fl_name='Fl_I')
        self.add_subsystem('flow_in', flow_in, promotes=['Fl_I:tot:*', 'Fl_I:stat:*'])

        # Perform inlet engineering calculations
        self.add_subsystem('calcs_inlet', Calcs(),
                           promotes_inputs=['ram_recovery', ('Pt_in', 'Fl_I:tot:P'),
                                            ('V_in', 'Fl_I:stat:V'), ('W_in', 'Fl_I:stat:W')],
                           promotes_outputs=['F_ram'])

        # Calculate real flow station properties
        real_flow = Thermo(mode='total_TP', fl_name='Fl_O:tot', 
                           method=thermo_method, 
                           thermo_kwargs={'elements':elements, 
                                          'spec':thermo_data})
        self.add_subsystem('real_flow', real_flow,
                           promotes_inputs=[('T', 'Fl_I:tot:T'), ('composition', 'Fl_I:tot:composition')],
                           promotes_outputs=['Fl_O:*'])


        self.connect("calcs_inlet.Pt_out", "real_flow.P")

        if statics:
            if design:
                #   Calculate static properties

                out_stat = Thermo(mode='static_MN', fl_name='Fl_O:stat', 
                                  method=thermo_method, 
                                  thermo_kwargs={'elements':elements, 
                                                 'spec':thermo_data})
                self.add_subsystem('out_stat', out_stat,
                                   promotes_inputs=[('composition', 'Fl_I:tot:composition'), ('W', 'Fl_I:stat:W'), 'MN'],
                                   promotes_outputs=['Fl_O:stat:*'])

                self.connect('Fl_O:tot:S', 'out_stat.S')
                self.connect('Fl_O:tot:h', 'out_stat.ht')
                self.connect('Fl_O:tot:P', 'out_stat.guess:Pt')
                self.connect('Fl_O:tot:gamma', 'out_stat.guess:gamt')

            else:
                # Calculate static properties
                out_stat = Thermo(mode='static_A', fl_name='Fl_O:stat', 
                                  method=thermo_method, 
                                  thermo_kwargs={'elements':elements, 
                                                 'spec':thermo_data})
                prom_in = [('composition', 'Fl_I:tot:composition'),
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

        super().setup()
        
    def pyc_setup_thermo(self, upstream):
        elements = self.options['elements']
        
        self.Fl_O_data = {
            'Fl_O': upstream['Fl_I']
        }

if __name__ == "__main__":
    from pycycle import constants

    p = om.Problem()
    p.model = Inlet()

    thermo = species_data.Properties(species_data.janaf, constants.AIR_ELEMENTS)
    p.model.set_input_defaults('Fl_I:tot:T', 284, units='degK')
    p.model.set_input_defaults('Fl_I:tot:P', 5.0, units='lbf/inch**2')
    p.model.set_input_defaults('Fl_I:stat:V', 0.0, units='ft/s')#keep
    p.model.set_input_defaults('Fl_I:stat:W', 1, units='kg/s')

    p.setup()

    #view_model(p)
    p.run_model()
    # print(p.get_val('Fl_I:tot:T', units='degK'))
    p.model.list_outputs(units=True)

    # generates regression testing setup
    #regression_generator(p)
