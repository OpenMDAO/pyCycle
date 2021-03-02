""" Class definition for Inlet."""

import openmdao.api as om

from pycycle.constants import g_c

from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo

from pycycle.flow_in import FlowIn
from pycycle.passthrough import PassThrough
from pycycle.element_base import Element

#from pycycle.elements.test.util import regression_generator

class MilSpecRecovery(om.ExplicitComponent):
    """
    Performs subsonic, supersonic, and hypsersonic inlet ram recovery calculations.
    """

    def setup(self):
        # inputs
        self.add_input('MN', val=0.5, units=None, desc='Flight Mach Number')
        self.add_input('ram_recovery_base', units=None, desc='Base Inlet Ram Recovery')
        
        # outputs
        self.add_output('ram_recovery', val=1.0, units=None, desc='Mil Spec Ram Recovery')
        
        self.declare_partials('ram_recovery', ['ram_recovery_base' ,'MN'])

    def compute(self, inputs, outputs):
        
        MN = inputs['MN']
        ram_recovery_base = inputs['ram_recovery_base']
            
        if MN < 1.0:
            outputs['ram_recovery'] = ram_recovery_base
            
        elif MN >= 1.0 and MN < 5.0:
            outputs['ram_recovery'] = ram_recovery_base *(1-(0.075*((MN-1)**1.35)))
            
        elif MN >= 5.0:
            outputs['ram_recovery'] = 800/(MN**4 + 935)

    def compute_partials(self, inputs, J):
        
        MN = inputs['MN']
        ram_recovery_base = inputs['ram_recovery_base']
                
        if MN < 1.0:
            J['ram_recovery', 'ram_recovery_base'] = 1
            J['ram_recovery', 'MN'] = 0
            
        elif MN >= 1.0 and MN < 5.0:
            J['ram_recovery', 'ram_recovery_base'] = 1-(0.075*((MN-1)**1.35))
            J['ram_recovery', 'MN'] = -0.10125*ram_recovery_base*((MN-1)**0.35)
            
        elif MN >= 5.0:
            J['ram_recovery', 'ram_recovery_base'] = 0
            J['ram_recovery', 'MN'] = -(3200 * MN**3)/((MN**4 + 935)**2)

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
        self.options.declare('statics', default=True,
                              desc='If True, calculate static properties.')

        self.default_des_od_conns = [
            # (design src, off-design target)
            ('Fl_O:stat:area', 'area'),
        ]

        super().initialize()

    def pyc_setup_output_ports(self): 
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        statics = self.options['statics']
        design = self.options['design']

        # elements = self.options['elements']
        composition = self.Fl_I_data['Fl_I']

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
                           thermo_kwargs={'composition':composition, 
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
                                  thermo_kwargs={'composition':composition, 
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
                                  thermo_kwargs={'composition':composition, 
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

