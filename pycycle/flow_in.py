""" FlowIN component which serves as an input flowstation for cycle components.
"""
import numpy as np

import openmdao.api as om 


class FlowIn(om.ExplicitComponent):
    """
    Provides a central place to connect flow information to in a component
    but doesn't actually do anything on its own
    """

    def initialize(self):
        self.options.declare('fl_name', default='flow',
                              desc='thermodynamic data set')

    def setup(self):
        fl_name = self.options['fl_name']

        self.add_output('foo', val=1.,
            desc="dummy output that is NOT used for anything other than to keep the framework happy. ")

        self.add_input('%s:tot:h'%fl_name, val=1.0, desc='total enthalpy', units='Btu/lbm')
        self.add_input('%s:tot:T'%fl_name, val=518., desc='total temperature', units='degR')
        self.add_input('%s:tot:P'%fl_name, val=1., desc='total pressure', units='lbf/inch**2')
        self.add_input('%s:tot:rho'%fl_name, val=1.0, desc='total density', units='lbm/ft**3')
        self.add_input('%s:tot:gamma'%fl_name, val=1.4, desc='total gamma')
        self.add_input('%s:tot:Cp'%fl_name, val=1.0, desc='total Specific heat at constant pressure', units='Btu/(lbm*degR)')
        self.add_input('%s:tot:Cv'%fl_name, val=1.0, desc='total Specific heat at constant volume', units='Btu/(lbm*degR)')
        self.add_input('%s:tot:S'%fl_name, val=1.0, desc='total entropy', units='Btu/(lbm*degR)')
        self.add_input('%s:tot:R'%fl_name, val=1.0, desc='total gas constant', units='Btu/(lbm*degR)')
        self.add_input('%s:tot:composition'%fl_name, shape_by_conn=True, desc='flow composition vector')

        self.add_input('%s:stat:h'%fl_name, val=1.0, desc='static enthalpy', units='Btu/lbm')
        self.add_input('%s:stat:T'%fl_name, val=518., desc='static temperature', units='degR')
        self.add_input('%s:stat:P'%fl_name, val=1.0, desc='static pressure', units='lbf/inch**2')
        self.add_input('%s:stat:rho'%fl_name, val=1.0, desc='static density', units='lbm/ft**3')
        self.add_input('%s:stat:gamma'%fl_name, val=1.4, desc='static gamma')
        self.add_input('%s:stat:Cp'%fl_name, val=1.0, desc='static Specific heat at constant pressure', units='Btu/(lbm*degR)')
        self.add_input('%s:stat:Cv'%fl_name, val=1.0, desc='static Specific heat at constant volume', units='Btu/(lbm*degR)')
        self.add_input('%s:stat:S'%fl_name, val= 0.0, desc='static entropy', units='Btu/(lbm*degR)')
        self.add_input('%s:stat:R'%fl_name, val=1.0, desc='static gas constant', units='Btu/(lbm*degR)')
        self.add_input('%s:stat:composition'%fl_name, shape_by_conn=True, desc='flow composition vector')

        # TODO takes these out of static (keep them top level)
        self.add_input('%s:stat:V'%fl_name, val=1.0, desc='Velocity', units='ft/s')
        self.add_input('%s:stat:Vsonic'%fl_name, val=1.0, desc='Speed of sound', units='ft/s')
        self.add_input('%s:stat:MN'%fl_name, val=1.0, desc='Mach number')
        self.add_input('%s:stat:area'%fl_name, val=1.0, desc='flow area', units='inch**2')
        self.add_input('%s:stat:Wc'%fl_name, val=1.0, desc='corrected weight flow', units='lbm/s')
        self.add_input('%s:stat:W'%fl_name, val= 0.0, desc='weight flow', units='lbm/s')
        self.add_input('%s:FAR'%fl_name, val=0.0, desc='fuel to air ratio')
        # self.add_input('%s:WAR'%fl_name, val  = 0.0, desc='water to air ratio')
        # self.add_input('%s:nu', %nameval=1.0, desc='dynamic viscosity', units='lbm/(s*ft)')

    def compute(self, inputs, outputs):
        pass
