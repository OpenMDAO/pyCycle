""" Class definition for Nozzle."""

import openmdao.api as om

from pycycle.constants import g_c
from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import Thermo
from pycycle.flow_in import FlowIn
from pycycle.element_base import Element

class PR_bal(om.ImplicitComponent):

    """
    Balance which varies the nozzle pressure ratio until the calculated exhaust static pressure matches
    the inputs exhaust static pressure.  This balance is needed to handle situations where the nozzle
    pressure ratio (Pt/Ps) drops below 1.
    """

    def setup(self):
        self.add_input('Ps_exhaust', val=5.0, units='lbf/inch**2', desc='Exhaust static pressure')
        self.add_input('Ps_calc', val=5.0, units='lbf/inch**2', desc='Calculated exhaust static pressure')

        self.add_output('PR', val=2.0, lower=1.000001, units=None, desc='Total-to-staic pressure ratio')

        self.declare_partials('PR', 'Ps_exhaust', val=1.0)
        self.declare_partials('PR', 'Ps_calc', val=-1.0)

    def apply_nonlinear(self, inputs, outputs, resids):
        resids['PR'] = inputs['Ps_exhaust'] - inputs['Ps_calc']

    def linearize(self, inputs, outputs, J):
        pass


class PressureCalcs(om.ExplicitComponent):
    """
    Performs pressure calculations to get throat conditions.
    """

    def setup(self):
        # inputs
        self.add_input('Pt_in', val=10.0, units='lbf/inch**2', desc='Entrance total pressure')
        self.add_input('PR', val=2.0, desc='Total-to-staic pressure ratio')
        self.add_input('dPqP', val=0.0, desc='Total pressure loss from inlet to throat')

        # outputs
        self.add_output('Pt_th', shape=1, units='lbf/inch**2', desc='Throat total pressure', lower=1e-3)
        self.add_output('Ps_calc', val=5.0, units='lbf/inch**2', desc='Calculated exhaust static pressure')

        self.declare_partials('Pt_th', ['Pt_in', 'dPqP'])
        self.declare_partials('Ps_calc', ['Pt_in', 'PR', 'dPqP'])

    def compute(self, inputs, outputs):
        outputs['Pt_th'] = inputs['Pt_in'] * (1. - inputs['dPqP'])
        outputs['Ps_calc'] = outputs['Pt_th'] / inputs['PR']

    def compute_partials(self, inputs, J):
        J['Pt_th', 'Pt_in'] = 1. - inputs['dPqP']
        J['Pt_th', 'dPqP'] = -inputs['Pt_in']
        J['Ps_calc', 'Pt_in'] = (1. - inputs['dPqP']) / inputs['PR']
        J['Ps_calc', 'PR'] = -inputs['Pt_in'] * (1. - inputs['dPqP']) * inputs['PR']**-2.
        J['Ps_calc', 'dPqP'] = -inputs['Pt_in'] / inputs['PR']


class PerformanceCalcs(om.ExplicitComponent):
    """
    Performs performance calculations for the nozzle.
    """

    def initialize(self):
        self.options.declare('lossCoef', default='Cfg',
                              desc='If set to "Cfg", then Gross Thrust Coefficient is an input.')

    def setup(self):
        lossCoef = self.options['lossCoef']

        if not (lossCoef=="Cfg" or lossCoef=="Cv"):
            raise ValueError("lossCoef must be 'Cfg' or 'Cv', but '{}' was given.".format(lossCoef))

        # input
        self.add_input('W_in', val=1.0, units='lbm/s', desc='incoming Mass flow rate')
        self.add_input('Ps_calc', val=5.0, units='lbf/inch**2', desc='Exhaust static pressure')
        self.add_input('V_ideal', val=10.0, units='ft/s', desc='Ideal exit velocity')
        # self.add_input('A_ideal', val=1.0, units='inch**2', desc='Ideal exit area')

        if lossCoef == 'Cfg':
            self.add_input('Cfg', val=1.0, desc='Gross thrust coefficient')
        else:
            self.add_input('Cv', val=1.0, desc='Velocity coefficient')
            self.add_input('Cang', val=1.0, desc='Angle coefficient')
            self.add_input('CmixCorr', val=1.0, desc='Mix efficiency coefficient')
            self.add_input('V_actual', val=10.0, units='ft/s', desc='Actual exit velocity')
            self.add_input('A_actual', val=1.0, units='inch**2', desc='Actal exit area')
            self.add_input('Ps_actual', val=5.0, units='lbf/inch**2', desc='Actual exit static pressure')

        # output
        self.add_output('Fg_ideal', val=12000.0, shape=1, units='lbf', desc='Ideal gross thrust', ref=1e2, res_ref=1e3)
        self.add_output('Fg', val=11800.0, shape=1, units='lbf', desc='Gross thrust', ref=1e2, res_ref=1e3)

        self.declare_partials('Fg_ideal', ['W_in', 'V_ideal'])
        if lossCoef == 'Cfg':
            # self.declare_partials('Fg', ['W_in', 'V_ideal', 'Ps_calc', 'A_ideal', 'Cfg'])
            self.declare_partials('Fg', ['W_in', 'V_ideal', 'Ps_calc', 'Cfg'])
        else:
            self.declare_partials('Fg', ['W_in', 'V_actual', 'Cv', 'Cang', 'CmixCorr', 'Ps_actual', 'Ps_calc', 'A_actual'])


    def compute(self, inputs, outputs):
        lossCoef = self.options['lossCoef']

        # Calculate nozzle performance parameters
        outputs['Fg_ideal'] = (inputs['W_in'] / g_c) * inputs['V_ideal']

        if lossCoef == 'Cfg':
            outputs['Fg'] = outputs['Fg_ideal'] * inputs['Cfg']
        else:
            outputs['Fg'] = (inputs['W_in'] / g_c) * inputs['V_actual'] * inputs['Cv'] * inputs['Cang'] * inputs['CmixCorr'] + \
                            (inputs['Ps_actual'] - inputs['Ps_calc']) * inputs['A_actual']

    def compute_partials(self, inputs, J):
        lossCoef = self.options['lossCoef']

        J['Fg_ideal', 'W_in'] = 1./ g_c * inputs['V_ideal']
        J['Fg_ideal', 'V_ideal'] = inputs['W_in'] / g_c

        if lossCoef == 'Cfg':
            J['Fg', 'W_in'] = 1./ g_c * inputs['V_ideal'] * inputs['Cfg']
            J['Fg', 'V_ideal'] = inputs['W_in'] / g_c * inputs['Cfg']
            J['Fg', 'Cfg'] = (inputs['W_in'] / g_c) * inputs['V_ideal']
        else:
            J['Fg', 'W_in'] = 1./g_c * inputs['V_actual'] * inputs['Cv'] * inputs['Cang'] * inputs['CmixCorr']
            J['Fg', 'V_actual'] = (inputs['W_in'] / g_c) * inputs['Cv'] * inputs['Cang'] * inputs['CmixCorr']
            J['Fg', 'Cv'] = (inputs['W_in'] / g_c) * inputs['V_actual'] * inputs['Cang'] * inputs['CmixCorr']
            J['Fg', 'Cang'] = (inputs['W_in'] / g_c) * inputs['V_actual'] * inputs['Cv'] * inputs['CmixCorr']
            J['Fg', 'CmixCorr'] = (inputs['W_in'] / g_c) * inputs['V_actual'] * inputs['Cv'] * inputs['Cang']
            J['Fg', 'Ps_actual'] = inputs['A_actual']
            J['Fg', 'Ps_calc'] = -inputs['A_actual']
            J['Fg', 'A_actual'] = inputs['Ps_actual'] - inputs['Ps_calc']


class Mux(om.ExplicitComponent):
    """
    Determines the appropriate throat and nozzle exit flow properties.
    """

    def initialize(self):
        self.options.declare('nozzType', default='CV',
                              desc='Nozzle type: CD, CV, or CD_CV.')
        self.options.declare('fl_out_name', default='Fl_O',
                              desc='Outflow station prefix.')

    def setup(self):
        nozzType = self.options['nozzType']
        fl_out_name = self.options['fl_out_name']

        if not (nozzType in ["CV", "CD", "CD_CV"]):
            msg = "nozzType must be 'CV', 'CD' or 'CD_CV', but '{}' was given.".format(nozzType)
            raise ValueError(msg)

        # input
        self.add_input('Ps_calc', val=5.0, units='lbf/inch**2', desc='Exhaust static pressure')
        self.add_input('S', val=0.0, desc='entropy', units='Btu/(lbm*degR)')

        for prefix in ('Ps', 'MN'):
            self.add_input('%s:h' % prefix, val=0.0, desc='static enthalpy', units='Btu/lbm')
            self.add_input('%s:T' % prefix, val=0.0, desc='static temperature', units='degR')
            self.add_input('%s:P' % prefix, val=0.0, desc='static pressure', units='lbf/inch**2')
            self.add_input('%s:rho' % prefix, val=0.0, desc='static density', units='lbm/ft**3')
            self.add_input('%s:gamma' % prefix, val=0.0, desc='static gamma')
            self.add_input('%s:V' % prefix, val=0.0, desc='Velocity', units='ft/s')
            self.add_input('%s:Vsonic' % prefix, val=0.0, desc='Speed of sound', units='ft/s')
            self.add_input('%s:MN' % prefix, val=0.0, desc='Mach number')
            self.add_input('%s:area' % prefix, val=0.0, desc='Flow area', units='inch**2')
            self.add_input('%s:Cp' % prefix, val=0.0, desc='specific heat at constant pressure', units='Btu/(lbm*degR)')
            self.add_input('%s:Cv' % prefix, val=0.0, desc='specific heat at constant volume', units='Btu/(lbm*degR)')
            self.add_input('%s:W' % prefix, val=0.0, desc='Mass flow rate', units='lbm/s')

        # output
        self.add_output('choked', shape=1, desc='Flag for choked flow')

        for prefix in ('Throat', fl_out_name):
            self.add_output('%s:stat:h' % prefix, shape=1, desc='static enthalpy', units='Btu/lbm')
            self.add_output('%s:stat:T' % prefix, shape=1, desc='static temperature', units='degR')
            self.add_output('%s:stat:P' % prefix, shape=1, desc='static pressure', units='lbf/inch**2')
            self.add_output('%s:stat:rho' % prefix, shape=1, desc='static density', units='lbm/ft**3')
            self.add_output('%s:stat:gamma' % prefix, shape=1, desc='static gamma')
            self.add_output('%s:stat:S' % prefix, shape=1, desc='entropy', units='Btu/(lbm*degR)')
            self.add_output('%s:stat:Cp' % prefix, shape=1, desc='specific heat at constant pressure', units='Btu/(lbm*degR)')
            self.add_output('%s:stat:Cv' % prefix, shape=1, desc='specific heat at constant volume', units='Btu/(lbm*degR)')
            self.add_output('%s:stat:V' % prefix, shape=1, desc='Velocity', units='ft/s')
            self.add_output('%s:stat:Vsonic' % prefix, shape=1, desc='Speed of sound', units='ft/s')
            self.add_output('%s:stat:MN' % prefix, shape=1, desc='Mach number')
            self.add_output('%s:stat:area' % prefix, shape=1, desc='Flow area', units='inch**2')
            self.add_output('%s:stat:W' % prefix, shape=1, desc='Mass Flow Rate', units='lbm/s')

        self.flow_out = ['h', 'T', 'P', 'rho', 'gamma', 'Cp', 'Cv', 'V', 'Vsonic', 'MN', 'area', 'W']

        self.declare_partials('*:stat:h', '*:h')
        self.declare_partials('*:stat:T', '*:T')
        self.declare_partials('*:stat:P', '*:P')
        self.declare_partials('*:stat:rho', '*:rho')
        self.declare_partials('*:stat:gamma', '*:gamma')
        self.declare_partials('*:stat:S', 'S')
        self.declare_partials('*:stat:Cp', '*:Cp')
        self.declare_partials('*:stat:Cv', '*:Cv')
        self.declare_partials('*:stat:V', '*:V')
        self.declare_partials('*:stat:Vsonic', '*:Vsonic')
        self.declare_partials('*:stat:MN', '*:MN')
        self.declare_partials('*:stat:area', '*:area')
        self.declare_partials('*:stat:W', '*:W')

    def compute(self, inputs, outputs):
        nozzType = self.options['nozzType']
        fl_out_name = self.options['fl_out_name']

        # Determine if nozzle is choked and pass appropriate flow parameters to nozzle exit
        if nozzType == "CV":
            if inputs['Ps_calc'] < inputs['MN:P']:
                prefix = "MN"
            else:
                prefix = "Ps"

            for p in self.flow_out:
                outputs['Throat:stat:%s' %p] = inputs['%s:%s' %(prefix, p)]
                outputs['%s:stat:%s' %(fl_out_name, p)] = inputs['%s:%s' %(prefix, p)]

            outputs['Throat:stat:S'] = inputs['S']
            outputs['%s:stat:S' %fl_out_name] = inputs['S']

        elif nozzType == "CD":
            for p in self.flow_out:
                outputs['Throat:stat:%s' %p] = inputs['MN:%s' %p]
                outputs['%s:stat:%s' %(fl_out_name, p)] = inputs['Ps:%s' %p]

            outputs['Throat:stat:S'] = inputs['S']
            outputs['%s:stat:S' %fl_out_name] = inputs['S']

        elif nozzType == "CD_CV":
            if inputs['Ps_calc'] < inputs['MN:P']:
                prefix = "MN"
            else:
                prefix = "Ps"

            for p in self.flow_out:
                outputs['Throat:stat:%s' %p] = inputs['%s:%s' %(prefix, p)]
                outputs['%s:stat:%s' %(fl_out_name, p)] = inputs['Ps:%s' %p]

            outputs['Throat:stat:S'] = inputs['S']
            outputs['%s:stat:S' %fl_out_name] = inputs['S']

    def compute_partials(self, inputs, J):
        nozzType = self.options['nozzType']
        fl_out_name = self.options['fl_out_name']

        if nozzType == "CV":

            if inputs['Ps_calc'] < inputs['MN:P']:
                prefix = "MN"
                other = "Ps"
            else:
                prefix = "Ps"
                other = "MN"

            for p in self.flow_out:
                J['Throat:stat:%s' %p, '%s:%s' %(prefix, p)] = 1.
                J['%s:stat:%s' %(fl_out_name, p), '%s:%s' %(prefix, p)] = 1.

                J['Throat:stat:%s' %p, '%s:%s' %(other, p)] = 0.
                J['%s:stat:%s' %(fl_out_name, p), '%s:%s' %(other, p)] = 0.

            J['Throat:stat:S', 'S'] = 1.
            J['%s:stat:S' %fl_out_name, 'S'] = 1.

        elif nozzType == "CD":
            for p in self.flow_out:
                J['Throat:stat:%s' %p, 'MN:%s' %p] = 1.0
                J['%s:stat:%s' %(fl_out_name, p), 'Ps:%s' %p] = 1.0

            J['Throat:stat:S', 'S'] = 1.0
            J['%s:stat:S' %fl_out_name, 'S'] = 1.0

        elif nozzType == "CD_CV":
            if inputs['Ps_calc'] < inputs['MN:P']:
                prefix = "MN"
                other = "Ps"
            else:
                prefix = "Ps"
                other = "MN"

            for p in self.flow_out:
                J['Throat:stat:%s' %p, '%s:%s' %(prefix, p)] = 1.0
                J['%s:stat:%s' %(fl_out_name, p), 'Ps:%s' %p] = 1.0

                J['Throat:stat:%s' %p, '%s:%s' %(other, p)] = 0.0

            J['Throat:stat:S', 'S'] = 1.0
            J['%s:stat:S' %fl_out_name, 'S'] = 1.0

class Nozzle(Element):
    """
    An assembly that models a convergent Nozzle.
    """

    def initialize(self):
        self.options.declare('nozzType', default='CV',
                              desc='Nozzle type: CD, CV, or CD_CV.')
        self.options.declare('lossCoef', default='Cv',
                              desc='If set to "Cfg", then Gross Thrust Coefficient is an input.')
        self.options.declare('internal_solver', default=False)

        super().initialize()

    def pyc_setup_output_ports(self): 
        
        self.copy_flow('Fl_I', 'Fl_O')

    def setup(self):
        thermo_method = self.options['thermo_method']
        thermo_data = self.options['thermo_data']
        nozzType = self.options['nozzType']
        lossCoef = self.options['lossCoef']

        # elements = self.options['elements']
        composition = self.Fl_I_data['Fl_I']

        self.add_subsystem('mach_choked', om.IndepVarComp('MN', 1.000, ))

        # Create inlet flow station
        in_flow = FlowIn(fl_name="Fl_I")
        self.add_subsystem('in_flow', in_flow, promotes_inputs=['Fl_I:*'])

        # PR_bal = self.add_subsystem('PR_bal', BalanceComp())
        # PR_bal.add_balance('PR', units=None, eq_units='lbf/inch**2', lower=1.001)
        # self.connect('PR_bal.PR', 'PR')
        # self.connect('Ps_exhaust', 'PR_bal.lhs:PR')
        # self.connect('Ps_calc', 'PR_bal.rhs:PR')

        self.add_subsystem('PR_bal', PR_bal(), promotes_inputs=['*'], promotes_outputs=['*'] )

        # Calculate pressure at the throat
        prom_in = [('Pt_in', 'Fl_I:tot:P'),
                   'PR', 'dPqP']
        self.add_subsystem('press_calcs', PressureCalcs(), promotes_inputs=prom_in,
                           promotes_outputs=['Ps_calc'])

        # Calculate throat total flow properties
        throat_total = Thermo(mode='total_hP', fl_name='Fl_O:tot', 
                              method=thermo_method, 
                              thermo_kwargs={'composition':composition, 
                                             'spec':thermo_data})
        prom_in = [('h', 'Fl_I:tot:h'),
                   ('composition', 'Fl_I:tot:composition')]
        self.add_subsystem('throat_total', throat_total, promotes_inputs=prom_in,
                           promotes_outputs=['Fl_O:*'])
        self.connect('press_calcs.Pt_th', 'throat_total.P')

        # Calculate static properties for sonic flow
        throat_static_MN = Thermo(mode='static_MN', 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':composition, 
                                                 'spec':thermo_data})
        prom_in = [('ht', 'Fl_I:tot:h'),
                   ('W', 'Fl_I:stat:W'),
                   ('composition', 'Fl_I:tot:composition')]
        self.add_subsystem('staticMN', throat_static_MN,
                           promotes_inputs=prom_in)
        self.connect('throat_total.S', 'staticMN.S')
        self.connect('mach_choked.MN', 'staticMN.MN')
        self.connect('press_calcs.Pt_th', 'staticMN.guess:Pt')
        self.connect('throat_total.gamma', 'staticMN.guess:gamt')
        # self.connect('Fl_I.flow:flow_products','staticMN.init_prod_amounts')

        # Calculate static properties based on exit static pressure
        throat_static_Ps = Thermo(mode='static_Ps', 
                                  method=thermo_method, 
                                  thermo_kwargs={'composition':composition, 
                                                 'spec':thermo_data})
        prom_in = [('ht', 'Fl_I:tot:h'),
                   ('W', 'Fl_I:stat:W'),
                   ('Ps', 'Ps_calc'),
                   ('composition', 'Fl_I:tot:composition')]
        self.add_subsystem('staticPs', throat_static_Ps,
                           promotes_inputs=prom_in)
        self.connect('throat_total.S', 'staticPs.S')
        # self.connect('press_calcs.Ps_calc', 'staticPs.Ps')
        # self.connect('Fl_I.flow:flow_products','staticPs.init_prod_amounts')

        # Calculate ideal exit flow properties
        ideal_flow = Thermo(mode='static_Ps', 
                            method=thermo_method, 
                            thermo_kwargs={'composition':composition, 
                                                 'spec':thermo_data})
        prom_in = [('ht', 'Fl_I:tot:h'),
                   ('S', 'Fl_I:tot:S'),
                   ('W', 'Fl_I:stat:W'),
                   ('Ps', 'Ps_calc'),
                   ('composition', 'Fl_I:tot:composition')]
        self.add_subsystem('ideal_flow', ideal_flow,
                           promotes_inputs=prom_in)
        # self.connect('press_calcs.Ps_calc', 'ideal_flow.Ps')
        # self.connect('Fl_I.flow:flow_products','ideal_flow.init_prod_amounts')

        # Determine throat and exit flow properties based on nozzle type and exit static pressure
        mux = Mux(nozzType=nozzType, fl_out_name='Fl_O')
        prom_in = [('Ps:W', 'Fl_I:stat:W'),
                   ('MN:W', 'Fl_I:stat:W'),
                   ('Ps:P', 'Ps_calc'),
                   'Ps_calc']
        self.add_subsystem('mux', mux, promotes_inputs=prom_in, promotes_outputs=['*:stat:*'])
        self.connect('throat_total.S', 'mux.S')
        self.connect('staticPs.h', 'mux.Ps:h')
        self.connect('staticPs.T', 'mux.Ps:T')
        self.connect('staticPs.rho', 'mux.Ps:rho')
        self.connect('staticPs.gamma', 'mux.Ps:gamma')
        self.connect('staticPs.Cp', 'mux.Ps:Cp')
        self.connect('staticPs.Cv', 'mux.Ps:Cv')
        self.connect('staticPs.V', 'mux.Ps:V')
        self.connect('staticPs.Vsonic', 'mux.Ps:Vsonic')
        self.connect('staticPs.MN', 'mux.Ps:MN')
        self.connect('staticPs.area', 'mux.Ps:area')

        self.connect('staticMN.h', 'mux.MN:h')
        self.connect('staticMN.T', 'mux.MN:T')
        self.connect('staticMN.Ps', 'mux.MN:P')
        self.connect('staticMN.rho', 'mux.MN:rho')
        self.connect('staticMN.gamma', 'mux.MN:gamma')
        self.connect('staticMN.Cp', 'mux.MN:Cp')
        self.connect('staticMN.Cv', 'mux.MN:Cv')
        self.connect('staticMN.V', 'mux.MN:V')
        self.connect('staticMN.Vsonic', 'mux.MN:Vsonic')
        self.connect('mach_choked.MN', 'mux.MN:MN')
        self.connect('staticMN.area', 'mux.MN:area')

        # Calculate nozzle performance paramters based on
        perf_calcs = PerformanceCalcs(lossCoef=lossCoef)
        if lossCoef == "Cv":
            other_inputs = ['Cv', 'Ps_calc']
        else:
            other_inputs = ['Cfg', 'Ps_calc']
        prom_in = [('W_in', 'Fl_I:stat:W')] + other_inputs
        self.add_subsystem('perf_calcs', perf_calcs, promotes_inputs=prom_in,
                           promotes_outputs=['Fg'])
        self.connect('ideal_flow.V', 'perf_calcs.V_ideal')
        # self.connect('ideal_flow.area', 'perf_calcs.A_ideal')

        if lossCoef == 'Cv':
            self.connect('Fl_O:stat:V', 'perf_calcs.V_actual')
            self.connect('Fl_O:stat:area', 'perf_calcs.A_actual')
            self.connect('Fl_O:stat:P', 'perf_calcs.Ps_actual')

        if self.options['internal_solver']:
            newton = self.nonlinear_solver = om.NewtonSolver()
            newton.options['atol'] = 1e-10
            newton.options['rtol'] = 1e-10
            newton.options['maxiter'] = 20
            newton.options['iprint'] = 2
            newton.options['solve_subsystems'] = True
            newton.options['reraise_child_analysiserror'] = False
            newton.linesearch = om.BoundsEnforceLS()
            newton.linesearch.options['bound_enforcement'] = 'scalar'

            newton.linesearch.options['iprint'] = -1
            self.linear_solver = om.DirectSolver(assemble_jac=True)

        super().setup()
