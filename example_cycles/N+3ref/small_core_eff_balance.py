from openmdao.api import ImplicitComponent
import numpy as np
from  scipy.interpolate import Akima1DInterpolator as Akima


""" Create tables for table lookup functions """
# Small engines polytripic efficiency values
Wc_SE = np.array([0, 0.205, 0.63,    1.0,   1.5,    2.,    2.5,    3.,    4.,    5.,   30., 200])
# TGL 0 - current technology level
EtaPoly_SE0 =np.array([0,  0.82, 0.86,  0.871, 0.881, 0.885, 0.8875, 0.889, 0.892, 0.894, 0.895, 0.895])
# TGL 1 - next generation technology level ~2% better
EtaPoly_SE1 =np.array([0, 0.84,  0.88,  0.891, 0.901, 0.905, 0.9075, 0.909, 0.912, 0.914, 0.915, 0.915 ])
# TGL 2 - beyond next generation technology level ~4% better
EtaPoly_SE2 =np.array([0, 0.855, 0.900, 0.912, 0.917, 0.920, 0.922, 0.9235, 0.926, 0.930, 0.931, 0.931])

# Create continuously differentiable interpolations
EtaPoly_SE0_interp = Akima(Wc_SE, EtaPoly_SE0)
EtaPoly_SE1_interp = Akima(Wc_SE, EtaPoly_SE1)
EtaPoly_SE2_interp = Akima(Wc_SE, EtaPoly_SE2)

# gather derivatives
EtaPoly_SE0_interp_deriv = EtaPoly_SE0_interp.derivative(1)
EtaPoly_SE1_interp_deriv = EtaPoly_SE1_interp.derivative(1)
EtaPoly_SE2_interp_deriv = EtaPoly_SE2_interp.derivative(1)

class SmallCoreEffBalance(ImplicitComponent):
    """ Polytropic/ Adiabatic efficiency balance. """

    def initialize(self):
        self.options.declare('tech_level', default=0, values=[0,1,2],
                              desc='Set Technology level, 0 - current tech, 1 - next gen ~2% better, 2 - beyond next gen ~4% better')
        self.options.declare('eng_type', default='large', values=['large', 'small'],
                              desc='Set engine type, which changes the polytropic eff curve')
    def setup(self):
        self.add_input('CS',val = 1.0,units='lbm/s', desc='core size or corrected mass flow on the high pressure side of the HPC')
        self.add_input('eta_p', val = 1.0, units=None, desc='polytropic efficiency')

        self.add_output('eta_a', val = 0.9, units=None, desc='adiabatic efficiency', upper=1, lower=0.8)

        self.declare_partials('eta_a', ['CS', 'eta_p'])

    def apply_nonlinear(self, inputs, outputs, residuals):
        """ Calulate residuals for each balance """
        TGL = self.options['tech_level']
        Type  = self.options['eng_type']
        CS = inputs['CS']

        if Type == 'small':
            if TGL == 1:
                EtaPoly_Calc = EtaPoly_SE1_interp(CS)
            elif TGL == 2:
                EtaPoly_Calc = EtaPoly_SE2_interp(CS)
            else:
                EtaPoly_Calc = EtaPoly_SE0_interp(CS)
        else:
            if CS < 5.30218862:
                EtaPoly_Calc = -9.025e-4*(CS**4.) + 0.01816*(CS**3.) - 0.1363*(CS**2.) + 0.4549*(CS) + 0.33620
            else:
                EtaPoly_Calc = 0.91
            if TGL == 1:
                EtaPoly_Calc += 0.02
            elif TGL == 2:
                EtaPoly_Calc += 0.04
        EtaPoly = inputs['eta_p']
        CS      = inputs['CS']

        residuals['eta_a'] = EtaPoly - EtaPoly_Calc

    def linearize(self, inputs, outputs, J):
        TGL = self.options['tech_level']
        CS = inputs['CS']
        Type  = self.options['eng_type']

        if Type == 'small':
            if TGL == 1:
                partl = EtaPoly_SE1_interp_deriv(CS).reshape(1,)[0]
            elif TGL == 2:
                partl = EtaPoly_SE2_interp_deriv(CS).reshape(1,)[0]
            else:
                partl = EtaPoly_SE0_interp_deriv(CS).reshape(1,)[0]
        else:
            if CS < 5.30218862:
                partl = -0.00361*(CS**3.) + 0.05448*(CS**2.) - 0.2726*CS + 0.4549
            else:
                partl = 0.0

        J['eta_a','CS'] = -partl
        J['eta_a','eta_p'] = 1