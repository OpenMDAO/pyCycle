import numpy as np
from scipy.optimize import fsolve

import openmdao.api as om

from pycycle.constants import R_UNIVERSAL_SI

class PsResid(om.ImplicitComponent):
    """Actual implicit relationship for when Mach number is specified"""

    def initialize(self):
        self.options.declare('mode', values=['MN', 'area'])

    def setup(self):


        self.add_input('Ts', val=518., units="degK", desc="Static temp")
        self.add_input('ht', val=1., units="J/kg", desc="Total enthalpy reference condition")
        self.add_input('hs', val=1., units="J/kg", desc="Static enthalpy")
        # self.add_input('n_moles', shape=1)
        self.add_input('R', val=0.0, units='J/kg/degK')
        self.add_input('gamma', val=1.4)
        self.add_input('W', val=1., desc="mass flow rate", units="kg/s")
        self.add_input('rho', val=1., desc="density", units="kg/m**3")

        # used for computing initial guess
        self.add_input('guess:gamt', val=1.4, desc="gamma computed from set total")
        self.add_input('guess:Pt', val=1.0, units="bar", desc="total pressure")

        self.add_output('Ps', lower=1e-4, upper=5e4, val=.001, units="bar",
                        desc="static pressure state variable",
                        ref0=1e-3)
        self.add_output('V', val=100.0, shape=1, desc="velocity", units="m/s",
                        res_ref=1e3)
        self.add_output('Vsonic', val=330.0, shape=1, desc="computed speed of sound", units="m/s",
                        res_ref=1e3)

        self.declare_partials('Ps', ['ht', 'hs'])
        self.declare_partials('Vsonic', ['gamma', 'R', 'Ts', 'Vsonic'])
        self.declare_partials('V', 'V', val=-1.0)

        mode = self.options['mode']
        if mode == "MN":
            self.add_input('MN', val=.5, desc="target mach number")
            self.add_output('area', shape=1, desc="flow area", units="m**2", lower=1e-5)

            self.declare_partials('area', ['area', 'W', 'rho', 'gamma', 'R', 'Ts', 'MN'])
            self.declare_partials('Ps', ['MN', 'R', 'gamma', 'Ts'])
            self.declare_partials('V', ['MN', 'R', 'gamma', 'Ts'])

        elif mode == "area":
            self.add_output('MN', val=.5, desc="target mach number", lower=1e-3)
            self.add_input('guess:MN', val=0.5, desc="Guess for Mach number.")
            self.add_input('area', val=np.inf, desc="flow area", units="m**2")

            self.declare_partials('MN', ['MN', 'area', 'Ts', 'R', 'W', 'gamma', 'rho'])
            self.declare_partials('Ps', ['area', 'Ts', 'R', 'W', 'gamma', 'rho'])
            self.declare_partials('V', ['area', 'Ts', 'R', 'W', 'gamma', 'rho'])

        else:
            raise ValueError('mode must be either "MN" or "area", but "%s" was given' % mode)



        # self.deriv_options['check_type'] = 'cs'
        # self.deriv_options['check_step_size'] = 1e-40
        # self.deriv_options['check_type'] = 'fd'
        # self.deriv_options['check_step_size'] = 1e-3

        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['step_size'] = 1e-7

        # cache the old guess and only re-apply it if it changes.
        # This lets us start from our last converged point, but
        # only if a new guess isn't present. If a new guess is present,
        # its a safe bet that we've moved a lot and our old converged point isn't
        # as good as our new guess
        self._ps_guess_cache = -1.

    def guess_nonlinear(self, inputs, outputs, resids):
        gamt = inputs['guess:gamt']
        if self.options['mode'] == "MN":
            ps_guess = inputs['guess:Pt'] * (1 + (gamt-1)/2 * inputs['MN']**2)**(-gamt/(gamt-1))
            if np.abs(ps_guess - self._ps_guess_cache) > 1e-10:
                if self._ps_guess_cache == -1:
                    outputs['Ps'] = ps_guess
                    self._ps_guess_cache = ps_guess

        else:

            def equations(params):
                ps, MN = params
                f1 = ps - inputs['guess:Pt'] * (1 + (gamt-1)/2 * M_guess**2)**(-gamt/(gamt-1))
                f2 = MN - inputs['W']*(R_UNIVERSAL_SI*inputs['Ts'])**0.5/(ps_guess*1.0e6*inputs['area']*gamt**0.5)
                return (f1[0], f2[0])

            M_guess = inputs['guess:MN']
            ps_guess = inputs['guess:Pt'] * (1 + (gamt-1)/2 * M_guess**2)**(-gamt/(gamt-1))
            ps_guess, M_guess = fsolve(equations, (ps_guess, M_guess))

            # print('foobar', self.pathname, np.abs(ps_guess - self._ps_guess_cache), inputs['W'], inputs['area'], inputs['Ts'])
            if np.abs(ps_guess - self._ps_guess_cache) > 1e-10:
                outputs['Ps'] = ps_guess
                self._ps_guess_cache = ps_guess

    def _compute_outputs_MN(self, i):

        Vsonic = (i['gamma']*i['R']*i['Ts'])**0.5
        try:
            np.seterr(all='raise')
            Vsonic = (i['gamma']*i['R']*i['Ts'])**0.5
            np.seterr(all='warn')
        except:
            np.seterr(all='warn')
            print(self.pathname, i['gamma'], i['R'], i['Ts'])

        MN = i['MN']
        if MN < 1e-16:
            area = np.inf
        else:
            area = i['W']/(i['rho']*Vsonic*MN)

        V = MN*Vsonic
        return Vsonic, V, area

    def _compute_outputs_area(self, i):
        # print('foo', i['gamma'], i['R'], i['Ts'])
        Vsonic = (i['gamma']*i['R']*i['Ts'])**0.5
        area = i['area']
        if area == np.inf:
            MN = 0.
        else:
            #MN = i['W']/(i['rho']*Vsonic*i['area'])
            #print("MN_calc", self.pathname, i['W'], i['rho'], Vsonic, i['area'])

            try:
                np.seterr(all='raise')
                MN = i['W']/(i['rho']*Vsonic*i['area'])
                np.seterr(all='warn')
            except:
                np.seterr(all='warn')
                print("MN_calc", self.pathname, i['W'], i['rho'], Vsonic, i['area'])
                MN = 5.

        V = MN*Vsonic
        
        return MN, Vsonic, V

    def solve_nonlinear(self, inputs, outputs):

        try:
            if self.options['mode'] == "MN":
                outputs['Vsonic'], outputs['V'], outputs['area'] = self._compute_outputs_MN(inputs)
            else:
                outputs['MN'], outputs['Vsonic'], outputs['V'] = self._compute_outputs_area(inputs)
        except FloatingPointError:
            raise om.AnalysisError('Bad values flow states in {}: Ts={}'.format(self.pathname, inputs['Ts']))

    def apply_nonlinear(self, inputs, outputs, resids):

        Ts = inputs['Ts']
        R = inputs['R']
        gamma = inputs['gamma']

        # explicit vars
        if self.options['mode'] == "MN":
            Vsonic, V, area = self._compute_outputs_MN(inputs)
            if area != np.inf:
                resids['area'] = area - outputs['area']
            else:
                resids['area'] = 0.
            MN = inputs['MN']
        else:
            MN, Vsonic, V = self._compute_outputs_area(inputs)
            resids['MN'] = MN - outputs['MN']
            # print "MN resid", self.pathname, MN, outputs['MN']

        resids['Vsonic'] = Vsonic - outputs['Vsonic']
        resids['V'] = V - outputs['V']
        # print(resids['Vsonic'], resids['V'], resids['area'])

        # actual residual for Ps
        RT_q_MW = Ts*R
        MN_squared_q2 = (MN**2)/2.
        # self.dh_dlnP = RT_q_MW*(1+MN_squared_q2*(inputs['gamma']-1))
        ht_calc = inputs['hs'] + MN_squared_q2 * gamma * RT_q_MW
        # ^ TN_D-132 Equation (85) for h*

        resids['Ps'] = (ht_calc - inputs['ht'])/inputs['ht']
        # print "foobar", self.pathname, outputs['Ps'], resids['Ps'], ht_calc, inputs['ht']

        # try:
        #     np.seterr(all="raise")
        #     resids['Ps'] = (self.ht_calc - inputs['ht'])/inputs['ht']
        #     np.seterr(all="ignore")
        # except:
        #     print self.pathname, resids['Ps'], inputs['ht']
        #     resids['Ps'] = (self.ht_calc - inputs['ht'])/inputs['ht']

        # print "ps_resid: ", self.pathname, self.ht_calc, resids['Ps'], inputs['hs']

    def linearize(self, inputs, outputs, J):

        mode = self.options['mode']

        gamma = inputs['gamma']
        Ts = inputs['Ts']
        R = inputs['R']
        rho = inputs['rho']
        W = inputs['W']
        ht = inputs['ht']
        gamma = inputs['gamma']

        if mode == "MN":
            MN_squared_q2 = inputs['MN']**2/2.
        else:
            MN_squared_q2 = outputs['MN']**2/2.

        RT_q_MW = R*Ts
        ht_calc = inputs['hs'] + MN_squared_q2 * gamma * RT_q_MW

        J['Ps', 'ht'] = -ht_calc/ht**2
        J['Ps', 'hs'] = 1/ht

        # Derivatives of outputs
        part = .5*(gamma*R*Ts)**-.5
        J['Vsonic', 'gamma'] = dVs_dgamma = part*R*Ts
        J['Vsonic', 'R'] = part*gamma*Ts
        J['Vsonic', 'Ts'] = part*gamma*R
        J['Vsonic', 'Vsonic'] = -1.
        # J['V', 'V'] = -1.

        if mode=="MN":
            MN = inputs['MN']

            Vsonic, V, area = self._compute_outputs_MN(inputs)

            J['area', 'area'] = -1.

            J['Ps', 'MN'] = MN*gamma*RT_q_MW/ht
            J['Ps', 'R'] = Ts*MN_squared_q2*gamma/ht
            J['Ps', 'gamma'] = RT_q_MW*MN_squared_q2/ht
            J['Ps', 'Ts'] = MN_squared_q2*gamma*R/ht

            if MN >= 1e-16:
                J['area', 'W'] = 1.0/(rho*Vsonic*MN)
                J['area', 'rho'] = -W/(Vsonic*MN*rho**2)

                part = -W/(rho*Vsonic**2*MN) * 0.5*(R*gamma*Ts)**-.5
                J['area', 'gamma'] = part*R*Ts
                J['area', 'R'] = part*gamma*Ts
                J['area', 'Ts'] = part*gamma*R
                J['area', 'MN'] = -W/rho/Vsonic/MN**2

                J['V', 'MN'] = Vsonic
                J['V', 'Ts'] = MN * J['Vsonic', 'Ts']
                J['V', 'R'] = MN * J['Vsonic', 'R']
                J['V', 'gamma'] = MN * J['Vsonic', 'gamma']

        else:
            MN, Vsonic, V = self._compute_outputs_area(inputs)
            area = inputs['area']

            J['MN', 'MN'] = -1

            dresid_dMN = (MN*gamma*RT_q_MW)

            try:
                dMN_dA = -(W/rho/Vsonic/area**2)
            except FloatingPointError:
                raise om.AnalysisError('{} Bad value in static calc: rho={}, Vsonic={}, area={}'.format(self.pathname, rho, Vsonic, area))

            J['Ps', 'area'] = dMN_dA*dresid_dMN/ht
            J['V', 'area'] = dMN_dA*Vsonic
            J['MN', 'area'] = dMN_dA

            dMN_dVs = -W/(rho*area*Vsonic**2)
            dVs_dTs = 0.5*gamma*R/Vsonic
            J['Ps', 'Ts'] = gamma*(MN*dMN_dVs*dVs_dTs*RT_q_MW + MN_squared_q2*R)/ht
            J['V', 'Ts'] = -(dMN_dVs*dVs_dTs*Vsonic + MN*dVs_dTs)
            J['MN', 'Ts'] = dMN_dVs*dVs_dTs

            dVs_dR = 0.5*gamma*Ts/Vsonic
            J['Ps', 'R'] = -gamma*(MN*RT_q_MW*dMN_dVs*dVs_dR + MN_squared_q2*Ts)/ht
            J['V', 'R'] = dMN_dVs*dVs_dR*Vsonic + dVs_dR*MN  # works out to exactly 0
            J['MN', 'R'] = dMN_dVs*dVs_dR

            dMN_dW = (1/rho/Vsonic/area)
            J['Ps', 'W'] = dMN_dW*dresid_dMN/ht
            J['V', 'W'] = dMN_dW*Vsonic
            J['MN', 'W'] = dMN_dW

            dMN_dgamma = dMN_dVs*dVs_dgamma
            J['MN', 'gamma'] = dMN_dgamma
            J['Ps', 'gamma'] = RT_q_MW/ht*(MN_squared_q2 + gamma*MN*dMN_dgamma)
            J['V', 'gamma'] = dMN_dgamma*Vsonic + dVs_dgamma*MN

            dMN_drho = -W/(area*Vsonic*rho**2)
            J['Ps', 'rho'] = MN*gamma*R*Ts*dMN_drho/ht
            J['V', 'rho'] = dMN_drho*Vsonic
            J['MN', 'rho'] = dMN_drho
