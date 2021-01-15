import numpy as np

from openmdao.api import ExplicitComponent

from pycycle.constants import P_REF, R_UNIVERSAL_ENG, R_UNIVERSAL_SI, MIN_VALID_CONCENTRATION


class PropsCalcs(ExplicitComponent):
    """computes, S, H, Cp, Cv, gamma, given a converged equilibirum mixture"""

    def initialize(self):
        self.options.declare('thermo', desc='thermodynamic data object', recordable=False)

    def setup(self):

        thermo = self.options['thermo']

        self.add_input('T', val=284., units="degK", desc="Temperature")
        self.add_input('P', val=1., units='bar', desc="Pressure")
        self.add_input('n', val=np.ones(thermo.num_prod),
                       desc="molar concentration of the mixtures, last element is the total molar concentration")
        self.add_input('n_moles', val=1., desc="1/molar_mass for gaseous mixture")

        ne1 = thermo.num_element + 1
        self.add_input('result_T', val=np.ones(ne1),
                       desc="result of the linear solve for T", shape=ne1)
        self.add_input('result_P', val=np.ones(ne1),
                       desc="result of the linear solve for T", shape=ne1)

        self.add_output('h', val=1., units="cal/g", desc="enthalpy")
        self.add_output('S', val=1., units="cal/(g*degK)", desc="entropy")
        self.add_output('gamma', val=1.4, lower=1.0, upper=2.0, desc="ratio of specific heats")
        self.add_output('Cp', val=1., units="cal/(g*degK)", desc="Specific heat at constant pressure")
        self.add_output('Cv', val=1., units="cal/(g*degK)", desc="Specific heat at constant volume")
        self.add_output('rho', val=0.0004, units="g/cm**3", desc="density")

        self.add_output('R', val=1., units='(N*m)/(kg*degK)', desc='Specific gas constant')
        # self.deriv_options['check_type'] = "cs"

        # partial derivs setup
        self.declare_partials('h', ['n', 'T'])
        self.declare_partials('S', ['n', 'T', 'P'])
        self.declare_partials('S', 'n_moles')
        self.declare_partials('Cp', ['n', 'T', 'result_T'])
        self.declare_partials('rho', ['T', 'P', 'n_moles'])
        self.declare_partials('gamma', ['n', 'n_moles', 'T', 'result_T', 'result_P'])
        self.declare_partials('Cv', ['n', 'n_moles', 'T', 'result_T', 'result_P'])

        self.declare_partials('R', 'n_moles', val=R_UNIVERSAL_SI)


    def compute(self, inputs, outputs):
        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']
        result_T = inputs['result_T']

        nj = inputs['n'][:num_prod]
        # nj[nj<0] = 1e-10 # ensure all concentrations stay non-zero
        n_moles = inputs['n_moles']

        self.dlnVqdlnP = dlnVqdlnP = -1 + inputs['result_P'][num_element]
        self.dlnVqdlnT = dlnVqdlnT = 1 - result_T[num_element]

        self.Cp0_T = Cp0_T = thermo.Cp0(T)
        Cpf = np.sum(nj*Cp0_T)

        self.H0_T = H0_T = thermo.H0(T)
        self.S0_T = S0_T = thermo.S0(T)
        self.nj_H0 = nj_H0 = nj*H0_T

        # Cpe = 0
        # for i in range(0, num_element):
        #     for j in range(0, num_prod):
        #         Cpe -= thermo.aij[i][j]*nj[j]*H0_T[j]*self.result_T[i]
        # vectorization of this for loop for speed
        Cpe = -np.sum(np.sum(thermo.aij*nj_H0, axis=1)*result_T[:num_element])
        Cpe += np.sum(nj_H0*H0_T)  # nj*H0_T**2
        Cpe -= np.sum(nj_H0)*result_T[num_element]

        outputs['h'] = np.sum(nj_H0)*R_UNIVERSAL_ENG*T

        try:
            val = (S0_T+np.log(n_moles/nj/(P/P_REF)))
        except FloatingPointError:
            P = 1e-5
            val = (S0_T+np.log(n_moles/nj/(P/P_REF)))


        outputs['S'] = R_UNIVERSAL_ENG * np.sum(nj*val)
        outputs['Cp'] = Cp = (Cpe+Cpf)*R_UNIVERSAL_ENG
        outputs['Cv'] = Cv = Cp + n_moles*R_UNIVERSAL_ENG*dlnVqdlnT**2/dlnVqdlnP

        outputs['gamma'] = -1*Cp/Cv/dlnVqdlnP

        outputs['rho'] = P/(n_moles*R_UNIVERSAL_SI*T)*100  # 1 Bar is 100 Kpa

        outputs['R'] = R_UNIVERSAL_SI*n_moles  #(m**3 * Pa)/(mol*degK)

    def compute_partials(self, inputs, J):

        thermo = self.options['thermo']
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        T = inputs['T']
        P = inputs['P']
        nj = inputs['n']
        n_moles = inputs['n_moles']
        result_T = inputs['result_T']
        result_T_last = result_T[num_element]
        result_T_rest = result_T[:num_element]

        dlnVqdlnP = -1 + inputs['result_P'][num_element]
        dlnVqdlnT = 1 - result_T_last

        Cp0_T = thermo.Cp0(T)
        Cpf = np.sum(nj * Cp0_T)

        H0_T = thermo.H0(T)
        S0_T = thermo.S0(T)
        nj_H0 = nj * H0_T

        # Cpe = 0
        # for i in range(0, num_element):
        #     for j in range(0, num_prod):
        #         Cpe -= thermo.aij[i][j]*nj[j]*H0_T[j]*self.result_T[i]
        # vectorization of this for loop for speed
        Cpe = -np.sum(np.sum(thermo.aij * nj_H0, axis=1) * result_T_rest)
        Cpe += np.sum(nj_H0 * H0_T)  # nj*H0_T**2
        Cpe -= np.sum(nj_H0) * result_T_last

        Cp = (Cpe + Cpf) * R_UNIVERSAL_ENG
        Cv = Cp + n_moles * R_UNIVERSAL_ENG * dlnVqdlnT ** 2 / dlnVqdlnP

        dH0_dT = thermo.H0_applyJ(T, 1.)
        dS0_dT = thermo.S0_applyJ(T, 1.)
        dCp0_dT = thermo.Cp0_applyJ(T, 1.)
        sum_nj_R = n_moles*R_UNIVERSAL_SI

        drho_dT = P/(sum_nj_R*T**2)*100
        drho_dnmoles = -P/(n_moles**2*R_UNIVERSAL_SI*T)*100

        dCpe_dT = 2*np.sum(nj*H0_T*dH0_dT)
        # for i in range(num_element):
        #     self.dCpe_dT -= np.sum(aij[i]*nj*self.dH0_dT)*self.result_T[i]
        dCpe_dT -= np.sum(np.sum(thermo.aij*nj*dH0_dT, axis=1)*result_T_rest)
        dCpe_dT -= np.sum(nj*dH0_dT)*result_T_last

        dCpf_dT = np.sum(nj*dCp0_dT)

        J['h', 'T'] = R_UNIVERSAL_ENG*(np.sum(nj*dH0_dT)*T + np.sum(nj*H0_T))
        J['h', 'n'] = R_UNIVERSAL_ENG*T*H0_T

        J['S', 'n'] = R_UNIVERSAL_ENG*(S0_T + np.log(n_moles) - np.log(P/P_REF) - np.log(nj) - 1)
        # zero out any derivs w.r.t trace species
        _trace = np.where(nj <= MIN_VALID_CONCENTRATION+1e-20)
        J['S', 'n'][0, _trace] = 0
        J['S', 'T'] = R_UNIVERSAL_ENG*np.sum(nj*dS0_dT)
        J['S', 'P'] = -R_UNIVERSAL_ENG*np.sum(nj/P)
        J['S', 'n_moles'] = R_UNIVERSAL_ENG*np.sum(nj)/n_moles
        J['rho', 'T'] = -P/(sum_nj_R*T**2)*100
        J['rho', 'n_moles'] = -P/(n_moles**2*R_UNIVERSAL_SI*T)*100
        J['rho', 'P'] = 1/(sum_nj_R*T)*100

        dCp_dnj = R_UNIVERSAL_ENG*(Cp0_T + H0_T**2)
        for j in range(num_prod):
            for i in range(num_element):
                dCp_dnj[j] -= R_UNIVERSAL_ENG*thermo.aij[i][j]*H0_T[j]*result_T[i]
        dCp_dnj -= R_UNIVERSAL_ENG * H0_T * result_T_last
        J['Cp', 'n'] = dCp_dnj


        dCp_dresultT = np.zeros(num_element+1)
        # for i in range(num_element):
        #     self.dCp_dresultT[i] = -R_UNIVERSAL_ENG*np.sum(aij[i]*nj_H0)
        dCp_dresultT[:num_element] = -R_UNIVERSAL_ENG*np.sum(thermo.aij*nj_H0, axis=1)
        dCp_dresultT[num_element] = - R_UNIVERSAL_ENG*np.sum(nj_H0)
        J['Cp', 'result_T'] = dCp_dresultT

        dCp_dT = (dCpe_dT + dCpf_dT)*R_UNIVERSAL_ENG
        J['Cp', 'T'] = dCp_dT

        J['Cv', 'n'] = dCp_dnj

        dCv_dnmoles = R_UNIVERSAL_ENG*dlnVqdlnT**2/dlnVqdlnP
        J['Cv', 'n_moles'] = dCv_dnmoles
        J['Cv', 'T'] = dCp_dT


        dCv_dresultP = np.zeros((1, num_element+1))
        dCv_dresultP[0, -1] = -R_UNIVERSAL_ENG*n_moles*(dlnVqdlnT/dlnVqdlnP)**2
        J['Cv', 'result_P'] = dCv_dresultP

        J['Cv', 'result_T'] = dCp_dresultT
        J['Cv', 'result_T'][0, -1] -= n_moles*R_UNIVERSAL_ENG/dlnVqdlnP*(2*dlnVqdlnT)
        dCv_dresultT_last = J['Cv', 'result_T'][0, -1]

        J['gamma', 'n'] = dCp_dnj*(Cp/Cv-1)/(dlnVqdlnP*Cv)
        J['gamma', 'n_moles'] = Cp/dlnVqdlnP/Cv**2*dCv_dnmoles
        J['gamma', 'T'] = dCp_dT/dlnVqdlnP/Cv*(Cp/Cv-1)


        dgamma_dresultT = np.zeros((1, num_element+1))
        dgamma_dresultT[0, :num_element] = 1/Cv/dlnVqdlnP*dCp_dresultT[:num_element]*(Cp/Cv-1)
        dgamma_dresultT[0, -1] = (-dCp_dresultT[-1]/Cv+Cp/Cv**2*dCv_dresultT_last)/dlnVqdlnP
        J['gamma', 'result_T'] = dgamma_dresultT

        gamma_dresultP = np.zeros((1, num_element+1))
        gamma_dresultP[0, num_element] = Cp/Cv/dlnVqdlnP*(dCv_dresultP[0, -1]/Cv + 1/dlnVqdlnP)
        J['gamma', 'result_P'] = gamma_dresultP


if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp

    from pycycle.cea import species_data

    thermo = species_data.Properties(species_data.co2_co_o2)

    p = Problem()
    model = p.model = Group()

    indeps = model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('T', 2761.56784655, units='degK')
    indeps.add_output('P', 1.034210, units='bar')
    indeps.add_output('n', val=np.array([2.272e-02, 1.000e-10, 1.136e-02]))
    indeps.add_output('n_moles', val=0.0340831628675)

    indeps.add_output('result_T', val=np.array([-3.02990116, 1.95459777, -0.05024694]))
    indeps.add_output('result_P', val=np.array([0.53047724, 0.48627081, -0.00437025]))

    model.add_subsystem('calcs', PropsCalcs(thermo=thermo), promotes=['*'])

    p.setup()
    p.run_model()

    print("outputs")
    print('h', p['h'])
    print('S', p['S'])
    print('gamma', p['gamma'])
    print('Cp', p['Cp'])
    print('Cv', p['Cv'])
    print('rho', p['rho'])
    print()
    print()
    print('############################################')

    p.model.run_linearize()

    jac = p.model.get_subsystem('calcs').jacobian._subjacs
    for pair in jac:
        print(pair)
        print(jac[pair])
        print
