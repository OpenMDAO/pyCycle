import numpy as np

from openmdao.api import ExplicitComponent

from pycycle.constants import R_UNIVERSAL_ENG, R_UNIVERSAL_SI, MIN_VALID_CONCENTRATION
from pycycle.thermo.cea import species_data


class PropsRHS(ExplicitComponent):

    def __init__(self, thermo):
        super(PropsRHS, self).__init__()
        self.thermo = thermo

    def setup(self):

        thermo = self.thermo
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        self.add_input('T', val=284., units="degK", desc="Total Temperature")
        self.add_input('n', val=np.zeros(num_prod),
                       desc="molar concentration of the mixtures, last element is "
                       "the total molar concentration")  # kg-mol/kg
        self.add_input('n_moles', val=1., desc="1/molar_mass for gaseous mixture")
        self.add_input('composition', val=thermo.b0,
                       desc="assigned kg-atoms of element i per total kg of reactant")  # kg-atom/kg

        self.add_output('rhs_T', val=np.zeros(num_element+1),
                        desc="rhs for the T solve")
        self.add_output('rhs_P', val=np.zeros(num_element+1),
                        desc="rhs for the P solve")
        self.add_output('lhs_TP', val=np.eye(num_element+1),
                        desc="A matrix for the totals linear solve")

        thermo = self.thermo
        num_prod = thermo.num_prod
        num_element = thermo.num_element

        self.drhsT_dT = np.empty(num_element+1)
        self.drhsT_dn = np.empty((num_element+1, num_prod))

        drhsP_dnmoles = np.zeros(num_element+1)
        drhsP_dnmoles[-1] = 1.
        # self.declare_partials('rhs_P', 'n_moles',
        #                       val=1, rows=(num_element,), cols=(0,))
        self.declare_partials('rhs_P', 'n_moles', val=drhsP_dnmoles)

        drhsP_db0 = np.zeros((num_element+1, num_element))
        np.fill_diagonal(drhsP_db0[:num_element, :num_element], 1)
        self.declare_partials('rhs_P', 'composition', val=drhsP_db0)

        # JSG: The for loops are slow, but its ok because we only do them one time
        ne1 = num_element+1
        dlhs_dn = np.zeros((ne1**2, num_prod))
        # val, idx_row, idx_col = [], [], []
        for i in range(num_element):
            for j in range(num_element):
                for k in range(num_prod):
                    dlhs_dn[ne1*i+j, k] = thermo.aij_prod[i][j, k]
                    # val.append(thermo.aij_prod[i][j, k])
                    # idx_row.append(3*i+j)
                    # idx_col.append(k)
        self.declare_partials('lhs_TP', 'n', val=dlhs_dn)

        dlhs_db0 = np.zeros((ne1**2, num_element))
        for i in range(num_element):
            for j in range(num_element):
                    dlhs_db0[ne1*num_element+j, j] = 1
                    dlhs_db0[ne1*j+num_element, j] = 1
        self.declare_partials('lhs_TP', 'composition', val=dlhs_db0)

        self.declare_partials('rhs_T', 'T')
        self.declare_partials('rhs_T', 'n')
        # self.approx_partials('*', '*')

    def compute(self, inputs, outputs):

        thermo = self.thermo
        num_element = thermo.num_element
        T = inputs['T']
        n = inputs['n']
        b0 = inputs['composition']

        for i in range(num_element):
            # outputs['lhs_TP'][i][:num_element] = np.sum(thermo.aij_prod[i] * n, axis=1)
            outputs['lhs_TP'][i][:num_element] = np.dot(thermo.aij_prod[i], n)

        # determine the delta coeff for 2.24 and pi coef for 2.26\
        # at the converged state, b = b0 by definition

        outputs['lhs_TP'][num_element, :num_element] = b0
        outputs['lhs_TP'][:num_element, num_element] = b0

        outputs['lhs_TP'][num_element, num_element] = 0

        # rhs for P
        outputs['rhs_P'][:num_element] = b0
        outputs['rhs_P'][num_element] = inputs['n_moles']

        # rhs for T
        self.H0_T = H0_T = thermo.H0(T)
        n_H0 = n*H0_T
        outputs['rhs_T'][:num_element] = np.sum(thermo.aij*n_H0, axis=1)
        outputs['rhs_T'][num_element] = np.sum(n_H0)

    def compute_partials(self, inputs, J):

        thermo = self.thermo
        num_element = thermo.num_element
        aij = thermo.aij

        if inputs._under_complex_step:
            self.drhsT_dT = self.drhsT_dT.astype(np.complex)
            self.drhsT_dn = self.drhsT_dn.astype(np.complex)
        else:
            self.drhsT_dT = self.drhsT_dT.real
            self.drhsT_dn = self.drhsT_dn.real

        T = inputs['T']
        nj = inputs['n']

        H0_T = self.H0_T
        nj_dH0dT = thermo.H0_applyJ(T, nj)

        self.drhsT_dT[:num_element] = np.sum(aij*nj_dH0dT, axis=1)
        self.drhsT_dT[num_element] = np.sum(nj_dH0dT)

        self.drhsT_dn[:num_element] = aij*H0_T
        self.drhsT_dn[num_element] = H0_T

        J['rhs_T', 'T'] = self.drhsT_dT.reshape((-1, 1))
        J['rhs_T', 'n'] = self.drhsT_dn

        # derivs of rhsP are constants, specified in setup

        # derivs of lhs_TP are constants, specified in setup


if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp, LinearSystemComp

    thermo = species_data.Properties(species_data.co2_co_o2)

    p = Problem()
    p.model = Group()

    indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('T', val=2761.56784655, units='degK')
    indeps.add_output('n', val=np.array([2.272e-02, 1.000e-10, 1.136e-02]))
    indeps.add_output('composition', val=np.array([0.023, 0.045]))
    indeps.add_output('n_moles', val=0.0340831628675)

    props_rhs = p.model.add_subsystem('props_rhs', PropsRHS(thermo=thermo), promotes=["*"])

    p.model.add_subsystem('ln_T', LinearSystemComp(size=thermo.num_element+1))
    p.model.connect('lhs_TP', 'ln_T.A')
    p.model.connect('rhs_T', 'ln_T.b')

    p.model.add_subsystem('ln_P', LinearSystemComp(size=thermo.num_element+1))
    p.model.connect('lhs_TP', 'ln_P.A')
    p.model.connect('rhs_P', 'ln_P.b')

    p.setup()
    p.run_model()

    print("outputs")
    print("rhs_t", repr(p['rhs_T']))
    print("rhs_P", repr(p['rhs_P']))
    print("lhs_TP", repr(p['lhs_TP']))
    print("result_T", repr(p['ln_T.x']))
    print("result_P", repr(p['ln_P.x']))

    p.model.run_linearize()

    jac = p.model.get_subsystem('props_rhs').jacobian._subjacs
    for pair in jac:
        print(pair)
        print(jac[pair])
        print()
