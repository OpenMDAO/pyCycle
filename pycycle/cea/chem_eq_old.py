from __future__ import division, print_function
import numpy as np
from scipy.linalg import lu_factor, lu_solve

from openmdao.api import Component, ImplicitComponent, ExplicitComponent

from pycycle.constants import R_UNIVERSAL_ENG, P_REF, MIN_VALID_CONCENTRATION


def resid_weighting(n):
    np.seterr(under='ignore')
    w = (1 / (1 + np.exp(-1e5 * n)) - .5) * 2
    return w


class ChemEq(Component):
    """ Find the equilibirum composition for a given gaseous mixture """

    def __init__(self, thermo, mode="T"):
        super(ChemEq, self).__init__()

        self.thermo = thermo
        self.mode = mode
        self.num_prod = num_prod = thermo.num_prod
        self.num_element = num_element = thermo.num_element

        from openmdao.utils.generalized_dict import GeneralizedDictionary
        self.solver_options = GeneralizedDictionary()
        self.solver_options.declare(
            'ftol', value=1e-12, desc="solver tolerance on the residual values")
        self.solver_options.declare(
            'xtol', value=1e-8, desc="solver tolerance on the change of the state variables")
        self.solver_options.declare(
            'maxiter', value=300, desc="internal solver maximum iteration")

        # Input vars
        self.add_param('init_prod_amounts', val=thermo.init_prod_amounts, shape=(num_prod,),
                       desc="initial mass fractions of products, before equilibrating")

        self.add_param('P', val=1.0, units="bar", desc="Pressure")

        if mode == "T":
            self.add_param('T', val=400., units="degK", desc="Temperature")
        elif mode == "h" or mode == "S":

            if mode == "h":
                self.add_param('h', val=0., units="cal/g",
                               desc="Enthalpy")
            elif mode == "S":
                self.add_param('S', val=0., units="cal/(g*degK)",
                               desc="Entropy")

            self.T_idx = num_prod + num_element
            self.add_state('T', lower=1e-10, val=400.,
                           units="degK", desc="Temperature")
        else:
            raise ValueError(
                'Only "T", "h", or "S" are allowed values for mode')

        # State vars
        self.n_init = np.ones(num_prod) / num_prod / 10  # thermo.init_prod_amounts
        self.n_idx = slice(0, num_prod)
        self.add_state('n', lower=MIN_VALID_CONCENTRATION, val=self.n_init, shape=num_prod,
                       desc="mole fractions of the mixture")
        self.pi_idx = slice(num_prod, num_prod + num_element)
        self.add_state('pi', val=np.zeros(num_element),
                       desc="modified lagrange multipliers from the Gibbs lagrangian")

        # Outputs
        self.add_output('b0', shape=num_element,  # when converged, b0=b
                        desc='assigned kg-atoms of element i per total kg of reactant for the initial prod amounts')
        self.add_output('n_moles', lower=0.0, val=0.034, shape=1,
                        desc="1/molecular weight of gas")

        # allocate the newton Jacobian
        self.size = size = num_prod + num_element
        if mode != "T":
            size += 1  # added T as a state variable

        self._dRdy = np.zeros((size, size))
        self._R = np.zeros(size)
        self.total_iters = 0

        self.DEBUG = False

        # Cached stuff for speed
        self.H0_T = None
        self.S0_T = None
        self.dH0_dT = None
        self.dS0_dT = None
        self.sum_n_H0_T = None

        # self.deriv_options['check_type'] = 'cs'
        # self.deriv_options['check_step_size'] = 1e-50
        # self.deriv_options['type'] = 'fd'
        # self.deriv_options['step_size'] = 1e-5

        # Once the concentration of a species reaches its minimum, we
        # can essentially remove it from the problem. This switch controls
        # whether to do this.
        self.remove_small_species = True

    def apply_nonlinear(self, params, unknowns, resids):
        thermo = self.thermo
        P = params['P'] / P_REF
        n = unknowns['n']
        n_moles = np.sum(n)
        pi = unknowns['pi']
        MW = n_moles

        if self.mode != "T":
            T = unknowns['T']
        else:
            T = params['T']

        # Output equation for n_moles
        resids['n_moles'] = n_moles - unknowns['n_moles']

        # Output equation for b0
        b0 = np.sum(thermo.aij * params['init_prod_amounts'], axis=1)
        resids['b0'] = b0 - unknowns['b0']

        self.H0_T = H0_T = thermo.H0(T)
        self.S0_T = S0_T = thermo.S0(T)

        self.mu = H0_T - S0_T + np.log(n) + np.log(P) - np.log(n_moles)

        self.weights = resid_weighting(n * MW)
        resids['n'] = (self.mu - np.sum(pi * thermo.aij.T, axis=1)) * self.weights

        # Zero out resids when a concentration drops too low.
        if self.remove_small_species:
            for j in range(len(n)):
                if n[j] <= 1.0e-10:
                    resids['n'][j] = 0.0

        # residuals from the conservation of mass
        resids['pi'] = np.sum(thermo.aij * n, axis=1) - b0

        # residuals from temperature equation when T is a state
        if self.mode == "h":
            self.sum_n_H0_T = np.sum(n * H0_T)
            h = params['h']
            resids['T'] = (h - self.sum_n_H0_T * R_UNIVERSAL_ENG * T)/h
            # if h < 1e-10:
            #     print self.pathname, h
        elif self.mode == "S":
            S = params['S']
            resids['T'] = (S - R_UNIVERSAL_ENG * np.sum(n * (S0_T - np.log(n) + np.log(n_moles) - np.log(P))))/S

    def solve_nonlinear(self, params, unknowns, resids):
        R_gibbs = resids['n']

        self.apply_nonlinear(params, unknowns, resids)
        unknowns['b0'] = np.sum(
            self.thermo.aij * params['init_prod_amounts'], axis=1)

        n = unknowns['n']
        n_min = MIN_VALID_CONCENTRATION

        R = self._build_R_vec(resids)
        ferr = np.linalg.norm(R) / np.sqrt(self.num_prod)
        xerr = 100000.

        # if ferr > 1e4: # sometimes the last converged point is not a good guess
        #     print "err too big", self.n_init, ferr

        #     n = unknowns['n'] = self.n_init.copy()
        #     self.apply_nonlinear(params, unknowns, resids)
        #     R = self._build_R_vec(resids)
        #     ferr = np.linalg.norm(R)/self.num_prod
        #     if self.mode != "T":
        #         unknowns['T'] = 100. #try this as a guess

        self.iter_count = 1

        ftol = self.solver_options['ftol']
        xtol = self.solver_options['xtol']
        maxiter = self.solver_options['maxiter']

        # print "barfoo", unknowns['n'], ferr > ftol, self.iter_count < maxiter
        while ferr > ftol and xerr > xtol and self.iter_count < maxiter:
            #if self.DEBUG:
            #    print(self.iter_count, R, ferr, xerr)
            #    # raw_input()

            self._calc_dRdy(params, unknowns)
            R = self._build_R_vec(resids)

            self.lup = lu_factor(self._dRdy)
            delta_y = lu_solve(self.lup, -R, trans=0)


            R_max = abs(5 * R_gibbs[-1])
            R_max = max(R_max, np.max(np.abs(resids['n'])))

            # lambdaf is an under-relaxation factor that will be 1 except in
            # very unconverged states
            lambdaf = 2.0 / (R_max + 1e-20)
            if (lambdaf > 1):
                lambdaf = 1.0

            # n and T are logrithmic steps, pi is regular step
            n_step = delta_y[:self.num_prod] / unknowns['n']
            n_step[n_step > 1e1] = 1e1  # prevent huge steps
            delta_n = np.exp(lambdaf * n_step)
            n *= delta_n

            # numerical limiting to prevent concentrations from going to low
            trace = n < n_min
            n[trace] = n_min

            if self.mode != "T":  # T is a state variable
                pi_step = delta_y[self.num_prod:-1]
                unknowns['pi'] += pi_step

                T_step = delta_y[-1] / unknowns['T']
                delta_T = np.exp(lambdaf * T_step)
                unknowns['T'] *= delta_T

            else:
                pi_step = delta_y[self.num_prod:]
                unknowns['pi'] += pi_step

            self.apply_nonlinear(params, unknowns, resids)

            R = self._build_R_vec(resids)
            ferr = np.linalg.norm(R) / np.sqrt(self.num_prod)
            xerr = np.linalg.norm(delta_y) / np.sqrt(self.num_prod)
            # if self.mode != "T":
            #     xerr = np.linalg.norm(delta_n - 1)/np.sqrt(self.num_prod) + (delta_T-1) + np.linalg.norm(pi_step)/np.sqrt(self.num_element)
            # else:
            #     xerr = np.linalg.norm(delta_n - 1)/np.sqrt(self.num_prod) + np.linalg.norm(pi_step)/np.sqrt(self.num_element)
            self.iter_count += 1

        # print "foobar", ferr, xerr, self.iter_count
        unknowns['n_moles'] = np.sum(unknowns['n'])

    def linearize(self, params, unknowns, resids):
        """
        Total derivs = dF/dX - dF/dy[(dR/dy)^-1 (dR/dx)]
        assumed ordering in all arrays:
            params:
                mode var (T, h, or S), P, init_prod_amounts
            states:
                n, pi, T (if mode is h or S)
            outputs:
                b0, n_moles
        """
        self._calc_dRdy(params, unknowns)
        dRdy = self._dRdy

        # lu factorization for use with solve_linear
        # self.lup = lu_factor(dRdy)
        self.dRdy_inv = np.linalg.inv(self._dRdy)

        num_element = self.num_element
        num_prod = self.num_prod

        P = params['P'] / P_REF
        n = unknowns['n']
        n_moles = unknowns['n_moles']
        qP = 1 / P  # quotient_P or 1/P

        J = {}

        end_element = num_prod + num_element

        J['n', 'n'] = dRdy[:num_prod, :num_prod]

        J['n', 'pi'] = dRdy[:num_prod, num_prod: end_element]

        J['n', 'P'] = (self.weights * qP).reshape((-1, 1))
        if self.mode != 'T':  # can only use the dRdy vals when T is a state. Otherwise its not computed
            J['n', 'T'] = dRdy[:num_prod, -1].reshape((num_prod, 1))
        else:
            T = params['T']
            dH0_dT = self.thermo.H0_applyJ(T, 1)
            dS0_dT = self.thermo.S0_applyJ(T, 1)
            J['n', 'T'] = ((dH0_dT - dS0_dT) * self.weights).reshape((num_prod, 1))

        # J['n', 'init_prod_amounts'] =

        # Replace J for tiny values of n with identity
        if self.remove_small_species:
            for j in range(num_prod):
                if unknowns['n'][j] <= 1.0e-10:
                    J['n', 'P'][j, :] = 0.0
                    J['n', 'T'][j, 0] = 0.0
                    # J['n', 'n'][j, :] = 0.0
                    # J['n', 'n'][j, j] = 1.0
                    # J['n', 'pi'][j, :] = 0.0

        J['pi', 'n'] = dRdy[num_prod:end_element, :num_prod]
        # J['pi','pi'] = dRdy[num_prod:end_element,num_prod:end_element] # zero
        J['pi', 'init_prod_amounts'] = -self.thermo.aij
        J['b0', 'init_prod_amounts'] = self.thermo.aij

        if self.mode == 'h':
            J['T', 'n'] = dRdy[-1, :num_prod].reshape(1, num_prod)
            J['T', 'h'] = (self.sum_n_H0_T * R_UNIVERSAL_ENG * unknowns['T'])/params['h']**2
            J['T', 'T'] = dRdy[-1, -1]

        elif self.mode == 'S':
            S = params['S']
            J['T', 'n'] = dRdy[-1, :num_prod].reshape(1, num_prod)
            J['T', 'S'] = (R_UNIVERSAL_ENG * np.sum(n * (self.S0_T - np.log(n) + np.log(n_moles) - np.log(P))))/S**2
            J['T', 'T'] = dRdy[-1, -1]
            J['T', 'P'] = R_UNIVERSAL_ENG * unknowns['n_moles'] / P / S

        J['n_moles', 'n'] = np.ones((1, num_prod))

        return J

    def solve_linear(self, dumat, drmat, vois, mode):
        # NOTE: it seems to be significantly slower to do it this way than to
        # just use gmres

        # self.lup = lu_factor(self._dRdy)
        # self.dRdy_inv = np.linalg.inv(self._dRdy)

        if mode == 'fwd':
            sol_vec, rhs_vec = self.dumat, self.drmat
            t = 0
        else:
            sol_vec, rhs_vec = self.drmat, self.dumat
            t = 1

        for voi in vois:
            rhs = np.zeros(self._dRdy.shape[0])
            rhs[:self.num_prod] = rhs_vec[voi]['n']
            if self.mode != 'T':
                rhs[self.num_prod:-1] = rhs_vec[voi]['pi']
                rhs[-1] = rhs_vec[voi]['T']
            else:
                rhs[self.num_prod:] = rhs_vec[voi]['pi']

            # LU decomp seems to be just slightly faster
            # sol = lu_solve(self.lup, rhs, trans=t)
            sol = self.dRdy_inv.dot(rhs)
            # sol, info = gmres(self._dRdy, rhs)

            sol_vec[voi]['n'] = sol[:self.num_prod]
            if self.mode != 'T':
                sol_vec[voi]['pi'] = sol[self.num_prod:-1]
                sol_vec[voi]['T'] = sol[-1]
            else:
                sol_vec[voi]['pi'] = sol[self.num_prod:]

    def _build_R_vec(self, resids):
        """ Assemble the residual vector for the solver."""
        R = self._R
        R[:self.num_prod] = resids['n']
        if self.mode != 'T':
            R[self.num_prod:-1] = resids['pi']
            R[-1] = resids['T']
        else:
            R[self.num_prod:] = resids['pi']
        return R

    def _calc_dRdy(self, params, unknowns):
        """ Computes the jacobian for the newton solver. This Jacobian
        contains the derivatives of all residual equations with respect to
        the state varaibles, which are ['n', 'pi', and sometimes 'T'] """

        thermo = self.thermo
        aij = thermo.aij
        num_prod = self.num_prod
        mode = self.mode

        n = unknowns['n']
        n_moles = np.sum(n)
        pi = unknowns['pi']


        dRdy = self._dRdy

        # dRgibbs_dn

        MW = 1 / n_moles
        dRdy[:num_prod, :num_prod] = (-MW)
        diag = (1 / n - MW)
        np.fill_diagonal(dRdy[:num_prod, :num_prod], diag)
        # multiples each row by one element of the vector
        dRdy[:num_prod, :num_prod] *= self.weights[:, np.newaxis]

        end_element = num_prod + self.num_element
        # dRgibbs_dpi
        dRdy[:num_prod,
             num_prod:end_element] = (-aij.T) * self.weights[:, np.newaxis]

        if mode != "T":
            # dRgibbs_dT
            T = unknowns['T']
            self.dH0_dT = thermo.H0_applyJ(T, 1)
            self.dS0_dT = thermo.S0_applyJ(T, 1)
            dRdy[:num_prod, -1] = (self.dH0_dT - self.dS0_dT) * self.weights
            # dRmass_dT = 0

        if mode == "h":
            h = params['h']
            # dRT_dn
            dRdy[-1, :num_prod] = (- R_UNIVERSAL_ENG * T * self.H0_T)/h
            # dRT_dT
            dRdy[-1, -1] = (-R_UNIVERSAL_ENG * \
                (T * np.sum(n * self.dH0_dT) + self.sum_n_H0_T))/h

        elif mode == "S":
            P = params['P'] / P_REF
            n_moles = unknowns['n_moles']
            S = params['S']
            # dRT_dn
            # resids['T'] = params['S'] - R_UNIVERSAL_ENG*np.sum(n*(S0_T - np.log(n) + np.log(n_moles) - np.log(P)))
            dRdy[-1, :num_prod] = -R_UNIVERSAL_ENG * \
                (self.S0_T - np.log(n) + np.log(n_moles) -
                 np.log(P) ) / S


            # dRT_dT
            # uc*(S0_T + np.log(sum_nj) - np.log(P) - np.log(nj))
            # valid_products = n > MIN_VALID_CONCENTRATION
            dRdy[-1, -1] = -R_UNIVERSAL_ENG * \
                np.sum(n * self.dS0_dT) / S

        # dRmass_dn
        dRdy[num_prod:end_element, :self.num_prod] = aij

        # Replace J for tiny values of n with identity
        if self.remove_small_species:
            for j in range(num_prod):
                if unknowns['n'][j] <= 1.0e-10:
                    dRdy[j, :] = 0.0
                    dRdy[j, j] = -1.0


if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp, DirectSolver, NewtonSolver, BacktrackingLineSearch
    from pycycle import species_data

    # thermo = species_data.Thermo(species_data.co2_co_o2)
    thermo = species_data.Thermo(species_data.janaf)

    prob = Problem()
    prob.root = Group()

    # prob.root.add('chemeq', ChemEq(thermo, mode="T"), promotes=["*"])
    # params = (
    #     ('P', 1.034210, {'units':'bar'}
    #     ('T', 1500., {'units':'degK'}),
    #     ('init_prod_amounts', thermo.init_prod_amounts),
    # )

    # prob.root.add('chemeq', ChemEq(thermo, mode="S"), promotes=["*"])
    # params = (
    #     ('P', 1.034210, {'units':'psi'}),
    #     ('S', 1.58473968216, {'units':'cal/(g*degK)'}),
    #     ('init_prod_amounts', thermo.init_prod_amounts),
    # )

    params = (
        ('P', 1.034210, {'units': 'psi'}),
        ('h', 100.26682261, {'units': 'cal/g'}),
        ('init_prod_amounts', thermo.init_prod_amounts),
    )
    prob.root.add('des_vars', IndepVarComp(params), promotes=["*"])

    prob.root.add('chemeq', ChemEq(thermo, mode="h"), promotes=["*"])

    # direct = prob.root.linear_solver = DirectSolver()
    # direct.options['method'] = 'LU'

    # newton = prob.root.nonlinear_solver = NewtonSolver()
    # newton.options['maxiter'] = 30
    # ln_bt = newton.set_subsolver('linesearch', BacktrackingLineSearch(rtol=0.9))
    # ln_bt.options['bound_enforcement'] = 'scalar'
    # ln_bt.options['maxiter'] = 5

    prob.setup(check=False)

    import time
    st = time.time()
    prob.run_model()
    print("time: ", time.time()-st)

    print(prob['n'])

    # print(prob['T'], prob.root._residuals['T'])
    # print(prob['n'], prob.root._residuals['n'])
    # print(prob['pi'], prob.root._residuals['pi'])