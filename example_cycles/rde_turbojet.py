"""
Step 6: Low-CPR Rotating Detonation Engine (RDE) Turbojet Cycle.

Simulates and compares hybrid RDE-gas turbine cycle configurations:
1. Conventional Turbojet (CPR = 13.5, Isobaric Combustor with 3% pressure loss).
2. RDE Turbojet (CPR = 13.5, RDECombustor with pressure gain).
3. Low-CPR RDE Turbojet (CPR = 8.0, RDECombustor with pressure gain, reduced stages/weight).

Architecture:
[FlightConditions] -> [Inlet] -> [Compressor] -> [RDECombustor / Combustor] -> [Turbine] -> [Nozzle]
                                      ^                                            |
                                      +-----------------[Shaft]--------------------+
"""

import sys
import numpy as np
import openmdao.api as om
import pycycle.api as pyc

from pycycle.elements.rde_combustor import RDECombustor, print_rde


class HybridTurbojet(pyc.Cycle):
    """
    Single-spool turbojet cycle supporting either a conventional or RDE combustor.
    """

    def initialize(self):
        self.options.declare('use_rde', default=True, desc='Use RDE combustor if True, conventional if False')
        self.options.declare('fuel_type', default='FAR', desc='Fuel type for thermodynamics')
        super().initialize()

    def setup(self):
        use_rde = self.options['use_rde']
        fuel_type = self.options['fuel_type']
        design = self.options['design']

        # Fast TABULAR thermodynamics
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        # 1. Add cycle components
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('comp', pyc.Compressor(map_data=pyc.AXI5, map_extrap=True),
                           promotes_inputs=['Nmech'])

        if use_rde:
            self.add_subsystem('burner', RDECombustor(fuel_type=fuel_type, dia_annulus=12.0))
        else:
            self.add_subsystem('burner', pyc.Combustor(fuel_type=fuel_type))

        self.add_subsystem('turb', pyc.Turbine(map_data=pyc.LPT2269),
                           promotes_inputs=['Nmech'])
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('shaft', pyc.Shaft(num_ports=2), promotes_inputs=['Nmech'])
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # 2. Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'comp.Fl_I')
        self.pyc_connect_flow('comp.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'turb.Fl_I')
        self.pyc_connect_flow('turb.Fl_O', 'nozz.Fl_I')

        # 3. Turbomachinery torque to shaft
        self.connect('comp.trq', 'shaft.trq_0')
        self.connect('turb.trq', 'shaft.trq_1')

        # 4. Nozzle exhaust backpressure
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # 5. Performance connections
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('comp.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        # 6. Balances
        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            # Match airflow to design net thrust Fn
            balance.add_balance('W', units='lbm/s', eq_units='lbf', rhs_name='Fn_target')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('perf.Fn', 'balance.lhs:W')

            # Match FAR to design turbine inlet temperature T4
            balance.add_balance('FAR', eq_units='degR', lower=1e-4, val=0.0175, rhs_name='T4_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

            # Match turbine PR to net shaft power = 0
            balance.add_balance('turb_PR', val=3.5, lower=1.001, upper=12.0, eq_units='hp', rhs_val=0.0)
            self.connect('balance.turb_PR', 'turb.PR')
            self.connect('shaft.pwr_net', 'balance.lhs:turb_PR')

        else:
            # Off-design: match FAR to target thrust
            balance.add_balance('FAR', eq_units='lbf', lower=1e-4, val=0.02, rhs_name='Fn_target')
            self.connect('balance.FAR', 'burner.Fl_I:FAR')
            self.connect('perf.Fn', 'balance.lhs:FAR')

            # Shaft mechanical speed balance
            balance.add_balance('Nmech', val=8070.0, units='rpm', lower=500.0, eq_units='hp', rhs_val=0.0)
            self.connect('balance.Nmech', 'Nmech')
            self.connect('shaft.pwr_net', 'balance.lhs:Nmech')

            # Choked throat area continuity balance
            balance.add_balance('W', val=140.0, units='lbm/s', eq_units='inch**2')
            self.connect('balance.W', 'inlet.Fl_I:stat:W')
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')

        # 7. Non-linear solver configuration
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = -1
        newton.options['maxiter'] = 40
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 100
        newton.options['reraise_child_analysiserror'] = False

        self.linear_solver = om.DirectSolver()

        super().setup()


class MPHybridTurbojet(pyc.MPCycle):
    """
    Multi-point cycle wrapper for design and off-design evaluation.
    """

    def initialize(self):
        self.options.declare('use_rde', default=True)
        super().initialize()

    def setup(self):
        use_rde = self.options['use_rde']

        # Add DESIGN point
        self.pyc_add_pnt('DESIGN', HybridTurbojet(use_rde=use_rde))

        self.set_input_defaults('DESIGN.Nmech', 8070.0, units='rpm')
        self.set_input_defaults('DESIGN.inlet.MN', 0.60)
        self.set_input_defaults('DESIGN.comp.MN', 0.020)
        self.set_input_defaults('DESIGN.burner.MN', 0.020)
        self.set_input_defaults('DESIGN.turb.MN', 0.40)

        if use_rde:
            self.pyc_add_cycle_param('burner.dPqP_inj', 0.12)
            self.pyc_add_cycle_param('burner.eta_rde', 0.85)
            self.pyc_add_cycle_param('burner.dia_annulus', 12.0)
            self.pyc_add_cycle_param('burner.N_waves', 1.0)
        else:
            self.pyc_add_cycle_param('burner.dPqP', 0.03)

        self.pyc_add_cycle_param('nozz.Cv', 0.99)

        # Off-design point (OD0: Mach 0.20 at 5,000 ft)
        self.od_pts = ['OD0']
        self.pyc_add_pnt('OD0', HybridTurbojet(design=False, use_rde=use_rde))
        self.set_input_defaults('OD0.fc.MN', val=0.20)
        self.set_input_defaults('OD0.fc.alt', 5000.0, units='ft')
        self.set_input_defaults('OD0.balance.Fn_target', 8000.0, units='lbf')

        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')

        super().setup()


def run_turbojet_simulation(use_rde=True, cpr=13.5, fn_target=11800.0, t4_target=2370.0):
    """
    Sets up and solves the turbojet cycle for a given CPR and burner configuration.
    """
    prob = om.Problem()
    prob.model = MPHybridTurbojet(use_rde=use_rde)

    prob.set_solver_print(level=-1)
    prob.setup(check=False)

    # Design point conditions
    prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
    prob.set_val('DESIGN.fc.MN', 0.000001)
    prob.set_val('DESIGN.balance.Fn_target', fn_target, units='lbf')
    prob.set_val('DESIGN.balance.T4_target', t4_target, units='degR')
    prob.set_val('DESIGN.comp.PR', cpr)
    prob.set_val('DESIGN.comp.eff', 0.83)
    prob.set_val('DESIGN.turb.eff', 0.86)

    # Balance initial guesses
    prob['DESIGN.balance.W'] = 145.0 if not use_rde else 125.0
    prob['DESIGN.balance.FAR'] = 0.01755
    prob['DESIGN.balance.turb_PR'] = 3.85 if not use_rde else 4.20
    prob['DESIGN.fc.balance.Pt'] = 14.696
    prob['DESIGN.fc.balance.Tt'] = 518.67

    for pt in ['OD0']:
        prob[pt + '.balance.W'] = 140.0 if not use_rde else 120.0
        prob[pt + '.balance.FAR'] = 0.0168
        prob[pt + '.balance.Nmech'] = 8100.0
        prob[pt + '.fc.balance.Pt'] = 13.0
        prob[pt + '.fc.balance.Tt'] = 530.0
        prob[pt + '.turb.PR'] = 4.0

    prob.run_model()

    pt = 'DESIGN'
    pt2 = prob.get_val(f'{pt}.inlet.Fl_O:tot:P', units='psi')[0]
    pt3 = prob.get_val(f'{pt}.comp.Fl_O:tot:P', units='psi')[0]
    pt4 = prob.get_val(f'{pt}.burner.Fl_O:tot:P', units='psi')[0]
    tt4 = prob.get_val(f'{pt}.burner.Fl_O:tot:T', units='degR')[0]
    pt5 = prob.get_val(f'{pt}.turb.Fl_O:tot:P', units='psi')[0]
    w_air = prob.get_val(f'{pt}.inlet.Fl_O:stat:W', units='lbm/s')[0]
    w_fuel = prob.get_val(f'{pt}.perf.Wfuel', units='lbm/s')[0]
    fn = prob.get_val(f'{pt}.perf.Fn', units='lbf')[0]
    tsfc = prob.get_val(f'{pt}.perf.TSFC', units='lbm/(h*lbf)')[0]
    turb_pr = prob.get_val(f'{pt}.turb.PR')[0]
    throat_area = prob.get_val(f'{pt}.nozz.Throat:stat:area', units='inch**2')[0]
    comp_pwr = prob.get_val(f'{pt}.shaft.pwr_out', units='hp')[0]

    diagnostics = {
        'use_rde': use_rde,
        'CPR': cpr,
        'Pt2_psi': pt2,
        'Pt3_psi': pt3,
        'Pt4_psi': pt4,
        'PR_burner': pt4 / pt3,
        'Pt5_psi': pt5,
        'Tt4_degR': tt4,
        'W_air': w_air,
        'W_fuel': w_fuel,
        'Fn_lbf': fn,
        'TSFC': tsfc,
        'Turb_PR': turb_pr,
        'Comp_pwr_hp': comp_pwr,
        'Throat_area_in2': throat_area
    }

    if use_rde:
        diagnostics['D_cj'] = prob.get_val(f'{pt}.burner.D_cj', units='ft/s')[0]
        diagnostics['f_rde'] = prob.get_val(f'{pt}.burner.f_rde', units='Hz')[0]
        diagnostics['Pt_inj'] = prob.get_val(f'{pt}.burner.Pt_inj', units='psi')[0]

    return prob, diagnostics


def viewer(prob, pt, file=sys.stdout):
    """
    print a report of all the relevant cycle properties
    """

    summary_data = (prob[pt+'.fc.Fl_O:stat:MN'], prob[pt+'.fc.alt'], prob[pt+'.inlet.Fl_O:stat:W'],
                    prob[pt+'.perf.Fn'], prob[pt+'.perf.Fg'], prob[pt+'.inlet.F_ram'],
                    prob[pt+'.perf.OPR'], prob[pt+'.perf.TSFC'])
    summary_data = tuple(float(np.asarray(x).item()) if hasattr(x, 'item') else float(x) for x in summary_data)

    print(file=file, flush=True)
    print(file=file, flush=True)
    print(file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                              POINT:", pt, file=file, flush=True)
    print("----------------------------------------------------------------------------", file=file, flush=True)
    print("                       PERFORMANCE CHARACTERISTICS", file=file, flush=True)
    print("    Mach      Alt       W      Fn      Fg    Fram     OPR     TSFC  ", file=file, flush=True)
    print(" %7.5f  %7.1f %7.3f %7.1f %7.1f %7.1f %7.3f  %7.5f" % summary_data, file=file, flush=True)

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'comp.Fl_O', 'burner.Fl_O',
                'turb.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.print_compressor(prob, comp_full_names, file=file)

    burner = prob.model._get_subsystem(f'{pt}.burner')
    if isinstance(burner, RDECombustor):
        pyc.print_rde(prob, [f'{pt}.burner'], file=file)
    else:
        pyc.print_burner(prob, [f'{pt}.burner'], file=file)

    turb_names = ['turb']
    turb_full_names = [f'{pt}.{t}' for t in turb_names]
    pyc.print_turbine(prob, turb_full_names, file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)

    shaft_names = ['shaft']
    shaft_full_names = [f'{pt}.{s}' for s in shaft_names]
    pyc.print_shaft(prob, shaft_full_names, file=file)

    pyc.print_balances(prob, pt, file=file)


def map_plots(prob, pt):
    comp_names = ['comp']
    comp_full_names = [f'{pt}.{c}' for c in comp_names]
    pyc.plot_compressor_maps(prob, comp_full_names)

    turb_names = ['turb']
    turb_full_names = [f'{pt}.{c}' for c in turb_names]
    pyc.plot_turbine_maps(prob, turb_full_names)


RDETurbojet = HybridTurbojet
MPRDETurbojet = MPHybridTurbojet


if __name__ == "__main__":

    import time

    prob = om.Problem()

    mp_turbojet = prob.model = MPHybridTurbojet(use_rde=True)

    prob.setup(check=False)

    # Define the design point
    prob.set_val('DESIGN.fc.alt', 0.0, units='ft')
    prob.set_val('DESIGN.fc.MN', 0.000001)
    prob.set_val('DESIGN.balance.Fn_target', 11800.0, units='lbf')
    prob.set_val('DESIGN.balance.T4_target', 2370.0, units='degR')
    prob.set_val('DESIGN.comp.PR', 13.5)
    prob.set_val('DESIGN.comp.eff', 0.83)
    prob.set_val('DESIGN.turb.eff', 0.86)

    # Set initial guesses for balances
    prob['DESIGN.balance.FAR'] = 0.01755
    prob['DESIGN.balance.W'] = 125.0
    prob['DESIGN.balance.turb_PR'] = 4.20
    prob['DESIGN.fc.balance.Pt'] = 14.696
    prob['DESIGN.fc.balance.Tt'] = 518.67

    for i, pt in enumerate(mp_turbojet.od_pts):
        # initial guesses
        prob[pt + '.balance.W'] = 120.0
        prob[pt + '.balance.FAR'] = 0.01680
        prob[pt + '.balance.Nmech'] = 8197.38
        prob[pt + '.fc.balance.Pt'] = 15.703
        prob[pt + '.fc.balance.Tt'] = 558.31
        prob[pt + '.turb.PR'] = 4.6690

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()

    for pt in ['DESIGN'] + mp_turbojet.od_pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)
