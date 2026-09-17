"""
Step 5: Air-Breathing Rotating Detonation Engine (RDE) Ramjet Cycle.

Simulates and compares a supersonic air-breathing ramjet at Mach 2.5, 40,000 ft:
1. Conventional Isobaric Ramjet (Brayton cycle with 5% combustor pressure loss).
2. Rotating Detonation Engine Ramjet (Humphrey/CJ cycle with pressure gain).

Architecture:
[FlightConditions] -> [Inlet] -> [RDECombustor / Combustor] -> [Nozzle] -> [Performance]
"""

import sys
import numpy as np
import openmdao.api as om
import pycycle.api as pyc

from pycycle.elements.rde_combustor import RDECombustor, print_rde


class RamjetCycle(pyc.Cycle):
    """
    Air-breathing ramjet propulsion cycle supporting either conventional or RDE combustor.
    """

    def initialize(self):
        self.options.declare('use_rde', default=True, desc='Use RDE combustor if True, conventional if False')
        self.options.declare('fuel_type', default='FAR', desc='Fuel type for thermodynamics')
        super().initialize()

    def setup(self):
        use_rde = self.options['use_rde']
        fuel_type = self.options['fuel_type']

        # Use fast TABULAR thermo for air + Jet-A
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        design = self.options['design']

        # 1. Add cycle components
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())

        if use_rde:
            self.add_subsystem('burner', RDECombustor(fuel_type=fuel_type, dia_annulus=14.0))
        else:
            self.add_subsystem('burner', pyc.Combustor(fuel_type=fuel_type))

        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CD', lossCoef='Cv'))
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        # 2. Connect flow stations
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        self.pyc_connect_flow('inlet.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'nozz.Fl_I')

        # 3. Connect ambient pressure to nozzle backpressure
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')

        # 4. Connect performance metrics
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('burner.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')

        # Resolve unit ambiguity on inlet flow rate
        self.set_input_defaults('inlet.Fl_I:stat:W', 100.0, units='lbm/s')
        self.set_input_defaults('burner.Fl_I:FAR', 0.028)
        self.set_input_defaults('inlet.MN', 0.35)
        self.set_input_defaults('burner.MN', 0.30)

        # 5. Solver configuration
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = -1
        newton.options['maxiter'] = 30
        newton.options['solve_subsystems'] = True
        newton.options['reraise_child_analysiserror'] = False

        self.linear_solver = om.DirectSolver()

        super().setup()


def run_ramjet_case(use_rde=True, far_val=0.028, alt_ft=40000.0, mach=2.5, w_air=100.0):
    """
    Sets up and executes a ramjet simulation at the given flight condition.
    """
    prob = om.Problem()
    prob.model = RamjetCycle(use_rde=use_rde, fuel_type='FAR')

    prob.set_solver_print(level=-1)
    prob.setup(check=False)

    # Flight conditions: Mach 2.5 at 40,000 ft
    prob.set_val('fc.alt', alt_ft, units='ft')
    prob.set_val('fc.MN', mach)

    # Inlet conditions (supersonic diffusion to combustor face)
    prob.set_val('inlet.Fl_I:stat:W', w_air, units='lbm/s')
    prob.set_val('inlet.MN', 0.35)
    prob.set_val('inlet.ram_recovery', 0.90)

    # Combustor parameters
    prob.set_val('burner.Fl_I:FAR', far_val)
    prob.set_val('burner.MN', 0.30)

    if use_rde:
        prob.set_val('burner.dPqP_inj', 0.12)
        prob.set_val('burner.eta_rde', 0.85)
        prob.set_val('burner.dia_annulus', 14.0, units='inch')
        prob.set_val('burner.N_waves', 1.0)
    else:
        prob.set_val('burner.dPqP', 0.05)  # 5% conventional pressure loss

    # Nozzle velocity coefficient
    prob.set_val('nozz.Cv', 0.98)

    prob.run_model()

    # Extract metrics
    p0 = prob.get_val('fc.Fl_O:stat:P', units='psi')[0]
    pt0 = prob.get_val('fc.Fl_O:tot:P', units='psi')[0]
    v0 = prob.get_val('fc.Fl_O:stat:V', units='ft/s')[0]
    pt2 = prob.get_val('inlet.Fl_O:tot:P', units='psi')[0]
    pt4 = prob.get_val('burner.Fl_O:tot:P', units='psi')[0]
    tt4 = prob.get_val('burner.Fl_O:tot:T', units='degR')[0]
    wfuel = prob.get_val('burner.Wfuel', units='lbm/s')[0]
    fram = prob.get_val('inlet.F_ram', units='lbf')[0]
    fg = prob.get_val('nozz.Fg', units='lbf')[0]
    fn = prob.get_val('perf.Fn', units='lbf')[0]
    tsfc = prob.get_val('perf.TSFC', units='lbm/(h*lbf)')[0]
    isp = fn / wfuel  # Specific impulse (lbf-s / lbm)
    pr_burner = pt4 / pt2
    throat_area = prob.get_val('nozz.Throat:stat:area', units='inch**2')[0]
    exit_area = prob.get_val('nozz.Fl_O:stat:area', units='inch**2')[0]

    diagnostics = {
        'use_rde': use_rde,
        'P0_psi': p0,
        'Pt0_psi': pt0,
        'V0_fts': v0,
        'Pt2_psi': pt2,
        'Pt4_psi': pt4,
        'PR_burner': pr_burner,
        'Tt4_degR': tt4,
        'W_air': w_air,
        'W_fuel': wfuel,
        'F_ram': fram,
        'F_g': fg,
        'F_n': fn,
        'TSFC': tsfc,
        'Isp_s': isp,
        'Throat_area_in2': throat_area,
        'Exit_area_in2': exit_area
    }

    if use_rde:
        diagnostics['D_cj'] = prob.get_val('burner.D_cj', units='ft/s')[0]
        diagnostics['f_rde'] = prob.get_val('burner.f_rde', units='Hz')[0]
        diagnostics['Pt_inj'] = prob.get_val('burner.Pt_inj', units='psi')[0]

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

    fs_names = ['fc.Fl_O', 'inlet.Fl_O', 'burner.Fl_O', 'nozz.Fl_O']
    fs_full_names = [f'{pt}.{fs}' for fs in fs_names]
    pyc.print_flow_station(prob, fs_full_names, file=file)

    burner = prob.model._get_subsystem(f'{pt}.burner')
    if isinstance(burner, RDECombustor):
        pyc.print_rde(prob, [f'{pt}.burner'], file=file)
    else:
        pyc.print_burner(prob, [f'{pt}.burner'], file=file)

    noz_names = ['nozz']
    noz_full_names = [f'{pt}.{n}' for n in noz_names]
    pyc.print_nozzle(prob, noz_full_names, file=file)


class MPRDERamjet(pyc.MPCycle):

    def setup(self):
        self.pyc_add_pnt('DESIGN', RamjetCycle(use_rde=True, fuel_type='FAR'))

        self.set_input_defaults('DESIGN.inlet.MN', 0.35)
        self.set_input_defaults('DESIGN.burner.MN', 0.30)
        self.set_input_defaults('DESIGN.inlet.Fl_I:stat:W', 100.0, units='lbm/s')
        self.set_input_defaults('DESIGN.burner.Fl_I:FAR', 0.028)
        self.set_input_defaults('DESIGN.inlet.ram_recovery', 0.90)

        self.pyc_add_cycle_param('burner.dPqP_inj', 0.12)
        self.pyc_add_cycle_param('burner.eta_rde', 0.85)
        self.pyc_add_cycle_param('burner.dia_annulus', 14.0)
        self.pyc_add_cycle_param('burner.N_waves', 1.0)
        self.pyc_add_cycle_param('nozz.Cv', 0.98)

        self.od_pts = []

        super().setup()


RDERamjet = RamjetCycle


if __name__ == "__main__":

    import time

    prob = om.Problem()

    mp_ramjet = prob.model = MPRDERamjet()

    prob.setup(check=False)

    # Define the design point
    prob.set_val('DESIGN.fc.alt', 40000.0, units='ft')
    prob.set_val('DESIGN.fc.MN', 2.50)

    st = time.time()

    prob.set_solver_print(level=-1)
    prob.set_solver_print(level=2, depth=1)

    prob.run_model()

    for pt in ['DESIGN'] + mp_ramjet.od_pts:
        viewer(prob, pt)

    print()
    print("time", time.time() - st)
