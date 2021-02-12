"""phi = ( Tgas' - Tmetal )/ ( Tgas' - Tcool )
phi' = ( profilefactor + phi )/( profilefactor + 1. )

where
Tgas' = 1.00*Tgas + 150R for vanes
Tgas' = 0.92*Tgas + 150R for rotors
and profilefactor = 0.30 for the first vane, 0.13 afterwards; the 150R is a factor of safety delta.

the cooling flow is then
Wc = Win*0.022*xFactor*(4./3.) ( phi' /( 1. - phi' ))**1.25

xFactor=0.90 is my improved technology N+3 fudge factor and the (4./3.) is the original fudge factor for leakage, I think.

The exit temperature from  blade row Tout is based on the enthalpy balance between the two flows minus any work done from rotors.
The Tout is then used as the Tgas for the next blade row, and Wout is used as the next Win.

The results for the N+3 are:
row       Tgas'    Tmetal       Tcool             phi            phi'          Win            Wc          Tout
   1     3550.00   2460.00      1721.97           0.596271      0.689439       62.15          4.44635     3299.28
   2     3185.33   2460.00      1721.97           0.495663      0.553684       66.60          2.30197     2846.45
   3     2996.45   2460.00      1721.97           0.420919      0.487539       68.90          1.70911     2821.58
   4     2745.85   2460.00      1721.97           0.279185      0.362111       70.61          0.91853     2412.12
Wout  = 71.53

************************************************************************************************************************************
Date:07/17/17    Time:10:57:07    Model:          N+3 Gear Drive Turbofan Engine for Small Core Research    converge = 1   CASE:   0
Version:            NPSS_2.6          Gas Package:  Janaf        iter/pass/Jacb/Broy= 24/132/ 2/21        Run by:           smjones1

                                       SUMMARY OUTPUT DATA
    MN       alt    dTamb         W         Fn      TSFC      Wfuel       BPR      VTAS      OPR       T4      T41      T49
 0.250       0.0    27.00   1903.72    22800.0    0.2891    6590.89   25.7674    169.59   42.892   3400.0   3299.5   2431.0
                          core size: 2.96     Q:  92.57     Maximum Thrust:     0.0     % Thrust:  0.0     Power code: 50.0

                                        FLOW STATION DATA
                                W        Pt        Tt       ht     FAR       Wc        Ps        Ts      rhos     Aphy      MN      gamt
FS_1   PNT2.start.Fl_O    1903.72    15.349    552.49     1.94  0.0000  1881.21    14.696    545.67  0.072692  13173.8  0.2500   1.39973
FS_2   PNT2.inlet.Fl_O    1903.72    15.303    552.49     1.94  0.0000  1886.87    12.673    523.51  0.065338   7109.8  0.5261   1.39973
FS_21  PNT2.fan.Fl_O      1903.72    18.633    585.49     9.86  0.0000  1595.26    16.496    565.49  0.078738   7098.0  0.4208   1.39912
FS_13  PNT2.splitter.Fl>  1832.60    18.633    585.49     9.86  0.0000  1535.67    16.483    565.36  0.078690   6813.9  0.4223   1.39912
FS_22  PNT2.splitter.Fl>    71.12    18.633    585.49     9.86  0.0000    59.60    16.808    568.53  0.079799    284.1  0.3866   1.39912
FS_23  PNT2.duct2.Fl_O      71.12    18.495    585.49     9.86  0.0000    60.04    16.695    568.63  0.079246    286.9  0.3854   1.39912
FS_24a PNT2.LPC.Fl_O        71.12    49.276    789.40    59.14  0.0000    26.17    43.317    761.21  0.153594    113.8  0.4340   1.39228
FS_24  PNT2.bld25.Fl_O      71.12    49.276    789.40    59.14  0.0000    26.17    43.317    761.21  0.153594    113.8  0.4340   1.39228
FS_25  PNT2.duct25.Fl_O     71.12    48.588    789.40    59.14  0.0000    26.54    42.727    761.29  0.151488    115.6  0.4335   1.39228
FS_3   PNT2.HPC.Fl_O        69.70   642.433   1721.97   298.48  0.0000     2.91   603.389   1694.79  0.960952     17.2  0.3070   1.33952
FS_36  PNT2.bld3.Fl_O       60.32   642.433   1721.97   298.48  0.0000     2.51   603.389   1694.79  0.960952     14.9  0.3070   1.33952
FS_4   PNT2.burner.Fl_O     62.15   616.736   3400.00   256.40  0.0304     3.79   612.934   3395.48  0.487315     67.5  0.1002   1.26566
FS_45  PNT2.HPT.Fl_O        71.53   149.113   2430.94    22.86  0.0263    15.26   140.578   2398.29  0.158246     92.9  0.3035   1.29490
FS_48  PNT2.duct45.Fl_O     71.53   148.350   2430.96    22.86  0.0263    15.34   129.992   2358.32  0.148810     66.3  0.4559   1.29490
FS_5   PNT2.LPT.Fl_O        72.95    17.817   1517.63  -231.60  0.0257   102.93    16.988   1499.69  0.030582    691.6  0.2682   1.33189
FS_7   PNT2.duct5.Fl_O      72.95    17.713   1517.69  -231.60  0.0257   103.54    17.277   1508.31  0.030925    945.0  0.1936   1.33188
FS_9   PNT2.core_nozz.F>    72.95    17.713   1517.75  -231.60  0.0257   103.54    14.696   1448.46  0.027391    393.4  0.5350   1.33188
FS_15  PNT2.bypBld.Fl_O   1832.60    18.633    585.49     9.86  0.0000  1535.67    16.483    565.36  0.078690   6813.9  0.4223   1.39912
FS_17  PNT2.duct17.Fl_O   1832.60    18.387    585.49     9.86  0.0000  1556.23    16.274    565.45  0.077681   6917.7  0.4214   1.39912
FS_19  PNT2.byp_nozz.Fl>  1832.60    18.387    585.49     9.86  0.0000  1556.23    14.696    549.22  0.072222   5531.9  0.5750   1.39912

TURBOMACHINERY PERFORMANCE DATA
               Wc      PR      eff         Nc       TR   efPoly        pwr     SMN     SMW
PNT2.fan  1886.87   1.218   0.9676   2073.543   1.0597   0.9685   -21354.3   17.09   11.73
PNT2.LPC    60.04   2.664   0.9207   6244.222   1.3483   0.9307    -4958.0   38.48   31.75
PNT2.HPC    26.54  13.222   0.8497  18051.144   2.1814   0.8906   -23843.4   22.19   22.29
PNT2.HPT     3.79   4.136   0.9320    381.916   1.3282   0.9071    24193.5
PNT2.LPT    15.34   8.326   0.9439    134.556   1.5971   0.9264    26578.0

TURBOMACHINERY MAP DATA
            WcMap   PRmap   effMap      NcMap    R/Parm     s_WcDes   s_PRdes  s_effDes    s_NcDes
PNT2.fan  2455.69   1.283   0.9338      0.879    1.7500      0.7684    0.7692    1.0362   2359.972
PNT2.LPC   199.65   1.380   0.9030      0.976    2.0112      0.3007    4.3745    1.0195   6398.438
PNT2.HPC   199.86  21.521   0.8548      0.989    2.0660      0.1328    0.5956    0.9940  18242.892
PNT2.HPT    30.11   5.028   0.9335    102.702    5.0285      0.1951    0.7785    0.9984      3.719
PNT2.LPT   151.08   5.722   0.9175     93.948    5.7219      0.1573    1.5515    1.0288      1.432

===INLETS====    eRam       Afs      Fram
PNT2.inlet     0.9970  13173.76   16938.1        BLEEDS - interstg  Wb/Win    BldWk     BldP          W        Tt       ht        Pt
                                                 CstmrBld  PNT2.>   0.0000   0.3500   0.1465     0.0000   1127.70   142.91   135.587
====DUCTS====  dPnorm        MN      Aphy        C_LPTexit PNT2.>   0.0200   0.5000   0.1465     1.4224   1268.62   178.81   135.587
PNT2.duct2     0.0074    0.3866    284.06        C_LPTinlt PNT2.>   0.0000   0.5000   0.1465     0.0000   1268.62   178.81   135.587
PNT2.duct25    0.0140    0.4340    113.85
PNT2.duct45    0.0051    0.3035     92.87        BLEEDS - output    Wb/Win   hscale   Pscale          W        Tt       ht        Pt
PNT2.duct5     0.0059    0.2682    691.63        LPC_SB    PNT2.>   0.0000   1.0000   1.0000     0.0000    789.40    59.14    49.276
PNT2.duct17    0.0132    0.4223   6813.90        C_HPTexit PNT2.>   0.0693   1.0000   1.0000     4.9296   1721.97   298.48   642.433
                                                 C_HPTinlt PNT2.>   0.0625   1.0000   1.0000     4.4464   1721.97   298.48   642.433
==SPLITTERS==     BPR    dP/P 1    dP/P 2        Duct17Lk  PNT2.>   0.0000   1.0000   1.0000     0.0000    585.49     9.86    18.633
PNT2.splitt>  25.7674    0.0000    0.0000

===SHAFTS====   Nmech    trq in    pwr in
PNT2.HP_sha>  22269.3    5705.9   24193.5
PNT2.LP_sha>   6634.3   21040.8   26578.0
PNT2.fan_sh>   2140.1   52407.2   21354.3


===BURNERS===   TtOut       eff    dPnorm      Wfuel       FAR    EINOx
PNT2.burner   3400.00    0.9990    0.0400    1.83080   0.03035   44.812


===NOZZLES===      PR       Cfg      CdTh       Cv        Ath      MNth     Vact        Fg       Vj ideal
Byp_Nozz        1.251    0.9975    1.0000   0.9975    5531.92     0.575    658.9   37528.0          660.5
Core_Nozz       1.205    0.9999    1.0000   0.9999     393.42     0.535    974.7    2210.1          974.8

"""

import unittest

import numpy as np

from openmdao.api import Problem, Group, IndepVarComp
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.assert_utils import assert_check_partials


from pycycle.elements import cooling, flow_start
from pycycle.thermo.cea import species_data
from pycycle.thermo.thermo import ThermoAdd
from pycycle.constants import CEA_AIR_COMPOSITION, CEA_AIR_FUEL_COMPOSITION


class Tests(unittest.TestCase):


    def test_cooling_calcs(self):
        """test the basic cooling requirement calculations"""
        p = Problem()

        p.model.set_input_defaults('x_factor', val=.9)
        p.model.set_input_defaults('W_primary', val=62.15, units='lbm/s')
        p.model.set_input_defaults('Tt_primary', val=3400.00, units='degR')
        p.model.set_input_defaults('Tt_cool', val=1721.97, units='degR')

        p.model.add_subsystem('w_cool', cooling.CoolingCalcs(n_stages=2, i_row=0,
                                                               T_metal=2460., T_safety=150.),
                                promotes=['*'])

        p.setup()
        p.set_solver_print(0)

        p.run_model()

        tol = 1e-4
        assert_near_equal(p['W_cool'], 4.44635, tol)
        # assert_near_equal(self, p['Pt_out'], 4.44635, tol) # TODO: set this
        # assert_near_equal(self, p['ht_out'], 4.44635, tol)

    

    def test_row(self):
        """test the flow mixing calculations for a single row"""
        p = Problem()

        # n_init = np.array([3.23319258e-04, 1.00000000e-10, 1.10131241e-05, 1.00000000e-10,
        #                    1.63212420e-10, 6.18813039e-09, 1.00000000e-10, 2.69578835e-02,
        #                    1.00000000e-10, 7.23198770e-03])
        # p.model.set_input_defaults('row.n_cool', val=n_init)  # product ratios for clean air
        b0_air = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]
        p.model.set_input_defaults('row.cool:composition', val=b0_air)
        p.model.set_input_defaults('Pt_in', val=616.736, units='psi')
        p.model.set_input_defaults('Pt_out', val=149.113, units='psi')
        p.model.set_input_defaults('row.x_factor', val=.9)
        p.model.set_input_defaults('row.W_primary', val=62.15, units='lbm/s')
        p.model.set_input_defaults('row.Tt_primary', val=3400.00, units='degR')
        p.model.set_input_defaults('row.Tt_cool', val=1721.97, units='degR')
        p.model.set_input_defaults('row.ht_cool', val=298.48, units='Btu/lbm')
        p.model.set_input_defaults('row.ht_primary', val=250.32757333, units='Btu/lbm')
        p.model.set_input_defaults('row.composition_primary', val=[0.00031378, 0.00211278, 0.00420881, 0.05232509, 0.01405863])

        p.model.add_subsystem(
            'row',
            cooling.Row(
                n_stages=2,
                i_row=0,
                T_metal=2460.,
                T_safety=150.,
                thermo_method='CEA',
                thermo_data=species_data.janaf,
                main_flow_composition=CEA_AIR_FUEL_COMPOSITION, 
                bld_flow_composition=CEA_AIR_COMPOSITION, 
                mix_flow_composition=CEA_AIR_FUEL_COMPOSITION),
            promotes=[
                'Pt_in',
                'Pt_out'])

        p.setup()


        p.set_solver_print(0)

        p.run_model()

        tol = 3e-4
        assert_near_equal(p['row.W_cool'], 4.44635, tol)
        # first row mass flow is primary + cooling
        assert_near_equal(p['row.W_out'], 66.60, tol)
        assert_near_equal(p['row.Fl_O:tot:T'], 3299.28, tol)

    def test_turbine_cooling(self):
        """test the flow calculations and final temperatures for multiple rows"""
        p = Problem()
        # p = self.prob

        # n_init = np.array([3.23319258e-04, 1.00000000e-10, 1.10131241e-05, 1.00000000e-10,
        #                    1.63212420e-10, 6.18813039e-09, 1.00000000e-10, 2.69578835e-02,
        #                    1.00000000e-10, 7.23198770e-03])


        # need ivc here, because b0 is shape_by_conn
        b0_air = [3.23319258e-04, 1.10132241e-05, 5.39157736e-02, 1.44860147e-02]
        b0_mix = [0.00031378, 0.00211278, 0.00420881, 0.05232509, 0.01405863]
        ivc = p.model.add_subsystem('ivc', IndepVarComp())
        ivc.add_output('air_composition', b0_air)
        ivc.add_output('mix_composition', b0_mix)


        # p.model.set_input_defaults('turb_cool.Fl_cool:tot:composition', val=b0_air)  # product ratios for clean air
        p.model.set_input_defaults('turb_cool.turb_pwr', val=24193.5, units='hp')
        p.model.set_input_defaults('turb_cool.Fl_turb_I:tot:P', val=616.736, units='psi')
        p.model.set_input_defaults('turb_cool.Fl_turb_O:tot:P', val=149.113, units='psi')
        p.model.set_input_defaults('turb_cool.Fl_turb_I:stat:W', val=62.15, units='lbm/s')
        p.model.set_input_defaults('turb_cool.Fl_turb_I:tot:T', val=3400.00, units='degR')
        p.model.set_input_defaults('turb_cool.Fl_cool:tot:T', val=1721.97, units='degR')
        p.model.set_input_defaults('turb_cool.Fl_turb_I:tot:h', val=250.097, units='Btu/lbm')
        p.model.set_input_defaults('turb_cool.Fl_cool:tot:h', val=298.48, units='Btu/lbm')

        cool_comp = p.model.add_subsystem('turb_cool',
                                          cooling.TurbineCooling(
                                            n_stages=2,
                                            T_metal=2460.,
                                            T_safety=150.,
                                            thermo_method='CEA',
                                            thermo_data=species_data.janaf))

        cool_comp.Fl_I_data['Fl_turb_I'] = CEA_AIR_FUEL_COMPOSITION
        cool_comp.Fl_I_data['Fl_cool'] = CEA_AIR_COMPOSITION
        cool_comp.Fl_I_data['Fl_turb_O'] = CEA_AIR_FUEL_COMPOSITION
        cool_comp.pyc_setup_output_ports()

        p.model.connect('ivc.mix_composition', ['turb_cool.Fl_turb_I:tot:composition', 
                                                     'turb_cool.Fl_turb_I:stat:composition' ])
        p.model.connect('ivc.mix_composition', ['turb_cool.Fl_turb_O:tot:composition', 
                                                     'turb_cool.Fl_turb_O:stat:composition'])
        p.model.connect('ivc.air_composition', ['turb_cool.Fl_cool:tot:composition', 'turb_cool.Fl_cool:stat:composition'])

        p.setup()
        p.set_solver_print(0)

        p.set_val('turb_cool.x_factor', .9)

        p.run_model()

        tol = 4e-4

        assert_near_equal(p['turb_cool.row_0.Fl_O:tot:T'], 3299.28, tol)
        assert_near_equal(p['turb_cool.row_1.Fl_O:tot:T'], 2846.45, tol)
        assert_near_equal(p['turb_cool.row_2.Fl_O:tot:T'], 2821.58, tol)
        assert_near_equal(p['turb_cool.row_3.Fl_O:tot:T'], 2412.12, tol)

        assert_near_equal(p['turb_cool.row_0.W_cool'], 4.44635, tol)
        assert_near_equal(p['turb_cool.row_1.W_cool'][0], 2.2981, tol)
        assert_near_equal(p['turb_cool.row_2.W_cool'], 1.7079, tol)
        assert_near_equal(p['turb_cool.row_3.W_cool'][0], 0.91799, tol)

        np.set_printoptions(precision=5)
        check = p.check_partials(includes=['turb_cool.row_0.cooling_calcs',
                                           'turb_cool.row_1.cooling_calcs'],
                                 out_stream=None)

        assert_check_partials(check, atol=1e-6, rtol=1e2)


if __name__ == "__main__":
    unittest.main()
