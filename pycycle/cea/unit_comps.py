from __future__ import print_function

import inspect
import numpy as np
from six import iteritems, PY3

from openmdao.api import ExplicitComponent

if PY3:
    _full_out_args = inspect.getfullargspec(ExplicitComponent.add_output)
    _allowed_out_args = set(_full_out_args.args[3:] + _full_out_args.kwonlyargs)
else:
    _full_out_args = inspect.getargspec(ExplicitComponent.add_output)
    _allowed_out_args = set(_full_out_args.args[3:])


class UnitCompBase(ExplicitComponent):

    def __init__(self, thermo, fl_name):

        super(UnitCompBase, self).__init__()

        self.thermo = thermo
        self.fl_name = fl_name

    def setup(self):
        rel2meta = self._var_rel2meta

        fl_name = self.fl_name

        for in_name in self._var_rel_names['input']:

            meta = rel2meta[in_name]
            val = meta['value'].copy()
            new_meta = {k:v for k, v in iteritems(meta) if k in _allowed_out_args}

            out_name = '{0}:{1}'.format(fl_name, in_name)
            self.add_output(out_name, val=val, **new_meta)

        rel2meta = self._var_rel2meta

        for in_name, out_name in zip(self._var_rel_names['input'], self._var_rel_names['output']):

            shape = rel2meta[in_name]['shape']
            size = np.prod(shape)
            row_col = np.arange(size, dtype=int)

            self.declare_partials(of=out_name, wrt=in_name,
                                  val=np.ones(size), rows=row_col, cols=row_col)

            # TODO-JSG: FD related bug?
            # self.approx_partials(of='*', wrt='*', step=1e-5)

    def compute(self, inputs, outputs):
        outputs._data[:] = inputs._data


class EngUnitProps(UnitCompBase):
    """only job is to provide unknowns in english units"""

    def setup(self):

        thermo = self.thermo

        self.add_input('T', val=284., units="degR", desc="Temperature")
        self.add_input('P', val=1., units='lbf/inch**2', desc="Pressure")
        self.add_input('h', val=1., units="Btu/lbm", desc="enthalpy")
        self.add_input('S', val=1., units="Btu/(lbm*degR)", desc="entropy")
        self.add_input('gamma', val=1.4, desc="ratio of specific heats")
        self.add_input('Cp', val=1., units="Btu/(lbm*degR)", desc="Specific heat at constant pressure")
        self.add_input('Cv', val=1., units="Btu/(lbm*degR)", desc="Specific heat at constant volume")
        self.add_input('rho', val=1., units="lbm/ft**3", desc="density")
        self.add_input('n', val=np.ones(thermo.num_prod))
        self.add_input('n_moles', val=1.)
        self.add_input('R', val=1.0, units="Btu/(lbm*degR)", desc='Total specific gas constant')

        super(EngUnitProps, self).setup()


class EngUnitStaticProps(UnitCompBase):

    def setup(self):

        self.add_input('area', val=1.0, units="inch**2")
        self.add_input('W', val=1.0, units="lbm/s")
        self.add_input('V', val=1.0, units="ft/s")
        self.add_input('Vsonic', val=1.0, units="ft/s")
        self.add_input('MN', val=0.5)

        super(EngUnitStaticProps, self).setup()


if __name__ == "__main__":

    from openmdao.api import Problem, Group, IndepVarComp
    from pycycle.cea import species_data

    thermo = species_data.Thermo(species_data.co2_co_o2)

    p = Problem()
    model = p.model = Group()
    indep = model.add_subsystem('indep', IndepVarComp(), promotes=['*'])
    # indep.add_output('T', val=100., units='degK')
    # indep.add_output('P', val=1., units='bar')
    indep.add_output('T', val=100., units='degR')
    indep.add_output('P', val=1., units='psi')

    model.add_subsystem('units', EngUnitProps(thermo=thermo), promotes=['*'])

    p.setup()

    p.run_model()

    p.model.run_linearize()
    jac = p.model.get_subsystem('units').jacobian._subjacs

    for pair in jac:
        print(pair)
        print(jac[pair])
        print()