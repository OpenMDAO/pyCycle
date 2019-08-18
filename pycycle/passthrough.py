import numpy as np

from openmdao.api import ExplicitComponent


class PassThrough(ExplicitComponent):
    """
    Helper component that is needed when variables must be passed directly from
    input to output of a cycle element with no other component in between
    """

    def __init__(self, i_var, o_var, val, units=None):
        super(PassThrough, self).__init__()
        self.i_var = i_var
        self.o_var = o_var
        self.units = units
        self.val = val

        if isinstance(val, (float, int)) or np.isscalar(val):
            size=1
        else:
            size = np.prod(val.shape)

        self.size = size

    def setup(self):
        if self.units is None:
            self.add_input(self.i_var, self.val)
            self.add_output(self.o_var, self.val)
        else:
            self.add_input(self.i_var, self.val, units=self.units)
            self.add_output(self.o_var, self.val, units=self.units)

        #partial derivs setup
        row_col = np.arange(self.size)
        self.declare_partials(of=self.o_var, wrt=self.i_var,
                              val=np.ones(self.size), rows=row_col, cols=row_col)

    def compute(self, inputs, outputs):

        outputs[self.o_var] = inputs[self.i_var]

    def compute_partials(self, inputs, J):
        pass


if __name__ == "__main__":

    from openmdao.api import Problem, IndepVarComp

    p = Problem()

    indeps = p.model.add_subsystem('indeps', IndepVarComp(), promotes=['*'])
    indeps.add_output('foo', val=np.ones(4))

    p.model.add_subsystem('pt', PassThrough("foo", "bar", val=np.ones(4)), promotes=['*'])

    p.setup()
    p.run_model()

    p.check_partial_derivatives()
