from openmdao.core.component import Component
from openmdao.core.system import System
import numpy as np

def check_element_partials(testcase, prob, depth = 2, tol = 1e-5):
    """
    gradient checking: analytic vs finite difference
    tests compare gradients for element-level calcs only (no thermo)
    """
    sys = prob.model.system_iter(recurse=True, typ=Component)
    comps = [i.pathname + '.*' for i in sys if len(i.pathname.split(".")) <= depth]

    derivs = prob.check_partials(out_stream=None, includes=comps)

    for name in derivs.keys():
        n_checks = 0
        for var_pair in derivs[name].keys():
            # use relative error if derivatives seem nonzero
            if derivs[name][var_pair]['magnitude'][-1] > tol:
                rel_err =  max(derivs[name][var_pair]['rel error'])
            # otherwise, use absolute error to avoid NaN in the checks
            else:
                rel_err = np.linalg.norm(derivs[name][var_pair]['J_fwd'])
            testcase.assertLessEqual(rel_err, tol, name +': '+ '  w.r.t '.join(var_pair))
            n_checks += 1
        # if n_checks > 0:
        #     print("%s : verified %d derivatives" % (name, n_checks))


def regression_generator(prob, exclude = ['.props.', '.chem_eq.']):
    """
    Generates code for regression testing a model against later
    versions of itself.
    """
    print("class TestGenerated(unittest.TestCase):")
    print("    # generated from from pycycle.elements.test.util.regression_generator")
    print("    def test_case0(self):")
    print("        # captured inputs:")
    for name in prob.model._inputs._views.keys():
        if any([pattern in name for pattern in exclude]):
            continue
        print("        prob['%s'] =" % name, "np." + np.array_repr(prob.model._inputs._views[name], precision=16))
    print()
    print("        # captured outputs:")
    for name in prob.model._outputs._views.keys():
        if any([pattern in name for pattern in exclude]):
            continue
        print("        assert_rel_error(self, prob['%s']," % name, "np." +np.array_repr(prob.model._outputs._views[name], precision=16) + ", tol)")
    print()
    print("        check_element_partials(self, prob, depth=1)")
