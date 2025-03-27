
***********************************
# Release Notes for 4.3.0

Mar 27, 2025

* Clarified how to test example cycles & made fixes to some related test scripts [#83](https://github.com/OpenMDAO/pyCycle/pull/83)
* Fix for SolverWarning on high_bypass_turbofan.py [#80](https://github.com/OpenMDAO/pyCycle/pull/80)
* Remove cast conversion of an array with ndim > 0 to scalar [#77](https://github.com/OpenMDAO/pyCycle/pull/77)
* Added missing design off design area connection in burner [#62](https://github.com/OpenMDAO/pyCycle/pull/62)
* For tabular aero, use the 3d-slinear if possible. [#59](https://github.com/OpenMDAO/pyCycle/pull/59)
* changed several places where np.seterr was being called in order to keep the changes localized. [#58](https://github.com/OpenMDAO/pyCycle/pull/58)
* fix tests that used component as model [#57](https://github.com/OpenMDAO/pyCycle/pull/57)
* Add a GitHub Actions test workflow [#56](https://github.com/OpenMDAO/pyCycle/pull/56)
* replace deprecated 'value' keyword with 'val' [#55](https://github.com/OpenMDAO/pyCycle/pull/55)
* Fixed incorrect ordering in compressor viewer [#54](https://github.com/OpenMDAO/pyCycle/pull/54)



***********************************
# Release Notes for 4.2.1

Jan 25, 2022

* updated setup.py with description and dynamic version

***********************************
# Release Notes for 4.2.0

Jan 25, 2022

* changed the package name to om-pycycle for release on pypi by @swryan in https://github.com/OpenMDAO/pyCycle/pull/52

***********************************
# Release Notes for 4.1.2

Jan 24, 2022

* updated setup.py with all modules/data and added optional test dependency by @swryan in https://github.com/OpenMDAO/pyCycle/pull/51

***********************************
# Release Notes for 4.1.1

Jan 21, 2022

* Added variables to viewer, removed IVC from compressor and turbine map, and lowered hpt map speed bound by @jdgratz10 in https://github.com/OpenMDAO/pyCycle/pull/49

***********************************
# Release Notes for 4.1.0

July 21, 2021

* Updating for API changes to OM 3.10 by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/45

***********************************
# Release Notes for 4.0.0

July 21, 2021

* Code cleanup - removing `n` from flow stations by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/28
* rename b0 to `composition` by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/29
* Refactoring of FlowIn component to remove dependence on thermo specific stuff by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/30
* fix for complex warnings. DO NOT merge this until the openmdao complex vector refactor is merged! by @naylor-b in https://github.com/OpenMDAO/pyCycle/pull/32
* Mil Spec Option for Inlet by @caflack in https://github.com/OpenMDAO/pyCycle/pull/27
* fixing n3ref exec-comp by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/33
* Tabular Thermo!! by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/37
* fixed the N3 benchmark by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/38
* Added tabular package to setup by @lamkina in https://github.com/OpenMDAO/pyCycle/pull/39
* fixing N+3 benchmark by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/40
* change the way tabular data is loaded, so it only gets imported once by @JustinSGray in https://github.com/OpenMDAO/pyCycle/pull/43
* updated compressor docstring by @anilyil in https://github.com/OpenMDAO/pyCycle/pull/44

***********************************
# Release Notes for 3.9.9

** Backwards incompatible API changes **

The thermo refactorization continues.
A lot more internal modularization happened,
and though it mostly does not impact user facing APIs for model building,
it does affect strongly the way that Elements are coded.
The goal continues to be to change things to make it possible to add swappable thermo but we are not quite there yet.

* `Element` base class has been introduced and must be used for anything that contains fluid ports.

* `Element` now have a `pyc_setup_output_ports` method that must be implemented to propagate
port setup data downstream.

* `pyc_add_element` has been deprecated. It was just added in V3.4.0, so it had a short but exciting life.
It is not needed any more because the presence of the `Element` class made it possible to handle all the details with the standard `add_subsystem` call.
It seems better to stick to stock OpenMDAO APIs where possible, so thats what we're going to do.
The old method is getting deprecated, and you'll get a noisy warning. Expect it to go away in V4.0

***********************************
# Release Notes for 3.4.0

* new `MPCycle` (stands for MultiPoint Cycle) and `Cycle` classes that you can optionally use to simplify your models and reduce boiler plate code associated with connecting data between design and off-design instances.
* major refactor of the thermodynamics library that won't directly affect your models, but is a major cleanup of the core thermo implementation. Though it is fully backwards compatible with 3.2.0, you may notice some small numerical differences due to slightly different solver structure for the CEA solver.

The thermo refactor has been specifically designed to make it easier to swap between multiple thermodynamics analyses
(i.e. simpler ones than CEA).
No other thermodynamic solvers are currently implemented, but they will be coming in future versions.

The features that will allow you to select from multiple thermodynamics libraries will be integrated in pyCycle 4.0.0.
This version will likely be slightly backwards incompatible, in terms of how you instantiate the elements.
If possible we'll provide a deprecations, but regardless it should be fairly simple to update to the new APIs.