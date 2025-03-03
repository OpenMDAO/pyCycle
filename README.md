[![Build Status](https://travis-ci.org/OpenMDAO/pyCycle.svg?branch=master)](https://travis-ci.org/OpenMDAO/pyCycle)

# pyCycle
--------------

This is a thermodynamic cycle modeling library, designed primarily to model jet engine performance.
It is built on top of the OpenMDAO framework and the design is heavily inspired by NASA's NPSS software.
You need to be at least familiar with either OpenMDAO or NPSS in order to be successful in using this library.

Disclosure: The docs are nearly non-existent. We're hoping to improve this, but for the moment this is what you get.
We suggest you look in the examples folder for some indications of how to run this code.
Also, you can read [the paper on pyCycle](https://www.mdpi.com/2226-4310/6/8/87/pdf) which goes into a lot of detail that is very relevant.

## OpenMDAO Version Compatibility
----------------------------------
pyCycle is built on top of OpenMDAO, and thus depends on it.
Here is the OpenMDAO version you need for the specific versions of pyCycle

| pyCycle version  | OpenMDAO version  |
| -----------------| ----------------  |
| 3.0.0            | 2.8.0 thru 3.1.1  |
| 3.2.0            | 3.2.0 thru 3.5.0  |
| 3.4.0            | 3.3.0 thru 3.5.0  |
| 3.5.0            | 3.5.0 thru 3.7.0  |
| 4.0.0            | 3.7.0 or greater  |
| 4.1.x            | 3.10.0 or greater |
| 4.2.0            | 3.10.0 or greater |

## Version 4.2 --- PyPI release
No significant code changes, but minor adjustments to the package name in `setup.py` to enable publishing to PyPI.

## Citation

If you use pyCycle, please cite this paper:

*E. S. Hendricks and J. S. Gray, “Pycycle: a tool for efficient optimization of gas turbine engine cycles,” Aerospace, vol. 6, iss. 87, 2019.*

    @article{Hendricks2019,
    author="Eric S. Hendricks and Justin S. Gray" ,
    title = "pyCycle: A Tool for Efficient Optimization of Gas Turbine Engine Cycles",
    journal = "Aerospace",
    year = "2019",
    day = "8",
    month = "August",
    volume = {6},
    number = {87},
    doi = {10.3390/aerospace6080087},
    }

## Installation

### PyPI

    If you want to install from PyPI then do the following:

    pip install om-pycycle

    or, if you want to install the (optional) additional testing tools

    pip install 'om-pycycle[all]'

Why is it `om-pycycle` on PyPI?
Because another package already claimed `pyCycle`!
Note that the import does not change though.
You still use `import pycycle` regardless.


### Clone

clone this repo, and checkout the specific version you want to run:

    git clone https://github.com/OpenMDAO/pyCycle
    cd pyCycle

You can see a list of all versions in the repo via:

    git tag

Select one of those tags (e.g. 3.0.0)

    git checkout 3.0.0

or for pyCycle V3.2.0:

    git checkout 3.2.0

or for pyCycle V4.0.0:

    git checkout 4.0.0

Use pip to install:

    pip install -e .[all]


## Testing

After installation if you wat to run the unit test suite you can do so via the `testflo` command:

    testflo pycycle

This will run all the unit tests within the pycycle repository, but note that it will NOT run the longer regression tests from the
`example_cycles` folder.  These tests are written as 'benchmark' tests.
If you want to run these tests, then you need to clone the repository, CD into the `example_cycles` folder and call

    testflo -b .


## Version 4.0 Announcements
Version 4.0 officially supports multiple thermodynamic packages.
Currently there are two: CEA (the original thermo solver) and the new TABULAR option.
Although these are the only two current thermo packages, the code has been setup so that it is expandable to more later.

The tabular thermodynamic is much simpler to use, and much faster to run.
The downside is that it is tied to a specific pre-computed thermodynamic data set that is valid for a specific fuel type, and within a specific temperature range.
We have included an [example script that shows how to generate your own tabular data set](example_cycles/tab_thermo_data_generator.py), which you would need to do for anything other than Jet-A fuel.
Additionally the default tabular thermo data only support fuel (no water injection).
If you want to use tabular thermo for a water injection case, you'll need to generate a new thermo data table.

## Different thermos will give different answers!
Please note that when you switch thermodynamics packages, you will get slightly different answers.
Depending on how finely you sample your thermo data for the tabular package, the differences could be small to modest.
If you see changes greater than 1% on any critical values then you should consider refining your thermodynamic data tables.

### V4 is modestly backwards incompatible
In order to modular thermodynamic happens, some modest changes to the API were needed.

- The `Cycle`, introduced in V3.5.0, is now mandatory. You must build your cycle in this, instead of a basic OpenMDAO `Group`.
- The `pyc_add_element` method has been deprecated (to be removed in version 4.1).
  Improvements to the cycle class made it possible to stick with standard `add_subsystem` calls instead.
- The arguments needed to be passed into Elements during instantiation have been changed (and for the most part significantly simplified).
  The biggest change is that you no longer need to pass element lists to each Element any more. All of the thermodynamic arguments have now been moved up to the `Cycle` group.
- There is a new `Element` class which must be the base class (or at least an ancestor class) for any component that contain flow-ports (anything you would point to in a call to `connect_flow` is an element).
  This new base class has one additional method, `pyc_setup_output_ports` that is required for initialization of the fluid port data.
  If you have developed any of your own custom Elements beyond the standard library, then note that you'll need to update them and define the new method in them.


Over all the, changes are pretty minor, but their impact is significant.
The changes to the Element initialization not only make models simpler,
they also make them thermo-agnostic.