[![Build Status](https://travis-ci.org/OpenMDAO/pyCycle.svg?branch=master)](https://travis-ci.org/OpenMDAO/pyCycle)

# pyCycle
--------------

This is a thermodynamic cycle modeling library, designed primarily to model jet engine performance. 
It is built on top of the OpenMDAO framework and the design is heavily inspired by NASA's NPSS software.
You need to be at least familiar with either OpenMDAO or NPSS in order to be successful in using this library. 

Disclosure: The docs are nearly non-existent. We're hoping to improve this, but for the moment this is what you get. 
We suggest you look in the examples folder for some indications of how to run this code. 
Also, you can read [the paper on pyCycle](https://www.mdpi.com/2226-4310/6/8/87/pdf) which goes into a lot of detail that is very relevant. 


# OpenMDAO Version Compatibility
----------------------------------
pyCycle is built on top of OpenMDAO, and thus depends on it. 
Here is the OpenMDAO version you need for the specific versions of pyCycle

| pyCycle version  | OpenMDAO version |
| -----------------| -------------    |
| 3.0.0            | 2.8.0 thru 3.1.1  |
| 3.2.0            | 3.2.0 or greater  |
| 3.4.0            | 3.3.0 or greater  |

## pyCycle 3.4.0 includes the following features: 
* new `MPCycle` (stands for MultiPoint Cycle) and `Cycle` classes that you can optionally use to simplify your models and reduce boiler plate code associated with connecting data between design and off-design instances. 
* major refactor of the thermodynamics library that won't directly affect your models, but is a major cleanup of the core thermo implementation. Though it is fully backwards compatible with 3.2.0, you may notice some small numerical differences due to slightly different solver structure for the CEA solver. 

The thermo refactor has been specifically designed to make it easier to swap between multiple thermodynamics analyses (i.e. simpler ones than CEA). 
No other thermodynamic solvers are currently implemented, but they will be coming in future versions. 

The features that will allow you to select from multiple thermodynamics libraries will be integrated in pyCycle 4.0.0. 
This version will likely be slightly backwards incompatible, in terms of how you instantiate the elements. 
If possible we'll provide a deprecations, but regardless it should be fairly simple to update to the new APIs. 


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

clone this repo, and checkout the specific version you want to run: 

    git clone https://github.com/OpenMDAO/pyCycle
    cd pyCycle

You can see a list of all versions in the repo via: 

    git tag

Select one of those tags (e.g. 3.0.0)

    git checkout 3.0.0

or for pyCycle V3.2.0: 

    git checkout 3.2.0

or for pyCycle V3.2.0: 

    git checkout 3.4.0

Use pip to install: 

    pip install -e .