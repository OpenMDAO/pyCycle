%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Documentation for pyCycle version: |release|
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

pyCycle is a open-source thermodynamic cycle analysis tool designed integrating this analysis into larger multidisciplinary optimization problems.  The code is written in Python and is built on top of the `OpenMDAO`_ framework.

.. _OpenMDAO: https://www.openmdao.org/

The pyCycle project implements 1D thermodynamic cycle analysis equations similar to the NPSS code and has been validated against this tool.  While the thermodynamic equations are similar to NPSS and other available cycle analysis tools, pyCycle is focused on applying advanced methods for supporting gradient-based optimization through the implementation of analytic derivatives. This feature enables efficient exploration of large design spaces when pyCycle is coupled with other disciplinary analysis tools.


.. warning:: 

    pyCycle is built on top of OpenMDAO and relies extensively its linear and nonlinear solver library. 
    pyCycle was also designed with a user interface that is very similar to the NPSS cycle modeling library.
    If you're not comfortable with either OpenMDAO or NPSS, then you may find this library difficult to understand and use. 

Installation
************
Installation of pyCycle requires two pieces of software.  First, if you do not have OpenMDAO you will need to install it following the instructions on the `OpenMDAO Getting Started`_ page.  Second, you will need to install pyCycle by typing the following command into your python environment (we recommend Anaconda):

.. code::

    pip install pycycle


.. _OpenMDAO Getting Started: http://openmdao.org/twodocs/versions/latest/getting_started/index.html

Tutorials
**********

This section provides several tutorials showing how to build thermodynamic cycle models in pyCycle.  
For users unfamiliar with OpenMDAO, a review of the OpenMDAO User Guide is highly recommended before completing the pyCycle tutorials.  
The tutorials in this section will demonstrate how models are constructed and executed in pyCycle, starting with a simple turboject then moving to a more complicated turbofan model.
Additional models showing more features available with pyCycle are provided in the Examples section.


.. toctree::
    :maxdepth: 1
    :name: tutorials

    tutorials/index.rst
    
Reference Guide
****************

The reference guide intended for users looking for explanation of a particular feature in detail or documentation of the arguments/options/settings for a specific cycle element, map or viewer.

.. toctree:: 
    :maxdepth: 1 
    :name: reference_guide

    reference_guide/elements/index.rst
    reference_guide/maps/index.rst
    reference_guide/viewers.rst



Examples 
********
The examples in this section provide a more comprehensive demsonstration of the features of pyCycle and advanced modeling methods.

.. toctree:: 
    :maxdepth: 1 
    :name: examples

    examples/index.rst









