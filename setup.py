from distutils.core import setup

setup(name='pycycle',
      version='4.1.1',
      packages=[
          'pycycle',
          'pycycle/thermo',
          'pycycle/thermo/cea',
          'pycycle/thermo/cea/thermo_data',
          'pycycle/elements', 
          'pycycle/maps',
          'pycycle/thermo/tabular'
      ],

      install_requires=[
        'openmdao>=3.13.0',
      ],
)
