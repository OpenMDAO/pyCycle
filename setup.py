from distutils.core import setup

setup(name='pycycle',
      version='3.0.0',
      packages=[
          'pycycle',
          'pycycle/thermo',
          'pycycle/thermo/cea',
          'pycycle/thermo/cea/thermo_data',
          'pycycle/elements', 
          'pycycle/maps'
      ],

      install_requires=[
        'openmdao>=3.2.0',
      ],
)
