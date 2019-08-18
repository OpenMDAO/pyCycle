from distutils.core import setup

setup(name='pycycle',
      version='3.0.0',
      packages=[
          'pycycle',
          'pycycle/cea',
          'pycycle/cea/thermo_data',
          'pycycle/elements', 
          'pycycle/maps'
      ],

      install_requires=[
        'openmdao>=2.8.0',
      ]
)
