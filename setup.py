from distutils.core import setup

# optional dependencies, by category (currently just 'test')
optional_dependencies = {
    'test': [
        'testflo>=1.3.6',
    ]
}

# Add an optional dependency that concatenates all others
optional_dependencies['all'] = sorted([
    dependency
    for dependencies in optional_dependencies.values()
    for dependency in dependencies
])

setup(name='om-pycycle',
      version='4.1.2',
      packages=[
          'pycycle',
          'pycycle.elements',
          'pycycle.elements.test',
          'pycycle.maps',
          'pycycle.maps.test',
          'pycycle.thermo',
          'pycycle.thermo.cea',
          'pycycle.thermo.cea.test',
          'pycycle.thermo.cea.thermo_data',
          'pycycle.thermo.tabular',
          'pycycle.thermo.tabular.test',
          'pycycle.thermo.test',
          'pycycle.tests',
      ],
      install_requires=[
        'openmdao>=3.10.0',
      ],
    package_data={
        'pycycle.elements.test': ['reg_data/*.csv'],
        'pycycle.thermo.test': ['*.csv'],
        'pycycle.thermo.tabular': ['*.pkl'],
    },
    extras_require=optional_dependencies,
)
