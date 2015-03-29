from distutils.core import setup
setup(name='haussmeister',
      version='0.1.0',
      packages=['haussmeister'],
      package_dir={'haussmeister': 'haussmeister'},
      package_data={'haussmeister': ['data/*.ttf']},
      )
