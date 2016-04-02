from setuptools import setup
setup(name='haussmeister',
      version='0.2.0',
      packages=['haussmeister'],
      package_dir={'haussmeister': 'haussmeister'},
      package_data={'haussmeister': ['data/*.ttf']},
      scripts=['haussmeister/thor2tiff.py'],
      install_requires=[
          'numpy',
          'pillow',
          'tables',
          'sima',
      ]
      )
