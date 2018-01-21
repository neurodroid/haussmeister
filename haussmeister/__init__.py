# -*- coding: utf-8 -*-
'''
Python module to read common electrophysiology file formats.
'''

from . import utils
from . import haussio
from . import movies
from . import scalebars
from . import pipeline2p
from . import spectral
from . import cnmf
try:
    from . import thor2tiff
except ImportError:
    print("Could not load thor2tiff")
from . import motion
from . import decode

__version__ = '0.2.0'
