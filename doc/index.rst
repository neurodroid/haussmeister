.. haussmeister documentation master file, created by
   sphinx-quickstart on Sun Mar 29 08:59:25 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Haussmeister: Handling 2-photon imaging datasets
================================================

:Release: |release|
:Date: |today|
:Email: christsc@gmx.de
:GitHub: https://github.com/neurodroid/haussmeister


Getting started
===============
Define your recording as a :obj:`pipeline2p.ThorExperiment`:

.. code-block:: python

   from haussio import pipeline2p as p2p

   ROOT_PATH = "/Volumes/fileserver/data"
                
   experiment = p2p.ThorExperiment(
       "2016/2016-03/2016-03-21/DKL7-verm6f_002_016", # Path to experiment (from root_path)
       "A", # Channel
       "CE", # Brain region
       "DKL7-verm6f_002_sync_016", # ThorSync file name
       "vr/20160321_0029", # Corresponding VR file
       mc_method="hmmc", # Motion correction method
       root_path="/Volumes/fileserver/data/", # Root path of your data
       seg_method="cnmf") # Segmentation method


Preprocess your data (currently this involves motion correction only):

.. code-block:: python

   p2p.thor_preprocess(experiment)


Segment data and extract fluorescence signals:
  
.. code-block:: python

   p2p.thor_extract_roi(data)


Contents
========
.. toctree::
   :maxdepth: 2

   usage/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

