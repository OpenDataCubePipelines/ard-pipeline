.. wagl documentation master file, created by
   sphinx-quickstart on Tue Jan  6 14:54:34 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to wagl's documentation!
================================

Contents:

.. toctree::
   :maxdepth: 2

   overview
   release_notes
   history
   algorithms
   dependencies
   directories
   acquisitions
   hdf5
   atmospherics
   scripts
   configuration
   logging
   unittesting
   modules
   appendix

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Bulding wagl
============

wagl uses several Fortran programs, C programs and modules built using `F2py <http://www.scipy.org/F2py>`_. The build process is performed using make, though if make is not available, it is pretty clear what needs to be done from the make files. Once the required bits and pieces are installed (see :doc:`dependencies </dependencies>`), building and generating documentation should be as easy as typing ``make`` in the top level directory.

wagl testing
============

Unit tests written using :py:mod:`unittest` for some modules can be found in :py:mod:`wagl.tests`. Some of these tests also depend on existing data computed from an earlier run in order to do comparisons. Please see the :py:mod:`unittest` documentation and the individual tests under wagl.tests on how to run these. More information can be found at :doc:`unittesting </unittesting>`.
