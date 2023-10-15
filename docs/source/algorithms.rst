Products produced by wagl
==============================

.. _lambertian-algorithm-label:

Lambertian Reflectance
----------------------

Similar to the NBAR product without the BRDF correction.



.. _nbar-algorithm-label:

Nadir BRDF-adjusted reflectance (NBAR)
--------------------------------------

An image taken by a satellite is affected by a range of factors, including:

* the angle at which a satellite views a pixel (view angle),
* illumination (determined largely by the solar angle),
* characteristics of the satellite, its sensor and its orbit,
* relative position of the sun and earth, and
* atmospheric conditions.

The NBAR algorithm attempts to normalise the imagery collected by the satellite to take account of these artifacts and hence render it more suitable for further analysis. The normalised imagery should appear as it would if the satellite were directly overhead (i.e. at `nadir <http://en.wikipedia.org/wiki/Nadir>`_) and at a specific solar angle (in the products produced here, 45 degrees).

The NBAR algorithm was developed by staff at GA and CSIRO and is described in :download:`An Evaluation of the Use of Atmospheric and BRDF Correction to Standardize Landsat Data <auxiliary/li_etal_2010_05422912.pdf>`.



.. _tc-algorithm-label:

Terrain Correction (TC)
-----------------------

Another major factor which affects the appearance of remotely sensed imagery is terrain. This changes the angle of incidence and introduces shadows. The TC algorithm extends the NBAR algorithm to take account of these effects.

The algorithm is described in :download:`A physics-based atmospheric and BRDF correction for Landsat data over mountainous terrain <auxiliary/1-s2.0-S0034425712002544-main.pdf>`. The document :download:`terrain process <auxiliary/terrain_process.doc>` was provided by `Fuqin Li <mailto:fuqin.li@ga.gov.au>`_ describing the (Fortran) executables and scripts used in running the Terrain Correction algorithm.
