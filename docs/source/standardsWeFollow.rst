Standards we follow
===================
Standards keep this a nice place to work, please play along

Variable Names
--------------
Variable names are preferred to be longer and explicit to help other developers understand what the variable
contains.  For example significantWaveHeight is preferred to Hs or waveHeight.  While this takes more typing
let's admit, we all have tab complete for variable names.

The other convention that we **try** to follow is called the `Camel Humps <https://en.wikipedia.org/wiki/Camel_case>`_
convention, specifically lowerCamelCase:

significant_wave_height  - No

SignificantWaveHeight - OK

significantWaveHeight - preferred

Doc Strings
-----------
The docstrings are a useful piece of a code.  The docstring comes up when you type help(function)
The details from the python environment protocol 257 is listed `here <https://www.python.org/dev/peps/pep-0257/>`_ which lays out what  docstring is

The convention by which these are written matters!  all of this documentation and API is based on the docString convention
This project follows the google convention due to its readability.

A link to google's documentation guidelines is shown `here <http://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_

MetaData
--------
Meta data is an important piece to help users understand what they're reading.  These meta data should be written into
all of the FRF's observation data as well as participants model output files. If this is confusing to you, feel free
to reach out and discuss.  We have tools that make them here makeNC_

.. _makeNC: dataInfrastructure.html#netCDF file creation software

`Climate and Forecast Standards <http://cfconventions.org/>`_

