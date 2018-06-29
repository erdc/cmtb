================
sub-Repositories
================

These are repositories that are not part of the main CMTB repository, they are instead treated as git submodules.
This allows more flexibility when using these for other projects or studies.

each one is shown below 

GetDataTestBed
--------------

This is a repo currently located here:

its role is to go get data between a time period of interest and return it in a standardized dictionary(specific to this project).

it is designed as a class that is initiated by passing two date time instances to it a start and end time.
It will then have sub functions that will go and get data from the FRF data server.  First it will try the
local server (only accessible from internal to the FRF network) and then it will try to go to the CHL THREDDS,
the public server.
get time is as the basis of this, and while functional, could potentially improved for speed.

getObs
______
This is primarily focused on getting observation data for initialization and validation

.. autoclass:: getdatatestbed.getDataFRF.getObs
   :members:
   :noindex:

getDataTestBed
______________
This is aimed at getting data primarily modeled data from the Test Bed server, crunchy!

.. autoclass:: getdatatestbed.getDataFRF.getDataTestBed
   :members:
   :noindex:


getOutsideData
______________
This is aimed at getting data that might be relavent to the test bed but from outside sources.  Potential places
might be cdip servers that redundantly store the wave rider data.  Places like NDBC buoys data from the continental
shelf near by or other buoys that may be of interest.  So far the work in this class has been focused on retrieving
the NOAA NCEP forcast for 44100 or the 26m wave rider.

.. autoclass:: getdatatestbed.getOutsideData.forecastData
   :members:
   :noindex:

test bed utils
--------------
This repo is stored here

This is a utility repository,  it has handy tricks that are not data specific, that are not specifically for
prepping data. ie rotating directions, spectrum.

of particular interest in here is the geoprocess library.  This library has various coordinate
transformation functions.  All of which are aimed at specifically the FRF.  The commonly used
coordinate system here is the FRF coordinate system which has an origin that is located at the south
end of the property in the dune and is oriented shore normal in x and positive y direction moving north
along the coast line.  The assumed shore normal angle is assumed to be the FRF pier.  This is 71.8 degrees
in geographic coordinate system and 69.98 in nc stateplane.  To be more accurate all coordinate transformations
to and from FRF coordinates happen through NC state plane


scaleCinterp
____________

this is used for bathymetry integration and interpolation.  from plant 2001, 2009 paper!