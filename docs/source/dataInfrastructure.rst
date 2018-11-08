==================
Data Infrastucture
==================

netCDF Files
------------
NetCDF files are developed out of Unidata and allow for easy sectioning off of data.  observation and model
data are stored as netCDF data.  Some of the advantage of netCDF files is that they are self describing binary
file that is platform independent.  The meta data standard followed by the FRF is the Climate and Forecast Standard.

For more information about netCDF files please visit  `Unidata <https://www.unidata.ucar.edu/software/netcdf/docs/>`_

netCDF file creation software
_____________________________
.. _makenc:

In python, these are done through the makenc.py module listed in the repository.  These use a global and variable
yaml file.  The global yaml file lays out the global metadata that should be included in the file.  The variable
yaml file will lay out the structure of each variable, variable names and any appropriate file structure.  The
dimensions of the variable need to be layed out 'manually' for each file.  We have tools that can generate CF compliant
meta-data standard compliantMetaData_

.. _compliantMetaData: standardsWeFollow.html#MetaData

If you'd like help in matlab (or python), we have similar scripts in place please reach out and we can help!

Data Dissemination
------------------
There is a google group where information about the data dissemination is shared.  Please sign up if you're interested

https://groups.google.com/d/forum/frfdata

ERDAPP Server
_____________
The FRF is experimenting internally with an ERDAPP server.  Stay tuned.

Thredds Server
______________
Recently the FRF has undergone a data overhaul in which the live data products have been converted
to netCDF files.  This is the primary data server of the CMTB project. These files are then served on
a public facing THREDDS server for easy access. Through the use of the .ncml files, data are more available
to easily ingest into the getdata scripts which are the basis for the test bed.

    `https://CHLthredds.erdc.dren.mil/
    <https://chlthredds.erdc.dren.mil/>`_

THREDDS interaction Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It's often easier for people that haven't worked with THREDDS or netCDF files to interact with the data via examples
the `getdata <https://github.com/sbfrf/getdata/>`_ [1]_ repository on github that has various examples of how to interact
with netCDF files and THREDDS server *via* OPeNDAP.

Contributions welcome!!!!!

Matlab
++++++
Matlab users need to be aware of a bug that is in the matlab source code related to netCDF openDAP calls as described
by this `bug report <https://www.mathworks.com/support/bugreports/1072120>`_

matlab functions to be aware of:
ncdisp, ncatt, ncvar

python
++++++
there are a few ways to deal with netCDF files and open dap delivery, the test bed utilizes the netCDF4 library as
distributed by anaconda or available through pip/conda installs.

::
    import netCDF4 as nc  # import the library
    import datetime as DT
    url = "https://chlthredds.erdc.dren.mil/thredds/frf/oceanography/waves/waverider-26m/waverider-26m.ncml" # use the ncml link
    ncfile = nc.Dataset(url)  # open the netCDF file (on the server)
    tt = ncfile['time']  # grab the time variable
    times = nc.num2date(tt[:], tt.units)  # convert to datetime: use the indicies of time, grab the unit meta data attribute in the conversion
    mask = times > DT.datetime(2017,12,1,12,14) # creates boolean compare
    idx = np.where(mask).squeeze()  #find my indices
    waveHeight = ncfile['waveHs'][idx]


Architecture
____________
The THREDDS is broken up into an architecture that is intended to be easily understood.  At the root level are various
projects that CHL is hosting (including the FRF data and the CMTB data).
inside the FRF folder is broken up by data type

    - geomorphology (profile lines, digital elevation maps (DEMS) etc)
    - oceanography (waves,  currents, water temps ...etc)
    - meteorology (winds, pressures, rain ...etc)

As a participant of the CMTB, data of each model run are shared back and hosted through the CHL thredds
currently (though still under construction), the architecture is as follows

- Wave models
    - STWAVE
    - CMS wave
    - SWAN
    - etc

- Circulation models
    - CMS wave
    - Delft 3D flow

- morphology models
    - Xbeach
    - CSHORE

- Coupled model systems
    - CSTORM (we hope to add)
    - COAWST (we hope to add)

- projects [2]_
    - This folder contains various experimental projects in which various people are working and sharing data

---------------------------------------------------------------------------

.. [1] Contributions welcome!
.. [2] these folders can and may be password protected for specific users
