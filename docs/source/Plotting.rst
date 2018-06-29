=========================
Plotting - Content Needed
=========================

There are various plots that are used we will go through a few of them that are used in the real time process
and what kind of data they expect some of which expect a logo as part of the plot
Most plots up to this point have been made for quick visualization, to ensure that the model is running properly

note most all of the spatial plots expect FRF coordinate systems

Make gif
--------
A good number of plots are started with the below plotting scripts that create pngs.  The make gif function in sblib
is

obs v mod
---------
this is a lovely plot that was created by david young. The plot is clean simple and displays both a times series
comparison and a 1 to 1 plot.  the plot lives currently in the plotting/cshoreplotlib.py file.  Data are expected
to be time matched (sblib.timematch) before being handed to this plot. the calls are a data dictionary, the output
file name and the path to the logo (designed for CHL logo.

|obs_v_mod|

.. autofunction:: plotting.operationalPlots.obs_V_mod_TS

spatial plots
_____________

this plot is a plot that will show the 2D spatial output of wave field (Hs, Tp, Tm, Dm, etc) some bulk statistic. '
it could be better built on basemap libarary, but is functional for smaller nearshore domains. internal to the function
are locations of points of interested in the area (kitty hawk road, Duck, corolla beach access, etc) to give the viewer
some frame of reference.  x and y coordinates are expected in FRF coordinate system

link to image

|obs_v_mod|:: images/plottingExamples/obs_v_mod.png