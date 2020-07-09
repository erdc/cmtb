import datetime as DT
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import image, tri
import matplotlib.dates as mdates
import os, pandas
from testbedutils.sblib import statsBryant


# these are all the ones that were formerly in plotFunctions.py
def plotTripleSpectra(fnameOut, time, Hs, raw, rot, interp, full=False):
    """
    This function takes various spectra, and plots them for QA/QC on the spectral inversion/rotation method

    :param fnameOut: file output
    :param time: time stamp for Hs/ spectra
    :param Hs:  wave Height value
    :param raw: this is un-rotated spectral input tuple (spec, direction bin, frequency bands)
    :param rot: this is rotated spectral input tuple (spec, direction bin, frequency bands)
    :param interp: this is interpolated spectral input tuple (spec, direction bin, frequency bands)
    :param full:  True/False boolean
    :return: NONE
        will create figure
    """
    pandas.plotting.register_matplotlib_converters()
    nlines = 15  # number of lines to span across Half planed spectra
    lw = 3  # this is the line width factor for showing the non shore perpendicular value
    assert raw[0].ndim == 2, 'first part of interp tuple should be 2 dimensional spectra'

    rawdwed = raw[0]  # input spectra
    rawDirBin = raw[1]  # direction bins associated
    rawFreqBin = raw[2]  # frequency bands associate

    assert rot[0].ndim == 2, 'first part of interp tuple should be 2 dimensional spectra'
    rot_dWED = rot[0]
    rotDirBin = rot[1]
    rotFreqBin = rot[2]

    assert interp[0].ndim == 2, 'first part of interp tuple should be 2 dimensional spectra'
    interp_dWED = interp[0]
    interpDirBin = interp[1]
    interpFreqBin = interp[2]

    HsInd = Hs[0]  # individual Wave Height associated with Time input of plot
    timeTS = Hs[1]  # time  series of Datetimes associated with Hs
    HsTs = Hs[2]  # total wave Height time series

    # %%%% plotting loop %%%%%
    # for zz in range(0, raw.shape[0]):
    # prep formatting for plt
    pltrawdWED = rawdwed  # [zz, :, :]
    pltrotdWED = rot_dWED  # [zz, :, :]
    pltintdWED = interp_dWED  # [zz, :, :]
    # now set the interpd dwed based oon full or half plane


    # getting proper colorbars and labels forthe contour plots
    cbar_min = np.nanmin(pltrawdWED)
    cbar_max = np.nanmax(pltrawdWED)
    levels = np.linspace(cbar_min, cbar_max, 35)  # the established levels to be plotted
    # levels = np.logspace(cbar_min, cbar_max**(1/cbar_max),num=35, endpoint=True, base=10)
    from matplotlib import colors
    norm = colors.LogNorm()  # mc.BoundaryNorm(levels, 256)  # color palate for contourplots

    # generating plot to compare input data
    fig = plt.figure(figsize=(12, 8.), dpi=80)
    fig.suptitle('Input Spectra to Wave Model at %s' % time,
                 fontsize='14', fontweight='bold', y=.975)
    # subplot 0 - wave height tracer
    sub0 = fig.add_subplot(2, 1, 1)
    sub0.plot(timeTS, HsTs, 'b-')
    sub0.plot(time, HsInd, 'r*', markersize=10)

    sub0.set_ylabel('Wave Height [m]')
    sub0.set_xlabel('time')

    # subplot 1 - measured spectra
    sub1 = fig.add_subplot(2, 3, 4)
    sub1.set_title('Measured Spectra', y=1.05)
    aaa = sub1.contourf(rawFreqBin, rawDirBin, pltrawdWED.T,
                        vmin=cbar_min, vmax=cbar_max, levels=levels, norm=norm)
    sub1.plot([0, 1], [70, 70], '--k', linewidth=lw)  # pier angle
    if full == False:
        bounds = [161.8, 341.8]
        diff = (bounds[1] - bounds[0]) / nlines
        # sub1.set_ylim(0,360)
        for iii in range(0, nlines):
            lineloc = bounds[0] + diff * iii
            sub1.plot([0, .5], [lineloc, lineloc], '--w', linewidth=lw)
    sub1.set_xlabel('Frequency (hz)')
    sub1.set_ylabel('Wave Direction - (0$\degree$=True North)')
    sub1.set_xlim(0.04, 0.5)
    sub1.set_ylim(0, 360)
    aaaa = plt.colorbar(aaa, format='%.1f')
    aaaa.set_label('$m^2/hz/rad$', rotation=90)
    # subplot 2
    sub2 = fig.add_subplot(2, 3, 5)
    sub2.set_title('Inverted Direction &\nShore Normal Sepectra', y=1.05)
    if full == False:
        bounds = [90, 270]
        diff = (bounds[1] - bounds[0]) / nlines
        #  sub1.set_ylim(0,360)
        for iii in range(0, nlines):
            lineloc = bounds[0] + diff * iii
            sub2.plot([0, 1], [lineloc, lineloc], '--w', linewidth=lw)
    bbb = sub2.contourf(rotFreqBin, rotDirBin, pltrotdWED.T,
                        vmin=cbar_min, vmax=cbar_max, levels=levels, norm=norm)
    sub2.set_ylabel('Wave Direction - (0$\degree$=Shore norm +south)')
    sub2.set_xlabel('Frequency(hz)')
    sub2.set_xlim(0.04, 0.5)
    # sub2.set_ylim(0, 180)
    bbbb = plt.colorbar(bbb, format='%.1f')
    bbbb.set_label('$m^2/hz/rad$', rotation=90)
    # subplot 3
    sub3 = fig.add_subplot(2, 3, 6)
    sub3.set_title('Centered Input Spectra', y=1.05)
    ccc = sub3.contourf(interpFreqBin, interpDirBin, pltintdWED.T,
                        vmin=cbar_min, vmax=cbar_max, levels=levels, norm=norm)
    sub3.plot([0, 1], [0, 0], '--k', linewidth=3.0)
    sub3.set_ylabel('Wave Direction - (0$\degree$ Shore Norm +south)')
    sub3.set_ylim(-180, 180)
    sub3.set_xlabel('Frequency(hz)')
    sub3.set_xlim(0.04, 0.5)
    cccc = plt.colorbar(ccc, format='%.1f')
    cccc.set_label('$m^2/hz/rad$', rotation=90)
    plt.subplots_adjust(wspace=0.45, hspace=.45, right=.95, left=.075)
    # plt.tight_layout(h_pad=0.4)
    plt.savefig(fnameOut)
    plt.close()

def plotSpatialFieldData(contourpacket, fieldpacket, prefix='', nested=True, **kwargs):
    """This function plots a 2D field of data

    Args:
      fieldpacket: field packet contains
        field:  field of data type: numpy array of [time, x coords, ycoords]

        title:  title for the plot

        xlabel: label for the x axis

        ylabel: label for the y axis

        xcoord: array of xcoordinates = number of cells in x direction

        ycoord: array of y coordinates = number of cells in y direction

        cblabel: label for the colorbar, the value being plotted

      prefix (str): prefix to savefile (path (Default value = '')
      contourpacket(dict):
        field:  field of data type: numpy array of [time, x coords, ycoords]

        title:  title for the plot

        xlabel: label for the x axis

        ylabel: label for the y axis

        xcoord: array of xcoordinates = number of cells in x direction

        ycoord: array of y coordinates = number of cells in y direction

        cblabel: label for the colorbar, the value being plotted

      nested (bool): demarks whether this is nested or not grid, will affect the gauge
                labels on the plot (Default value = 1)
    Keyword Args:
          directions: this is a spatial direction data of same dimensions of spatail wave height data (or other scalar)
            if directionalWaveGaugeList data should be wrt shore normal
          namebase (str): a base to create filenames with, datetime will be appended (Default value = 'file')
    
    Returns:
      a plot to file
      
      TODO:
      increase speed with this capability
      https://stackoverflow.com/questions/42386372/increase-the-speed-of-redrawing-contour-plot-in-matplotlib
    """
    namebase=kwargs.get('namebase', 'NAMEFILLER')
    # a place to manipulate axes - not manipulated now
    ycoord = fieldpacket['ycoord']
    xcoord = fieldpacket['xcoord']  # [::-1]
    ylabel = fieldpacket['ylabel']
    xlabel = fieldpacket['xlabel']
    title = fieldpacket['title']
    clabel_text = fieldpacket['cblabel']
    time = fieldpacket['time']
    numrecs = np.size(fieldpacket['field'], axis=0)

    dx = xcoord[1] - xcoord[0]
    dy = ycoord[1] - ycoord[0]
    if np.abs(round(dx)) == np.abs(round(dy)):
        dxdy = round(dx)
    elif (np.diff(xcoord) != np.median(xcoord)).all():
        dxdy = None
    else:
        print("spatial plotting function cannot currently handle dx != dy")
        raise NotImplementedError
    # applying colorbar labels
    cbar_max = np.ceil(np.max(fieldpacket['field']))
    cbar_min = np.floor(np.min(fieldpacket['field']))
    cbarlabels = np.linspace(cbar_min, cbar_max, num=5, endpoint=True)  # a list of labels

    if numrecs == 1:
        time = [time]
        # wave gauges in approx position
    x_pier = (0, 580)  # FRF pier in approximate locatioon
    y_pier = (516, 516)
    L_pier = 'FRF Pier'
    y_26m = (4375)
    x_26m = (16100)  # -4])
    L_26m = '26m Waverider'
    y_17m = (1303)
    x_17m = (3710)  # -252])
    L_17m = '17m Waverider'
    y_11m = (933)
    x_11m = (1302)  # -302])
    L_11m = '11m AWAC'
    y_8m = (915)
    x_8m = (825)  # -309])
    L_8m = '8m Array'
    y_5m = (937)
    x_5m = (606)  # -320])
    L_5m = '6m AWAC'
    y_4m = (939)
    x_4m = (400)  # -322])
    L_4m = '4.5m AWAC'
    y_3m = (940)
    x_3m = (306)  # -324])
    L_3m = '3.5m Aquadopp'
    x_xp200 = (200)
    y_xp200 = (940)
    L_xp200 = 'XP200m Paros'
    x_xp150 = (150)
    y_xp150 = (940)
    L_xp150 = 'XP150m Paros'
    x_xp125 = (125)
    y_xp125 = (950)
    L_xp125 = 'XP125m Paros'
    # default locations for 3 things with labels
    y_CBaccess = (0)
    x_CBaccess = (0)  # [-339])
    L_CBaccess = ''
    # Kitty hawk Rd
    y_KHrd = (0)
    x_KHrd = (0)  # [-294])
    L_KHrd = ''
    # town of Duck
    y_Duck = (0)
    x_Duck = (0)  # [-334])
    L_Duck = ''
    fgsize = (6, 9)
    if nested == 0:
        if dxdy == 50:
            # corrolla beach Access
            y_CBaccess = (fieldpacket['ycoord'][-221])
            x_CBaccess = (fieldpacket['xcoord'][3])  # [-339])
            L_CBaccess = 'Corolla Beach Access'
            # Kitty hawk Rd
            y_KHrd = (fieldpacket['ycoord'][-722])
            x_KHrd = (fieldpacket['xcoord'][48])  # [-294])
            L_KHrd = 'Kitty Hawk Rd'
            # town of Duck
            y_Duck = (fieldpacket['ycoord'][-469])
            x_Duck = (fieldpacket['xcoord'][8])  # [-334])
            L_Duck = 'Town of Duck'
            # NOTE Xcoord and YCOORD are grid coords not plot coords
        elif dxdy == None:  # this is the variable spaced grid from CMS

            fgsize = (8,8)
            # corrolla beach Access
            x_CBaccess = 0
            y_CBaccess = 0  # [-339])
            L_CBaccess = ''
            # Kitty hawk Rd
            y_KHrd = (-5000)
            x_KHrd = (0)  # [-294])
            L_KHrd = 'kitty Hawk road '
            # town of Duck
            y_Duck = (-200)# [295])
            x_Duck = (0)# [-387])  # [-334])
            L_Duck = 'Town of Duck'

            # NOTE Xcoord and YCOORD are grid coords not plot coords
            nested = False  # this changes to plot the nearshore wave gauges
    elif nested == 1:
        if dxdy == 5:
            L_26m = '' # don't plot 26m
            L_17m = ''
            fgsize = (6, 9)  # np.size(fieldpacket['xcoord']) / ratio, np.size(fieldpacket['ycoord']) / ratio,)
        elif dxdy == 10:
            L_26m = '' # don't plot 26m
            L_17m = ''
            fgsize = (6, 9)  # np.size(fieldpacket['xcoord']) / ratio, np.size(fieldpacket['ycoord']) / ratio,)

# prepping gneral variables dependant on nesting option for plotting
    import matplotlib.colors as mc
    levels = np.linspace(cbar_min, cbar_max, 35)  # draw 35 levels
    norm = mc.BoundaryNorm(levels, 256)

    # __LOOPING THROUGH PLOTS___
    for tt in range(0, numrecs):
        # print('\ntitle: %s plot \nsize: %s \ntime %s \ncbar_min %d cbar_max %d' %(title, fgsize, time[tt], cbar_min, cbar_max))

        plt.figure(figsize=fgsize, dpi=80, tight_layout=True)
        plt.title(title + '\n{}'.format(time[tt]))
        try:
            plt.contourf(xcoord, ycoord, fieldpacket['field'][int(tt), :, :], levels, vmin=cbar_min, vmax=cbar_max,
                     cmap='coolwarm', levels=levels, norm=norm)
        except TypeError:
            if isinstance(fieldpacket['field'], tuple):
                assert len(fieldpacket['field']) == 1, 'weirdness: bad error descritpion'
                fieldpacket['field'] = fieldpacket['field'][0]
            plt.contourf(xcoord, ycoord, fieldpacket['field'][tt, :, :], levels, vmin=cbar_min, vmax=cbar_max,
                     cmap='coolwarm', levels=levels, norm=norm)
        # plot pier

        # plot 8m
        if nested == 1:
            plt.plot(x_8m, y_8m, '+k')
            plt.text(x_8m, y_8m, L_8m, fontsize=12, va='bottom',
                     rotation=45, color='white', weight='bold')
            plt.plot(x_5m, y_5m, '+k')
            plt.text(x_5m, y_5m, L_5m, fontsize=12, va='bottom',
                     rotation=45, color='white', weight='bold')
            plt.plot(x_4m, y_4m, '+k')
            plt.text(x_4m, y_4m, L_3m, fontsize=12, va='bottom',
                     rotation=45, color='white', weight='bold')
            plt.plot(x_3m, y_3m, '+k')
            plt.text(x_3m, y_3m, L_3m, fontsize=12, va='bottom',
                     rotation=45, color='white', weight='bold')

            plt.plot(x_xp200, y_xp200, '+k')
            plt.text(x_xp200, y_xp200, L_xp200, rotation=45, fontsize=12, va='bottom',
                     color='white', weight='bold')
            plt.plot(x_xp150, y_xp150, '+k')
            plt.text(x_xp150, y_xp150, L_xp150, rotation=45, fontsize=12, va='bottom',
                     color='white', weight='bold')
            plt.plot(x_xp125, y_xp125, '+k')
            plt.text(x_xp125, y_xp125, L_xp125, rotation=45, fontsize=12,va='bottom',
                     color='white', weight='bold',ha='right')

            plt.plot(x_pier, y_pier, 'k-', linewidth=5)
            plt.text(x_pier[1], y_pier[1], L_pier, fontsize=12, va='bottom', ha='right',
                     color='black', rotation=315, weight='bold')
            cont_labels = [0, 2, 4, 6, 8, 10]  # labels for contours
        elif nested == 0:
            plt.plot(x_CBaccess, y_CBaccess, 'oy', ms=10)
            plt.text(x_CBaccess, y_CBaccess, L_CBaccess, ha='right', va='bottom', fontsize=12,
                     color='white', rotation=315, weight='bold')
            plt.plot(x_Duck, y_Duck, '*w', ms=10)
            plt.text(x_Duck, y_Duck, L_Duck, va='bottom', fontsize=12,
                     color='white', weight='bold')
            plt.plot(x_KHrd, y_KHrd, 'oc', ms=10)
            plt.text(x_KHrd, y_KHrd, L_KHrd, fontsize=12, color='white',
                     weight='bold', rotation=270, va='bottom', ha='left')
            plt.plot(x_26m, y_26m, '+k')
            plt.text(x_26m, y_26m, L_26m, va='bottom', ha='right', fontsize=12, color='white',
                     weight='bold', rotation=0)
            plt.plot(x_17m, y_17m, '+k')
            plt.text(x_17m, y_17m, L_17m, va='bottom', fontsize=12, color='white',
                     weight='bold')
            plt.plot(x_11m, y_11m, '+k')
            plt.text(x_11m, y_11m, L_11m, va='bottom', fontsize=12, color='white',
                     weight='bold', ha='right', rotation=90)
            plt.plot(x_pier, y_pier, 'k-', linewidth=5)
            # plt.text(x_pier[1], y_pier[1], L_pier, fontsize=12, va='bottom', ha='right',
            #          color='black', rotation=-20, weight='bold')
            cont_labels = [0, 10, 16, 24]  # labels for contours

        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        cbar = plt.colorbar()
        cbar.set_ticks(cbarlabels)
        cbar.set_ticklabels(cbarlabels)
        cbar.set_label(clabel_text, fontsize=12)
        con = plt.contour(xcoord, ycoord, contourpacket['bathy'][0], levels=cont_labels, colors='k')
        plt.clabel(con, inline=True, fmt='%d')
        # if type(time[tt]) == list:
        #     plt.savefig(prefix + namebase + '_%s.png' % time)
        try:
            plt.savefig(prefix + namebase + '_%s.png' % time[tt].strftime("%Y%m%d%H%M"))
        except AttributeError:
            plt.savefig(prefix + namebase + '_%s.png' % time[tt][0].strftime("%Y%m%d%H%M"))
        plt.close()

def plotWaveProfile(x, waveHs, bathyToPlot, fname):
    """
    This function will plot the Cross shore Wave profile at the FRF Xshore array
    :param waveHs: a 2 dimensional array of wave height
    :param x: the x coordinates of the plot
    :param yLocation: the location (in STWAVE longshore coord)
    of the profile of wave height to be tak en  default 142, is
    the nested grid of the xshore array
    :param bathyField: this is a 2 dimensional array of bathymetry with Positive up

    :return: a saved plot
    """
    profileToPlot = waveHs
    # if bathyField.ndim == 3:
    #     bathyToPlot = -bathyField[0, yLocation, :]
    # elif bathyField.ndim == 2:
    #     bathyToPlot = -bathyField[yLocation, :]
    # if profileToPlot.shape[0] == 200:
    #     dx = 5
    # elif profileToPlot.shape[0] == 100:
    #     dx = 10
    ## setup

    # figure
    fig, ax1 = plt.subplots(figsize=(8,4))
    ax1.plot(x, profileToPlot, 'b', lw=3)
    ax1.set_ylabel('Wave Height [m]', color='b')
    ax2 = ax1.twinx()
    ax2.plot(x, bathyToPlot, 'brown', lw=2)
    ax2.set_ylabel('Bathymetry Depth[m] NAVD88', color='brown')
    plt.xlabel('STWAVE Cross Shore Position [m]')
    plt.title('Wave Height at the FRF Cross Shore Array')
    plt.savefig(fname)
    plt.close()


# these are all the ones that were formerly in CSHORE_plotLib
def obs_V_mod_TS(ofname, p_dict, logo_path='ArchiveFolder/CHL_logo.png'):
    """
    This script basically just compares two time series, under
        the assmption that one is from the model and one a set of observations

    :param  file_path: this is the full file-path (string) to the location where the plot will be saved
    :param p_dict: has 6 keys to it.
        (1) a vector of datetimes ('time')
        (2) vector of observations ('obs')
        (3) vector of model data ('model')
        (4) variable name (string) ('var_name')
        (5) variable units (string!!) ('units') -> this will be put inside a tex math environment!!!!
        (6) plot title (string) ('p_title')
    :return: a model vs. observation time-series plot'
        the dictionary of the statistics calculated

    """
    # this function plots observed data vs. model data for time-series data and computes stats for it.

    assert len(p_dict['time']) == len(p_dict['obs']) == len(p_dict['model']), "Your time, model, and observation arrays are not all the same length!"
    assert sum([isinstance(p_dict['time'][ss], DT.datetime) for ss in range(0, len(p_dict['time']))]) == len(p_dict['time']), 'Your times input must be an array of datetimes!'
    # calculate total duration of data to pick ticks for Xaxis on time series plot
    totalDuration = p_dict['time'][-1] - p_dict['time'][0]
    if totalDuration.days > 365:  # this is a year +  of data
        # mark 7 day increments with monthly major lables
        majorTickLocator = mdates.MonthLocator(interval=3) # every 3 months
        minorTickLocator = mdates.AutoDateLocator() # DayLocator(7)
        xfmt = mdates.DateFormatter('%Y-%m')
    elif totalDuration.days > 30: # thie is months of data that is not a year
        # mark 12 hour with daily major labels
        majorTickLocator = mdates.DayLocator(1)
        minorTickLocator = mdates.HourLocator(12)
        xfmt = mdates.DateFormatter('%Y-%m-%d')
    elif totalDuration.days > 5:
        # mark 6 hours with daily major labels
        majorTickLocator = mdates.DayLocator(1)
        minorTickLocator = mdates.HourLocator(6)
        xfmt = mdates.DateFormatter('%Y-%m-%d %H:%M')
    else:
        # mark hourly with 6 hour labels major intervals
        tickInterval = 12  # hours?
        majorTickLocator = mdates.HourLocator(interval=tickInterval)
        minorTickLocator = mdates.HourLocator(1)
        xfmt = mdates.DateFormatter('%m/%d\n%H:%M')
    # DLY notes 12/17/2018 - I think this tick selection section still needs work,
    # it works fine in some cases but terrible in others

    ####################################################################################################################
    # Begin Plot
    ####################################################################################################################
    fig = plt.figure(figsize=(10, 10))
    if 'p_title' in p_dict.keys():
        fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # time series
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    min_val = np.nanmin([np.nanmin(p_dict['obs']), np.nanmin(p_dict['model'])])
    max_val = np.nanmax([np.nanmax(p_dict['obs']), np.nanmax(p_dict['model'])])
    if min_val < 0 and max_val > 0:
        ax1.plot(p_dict['time'], np.zeros(len(p_dict['time'])), 'k--')
    ax1.plot(p_dict['time'], p_dict['obs'], 'r.', label='Observed')
    ax1.plot(p_dict['time'], p_dict['model'], 'b.', label='Model')
    ax1.set_ylabel('%s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(p_dict['time']) - DT.timedelta(seconds=0.5 * (p_dict['time'][1] - p_dict['time'][0]).total_seconds()),
                  max(p_dict['time']) + DT.timedelta(seconds=0.5 * (p_dict['time'][1] - p_dict['time'][0]).total_seconds())])

    # this is what you change for time-series x-axis ticks!!!!!
    #
    # ax1.xaxis.set_major_locator(majorTickLocator)
    # ax1.xaxis.set_minor_locator(minorTickLocator)
    # ax1.xaxis.set_major_formatter(xfmt)
    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax1.minorticks_off()
    ax1.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., 0.102), loc=3, ncol=3, borderaxespad=0., fontsize=14)
    fig.autofmt_xdate()
    # Now working on the 1-1 comparison subplot
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='unity-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')
    ax2.plot(p_dict['obs'], p_dict['model'], 'r*')
    ax2.set_xlabel('Observed %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = statsBryant(p_dict['obs'], p_dict['model'])
    stats_dict['m_mean'] = np.nanmean(p_dict['model'])
    stats_dict['o_mean'] = np.nanmean(p_dict['obs'])

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    m_mean_str = '\n Model Mean $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['m_mean']), p_dict['units'])
    o_mean_str = '\n Observation Mean $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['o_mean']), p_dict['units'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    SI_str = '\n Similarity Index $=%s$' % ("{0:.2f}".format(stats_dict['scatterIndex']))
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['symSlope']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr']))
    RMSE_Norm_str = '\n %%RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSEnorm']), p_dict['units'])

    num_String = '\n Number of samples $= %s$' %len(stats_dict['residuals'])
    plot_str = m_mean_str + o_mean_str + bias_str + RMSE_str + RMSE_Norm_str + SI_str + sym_slp_str + corr_coef_str + num_String
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax4 = ax3.twinx()
    ax3.axis('off')
    ax4.axis('off')
    try:
        CHL_logo = image.imread(logo_path)
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print('Plot generated sans CHL Logo!')

    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=18,
             fontweight='bold')
    ax3.text(0.01, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()
    return stats_dict

def bc_plot(ofname, p_dict):
    """
    This is the script to plot some information about the boundary conditions that were put into the CSHORE infile..
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of datetimes ('time')
        (2) vector of bathymetry x-positions ('x')
        (3) vector of bathymetry bed elevations ('zb')
        (4) datetime that the bathy survey was DONE ('init_bathy_stime')
        (5) vector of water levels ('time-series') at the offshore boundary ('WL')
        (6) vector of significant wave heights (time-series) at the offshore boundary ('Hs')
        (7) vector of wave angles (time-series) at the offshore boundary ('angle')
        (8) vector of wave periods (time-series) at the offshore boundary ('Tp')
        (9) plot title ('string') ('p_title')
    :return: a plot of the boundary conditions for the simulation
    """

    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []

    # assert some stuff to throw errors if need be!
    assert len(p_dict['time']) == len(p_dict['Hs']) == len(p_dict['WL']) == len(p_dict['angle']), "Your time, Hs, wave angle, and WL arrays are not all the same length!"
    assert len(p_dict['x']) == len(p_dict['zb']), "Your x and zb arrays are not the same length!"
    assert sum([isinstance(p_dict['time'][ss], DT.datetime) for ss in range(0, len(p_dict['time']))]) == len(p_dict['time']), 'Your times input must be an array of datetimes!'

    xfmt = mdates.DateFormatter('%m/%d/%y %H:%M')
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(p_dict['p_title'], fontsize=14, fontweight='bold', verticalalignment='top')

    # Plotting Hs and WL
    ax1 = plt.subplot2grid((2, 3), (1, 0), colspan=3)
    a, = ax1.plot(p_dict['time'], p_dict['Hs'], 'r-', label='$H_{s}$')
    ax1.set_ylabel('$H_s$ [$m$]')
    ax1.tick_params('y', colors='r')
    ax1.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax1.set_ylim([0.9 * np.nanmin(p_dict['Hs']), 1.1 * np.nanmax(p_dict['Hs'])])
    ax1.yaxis.label.set_color('red')
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax1.xaxis.set_major_formatter(xfmt)

    # determine axis scale factor
    if np.min(p_dict['WL']) >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if np.max(p_dict['WL']) >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9

    ax2 = ax1.twinx()
    ax2.plot(p_dict['time'], np.zeros(len(p_dict['WL'])), 'b--')
    b, = ax2.plot(p_dict['time'], p_dict['WL'], 'b-', label='WL')
    ax2.set_ylabel('$WL$ [$m$]')
    ax2.tick_params('y', colors='b')
    ax2.set_ylim([sf1 * np.min(p_dict['WL']), sf2 * np.max(p_dict['WL'])])
    ax2.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax2.yaxis.label.set_color('blue')
    p = [a, b]
    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # determine axis scale factor
    if np.min(p_dict['zb']) >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if np.max(p_dict['zb']) >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9

    if len(p_dict['veg_ind']) > 0:
        ax3 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb'][p_dict['veg_ind']], 'g-', label='Vegetated')
        ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated')
        ax3.plot(p_dict['x'], np.mean(p_dict['WL']) * np.ones(len(p_dict['x'])), 'b-', label='Mean WL')
        ax3.set_ylabel('$Elevation$ [$m$]')
        ax3.set_xlabel('x [$m$]')
        ax3.set_xlim([np.min(p_dict['x']), np.max(p_dict['x'])])
        ax3.set_ylim([sf1 * np.min(p_dict['zb']), sf2 * np.max(p_dict['zb'])])
        Bathy_date = p_dict['init_bathy_stime'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.05, 0.85, 'Bathymetry Survey Time:\n' + Bathy_date, transform=ax3.transAxes, color='black', fontsize=12)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., fontsize=14)


    else:
        ax3 = plt.subplot2grid((2, 3), (0, 1), colspan=2)
        ax3.plot(p_dict['x'], p_dict['zb'], 'y-', label='Non-vegetated')
        ax3.plot(p_dict['x'], np.mean(p_dict['WL']) * np.ones(len(p_dict['x'])), 'b-', label='Mean WL')
        ax3.set_ylabel('$Elevation$ [$m$]')
        ax3.set_xlabel('x [$m$]')
        ax3.set_xlim([np.min(p_dict['x']), np.max(p_dict['x'])])
        ax3.set_ylim([sf1 * np.min(p_dict['zb']), sf2 * np.max(p_dict['zb'])])
        Bathy_date = p_dict['init_bathy_stime'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.05, 0.85, 'Bathymetry Survey Time:\n' + Bathy_date, transform=ax3.transAxes, color='black', fontsize=12)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # plotting Tp and angle
    ax4 = plt.subplot2grid((2, 3), (0, 0), colspan=1)
    a, = ax4.plot(p_dict['time'], p_dict['Tp'], 'b-', label='$T_{p}$')
    ax4.set_ylabel('$T_{p}$ [$m$]')
    ax4.tick_params('y', colors='b')
    ax4.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax4.set_ylim([0.9 * np.min(p_dict['Tp']), 1.1 * np.max(p_dict['Tp'])])
    ax4.yaxis.label.set_color('blue')
    ax4.xaxis.set_major_formatter(xfmt)
    ax4.xaxis.set_major_locator(mdates.HourLocator(interval=24))

    ax5 = ax4.twinx()
    b, = ax5.plot(p_dict['time'], p_dict['angle'], 'r-', label='Wave Angle')
    ax5.plot(p_dict['time'], np.zeros(len(p_dict['angle'])), 'r--')
    ax5.set_ylabel('$decimal$ $^{0}$')
    ax5.tick_params('y', colors='r')
    ax5.set_ylim([-180, 180])
    ax5.set_xlim([np.min(p_dict['time']), np.max(p_dict['time'])])
    ax5.yaxis.label.set_color('red')
    p = [a, b]
    ax4.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[.05, 0.05, 0.95, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def obs_V_mod_bathy(ofname, p_dict, obs_dict, logo_path='ArchiveFolder/CHL_logo.png', contour_s=3, contour_d=8):
    """
    This is a plot to compare observed and model bathymetry to each other
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of OBSERVED bathymetry bed elevations ('obs')
        (3) datetime of the OBSERVED bathymetry survey ('obs_time')
        (4) vector of MODEL bathymetry bed elevations ('model')
        (5) datetime of the MODEL bathymetry ('model_time')
        (6) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (7) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (8) time series of water level at the offshore boundary ('WL')
        (12) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
        (9) variable name ('var_name')
        (10) variable units ('units') (string) -> this will be put inside a tex math environment!!!!
        (11) plot title (string) ('p_title')

    :param logo_path: this is the path to get the CHL logo to display it on the plot!!!!
    :param contour_s: this is the INSIDE THE SANDBAR contour line (shallow contour line)
        we are going out to for the volume calculations (depth in m!!)
    :param contour_d: this is the OUTSIDE THE SANDBAR contour line (deep contour line)
        we are going out to for the volume calculations (depth in m!!)
    :return:
        model to observation comparison for spatial data - right now all it does is bathymetry?
        may need modifications to be more general, but I'm not sure what other
        kind of data would need to be plotted in a similar manner?
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict.get('Alt05', None)
    Alt04 = obs_dict.get('Alt04', None)
    Alt03 = obs_dict.get('Alt03', None)
    # wave data
    Adopp_35 = obs_dict.get('adop-3.5m', None)
    AWAC6m = obs_dict.get('AWAC6m', None)
    AWAC8m = obs_dict.get('AWAC8m', None)

    assert len(p_dict['sigma_Hs']) == len(p_dict['Hs']) == len(p_dict['x']) == len(p_dict['obs']) == len(
        p_dict['model']), "Your x, Hs, model, and observation arrays are not the same length!"

    min_val = np.min([np.min(p_dict['obs']), np.min(p_dict['model'])])
    max_val = np.max([np.max(p_dict['obs']), np.max(p_dict['model'])])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    crossShoreX = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    obs_date = p_dict['obs_time'].strftime('%Y-%m-%d %H:%M')
    model_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
    ################################3
    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # transects
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    a, = ax1.plot(crossShoreX, np.mean(p_dict['WL']) * np.ones(len(crossShoreX)), 'b-', label='Mean WL')
    # get the time strings!!
    b, = ax1.plot(p_dict['x'], p_dict['obs'], 'r-', label='Observed (initial) \n' + obs_date)
    c, = ax1.plot(p_dict['x'], p_dict['model'], 'y-', label='Model \n' + model_date)
    if 'obs2_time' in p_dict.keys():
        obs2_date = p_dict['obs2_time'].strftime('%Y-%m-%d %H:%M')
        r, = ax1.plot(p_dict['x'], p_dict['obs2'], 'r--', label='Observed (final) \n' + obs2_date)

    # add altimeter data!!
    if Alt05 is not None:
        # Alt05
        temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
        f, = ax1.plot(Alt05['xFRF'] * np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'k-',
                      label='Gauge Data')
        g, = ax1.plot(Alt05['xFRF'] * np.ones(1), temp05, 'k_', label='Gauge Data')
    if Alt04 is not None:
        temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
        # Alt04
        h, = ax1.plot(Alt04['xFRF'] * np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'k-',
                      label='Gauge Data')
        i, = ax1.plot(Alt04['xFRF'] * np.ones(1), temp04, 'k_', label='Gauge Data')
    if Alt03 is not None:
        temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
        # Alt03
        j, = ax1.plot(Alt03['xFRF'] * np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'k-',
                      label='Gauge Data')
        k, = ax1.plot(Alt03['xFRF'] * np.ones(1), temp03, 'k_', label='Gauge Data')

    ax5 = ax1.twinx()
    d, = ax5.plot(p_dict['x'], p_dict['Hs'], 'g-', label='Model $H_{s}$')
    e, = ax5.plot(p_dict['x'], p_dict['Hs'] + p_dict['sigma_Hs'], 'g--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax5.plot(p_dict['x'], p_dict['Hs'] - p_dict['sigma_Hs'], 'g--')

    # add wave data!!
    if Adopp_35 is not None:
        temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
        # Adopp_35
        l, = ax5.plot(Adopp_35['xFRF'] * np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])],
                      'k-', label='Gauge Data')
        m, = ax5.plot(Adopp_35['xFRF'] * np.ones(1), temp35, 'k_', label='Gauge Data')
    if AWAC6m is not None:
        temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
        # AWAC6m
        n, = ax5.plot(AWAC6m['xFRF'] * np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'k-',
                      label='Gauge Data')
        o, = ax5.plot(AWAC6m['xFRF'] * np.ones(1), temp6m, 'k_', label='Gauge Data')
    if AWAC8m is not None:
        temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]
        # AWAC8m
        p, = ax5.plot(AWAC8m['xFRF'] * np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])], 'k-',
                      label='Gauge Data')
        q, = ax5.plot(AWAC8m['xFRF'] * np.ones(1), temp8m, 'k_', label='Gauge Data')


    ax1.set_ylabel('Elevation (NAVD88) [$%s$]' % (p_dict['units']), fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_ylabel('$H_{s}$ [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(crossShoreX), max(crossShoreX)])
    ax1.tick_params('y', colors='r')
    ax1.yaxis.label.set_color('red')
    ax5.tick_params('y', colors='g')
    # ax5.set_ylim([-1.05 * np.nanmax(p_dict['Hs'] + p_dict['sigma_Hs']), 1.05 * np.nanmax(p_dict['Hs'] + p_dict['sigma_Hs'])])
    ylim = ax1.get_ylim()
    ax5.set_ylim(ylim)
    ax5.set_xlim([min(crossShoreX), max(crossShoreX)])
    ax5.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax5.tick_params(labelsize=14)

    if 'obs2_time' in p_dict.keys():
        if Alt05 is not None:
            p = [a, d, b, e, r, c, f]
        else:
            p = [a, d, b, e, r, c]
        ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=5,
                borderaxespad=0., fontsize=12, handletextpad=0.05)
    else:
        if Alt05 is not None:
            p = [a, b, c, f, d, e]
        else:
            p = [a, b, c, d, e]
        ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(p),
                borderaxespad=0., fontsize=12, handletextpad=0.05)

    # 1 to 1
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='$45^{0}$-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')

    if 'obs2_time' in p_dict.keys():
        ax2.plot(p_dict['obs2'], p_dict['model'], 'r*')
        ax2.set_xlabel('Observed %s (final) [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    else:
        ax2.plot(p_dict['obs'], p_dict['model'], 'r*')
        ax2.set_xlabel('Observed %s (initial) [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)

    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=16)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = sb.statsBryant(models=p_dict['model'], observations=p_dict['obs'])

    # volume change, shallow
    index_XXm = np.min(np.argwhere(p_dict[
                                       'obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:],
                         p_dict['x'][1] - p_dict['x'][0])
    stats_dict['vol_change_%sm' % (contour_s)] = vol_model - vol_obs
    # deep
    index_XXm = np.min(np.argwhere(p_dict[
                                       'obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][1] - p_dict['x'][0])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:],
                         p_dict['x'][1] - p_dict['x'][0])
    stats_dict['vol_change_%sm' % (contour_d)] = vol_model - vol_obs

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['sym_slp']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr_coef']))
    shall_vol_str = '\n $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (
    contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['vol_change_%sm' % (contour_s)]), p_dict['units'],
    p_dict['units'])
    deep_vol_str = '\n $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (
    contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['vol_change_%sm' % (contour_d)]), p_dict['units'],
    p_dict['units'])
    vol_expl_str = '*Note: volume change is defined as the \n $model$ volume minus the $observed$ volume'

    plot_str = bias_str + RMSE_str + sym_slp_str + corr_coef_str + shall_vol_str + deep_vol_str
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.axis('off')
    ax4 = ax3.twinx()
    ax3.axis('off')
    try:
        ax4.axis('off')
        dir_name = os.path.dirname(__file__).split('\\plotting')[0]
        CHL_logo = image.imread(os.path.join(dir_name, logo_path))
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print('Plot generated sans CHL logo!')
    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=18,
             fontweight='bold')
    ax3.text(0.00, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16)
    ax3.text(0.02, 0.43, vol_expl_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(ofname, dpi=300)
    plt.close()

def mod_results(ofname, p_dict, obs_dict):
    """
    This script just lets you visualize the model outputs at a particular time-step
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of MODEL bathymetry bed elevations ('zb_m')
        (3) vector of the standard deviation of the MODEL bathymetry bed elevations at each node! ('sigma_zbm')
        (4) datetime of the MODEL bathymetry ('model_time')
        (5) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (6) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (7) vector of the setup at EACH model NODE at the TIME of the MODEL BATHYMETRY ('setup_m')
            NOTE: the "setup" value output by the model is actually just the water surface elevation!!!!
            So if you want the actual "setup" you need to subtract off some reference water level!
            I used the water level at the offshore boundary at the same time-step,
                but we will need to check this once we resolve the model comparison issue with Brad!!
        (8) vector of the standard deviation of model setup at EACH model NODE ('sigma_setup')
            DONT have to subtract anything for standard deviation, it wont change....
        (9) plot title (string) ('p_title')
        (10) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
    :return: plot of a bunch of model results
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict.get('Alt05', None)
    Alt04 = obs_dict.get('Alt04', None)
    Alt03 = obs_dict.get('Alt03', None)
    # wave data
    Adopp_35 = obs_dict.get('adop-3.5m', None)
    AWAC6m = obs_dict.get('AWAC6m', None)
    AWAC8m = obs_dict.get('AWAC8m', None)

    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []

    assert len(p_dict['zb_m']) == len(p_dict['sigma_zbm']) == len(p_dict['x']) == len(p_dict['Hs_m']) == len(p_dict['sigma_Hs']) == len(p_dict['setup_m']) == len(p_dict['sigma_setup']), "Your x, Hs, zb, and setup arrays are not the same length!"

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # Hs
    ax1 = plt.subplot2grid((3, 1), (0, 0), colspan=1)
    min_val = np.nanmin(p_dict['Hs_m'] - p_dict['sigma_Hs'])
    max_val = np.nanmax(p_dict['Hs_m'] + p_dict['sigma_Hs'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)

    if min_val < 0 and max_val > 0:
        ax1.plot(dum_x, np.zeros(len(dum_x)), 'k--')

    ax1.plot(p_dict['x'], p_dict['Hs_m'] - p_dict['sigma_Hs'], 'r--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax1.plot(p_dict['x'], p_dict['Hs_m'] + p_dict['sigma_Hs'], 'r--')
    ax1.plot(p_dict['x'], p_dict['Hs_m'], 'b-', label='Model $H_{s}$')

    # observation plots HOOOOOOO!
    if Adopp_35 is not None:
        temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
        # Adopp_35
        ax1.plot(Adopp_35['xFRF']*np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])], 'k-', label='Gauge Data')
        ax1.plot(Adopp_35['xFRF']*np.ones(1), [temp35], 'k_')
    if AWAC6m is not None:
        temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
        # AWAC6m
        ax1.plot(AWAC6m['xFRF']*np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'k-')
        ax1.plot(AWAC6m['xFRF']*np.ones(1), [temp6m], 'k_')
    if AWAC8m is not None:
        temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]

        # AWAC8m
        ax1.plot(AWAC8m['xFRF']*np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])], 'k-')
        ax1.plot(AWAC8m['xFRF']*np.ones(1), [temp8m], 'k_')

    ax1.set_ylabel('$H_{s}$ [$m$]', fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax1.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, borderaxespad=0., fontsize=14)

    # Setup
    ax2 = plt.subplot2grid((3, 1), (1, 0), colspan=1)
    min_val = np.nanmin(p_dict['setup_m'] - p_dict['sigma_setup'])
    max_val = np.nanmax(p_dict['setup_m'] + p_dict['sigma_setup'])

    if min_val < 0 and max_val > 0:
        ax2.plot(dum_x, np.zeros(len(dum_x)), 'k--')

    ax2.plot(p_dict['x'], p_dict['setup_m'] - p_dict['sigma_setup'], 'r--', label='$W_{setup} \pm \sigma_{W_{setup}}$')
    ax2.plot(p_dict['x'], p_dict['setup_m'] + p_dict['sigma_setup'], 'r--')
    ax2.plot(p_dict['x'], p_dict['setup_m'], 'b-', label='Model $W_{setup}$')

    ax2.set_ylabel('$W_{setup}$ [$m$]', fontsize=16)
    ax2.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax2.set_ylim([sf1 * min_val, sf2 * max_val])
    ax2.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax2.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, borderaxespad=0., fontsize=14)

    # Zb
    ax3 = plt.subplot2grid((3, 1), (2, 0), colspan=1)
    min_val = np.nanmin(p_dict['zb_m'] - p_dict['sigma_zbm'])
    max_val = np.nanmax(p_dict['zb_m'] + p_dict['sigma_zbm'])

    if len(p_dict['veg_ind']) > 0:
        ax3.plot(p_dict['x'], p_dict['zb_m'] - p_dict['sigma_zbm'], 'r--', label='$z_{b} \pm \sigma_{z_{b}}$')
        ax3.plot(p_dict['x'], p_dict['zb_m'] + p_dict['sigma_zbm'], 'r--')
        ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$')
        ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$')
        # get the bathy date
        zb_date = p_dict['model_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.75, 0.75, 'Bathymetry Date:\n' + zb_date, transform=ax3.transAxes, color='black', fontsize=14)
        col_num = 4
    else:
        ax3.plot(p_dict['x'], p_dict['zb_m'] - p_dict['sigma_zbm'], 'r--', label='$z_{b} \pm \sigma_{z_{b}}$')
        ax3.plot(p_dict['x'], p_dict['zb_m'] + p_dict['sigma_zbm'], 'r--')
        ax3.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$')
        zb_date = p_dict['model_time'].strftime('%Y-%m-%dT%H:%M:%SZ')
        ax3.text(0.75, 0.75, 'Bathymetry Date:\n' + zb_date, transform=ax3.transAxes, color='black', fontsize=14)
        col_num = 3

    # add altimeter data!!
    # Alt05
    if Alt05 is not None:
        temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]

        ax3.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'k-', label='Gauge Data')
        ax3.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'k_')
    # Alt04
    if Alt04 is not None:
        temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
        ax3.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'k-')
        ax3.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'k_')
    # Alt03
    if Alt03 is not None:
        temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
        ax3.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'k-')
        ax3.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'k_')

    ax3.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    ax3.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax3.set_ylim([sf1 * min_val, sf2 * max_val])
    ax3.set_xlim([min(dum_x), max(dum_x)])

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax3.tick_params(labelsize=14)
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num, borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=0.6)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def als_results(ofname, p_dict, obs_dict):
    """
    This is just some script to visualize the alongshore current results from the model output at a particular time step
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!
        (2) vector of MODEL bathymetry bed elevations ('zb_m')
        (3) datetime of the MODEL bathymetry ('model_time')
        (4) vector of model alongshore velocity at EACH model NODE at the TIME of the MODEL BATHYMETRY ('vmean_m')
        (5) vector of the standard deviation of model alongshore velocity at EACH model NODE ('sigma_vm')
        (6) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (7) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (8) plot title (string) ('p_title')
        (9) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
    :return: plot of some alongshore current stuff
    """
    # Altimeter data!!!!!!!!
    Alt05 = obs_dict.get('Alt05', None)
    Alt04 = obs_dict.get('Alt04', None)
    Alt03 = obs_dict.get('Alt03', None)
    # wave data
    Adopp_35 = obs_dict.get('adop-3.5m', None)
    AWAC6m = obs_dict.get('AWAC6m', None)
    AWAC8m = obs_dict.get('AWAC8m', None)
    myMS = 10 # marker size for observations -- can turn this into a kwargs
    # get rid of this and include them in the handed dictionary if you want to include vegetation in the plots
    p_dict['veg_ind'] = []
    p_dict['non_veg_ind'] = []
    p  = [] # axis label for how many things i've plotted
    # assert stuff here....
    assert len(p_dict['zb_m']) == len(p_dict['x']) == len(p_dict['vmean_m']) == len(p_dict['sigma_vm']), "Your x, zb, and y-vel arrays are not the same length!"

    # zb
    min_val = np.nanmin(p_dict['zb_m'])
    max_val = np.nanmax(p_dict['zb_m'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    CrossShoreX = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    ####################################################################################
    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    ax1 = plt.subplot2grid((2, 1), (0, 0), colspan=1)

    if len(p_dict['veg_ind']) > 0:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax1.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$ ' + '(' + zb_date + ')')
        b, = ax1.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 5
        p.extend((a,b))
    else:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax1.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 4
        p.extend([a])

    # add altimeter data!!
    # Alt05
    if Alt05 is not None:
        temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
        c, = ax1.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'y-', ms=myMS,  label='Altimeter')
        d, = ax1.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'y_')
        p.extend((c,d))
    # Alt04
    if Alt04 is not None:
        temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
        e, = ax1.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'y-',ms=myMS)
        f, = ax1.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'y_')
        p.extend((e,f))

    # Alt03
    if Alt03 is not None:
        temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
        g, = ax1.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])],'y-', ms=myMS)
        h, = ax1.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'y_')
        p.extend((g,h))


    ax1.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    # ax1.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(CrossShoreX), max(CrossShoreX)])
    ax1.tick_params('y', colors='g')
    ax1.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)

    # y-vel
    min_val = np.nanmin(p_dict['vmean_m'] - p_dict['sigma_vm'])
    max_val = np.nanmax(p_dict['vmean_m'] + p_dict['sigma_vm'])
    ax2 = ax1.twinx()
    if min_val < 0 and max_val > 0:
        ax2.plot(CrossShoreX, np.zeros(len(CrossShoreX)), 'b--')

    i, = ax2.plot(p_dict['x'], p_dict['vmean_m'], 'b-', label='Model $V$')
    p.extend([i])

    # velocity data HOOOOOOOOO!
    if Adopp_35 is not None:
        # Adopp_35
        temp35V = Adopp_35['V'][Adopp_35['plot_ind_V'] == 1]
        j, = ax2.plot(Adopp_35['xFRF']*np.ones(2), [temp35V - np.std(Adopp_35['V']), temp35V + np.std(Adopp_35['V'])], 'b:',ms=myMS, label='Current obs')
        k, = ax2.plot(Adopp_35['xFRF']*np.ones(1), [temp35V], 'b_')
        p.extend((j,k))

    if AWAC6m is not None:
        # AWAC6m
        if Adopp_35 is None:
            label = 'Current Obs'
        else:
            label=None
        temp6mV = AWAC6m['V'][AWAC6m['plot_ind_V'] == 1]
        l, = ax2.plot(AWAC6m['xFRF']*np.ones(2), [temp6mV - np.std(AWAC6m['V']), temp6mV + np.std(AWAC6m['V'])], 'b:', ms=myMS, label=label)
        m, = ax2.plot(AWAC6m['xFRF']*np.ones(1), [temp6mV], 'b_')
        p.extend((l,m))

    if AWAC8m is not None:
        # AWAC8m
        temp8mV = AWAC8m['V'][AWAC8m['plot_ind_V'] == 1]
        n, = ax2.plot(AWAC8m['xFRF']*np.ones(2), [temp8mV - np.std(AWAC8m['V']), temp8mV + np.std(AWAC8m['V'])], 'b:', ms=myMS)
        o, = ax2.plot(AWAC8m['xFRF']*np.ones(1), [temp8mV], 'b_')
        p.extend((n,o))

    ax2.set_ylabel('Along-shore Current [$m/s$]', fontsize=16)
    # ax2.set_xlabel('Cross-shore Position [$m$]', fontsize=16)

    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    if ylims is None:
        ax2.set_ylim([sf1 * min_val, sf2 * max_val])
    else:
        ax2.set_ylim(ylims[0])
    ax2.set_xlim([min(CrossShoreX), max(CrossShoreX)])

    ax2.tick_params('y', colors='b')
    ax2.yaxis.label.set_color('blue')

    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)

    # if col_num == 5:
    #     if Alt05 is not None:
    #         p = [a, b, c, i, j]
    #     else:
    #         p = [a, b, i, j]
    # else:
    #     if Alt05 is not None:
    #         p = [a, c, i, j]
    #     else:
    #         p = [a, i, j]

    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num,
               borderaxespad=0., fontsize=14)

    ax3 = plt.subplot2grid((2, 1), (1, 0), colspan=1)

    # zb
    min_val = np.nanmin(p_dict['zb_m'])
    max_val = np.nanmax(p_dict['zb_m'])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    CrossShoreX = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    p = []
    if len(p_dict['veg_ind']) > 0:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax3.plot(p_dict['x'][p_dict['veg_ind']], p_dict['zb_m'][p_dict['veg_ind']], 'g-', label='Vegetated $z_{b}$ ' + '(' + zb_date + ')')
        b, = ax3.plot(p_dict['x'][p_dict['non_veg_ind']], p_dict['zb_m'][p_dict['non_veg_ind']], 'y-', label='Non-vegetated $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 5
        p.extend((a,b))
    else:
        zb_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
        a, = ax3.plot(p_dict['x'], p_dict['zb_m'], 'y-', label='Model $z_{b}$ ' + '(' + zb_date + ')')
        col_num = 4
        p = [a]
    # add altimeter data!!
    # Alt05
    if Alt05 is not None:
        temp05 = Alt05['zb'][Alt05['plot_ind'] == 1]
        c, = ax3.plot(Alt05['xFRF']*np.ones(2), [temp05 - np.std(Alt05['zb']), temp05 + np.std(Alt05['zb'])], 'y-', ms=myMS, label='Altimeter')
        d, = ax3.plot(Alt05['xFRF'] * np.ones(1), [temp05], 'y_')
        p.extend((c,d))
    # Alt04
    if Alt04 is not None:
        if Alt05 is None:
            label='altimeter'
        else:
            label=None
        temp04 = Alt04['zb'][Alt04['plot_ind'] == 1]
        e, = ax3.plot(Alt04['xFRF']*np.ones(2), [temp04 - np.std(Alt04['zb']), temp04 + np.std(Alt04['zb'])], 'y-', ms=myMS, label=label)
        f, = ax1.plot(Alt04['xFRF'] * np.ones(1), [temp04], 'y_')
        p.extend((e,f))
    # Alt03
    if Alt03 is not None:
        if Alt05 is None and Alt04 is None:
            label='altimeter'
        else:
            label=None
        temp03 = Alt03['zb'][Alt03['plot_ind'] == 1]
        g, = ax3.plot(Alt03['xFRF']*np.ones(2), [temp03 - np.std(Alt03['zb']), temp03 + np.std(Alt03['zb'])], 'y-',ms=myMS)
        h, = ax3.plot(Alt03['xFRF'] * np.ones(1), [temp03], 'y_')
        p.extend((g, h))

    ax3.set_ylabel('Elevation (NAVD88) [$m$]', fontsize=16)
    ax3.set_xlabel('Cross-shore Position [$m$]', fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax3.set_ylim([sf1 * min_val, sf2 * max_val])
    ax3.set_xlim([min(CrossShoreX), max(CrossShoreX)])
    ax3.tick_params('y', colors='g')
    ax3.yaxis.label.set_color('green')

    for tick in ax3.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax3.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax3.tick_params(labelsize=14)

    # Hs
    min_val = np.nanmin(p_dict['Hs_m'] - p_dict['sigma_Hs'])
    max_val = np.nanmax(p_dict['Hs_m'] + p_dict['sigma_Hs'])
    ax4 = ax3.twinx()
    if min_val < 0 and max_val > 0:
        ax4.plot(CrossShoreX, np.zeros(len(CrossShoreX)), 'b--')

    i, = ax4.plot(p_dict['x'], p_dict['Hs_m'], 'b-', label='Model $H_{s}$')
    p.extend([i])
    # observation plots HOOOOOOO!

    if Adopp_35 is not None:
        # Adopp_35
        temp35 = Adopp_35['Hs'][Adopp_35['plot_ind'] == 1]
        j, = ax4.plot(Adopp_35['xFRF'] * np.ones(2), [temp35 - np.std(Adopp_35['Hs']), temp35 + np.std(Adopp_35['Hs'])], 'b:', ms=myMS, label='Wave Observation')
        k, = ax4.plot(Adopp_35['xFRF'] * np.ones(1), [temp35], 'b_')
        p.extend((j,k))
    if AWAC6m is not None:
        if Adopp_35 is None:
            label = 'Wave Obs'
        else:
            label=None
        # AWAC6m
        temp6m = AWAC6m['Hs'][AWAC6m['plot_ind'] == 1]
        l, = ax4.plot(AWAC6m['xFRF'] * np.ones(2), [temp6m - np.std(AWAC6m['Hs']), temp6m + np.std(AWAC6m['Hs'])], 'b:', ms=myMS, label=label)
        m, = ax4.plot(AWAC6m['xFRF'] * np.ones(1), [temp6m], 'b_')
        p.extend((l,m))

    if AWAC8m is not None:
        # AWAC8m
        temp8m = AWAC8m['Hs'][AWAC8m['plot_ind'] == 1]
        n, = ax4.plot(AWAC8m['xFRF'] * np.ones(2), [temp8m - np.std(AWAC8m['Hs']), temp8m + np.std(AWAC8m['Hs'])],  'b:',ms=myMS)
        o, = ax4.plot(AWAC8m['xFRF'] * np.ones(1), [temp8m], 'b_')
        p.extend((n,o))
    ax4.set_ylabel('$H_{s}$ [$m$]', fontsize=16)
    ax4.set_xlabel('Cross-shore Position [$m$]', fontsize=16)

    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    if ylims is None:
        ax4.set_ylim([sf1 * min_val, sf2 * max_val])
    else:
        ax4.set_ylim(ylims[1])
    ax4.set_xlim([min(CrossShoreX), max(CrossShoreX)])

    ax4.tick_params('y', colors='b')
    ax4.yaxis.label.set_color('blue')

    for tick in ax4.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax4.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax4.tick_params(labelsize=14)

    # if col_num == 5:
    #     if Alt05 is not None:
    #         p = [a, b, c, i, j]
    #     else:
    #         p = [a, b, i, j]
    # else:
    #     if Alt05 is not None:
    #         p = [a, c, i, j]
    #     else:
    #         p = [a, i, j]

    ax3.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=col_num,
               borderaxespad=0., fontsize=14)

    fig.subplots_adjust(wspace=0.4, hspace=1.0)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    fig.savefig(ofname, dpi=300)
    plt.close()

def obs_V_mod_bathy_TN(ofname, p_dict, obs_dict, logo_path='ArchiveFolder/CHL_logo.png', contour_s=3, contour_d=8):
    """This is a plot to compare observed and model bathymetry to each other
    
    :param file_path: this is the full file-path (string) to the location where the plot will be saved i.e., C://users...
        DONT include the final '/' or the actual NAME of the plot!!!!!!
    :param p_dict:
        (1) a vector of x-positions for the bathymetry ('x')
            MAKE SURE THIS IS IN FRF COORDS!!!!

        (2) vector of initial OBSERVED bathymetry bed elevations ('i_obs')
        (3) datetime of the initial OBSERVED bathymetry survey ('i_obs_time')

        (3) vector of final OBSERVED bathymetry bed elevations ('f_obs')
        (4) datetime of the final OBSERVED bathymetry survey ('f_obs_time')

        (5) vector of MODEL bathymetry bed elevations ('model')
        (6) datetime of the MODEL bathymetry ('model_time')
        (7) vector of model Hs at EACH model NODE at the TIME of the MODEL BATHYMETRY ('Hs')
        (8) vector of the standard deviation of model Hs at EACH model NODE ('sigma_Hs')
        (9) time series of water level at the offshore boundary ('WL')
        (13) array of datetimes for the water level data ('time').
            AS A HEADS UP, THIS IS THE RANGE OF TIMES THAT WILL GO INTO getObs for the comparisons!!!
        (10) variable name ('var_name')
        (11) variable units ('units') (string) -> this will be put inside a tex math environment!!!!
        (12) plot title (string) ('p_title')

    :param logo_path: this is the path to get the CHL logo to display it on the plot!!!!
    :param contour_s: this is the INSIDE THE SANDBAR contour line (shallow contour line)
        we are going out to for the volume calculations (depth in m!!)
    :param contour_d: this is the OUTSIDE THE SANDBAR contour line (deep contour line)
        we are going out to for the volume calculations (depth in m!!)
    :return:
        model to observation comparison for spatial data - right now all it does is bathymetry?
        may need modifications to be more general, but I'm not sure what other
        kind of data would need to be plotted in a similar manner?
    """

    # Altimeter data!!!!!!!!
    Alt05 = obs_dict['Alt05']
    Alt04 = obs_dict['Alt04']
    Alt03 = obs_dict['Alt03']

    # wave data
    Adopp_35 = obs_dict['Adopp_35']
    AWAC6m = obs_dict['AWAC6m']
    AWAC8m = obs_dict['AWAC8m']

    assert len(p_dict['sigma_Hs']) == len(p_dict['Hs']) == len(p_dict['x']) == len(p_dict['i_obs']) == len(p_dict['f_obs']) == len(p_dict['model']), "Your x, Hs, model, and observation arrays are not the same length!"

    fig = plt.figure(figsize=(12, 12))
    fig.suptitle(p_dict['p_title'], fontsize=18, fontweight='bold', verticalalignment='top')

    # transects
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    min_val = np.min([np.min(p_dict['i_obs']), np.min(p_dict['model']), np.min(p_dict['f_obs'])])
    max_val = np.max([np.max(p_dict['i_obs']), np.max(p_dict['model']), np.max(p_dict['f_obs'])])
    min_x = np.min(p_dict['x'])
    max_x = np.max(p_dict['x'])
    dum_x = np.linspace(min_x - 0.05 * (max_x - min_x), max_x + 0.05 * (max_x - min_x), 100)
    a, = ax1.plot(dum_x, np.mean(p_dict['WL']) * np.ones(len(dum_x)), 'b-', label='Mean WL')
    # get the time strings!!
    i_obs_date = p_dict['i_obs_time'].strftime('%Y-%m-%d %H:%M')
    model_date = p_dict['model_time'].strftime('%Y-%m-%d %H:%M')
    f_obs_date = p_dict['f_obs_time'].strftime('%Y-%m-%d %H:%M')
    b, = ax1.plot(p_dict['x'], p_dict['i_obs'], 'r--', label='Initial Observed \n' + i_obs_date)
    c, = ax1.plot(p_dict['x'], p_dict['model'], 'y-', label='Model \n' + model_date)
    r, = ax1.plot(p_dict['x'], p_dict['f_obs'], 'r-', label='Final Observed \n' + f_obs_date)

    # add altimeter data!!
    # Alt05
    f, = ax1.plot(Alt05['xFRF'] * np.ones(2), [min(Alt05['zb']), max(Alt05['zb'])], 'k-', label='Gage Data')
    g, = ax1.plot(Alt05['xFRF'] * np.ones(1), Alt05['zb'][Alt05['plot_ind'] == 1], 'k_', label='Gage Data')
    # Alt04
    h, = ax1.plot(Alt04['xFRF'] * np.ones(2), [min(Alt04['zb']), max(Alt04['zb'])], 'k-', label='Gage Data')
    i, = ax1.plot(Alt04['xFRF'] * np.ones(1), Alt04['zb'][Alt04['plot_ind'] == 1], 'k_', label='Gage Data')
    # Alt03
    j, = ax1.plot(Alt03['xFRF'] * np.ones(2), [min(Alt03['zb']), max(Alt03['zb'])], 'k-', label='Gage Data')
    k, = ax1.plot(Alt03['xFRF'] * np.ones(1), Alt03['zb'][Alt03['plot_ind'] == 1], 'k_', label='Gage Data')

    ax5 = ax1.twinx()
    d, = ax5.plot(p_dict['x'], p_dict['Hs'], 'g-', label='Model $H_{s}$')
    e, = ax5.plot(p_dict['x'], p_dict['Hs'] + p_dict['sigma_Hs'], 'g--', label='$H_{s} \pm \sigma_{H_{s}}$')
    ax5.plot(p_dict['x'], p_dict['Hs'] - p_dict['sigma_Hs'], 'g--')

    # add wave data!!
    # Adopp_35
    l, = ax5.plot(Adopp_35['xFRF'] * np.ones(2), [Adopp_35['Hs'][Adopp_35['plot_ind'] == 1] - np.std(Adopp_35['Hs']), Adopp_35['Hs'][Adopp_35['plot_ind'] == 1] + np.std(Adopp_35['Hs'])], 'k-', label='Gage Data')
    m, = ax5.plot(Adopp_35['xFRF'] * np.ones(1), Adopp_35['Hs'][Adopp_35['plot_ind'] == 1], 'k_', label='Gage Data')
    # AWAC6m
    n, = ax5.plot(AWAC6m['xFRF'] * np.ones(2), [AWAC6m['Hs'][AWAC6m['plot_ind'] == 1] - np.std(AWAC6m['Hs']), AWAC6m['Hs'][AWAC6m['plot_ind'] == 1] + np.std(AWAC6m['Hs'])], 'k-', label='Gage Data')
    o, = ax5.plot(AWAC6m['xFRF'] * np.ones(1), AWAC6m['Hs'][AWAC6m['plot_ind'] == 1], 'k_', label='Gage Data')
    # AWAC8m
    p, = ax5.plot(AWAC8m['xFRF'] * np.ones(2), [AWAC8m['Hs'][AWAC8m['plot_ind'] == 1] - np.std(AWAC8m['Hs']), AWAC8m['Hs'][AWAC8m['plot_ind'] == 1] + np.std(AWAC8m['Hs'])], 'k-', label='Gage Data')
    q, = ax5.plot(AWAC8m['xFRF'] * np.ones(1), AWAC8m['Hs'][AWAC8m['plot_ind'] == 1], 'k_', label='Gage Data')

    ax1.set_ylabel('Elevation (NAVD88) [$%s$]' % (p_dict['units']), fontsize=16)
    ax1.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_ylabel('$H_{s}$ [$%s$]' % (p_dict['units']), fontsize=16)
    ax5.set_xlabel('Cross-shore Position [$%s$]' % (p_dict['units']), fontsize=16)
    # determine axis scale factor
    if min_val >= 0:
        sf1 = 0.9
    else:
        sf1 = 1.1
    if max_val >= 0:
        sf2 = 1.1
    else:
        sf2 = 0.9
    ax1.set_ylim([sf1 * min_val, sf2 * max_val])
    ax1.set_xlim([min(dum_x), max(dum_x)])
    ax1.tick_params('y', colors='r')
    ax1.yaxis.label.set_color('red')

    ax5.tick_params('y', colors='g')
    ax5.set_ylim([-1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs']), 1.05 * max(p_dict['Hs'] + p_dict['sigma_Hs'])])
    ax5.set_xlim([min(dum_x), max(dum_x)])
    ax5.yaxis.label.set_color('green')

    for tick in ax1.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax5.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax1.tick_params(labelsize=14)
    ax5.tick_params(labelsize=14)
    p = [a, b, c, r, f, d, e]
    ax1.legend(p, [p_.get_label() for p_ in p], bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=len(p), borderaxespad=0., fontsize=10, handletextpad=0.05)

    #go ahead and make sure they all cover the same space.
    o_mask = np.ma.getmask(p_dict['f_obs'])
    o_mask2 = o_mask.copy()
    p_dict['f_obs'] = p_dict['f_obs'][~o_mask2]
    p_dict['i_obs'] = p_dict['i_obs'][~o_mask2]
    p_dict['model'] = p_dict['model'][~o_mask2]
    p_dict['x'] = p_dict['x'][~o_mask2]




    # 1 to 1
    one_one = np.linspace(min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val), 100)
    ax2 = plt.subplot2grid((2, 2), (1, 0), colspan=1)
    ax2.plot(one_one, one_one, 'k-', label='$45^{0}$-line')
    if min_val < 0 and max_val > 0:
        ax2.plot(one_one, np.zeros(len(one_one)), 'k--')
        ax2.plot(np.zeros(len(one_one)), one_one, 'k--')
    ax2.plot(p_dict['f_obs'], p_dict['model'], 'r*')
    ax2.set_xlabel('Final Observed %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=14)
    ax2.set_ylabel('Model %s [$%s$]' % (p_dict['var_name'], p_dict['units']), fontsize=14)
    ax2.set_xlim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    ax2.set_ylim([min_val - 0.025 * (max_val - min_val), max_val + 0.025 * (max_val - min_val)])
    for tick in ax2.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax2.tick_params(labelsize=14)
    plt.legend(loc=0, ncol=1, borderaxespad=0.5, fontsize=14)

    # stats and stats text
    stats_dict = {}
    stats_dict['bias'] = np.mean(p_dict['f_obs'] - p_dict['model'])
    stats_dict['RMSE'] = np.sqrt((1 / (float(len(p_dict['f_obs'])) - 1)) * np.sum(np.power(p_dict['f_obs'] - p_dict['model'] - stats_dict['bias'], 2)))
    stats_dict['sym_slp'] = np.sqrt(np.sum(np.power(p_dict['f_obs'], 2)) / float(np.sum(np.power(p_dict['model'], 2))))
    # correlation coef
    dum = np.zeros([2, len(p_dict['model'])])
    dum[0] = p_dict['model'].flatten()
    dum[1] = p_dict['f_obs'].flatten()
    stats_dict['corr_coef'] = np.corrcoef(dum)[0, 1]

    # volume change
    # shallow - predicted
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['pred_vol_change_%sm' % (contour_s)] = vol_model - vol_obs
    # shallow - actual
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_s).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs_i = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_obs_f = np.trapz(p_dict['f_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['actual_vol_change_%sm' % (contour_s)] = vol_obs_f - vol_obs_i

    # deep - predicted
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_model = np.trapz(p_dict['model'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['pred_vol_change_%sm' % (contour_d)] = vol_model - vol_obs

    # deep - actual
    index_XXm = np.min(np.argwhere(p_dict['i_obs'] >= -1 * contour_d).flatten())  # ok, the indices currently count from offshore to onshore, so we want the SMALLEST index!
    vol_obs_i = np.trapz(p_dict['i_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    vol_obs_f = np.trapz(p_dict['f_obs'][index_XXm:] - min_val, p_dict['x'][index_XXm:], p_dict['x'][0] - p_dict['x'][1])
    stats_dict['actual_vol_change_%sm' % (contour_d)] = vol_obs_i - vol_obs_f

    header_str = '%s Comparison \nModel to Observations:' % (p_dict['var_name'])
    bias_str = '\n Bias $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['bias']), p_dict['units'])
    RMSE_str = '\n RMSE $=%s$ $(%s)$' % ("{0:.2f}".format(stats_dict['RMSE']), p_dict['units'])
    sym_slp_str = '\n Symmetric Slope $=%s$' % ("{0:.2f}".format(stats_dict['sym_slp']))
    corr_coef_str = '\n Correlation Coefficient $=%s$' % ("{0:.2f}".format(stats_dict['corr_coef']))

    shall_vol_str_p = '\n Modeled $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['pred_vol_change_%sm' % (contour_s)]), p_dict['units'], p_dict['units'])
    shall_vol_str_a = '\n Observed $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_s, p_dict['units'], "{0:.2f}".format(stats_dict['actual_vol_change_%sm' % (contour_s)]), p_dict['units'], p_dict['units'])

    deep_vol_str_p = '\n Modeled $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['pred_vol_change_%sm' % (contour_d)]), p_dict['units'], p_dict['units'])
    deep_vol_str_a = '\n Observed $%s$ $%s$ Volume Change $=%s$ $(%s^{3}/%s)$' % (contour_d, p_dict['units'], "{0:.2f}".format(stats_dict['actual_vol_change_%sm' % (contour_d)]), p_dict['units'], p_dict['units'])

    vol_expl_str = '*Note: volume change is defined as the \n $final$ volume minus the $initial$ volume'

    plot_str = bias_str + RMSE_str + sym_slp_str + corr_coef_str + shall_vol_str_p + shall_vol_str_a + deep_vol_str_p + deep_vol_str_a
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    ax3.axis('off')
    ax4 = ax3.twinx()
    ax3.axis('off')
    try:
        ax4.axis('off')
        dir_name = os.path.dirname(__file__).split('\\plotting')[0]
        CHL_logo = image.imread(os.path.join(dir_name, logo_path))
        ax4 = fig.add_axes([0.78, 0.02, 0.20, 0.20], anchor='SE', zorder=-1)
        ax4.imshow(CHL_logo)
        ax4.axis('off')
    except:
        print('Plot generated sans CHL logo!')
    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16, fontweight='bold')
    ax3.text(0.00, 0.90, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=14)
    ax3.text(0.02, 0.41, vol_expl_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=12)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.90])
    fig.savefig(ofname, dpi=300)
    plt.close()

def plotUnstructField(ofname, pDict):
    """
    This is a function to plot unstructured grid bathymetry data.
    It uses the matplotlib.tri package to triangulate your points onto a grid,
    then the tricontourf function to actually plot it.  The triangulation is a Delaunay triangulation

    :param ofname: complete filepath where the output will be stored, including extension!!!!!
    :param pDict:
        Keys:
        ptitle - plot title
        x - x-positions
        y - y-positions
        z - this can be any value that you want a 2D colorf plot of, but for our applications mainly depth or elevation
        xLabel - label for x-axis
        yLabel - label for y-axis
        cbarLabel - label for the color bar
        cbarMin - minumum value to show on colorbar
        cbarMax - maximum value to show on colorbar
        cbarColor - type of colorbar you want to use
        ncLev - number of contour "levels" you want to have.
                defaults to 100 to make it look like a continuous colorbar
        xbounds - (xmin, xmax) for your plot
        ybounds - (ymin, ymax) for your plot
    :return:
        saved contourf plot
    """

    # check for dictionary keys
    assert 'x' in pDict.keys(), "Error: x must be specified"
    assert 'y' in pDict.keys(), "Error: y must be specified"
    assert 'z' in pDict.keys(), "Error: z must be specified"

    # make assumptions if optional keys are blank
    if 'xLabel' not in pDict.keys():
        pDict['xLabel'] = 'x'
    if 'yLabel' not in pDict.keys():
        pDict['yLabel'] = 'y'
    if 'cbarLabel' not in pDict.keys():
        pDict['cbarLabel'] = 'z'
    if 'cbarMin' not in pDict.keys():
        pDict['cbarMin'] = np.nanmin(pDict['z'])
    if 'cbarMax' not in pDict.keys():
        pDict['cbarMax'] = np.nanmax(pDict['z'])
    if 'cbarColor' not in pDict.keys():
        pDict['cbarColor'] = 'RdYlBu'
    if 'ncLev' not in pDict.keys():
        pDict['ncLev'] = 100
    if 'xbounds' not in pDict.keys():
        pDict['xbounds'] = (np.min(pDict['x']), np.max(pDict['x']))
    if 'ybounds' not in pDict.keys():
        pDict['ybounds'] = (np.min(pDict['y']), np.max(pDict['y']))

    z = pDict['z'].copy()
    # NOTE - if pDict['z'] is a masked array, this script will MODIFY the mask!!!!!
    maskFlag = False
    if np.ma.is_masked(z):
        maskInd = np.ma.getmask(z).copy()
        maskFlag = True

    # if I have colorbar ranges, force the data to be within the min/max bounds
    z[z < pDict['cbarMin']] = pDict['cbarMin']
    z[z > pDict['cbarMax']] = pDict['cbarMax']

    # figure out how to force my colorbar ticks through zero
    if pDict['cbarMin'] > 0 or pDict['cbarMax'] < 0:
        v = np.linspace(pDict['cbarMin'], pDict['cbarMax'], 11, endpoint=True)
    else:
        # first guess at spacing
        s1 = (pDict['cbarMax'] - pDict['cbarMin'])/float(11)
        cnt = 0
        if s1 > 1:
            while s1 > 1:
                cnt = cnt + 1
                s1 = s1/float(10)
        elif s1 < 0.1:
            while s1 < 0.1:
                cnt = cnt -1
                s1 = s1 * float(10)
        else:
            pass
        # round to nearest quarter
        s1n = round(s1 * 4)/4
        if s1n == 0:
            s1n = round(s1, 1)

        # get it to the same decimal place it was before
        s1n = s1n*10**cnt

        # build stuff out of it....
        rL = np.arange(0, pDict['cbarMax'], s1n)
        lL = -1*np.arange(s1n, abs(pDict['cbarMin']), s1n)
        v = np.concatenate([lL, rL])


    # perform triangulation
    triang = tri.Triangulation(pDict['x'], pDict['y'])

    # do we re-apply the mask here?  does tricontourf work with masked arrays?
    if maskFlag:
        z = np.array(z)
        z[maskInd] = pDict['cbarMax'] + (pDict['cbarMax'] - pDict['cbarMin'])

    # generate the plot.
    axisAspect = (pDict['ybounds'][1] - pDict['ybounds'][0])/float(pDict['xbounds'][1] - pDict['xbounds'][0])
    plt.figure()
    plt.ylim([pDict['ybounds'][0], pDict['ybounds'][1]])
    plt.xlim([pDict['xbounds'][0], pDict['xbounds'][1]])
    plt.gca().set_aspect(axisAspect)
    clev = np.arange(pDict['cbarMin'], pDict['cbarMax'], (pDict['cbarMax'] - pDict['cbarMin'])/float(pDict['ncLev']))
    plt.tricontourf(triang, z, clev, cmap=plt.get_cmap(pDict['cbarColor']))
    plt.clim(pDict['cbarMin'], pDict['cbarMax'])
    cb1 = plt.colorbar(orientation='vertical', ticks=v)
    cb1.set_label(pDict['cbarLabel'], fontsize=12)
    if 'U' in pDict.keys() and 'V' in pDict.keys():

        # lets interpolate this onto a uniform grid?
        # build new grid
        stepsize = 250
        xP = []
        yP = []
        for x in range(pDict['xbounds'][0], pDict['xbounds'][1], stepsize):
            for y in range(pDict['ybounds'][0], pDict['ybounds'][1], stepsize):
                xP.append(x)
                yP.append(y)
        # do the interpolation
        points = (pDict['x'], pDict['y'])
        values = pDict['U']
        # do the interpolation
        uP = griddata(points, values, (xP, yP), method='linear')
        values = pDict['V']
        # do the interpolation
        vP = griddata(points, values, (xP, yP), method='linear')

        # plot quiver vectors
        if 'scaleP' in pDict.keys():
            Q = plt.quiver(xP, yP, uP, vP, scale=pDict['scaleP'])
        else:
            Q = plt.quiver(xP, yP, uP, vP)
        vMag = np.sqrt(np.power(uP, 2) + np.power(vP, 2))
        # what should the scale be?
        if 'scaleV' in pDict.keys():
            scaleV = round(pDict['scaleV'], 1)
        else:
            scaleV = round(np.nanmax(vMag), 1)
        plt.quiverkey(Q, pDict['xbounds'][0] + 0.05*(pDict['xbounds'][1] - pDict['xbounds'][0]), pDict['ybounds'][1] + 0.02*(pDict['ybounds'][1] - pDict['ybounds'][0]), scaleV, '%s $m/s$'%scaleV, linewidth=1, labelpos='E', coordinates='data')

    # DLY Note 12/19/2018 - the labeling of gauges is not flexible as currently constructed.  suggest switching to
    # different markers and legend.  as is the text will overlap without significant tinkering
    if 'gaugeLabels' in pDict.keys():
        if pDict['gaugeLabels']:

            gaugeNames = ['FRF Pier', '26m Waverider', '17m Waverider', '11m AWAC', '8m Array', '6m AWAC', '4.5m AWAC', '3.5m Aquadopp', '200m Paros', '150m Paros', '125m Paros']
            gaugeX = [[0, 580], 16100, 3710, 1302, 825, 606, 400, 306, 200, 150, 125]
            gaugeY = [[516, 516], 4375, 1303, 933, 915, 937, 939, 940, 940, 940, 950]

            # gauge label time!
            parosFlag = False
            for ii in range(0, len(gaugeNames)):

                if gaugeNames[ii] == 'FRF Pier':
                    plt.plot(gaugeX[ii], gaugeY[ii], 'k-', linewidth=5)
                    plt.text(gaugeX[ii][1], gaugeY[ii][1]-150, gaugeNames[ii], fontsize=8, va='bottom', ha='right',
                             color='black', rotation=0)

                elif 'Paros' in gaugeNames[ii]:
                    if gaugeX[ii] > pDict['xbounds'][0] and gaugeX[ii] < pDict['xbounds'][1] and gaugeY[ii] > pDict['ybounds'][0] and gaugeY[ii] < pDict['ybounds'][1]:
                        plt.plot(gaugeX[ii], gaugeY[ii], 'or')
                        parosFlag = True
                elif gaugeNames[ii] == '3.5m Aquadopp':
                    if gaugeX[ii] > pDict['xbounds'][0] and gaugeX[ii] < pDict['xbounds'][1] and gaugeY[ii] > pDict['ybounds'][0] and gaugeY[ii] < pDict['ybounds'][1]:
                        plt.plot(gaugeX[ii], gaugeY[ii], 'or')
                        plt.text(gaugeX[ii]-25, gaugeY[ii], gaugeNames[ii], fontsize=6, va='bottom', rotation=90, color='black')
                else:
                    if gaugeX[ii] > pDict['xbounds'][0] and gaugeX[ii] < pDict['xbounds'][1] and gaugeY[ii] > pDict['ybounds'][0] and gaugeY[ii] < pDict['ybounds'][1]:
                        plt.plot(gaugeX[ii], gaugeY[ii], 'or')
                        plt.text(gaugeX[ii], gaugeY[ii], gaugeNames[ii], fontsize=6, va='bottom', rotation=90, color='black')
            if parosFlag:
                plt.text(gaugeX[-1]-65, gaugeY[-1]-225, '125m, 150m,\n200 m Paros', fontsize=6, va='bottom', rotation=90, color='black')





    # set some other labels
    plt.ylabel(pDict['yLabel'], fontsize=12)
    plt.xlabel(pDict['xLabel'], fontsize=12)
    if 'ptitle' in pDict.keys():
        plt.title(pDict['ptitle'], fontsize=16)

    # save time
    plt.savefig(ofname, dpi=300, bbox_inches='tight')
    plt.close()


def generate_CrossShoreTimeseries(ofname, dataIn, bottomIn, xIn, **kwargs):
    """generates a water elevation cross-section, used single timesteps of
    phase resolving models

    Args:
        ofname (str): fullpath (or relative) file name
        dataIn (array):   value to plot (eta) -- size [xIn]
        bottomIn: elevations for the bottom (negative) -- size [xIn]
        xIn: coordinates positions for cross-shore

    Keyword Args:
         figsize (tuple): sets figure size (default = (8,4))

    Returns:
        a plot

    """
    figsize = kwargs.get('figsize', (8,4))
    beachColor = 'wheat'
    skyColor = 'aquamarine'
    waterColor = 'deepskyblue'
    if np.median(bottomIn) > 0:
        bottomIn = -bottomIn
    ###########################
    plt.figure(figsize=figsize)
    ax1 = plt.subplot(111)
    ax1.set_facecolor(skyColor)
    ax1.plot(xIn, dataIn)  # plot water line
    ax1.plot(xIn, bottomIn, color=beachColor)  # plot beach
    ax1.fill_betweenx(bottomIn, xIn, color=beachColor)  # fill in beach
    ax1.fill_between(xIn, bottomIn, dataIn, color=waterColor)  # fill in water
    ax1.set_xlim([np.min(xIn), np.max(xIn)])
    ax1.set_ylim([np.min(bottomIn), np.max(bottomIn) + 0.5])

    plt.savefig(ofname)
    plt.close()


def plotCrossShoreSummaryTS(ofname, xFRF, bathy, totalStatisticDict, SeaSwellStats, IGstats, setup, WL, **kwargs):
    """ plots a 4 panel plot summary of cross-shore performance of model that can resolve IG

    Args:
        ofname: output file name
        xFRF (array): array of cross-shore positions associated with bathy, all wave height profiles
        bathy: bathymetry values (function assumes positive down)
        totalStatisticDict (dict): total statistics;  has key 'Hm0' shaped as length xFRF
        SeaSwellStats (dict): sea/swell only statistics; has key 'Hm0' shaped as length xFRF
        IGstats (dict): Infragravity only statistics; has key 'Hm0' shaped as length xFRF
        setup: mean of the water level
        WL: offshore tide level (relative to bathy datum)

    Keyword Args:
        'obs' (dict): a nested dictionary of numerous observations dictionaries

    Returns:
        save's a plot

    """
    obs = kwargs.get('obs', None)
    HsTS = kwargs.get('HsTs', None)
    fs = kwargs.get('fontSize', 12)
    var = kwargs.get('plotVar', 'Hm0')
    beachColor = 'wheat'
    waterColor = 'aquamarine'
    setupColor = 'green'
    if obs is not None:
        raise NotImplementedError('Please add functionality to loop through obs stations')

    size = '41'  # adds flexibility if need to change number of subplots
    figsize = (12, 8)  # just a guess at size for the figure for now
    ########### make Figure ########################
    plt.figure(figsize=figsize);
    ax1 = plt.subplot(int(size + '1'))
    if HsTS is not None:
        ax1.plot(xFRF, HsTS, label='$Hs_{Ts}$')
    ax1.plot(xFRF, totalStatisticDict[var], label='$Hs_{Total}$')
    ax1.plot(xFRF, SeaSwellStats[var], label='$Hs_{seaSwell}$')
    ax1.plot(xFRF, IGstats[var], label='$Hs_{IG}$')
    ax1.legend(loc='upper left', fontsize=fs)
    ax1.set_ylabel('Wave Height $[m]$', fontsize=fs)

    ax2 = plt.subplot(int(size + '2'))
    ax2.plot(xFRF, IGstats[var], label='$Hs_{IG}$')
    ax2.set_ylabel('IG wave Height', fontsize=fs)

    ax3 = plt.subplot(int(size + '3'))
    ax3.plot(xFRF, setup)
    ax3.set_ylabel('$\eta$', fontsize=fs)

    ax4 = plt.subplot(int(size + '4'))
    ax4.plot(xFRF, -bathy,'-', lw=7, color=beachColor)
    ax4.plot(xFRF, np.tile(WL, (xFRF.shape[0])), color=waterColor, label='Water Level')
    ax4.plot(xFRF, setup, color=setupColor, label='TWL')
    ax4.set_ylabel('Z NAVD88 - [m]')
    ax4.set_xlabel('Cross-shore Location [m]')

    plt.savefig(ofname)
    plt.close()

def crossShoreSurfaceTS2D(ofname, eta, xFRF, time):
    """surface 2D timeseries

    Args:
        ofname: output file location
        eta:  2D array of xFRF
        xFRF:
        time:

    Returns:

    """
    eta= eta.squeeze()
    plt.figure()
    plt.pcolormesh(xFRF, time, eta, cmap='RdBu')
    plt.savefig(ofname)
    plt.close()

def crossShoreSpectrograph(ofname, xFRF, freqs, fspec, **kwargs):
    """ A cross shore evolution of spectra

    Args:
        ofname (str): output location/name of the file
        xFRF: cross-shore position
        freqs: frequency bands
        fspec: 2d array of frequency
    Keyword Args:
        'ylims': limits for the frequency space in the nearshore spectrogram

    Returns:
        a plot

    """
    ylims = kwargs.get('ylims', (0, 0.4))
    plt.figure();
    plt.pcolormesh(xFRF, freqs, fspec.T)
    plt.colorbar()
    plt.ylabel('frequency', fontsize=12)
    plt.xlabel('cross-shore location', fontsize=12)
    plt.ylim(ylims)
    plt.savefig(ofname);
    plt.close();
