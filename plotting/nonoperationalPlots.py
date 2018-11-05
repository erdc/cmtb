import matplotlib.colors as mc
from matplotlib.ticker import NullFormatter, MaxNLocator
from testbedutils import waveLib, sblib
from matplotlib import pyplot as plt
import numpy as np

def pltFRFgrid(xyzDict, save=False):
    """This function plots a dictionary of values with keys x, y, z

    Args:
      save: return: (Default value = False)
      xyzDict: 

    Returns:

    """
    x = xyzDict['x']
    y = xyzDict['y']
    z = xyzDict['z']

    levels = np.logspace(np.min(z), np.max(z), num=35, endpoint=True, base=10)
    norm = mc.BoundaryNorm(levels, 256)  # color palate for contourplots
    # plt.contourf(ycoord, xcoord, fieldpacket['field'][tt, :, :], levels, vmin=cbar_min, vmax=cbar_max,
    #              cmap='coolwarm', levels=levels, norm=norm)
    plt.pcolor(x, y, z, vmin=z.min(), vmax=z.max())

def plot2DcontourSpec(spec2D, freqBin, dirBin, fname, pathCHLlogo=None, **kwargs):
    """This function plots a 2d spectra showing the 1d direction and 1d frequency spectra on both sides of a 2
        dimensional spectra. idea and base functionwas taken from the below website

    References:
        http://www.astrobetter.com/blog/2014/02/10/visualization-fun-with-python-2d-histogram-with-1d-histograms-on-axes/

    Args:
      fname: outpufile name
      spec2D: 2 dimensional spectrum (single)
      freqBin: associated freuqncy bins
      dirBin: associated direction bins
      pathCHLlogo: defaults to None, but will put a logo at the top right if path is given
      **kwargs: 

    Returns:
      saves a plot to the fname location

    """
    # convert spectra from m2/rad to m2/hz
    spec2D = np.deg2rad(spec2D )  # inverse from rad2degree because rad is denom
    # Define the x and y data (sum the 2D Spectra)
    freqspec = spec2D.sum(axis=1) * np.median(np.diff(dirBin))
    dirspec = spec2D.sum(axis=0) * np.median(np.diff(freqBin))

    cbar_min = spec2D.min()
    cbar_max = np.ceil(spec2D.max()*100)/100 # round up, with two decimal points
    levels = np.geomspace(cbar_min, cbar_max, num=25, endpoint=True)
    # levels = np.linspace(cbar_min, cbar_max, num=25, endpoint=True)
    norm = mc.BoundaryNorm(levels, 256)  # color palate for contourplots

    # Set up default x and y limits
    freqLims = [min(freqBin), max(freqBin)]
    dirLims = [min(dirBin), max(dirBin)]

    # Set up your x and y labels
    xlabel = 'Freqeuency $hz$'
    ylabel = 'Direction $^\degree TN$'

    # Define the locations for the axes
    # mess with sizes here
    left, width = 0.12, 0.55
    bottom, height = 0.15, 0.55
    left_h = left + width  # + 0.02  # this .02 adds space between the plots
    bottom_h = bottom + height
    # Set up the geometry of the three plots
    SizeMain = [left, bottom, width, height]  # dimensions of Main Plot
    SizeTop = [left, bottom_h, width, 0.22]  # dimensions of the Top Plot
    SizeRight = [left_h, bottom, 0.22, height]  # dimensions of Right Plot

    # Set up the size of the figure
    fig = plt.figure(1, figsize=(10, 9))
    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = 'Wave Directional Spectrum Plot'
    fig.suptitle(title, size=16, weight='bold')
    fig.patch.set_facecolor('white')

    # Make the three plots
    axSpec = plt.axes(SizeMain)  # temperature plot
    axHistx = plt.axes(SizeTop)  # x histogram
    axHisty = plt.axes(SizeRight)  # y histogram

    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # Find the min/max of the data
    xmin = min(freqLims)
    xmax = max(freqLims)
    ymin = min(dirLims)
    ymax = max(dirLims)

    dirTicks = [0, 45, 90, 135, 180, 225, 270, 315]
    freqTicks = [.1, .2, .3, .4, .5]

    contAX = axSpec.contourf(freqBin, dirBin, spec2D.T, vmin=cbar_min, vmax=cbar_max,
                             levels=levels, norm=norm)

    axSpec.set_yticks(dirTicks)
    axSpec.set_yticklabels(dirTicks, size=13, family='serif')
    axSpec.set_xticks(freqTicks)
    axSpec.set_xticklabels(freqTicks, size=13, family='serif')
    # placing the colorbar
    cbaxes = fig.add_axes([0.13, 0.05, 0.53, 0.02])
    colorbarticks = np.linspace
    cb = plt.colorbar(contAX, ax=[axSpec, axHisty], cax=cbaxes, orientation='horizontal', format='%.1e')
    cb.ax.text(-0.2, 0.3, '$m^2/hz/deg$', size=14)

    # Plot the axes labels
    axSpec.set_xlabel(xlabel, fontsize=20)
    axSpec.set_ylabel(ylabel, fontsize=20)

    # Set up the plot limits
    axSpec.set_xlim(freqLims)
    axSpec.set_ylim(dirLims)

    # plotting 1D spectra now
    # stp.gradient_fill(x=freqBin, y=freqspec, ax=axHistx, zfunc=True)
    axHistx.fill_between(freqBin, freqspec, 0, alpha=0.4)
    axHistx.set_xlim(freqLims)
    axHistx.set_ylabel('$m^2/hz$', size=15)
    # plotting the direction spectra
    axHisty.fill_betweenx(dirBin, dirspec, 0, alpha=0.4)
    axHisty.set_ylim(dirLims)
    axHisty.set_xlabel('$m^2/deg$', size=15)

    # stp.gradient_fill(x=dirspec, y=dirBin,  ax=axHisty, zfunc=False)

    # Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')

    # Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')

    # Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))
    if pathCHLlogo != None:
        img = plt.imread(pathCHLlogo)
        ax4 = fig.add_axes([.7, .7, .2, .22], anchor='NE', zorder=-1)
        ax4.imshow(img)
        ax4.axis('off')

    ## Now put stats in text
    stats = waveLib.waveStat(np.expand_dims(spec2D, axis=0), freqBin, dirBin)

    textString = 'wave Height = %.2f m\nPeak Period = %.2f s\nPeak Dir = %.2f deg TN' % (stats['Hm0'], stats['Tp'], stats['Dp'] )
    plt.text(0, 400, textString , fontSize=16)
    plt.savefig(fname)

    plt.close()

def pltspec(dirbin, freqbin, spec, name, bounds=[161.8, 341.8], nlines=15, show=True, **kwargs):
    """this plots a single spectra in a single plot

    Args:
      dirbin:  direction bins
      freqbin: frequency bins
      spec: 2d wave Energy Spectra (or similar)
      name: title for plot
      bounds:  bounds for strike through lines (Default value = [161.8 341.8]):
      nlines:  number of lines to use to strike through (Default value = 15)
      show (bool): true or false to display (Default value = True)

    Keyword Args:
        fname: file name to save
    Returns:

    """

    diff = (bounds[1] - bounds[0]) / nlines

    specfig = plt.figure()
    specfig.suptitle(name)
    aaa = plt.contourf(freqbin, dirbin, spec)
    plt.plot([0, 1], [71.8, 71.8], '--k', linewidth=3.0)  # pier angle
    # sub1.set_ylim(0,360)
    for iii in range(0, nlines):
        lineloc = bounds[0] + diff * iii
        plt.plot([0, 1], [lineloc, lineloc], '-.w', linewidth=5)
    plt.xlabel('Frequency (hz)')
    plt.ylabel('Wave Direction - (0=True North)')
    plt.xlim(0.04, 0.5)
    aaaa = plt.colorbar(aaa)
    aaaa.set_label('$m^2/hz/rad$', rotation=90)
    if 'fname' in kwargs:
        plt.savefig(kwargs['fname'])
    if show == True:
        plt.show()
    plt.close()

def plot121(ofname, plotpacket1, plotpacket2, plotpacket3, **kwargs):
    """this is a plot function that will plot 2 dataList and plot them 1-1 as a means
    for comparison
    
    the first plot package is usually wave HS
    the second plot packages is usually wave Tp
    the thrid plot package is usually wave direction

    Args:
      plotpacket1: dictionary with the following keys
         obs: observations
         mod: model values
         title: label for subplot title (default = '')
      plotpacket1: dictionary with the following keys
        obs: observations
        mod: model values
        title: label for subplot title (default = '')
      plotpacket1: dictionary with the following keys
        obs: observations
        mod: model values
        title: label for subplot title (default = '')
      ofname: output file name

    Keyword Args:
        watermark (bool): True/False will put a watermark over the plot (default = True)
        stats (bool): True/False - will calculate stats for each of the comparison plots (default = False)

    Returns:
      3 dictionaries with associated comparison statistics from each plotpacket in

    """
    if 'stats' in kwargs:
        stats = kwargs['stats']
    else:
        stats = False
    if 'watermark' in kwargs:
        watermark= kwargs['watermark']
    else:
        watermark = True
    if 'pier' in kwargs:
        pier = kwargs['pier']
    else:
        pier = False
    # assigning a figure name from plot packet 1
    if 'title' in kwargs:
        figtitle = kwargs['title']
    else:
        figtitle = ''

    # data for first subplot (wave height)
    xdata1 = np.array(plotpacket1['obs'])
    ydata1 = np.array(plotpacket1['mod'])
    linestring1 = '.b'  # plotpacket1['linestring']
    if 'title' in plotpacket1:
        title1 = plotpacket1['title']
    else:
        title1 = ''
    xlabel1 = 'observations'
    ylabel1 = 'model'
    max1 = max(np.max(ydata1), np.max(xdata1)) * 1.15
    ####################################
    # data for 2nd subplot (period)
    xdata2 = np.array(plotpacket2['obs'])
    ydata2 = np.array(plotpacket2['mod'])
    linestring2 = '.b'
    if 'title' in plotpacket2:
        title2 = plotpacket2['title']
    else:
        title2 = ''
    xlabel2 = 'observations'
    ylabel2 = 'model'
    max2 = max(np.max(ydata2), np.max(xdata2)) * 1.15
    ####################################
    # data for 2nd subplot (period)
    # data for 3rd subplot
    xdata3 = np.array(plotpacket3['obs'])
    ydata3 = np.array(plotpacket3['mod'])
    linestring3 = '.b'
    if 'title' in plotpacket3:
        title3 = plotpacket3['title']
    else:
        title3 = ''
    xlabel3 = 'observations'
    ylabel3 = 'model'
    max3 = max(np.max(ydata3), np.max(xdata3)) * 1.15
    min3 = min(np.min(ydata3), np.min(xdata3)) * .85
    # calc stats
    if stats:
        statPacket1 = sblib.statsBryant(xdata1, ydata1)
        statString1 = 'Bias={:.2f}\nRMSE={:.2f}\n$r^2$={:.2f}\nn={}'.format(statPacket1['bias'], statPacket1['RMSE'], statPacket1['r2'], np.size(statPacket1['residuals']))
        statPacket2 = sblib.statsBryant(xdata2, ydata2)
        statString2 = 'Bias={:.2f}\nRMSE={:.2f}\n$r^2$={:.2f}\nn={}'.format(statPacket2['bias'], statPacket2['RMSE'], statPacket2['r2'], np.size(statPacket2['residuals']))
        statPacket3 = sblib.statsBryant(xdata3, ydata3)
        statString3 = 'Bias={:.2f}\nRMSE={:.2f}\n$r^2$={:.2f}\nn={}'.format(statPacket3['bias'], statPacket3['RMSE'], statPacket3['r2'], np.size(statPacket3['residuals']))

    # plotting
    yloc = 7.5/10.
    # 1st subplot ____
    one2one = plt.figure(figsize=(12, 4), dpi=80)
    sub1 = plt.subplot(1, 3, 1)
    if (xdata1 == 0).all() and (ydata1 == 0).all():
        sub1.text(0.5, 0.5, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata1 == 0).all():
        sub1.text(0.5, 0.5, 'NO OBSERVATIOIN data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (ydata1 == 0).all():
        sub1.text(0.5, 0.5, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:  # no errors, will plot
        line1, = sub1.plot(xdata1, ydata1, linestring1)
        plt.plot([0, max1], [0, max1], 'k-', linewidth=1)
        plt.xlim((0, max1))
        plt.ylim((0, max1))
        if stats and watermark:
            plt.text(max1 / 10, max1 * yloc, statString1)

    plt.xlabel(xlabel1, fontsize=12)
    plt.ylabel(ylabel1, fontsize=12)
    sub1.set_title(title1, fontsize=12)
    #  ____ 2nd subplot ____
    sub2 = plt.subplot(1, 3, 2)
    if (xdata2 == 0).all() and (ydata2 == 0).all():
        sub2.text(0.5, 0.5, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata2 == 0).all():
        sub2.text(0.5, 0.5, 'NO OBSERVATIOIN data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (ydata2 == 0).all():
        sub2.text(0.5, 0.5, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        line2, = plt.plot(xdata2, ydata2, linestring2)
        if watermark == True:
            sub2.text(max2 / 2, max2 / 2, 'Research Product', fontsize=20,
                  color='gray', ha='center', va='top', alpha=0.5)
        plt.plot([0, max2], [0, max2], 'k-', linewidth=1)
        plt.xlim((0, max2))
        plt.ylim((0, max2))
        if stats and watermark:
            plt.text(max2 / 10, max2 * yloc, statString2)

    plt.xlabel(xlabel2, fontsize=12)
    plt.ylabel(ylabel2, fontsize=12)
    sub2.set_title(figtitle + '\n' + title2, fontsize=12)
    # ____3rd subplot ___
    sub3 = plt.subplot(1, 3, 3)
    if (xdata3 == 0).all() and (ydata3 == 0).all():
        sub3.text(0.5, 0.5, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata3 == 0).all():
        sub3.text(0.5, 0.5, 'NO OBSERVATIOIN data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (ydata3 == 0).all():
        sub3.text(0.5, 0.5, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    else:
        line3 = plt.plot(xdata3, ydata3, linestring3)
        #            plt.plot([min3,max3],[min3,max3], 'k-', linewidth = 1)
        plt.plot([0, 360], [0, 360], 'k-', linewidth=1)
        plt.xlim(min3, max3)
        plt.ylim(min3, max3)
        if stats and watermark:
            plt.text((max3-min3)/10 + min3,  (max3-min3) * yloc + min3, statString3)
        if pier == True:
            plt.plot([0, 71], [71, 71], 'k--', label='Shore Normal')
            plt.plot([71, 71], [0, 71], 'k--')
    plt.xlabel(xlabel3, fontsize=12)
    plt.ylabel(ylabel3, fontsize=12)
    sub3.set_title(title3, fontsize=12)
    # saving and closing plot
    plt.tight_layout()
    plt.savefig(ofname)
    plt.close()

    # return data if it's calculated
    if stats:
        return statPacket1, statPacket2, statPacket3

def plotTS(ofname, plotpacket1, plotpacket2, plotpacket3, **kwargs):
    """this is a function that plots 3 time series comparison data
    could be used for example to plot wave height, mean period, mean direction on the same plot
    often paired with plot121 and statistics

    Args:
      ofname: plot file name out
      plotpacket1: a dictionary containing data to be plotted in first subplot
        'time_obs': x value for 1st overplot, plotted in red (generally obs

        'obs':  yvalue for 1st over plot, plotted in red (generally obs)

        'label_obs': legend label for 1st overplot

        'time_mod': x value for 2nd overplot, plotted in blue (generally model)

        'mod': y value for 2nd overplot, plotted in blue (generally model)

        'label_mod': legend label for 2nd overplot

        'TS_ylabel': y label for the whole plot

      plotpacket2: a dictionary containing data to be plotted in 2nd subplot
        'time_obs': x value for 1st overplot, plotted in red (generally obs

        'obs':  yvalue for 1st over plot, plotted in red (generally obs)

        'label_obs': legend label for 1st overplot

        'time_mod': x value for 2nd overplot, plotted in blue (generally model)

        'mod': y value for 2nd overplot, plotted in blue (generally model)

        'label_mod': legend label for 2nd overplot

        'TS_ylabel': y label for the whole plot

      plotpacket3: a dictionary containing data to be plotted in third subplot
        'time_obs': x value for 1st overplot, plotted in red (generally obs

        'obs':  yvalue for 1st over plot, plotted in red (generally obs)

        'label_obs': legend label for 1st overplot

        'time_mod': x value for 2nd overplot, plotted in blue (generally model)

        'mod': y value for 2nd overplot, plotted in blue (generally model)

        'label_mod': legend label for 2nd overplot

        'ylabel': y label for the whole plot
    
    Keyword Args:
        'title' (str):  will title the plot (centered) with the attached string

    Returns:
      plot located at ofname

    """
    # DEFINE plot variables

    if 'watermark' in kwargs:
        watermark= kwargs['watermark']
    else:
        watermark = True
    if 'pier' in kwargs:
        pier = kwargs['pier']
    else:
        pier = True
    #########################################3
    # first plot packet
    xdata1 = np.array(plotpacket1['time_obs'])
    ydata1 = np.array(plotpacket1['obs'])
    xdata11 = np.array(plotpacket1['time_mod'])
    ydata11 = np.array(plotpacket1['mod'])
    label1 = plotpacket1['label_obs']  # first legend label
    label11 = plotpacket1['label_mod']  # second legend label
    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = ''
    ylabel1 = plotpacket1['ylabel']
    fname = ofname
    ###########################################
    # 2nd plot packet
    xdata2 = np.array(plotpacket2['time_obs'])
    ydata2 = np.array(plotpacket2['obs'])
    xdata22 = np.array(plotpacket2['time_mod'])
    ydata22 = np.array(plotpacket2['mod'])
    ylabel2 = plotpacket2['ylabel']
    ymax2 = max(np.max(ydata2), np.max(ydata22)) * 1.15
    ymin2 = min(np.min(ydata2), np.min(ydata22)) * .85
    if watermark:
        try:
            xx = len(xdata2) / 2  # find the middle index for word placement
            xmid = xdata2[xx]
        except TypeError:
            xmid = 0.5
    ###########################################
    # 3rd plotpacket
    xdata3 = np.array(plotpacket3['time_obs'])
    ydata3 = np.array(plotpacket3['obs'])
    xdata33 = np.array(plotpacket3['time_mod'])
    ydata33 = np.array(plotpacket3['mod'])
    ylabel3 = plotpacket3['ylabel']
    ymin3 = max(np.max(ydata3), np.max(ydata33))
    ymax3 = min(np.min(ydata3), np.min(ydata33))
    ############################################
    # make plot
    # plot the defined variables
    ts = plt.figure(figsize=(12, 6), dpi=80)
    # subplot 1
    sub1 = plt.subplot(3, 1, 1)
    if (xdata1 == 0).all() and (xdata11 == 0).all():
        if watermark:
            sub1.text(0.5, 0.5, 'NO data', fontsize=20,
                      color='gray', ha='center', va='center', alpha=0.5)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
    elif (xdata1 == 0).all():
        if watermark:
            sub1.text(0.5, 0.5, 'NO OBSERVATIOIN data', fontsize=20,
                      color='gray', ha='center', va='center', alpha=0.5)
            plt.plot(xdata11, ydata11, 'b', label=label11)

    elif (xdata11 == 0).all():
        sub1.text(0.5, 0.5, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata1, ydata1, '.r', label=label1)
    else:
        plt.plot(xdata1, ydata1, '.r', label=label1)
        plt.plot(xdata11, ydata11, 'b', label=label11)
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=0,
                   ncol=2, mode='expand', borderaxespad=0.)
    plt.title(title, weight='bold')
    plt.gca().get_xaxis().set_visible(False)
    plt.ylabel(ylabel1)

    # 2nd subplot
    sub2 = plt.subplot(3, 1, 2)
    if (xdata2 == 0).all() and (xdata22 == 0).all():
        sub2.text(0.5, 0.5, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata2 == 0).all():
        sub2.text(0.5, 0.5, 'NO OBSERVATIOIN data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata22, ydata22, 'b')

    elif (xdata22 == 0).all():
        sub2.text(0.5, 0.5, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata2, ydata2, '.r')
    else:
        if watermark == True:
            sub2.text(xmid, (ymax2 + ymin2) / 2, 'Research Product', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata2, ydata2, '.r')
        plt.plot(xdata22, ydata22, 'b')
        plt.ylim(ymin2, ymax2)
    plt.ylabel(ylabel2)
    # 3rd subplot
    sub3 = plt.subplot(3, 1, 3)
    if (xdata3 == 0).all() and (xdata33 == 0).all():
        sub3.text(0.5, 180, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata3 == 0).all():
        sub3.text(0.5, 180, 'NO OBSERVATION data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata22, ydata22, 'b')

    elif (xdata33 == 0).all():
        sub3.text(0.5, 180, 'NO MODEL data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.plot(xdata1, ydata2, '.r')
    else:
        plt.plot(xdata3, ydata3, '.r')
        plt.plot(xdata33, ydata33, 'b')
        if pier == True:
            plt.plot([xdata3[0], xdata3[-1]], [71,71],'k-', label='Shore Normal')
    # plt.ylim(0, 360)

    plt.ylabel(ylabel3)
    plt.gcf().autofmt_xdate()
    ts.tight_layout()

    ts.savefig(fname)
    plt.close()

# these are all the ones that were formerly in gridTools!
def plotBathyInterp(ofname, dataDict, title):
    """This is a quick plot of the bathy interp, Not sure if its used in any work flow or was part of a quality check
    This can probably be moved to a plotting library maybe be a more generic
    
    designed to QA newly inserted bathy into background

    Args:
      ofname: file output name
      dataDict: a dictionary with keys:
        'newXfrf'  new x coords (1d)
        'newYfrf'  new y coords (1d)
        'newZfrf'  new Z values (2D of shape newXfrf, newYfrf)
        'newBathyGrid'  2 d array (newly interpolated values ) wth dimensions modelGridX, modelgridY
        'goodOldBathy'
        'modelGridX' 1 d array of model domain
        'modelGridY 1 d array of model domain
      title: Plot title

    Returns:

    """

    newXfrf = dataDict['newXfrf']
    newYfrf = dataDict['newYfrf']
    newZfrf = dataDict['newZfrf']
    newBathyGrid = dataDict['newBathyGrid']
    goodOldBathy = dataDict['goodOldBathy']
    modelGridX = dataDict['modelGridX']
    modelGridY = dataDict['modelGridY']

    # define function
    # prep plot
    lineCodes = 'k--'  # plotting code (line design)
    # establishing limits  for colorbars
    vmin = min(np.floor(goodOldBathy.min()), np.floor(newZfrf.min()))
    vmax = max(np.ceil(goodOldBathy.max()), np.ceil(newZfrf.max()))
    levels = np.linspace(vmin, vmax , 50, endpoint=True)  # making 50 levels of colors between limits above

    # create plotted box
    rectangle = []
    # do plot
    fig = plt.figure(figsize=(7,5))
    plt.suptitle(title, weight='bold')
    ax1 = plt.subplot(121)
    ax1contOld = ax1.pcolor(modelGridX, modelGridY, goodOldBathy,  vmin=vmin, vmax=vmax)
    # cbar = plt.colorbar(ax1cont)  # more sophicsticated color bar below
    # plot Box around removed data
    ax1.plot([newXfrf.min(), newXfrf.max()], [newYfrf.min(), newYfrf.min()], lineCodes, label='New data bounds')  # south Boundary
    ax1.plot([newXfrf.min(), newXfrf.min()], [newYfrf.min(), newYfrf.max()], lineCodes)  # dune Boundary
    ax1.plot([newXfrf.max(), newXfrf.max()], [newYfrf.min(), newYfrf.max()], lineCodes)  # offshore boundary
    ax1.plot([newXfrf.min(), newXfrf.max()], [newYfrf.max(), newYfrf.max()], lineCodes)  # north Boundary
    ax1contNew = ax1.pcolor(newXfrf, newYfrf, newZfrf, vmin=vmin, vmax=vmax) # levels=levels,
    ax1.legend()

    # second Subplot
    ax2 = plt.subplot(122, sharex=ax1)  # , sharey=ax1)  this removed the y axis from both plots
    ax2.pcolor(modelGridX, modelGridY, newBathyGrid, vmin=vmin, vmax=vmax)
    ax2.contour(modelGridX, modelGridY, newBathyGrid, levels= [-9,-6,-3,0], linestyles = '-', colors='k', labels=[-9,-6, -3, 0])
    ax2.set_yticks([])
    plt.subplots_adjust(wspace=0.05)
    plt.plot([0, 566], [515, 515], lw=5) # frf Pier
    # colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(ax1contOld, cax=cbar_ax)
    cbar.set_label('Elevation NAVD 88[m]')
    plt.savefig(ofname)
    plt.close(fig)

def CreateGridPlotinFRF(outi, outj, spacings, fname):
    """This function creates a plot of bathymetry grid.  The axis labels assume FRF coordinates

    Args:
      outi: a 2 by x array of grid nodes with the locations of interest in [:, 0]
      outj: a 2 by y array of grid nodes with the locations of interest in [:, 1]
      spacings (dict): a dictionary with keys
       'dx': the (constant) grid spacing in x

       'dy': the (constant) grid spacing in y

       'ni': the number of cells in x

       'nj': the number of cells in y

      fname: file name output

    Returns:
      a plot with file name fname

    """
    from matplotlib import pyplot as plt

    crossShoreWidth = spacings['dx'] * spacings['ni']
    alongShoreWidth = spacings['dy'] * spacings['nj']
    plt.figure()
    plt.plot(outj[:, 0], outj[:, 1], label='offshore Bound')
    plt.plot(outi[:, 0], outi[:, 1], label='Northern Bound')
    plt.plot(outj[:, 0] - crossShoreWidth, outj[:, 1], label='inshore Bound')
    plt.plot(outi[:, 0], outi[:, 1] - alongShoreWidth, label='Southern Bound')
    plt.plot([0, 560], [515, 515], 'k', lw=5, label='approximate FRF pier')
    plt.plot([0, 1000], [945, 945], 'k', label='Cross Shore Array')
    plt.legend()
    plt.savefig(fname)
    plt.close()

