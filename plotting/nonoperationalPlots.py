import matplotlib.colors as mc
from matplotlib.ticker import NullFormatter, MaxNLocator
from testbedutils import waveLib
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.tri as tri
from scipy.spatial import ConvexHull
import testbedutils.sblib as sb
import scipy.spatial
from matplotlib import ticker
import matplotlib.cm as cm
from scipy.interpolate import griddata
from numpy.random import *

def gradient_fill(x, y, fill_color=None, ax=None, zfunc=False, **kwargs):
    """
    This is a function that plots a gradient fill found here
    http://stackoverflow.com/questions/29321835/is-it-possible
    -to-get-color-gradients-under-curve-in-matplotlb?noredirect=1&lq=1
   Args:
        x:
        y:
        fill_color:
        ax:  The axis from the plot
        zfunc:

   Keyword Args
       keyword args from plt.plot

    Returns
        None

    """

    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFilter

    def zfunc(x, y, fill_color='k', alpha=1.0):
        scale = 10
        x = (x * scale).astype(int)
        y = (y * scale).astype(int)
        xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

        w, h = xmax - xmin, ymax - ymin
        z = np.empty((h, w, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:, :, :3] = rgb

        # Build a z-alpha array which is 1 near the line and 0 at the bottom.
        img = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(img)
        xy = (np.column_stack([x, y]))
        xy -= xmin, ymin
        # Draw a blurred line using PIL
        draw.line(map(tuple, xy.tolist()), fill=255, width=15)
        img = img.filter(ImageFilter.GaussianBlur(radius=100))
        # Convert the PIL image to an array
        zalpha = np.asarray(img).astype(float)
        zalpha *= alpha / zalpha.max()
        # make the alphas melt to zero at the bottom
        n = zalpha.shape[0] // 4
        zalpha[:n] *= np.linspace(0, 1, n)[:, None]
        z[:, :, -1] = zalpha
        return z

    if ax is None:
        ax = plt.gca()

    line, = ax.plot(x, y, **kwargs)
    if fill_color is None:
        fill_color = line.get_color()

    zorder = line.get_zorder()
    alpha = line.get_alpha()
    alpha = 1.0 if alpha is None else alpha

    h, w = 100, 1
    # do shading here
    if np.mean(np.diff(y)) >= 5:  # this should be directional plot
        z = np.empty((w, h, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, alpha, h)[:, None].T
    else:  # normal run (shade top to bottom
        z = np.empty((h, w, 4), dtype=float)
        rgb = mcolors.colorConverter.to_rgb(fill_color)
        z[:, :, :3] = rgb
        z[:, :, -1] = np.linspace(0, alpha, h)[:, None]

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
                   origin='lower', zorder=zorder)

    xy = np.column_stack([x, y])
    xy = np.vstack([[xmin, ymin], xy, [xmax, ymin], [xmin, ymin]])
    clip_path = patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True)
    ax.add_patch(clip_path)
    im.set_clip_path(clip_path)
    ax.autoscale(True)
    return line, im

def pltFRFgrid(xyzDict, savefname=None):
    """This function plots a dictionary of values with keys x, y, z

    Args:
        xyzDict: dictionary with x, y, z values
        savefname(bool): save file name

    Returns:
        None
    """
    x = xyzDict['x']
    y = xyzDict['y']
    z = xyzDict['z']

    levels = np.logspace(np.min(z), np.max(z), num=35, endpoint=True, base=10)
    norm = mc.BoundaryNorm(levels, 256)  # color palate for contourplots
    # plt.contourf(ycoord, xcoord, fieldpacket['field'][dir_ocean, :, :], levels, vmin=cbar_min, vmax=cbar_max,
    #              cmap='coolwarm', levels=levels, norm=norm)
    plt.pcolor(x, y, z, vmin=z.min(), vmax=z.max())
    if savefname is not None:
        plt.savefig(savefname)
    plt.close()

def halfPlanePolarPlot(spectra, frequencies, directions, lims=[-18, 162], **kwargs):
    """ creates single polar plot for spectra.  generally Half-planed

    Args:
        spectra (array): 2D array only
        frequencies (array): 1 d array of corresponding frequencies to spectra
        directions(array): directions associated with spectra
        lims (list): default is half plane for Duck (incident energy only), will NOT truncate spectra
            set to None if looking to plot whole 360 polar plot

    Keyword Args:
        'contour_levels'(list): a list of contour levels to color
        'figsize' (tup): a tuple of figure size eg. (12, 10)
        'fname' (str): file path save name
        'fontSize' (int): controls fontsize for labels
        
    Returns:
        Axis object: if you want to further modify the plot

    """
    # begin by checking inputs
    assert np.array(spectra).ndim == 2, 'spectra needs to be 2 dimensional'
    assert np.array(spectra).shape[0] == np.array(frequencies).shape[
        0], 'spectra should be shaped by freq then direction'
    assert np.array(spectra).shape[1] == np.array(directions).shape[
        0], 'spectra should be shaped by freq then direction'
    fontsize=kwargs.get('fontSize', 12)
    myTitle = kwargs.get('title', 'Polar Spectrum ')
    figSize = kwargs.get('figsize', (11,11))
    # pre-processing spectra
    Edarray = np.asarray(spectra, dtype=object)  # make spectra an array (if not already )
    Ednew = np.append(spectra, spectra[:, 0:1], axis=1)  # add extra directional band to get it to wrap
    Dmean_rad = np.deg2rad(np.append(directions, directions[0]))  # convert input directions to radian
    ## set Color-scale
    if 'contour_levels' in kwargs:  # manually set contours
        contour_levels = kwargs['contour_levels']
    else:  # automatically set contours
        Edmax = float(np.max(spectra))  # take max for colorbars
        contourNumber = 50  # set default number of contour levels
        minlevel = Edmax / contourNumber  # calculate min level
        maxlevel = Edmax  # calculate max level
        step = (maxlevel - minlevel) / contourNumber  # associated step
        contour_levels = np.arange(minlevel, maxlevel, step)  # create list/array of contour levels for plot
    ########################################################################
    fig = plt.figure(figsize=figSize)  # create figure
    thetas = Dmean_rad[:]  # in radian NOT DEGREES

    ax = plt.subplot(111, polar=True)  # create polar axis object
    ax.set_theta_direction(-1)  # set to counter clock-wise plot
    ax.set_theta_zero_location("N")  # set zero as up
    colorax = ax.contourf(thetas, frequencies, Ednew, contour_levels)  # make plot

    ## Set titles and colorbar
    plt.suptitle(myTitle, fontsize=22, y=0.95, x=0.45)
    cbar = fig.colorbar(colorax)
    cbar.set_label('Energy Density ($m^2/Hz/deg$)', rotation=270, fontsize=16)
    cbar.ax.get_yaxis().labelpad = 30

    #     degrange = range(0,360,30)
    #     lines, labels = plt.thetagrids(degrange, labels=None, frac = 1.07)
    if lims is not None:
        ax.set_thetalim(np.deg2rad(lims))
    if 'baseGridFname' in kwargs:
        plt.savefig(kwargs['baseGridFname']);
        plt.close()

    return ax

def plot2DcontourSpec(spec2D, freqBin, dirBin, fname, pathCHLlogo=None, **kwargs):
    """This function plots a 2d spectra showing the 1d direction and 1d frequency spectra on both sides, idea and base function
        was taken from the below website from

    Args:
        fname: outpufile name
        spec2D: 2 dimensional spectrum (single)
        freqBin: associated freuqncy bins
        dirBin: associated direction bins
        pathCHLlogo: will put a logo at the top right if path is given (default=None)

    Returns:
        None
    References:
        http://www.astrobetter.com/blog/2014/02/10/visualization-fun-with-python-2d-histogram-with-1d-histograms-on-axes/

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

def pltspec(dirbin, freqbin, spec, name, bounds=[161.8, 341.8], nlines=15, show=1):
    """this plots a single spectra
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
    if 'baseGridFname' in kwargs:
        plt.savefig(kwargs['baseGridFname'])
    if show == True:
        plt.show()

def plot121(plotpacket1, plotpacket2, plotpacket3):
    """ this is a plot fuction that will plot 2 dataList and plot them 1-1 as a means
    for comparison
    the first plot package is usually wave HS
    the second plot packages is ually wave Tp
    the thrid plot package is usually wave direction

    """
    # assigning a figure name from plot packet 1
    try:
        fname = plotpacket1['fname']
        if fname[-4] != '.png' or fname[-4] != '.jpg':
            fname = fname + '.png'
    except KeyError:
        fname = 'default.png'
    path = plotpacket1['path']
    figtitle = plotpacket1['figtitle']
    # data for first subplot (wave height)
    xdata1 = np.array(plotpacket1['xdata'])
    ydata1 = np.array(plotpacket1['ydata'])
    linestring1 = '.b'  # plotpacket1['linestring']
    title1 = plotpacket1['title']
    xlabel1 = plotpacket1['xlabel']
    ylabel1 = plotpacket1['ylabel']
    max1 = max(np.max(ydata1), np.max(xdata1)) * 1.15
    # data for 2nd subplot (period)
    xdata2 = np.array(plotpacket2['xdata'])
    ydata2 = np.array(plotpacket2['ydata'])
    linestring2 = '.b'
    title2 = plotpacket2['title']
    xlabel2 = plotpacket2['xlabel']
    ylabel2 = plotpacket2['ylabel']
    max2 = max(np.max(ydata2), np.max(xdata2)) * 1.15

    # data for 3rd subplot
    xdata3 = np.array(plotpacket3['xdata'])
    ydata3 = np.array(plotpacket3['ydata'])
    linestring3 = '.b'
    title3 = plotpacket3['title']
    xlabel3 = plotpacket3['xlabel']
    ylabel3 = plotpacket3['ylabel']
    max3 = max(np.max(ydata3), np.max(xdata3)) * 1.15
    min3 = min(np.min(ydata3), np.min(xdata3)) * .85

    # plotting
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
    else:
        line1, = plt.plot(xdata1, ydata1, linestring1)
        plt.plot([0, max1], [0, max1], 'k-', linewidth=1)
        plt.xlim((0, max1))
        plt.ylim((0, max1))
    plt.xlabel(xlabel1)
    plt.ylabel(ylabel1)
    sub1.set_title(title1)
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
        sub2.text(max2 / 2, max2 / 2, 'Research Product', fontsize=20,
                  color='gray', ha='center', va='top', alpha=0.5)
        plt.plot([0, max2], [0, max2], 'k-', linewidth=1)
        plt.xlim((0, max2))
        plt.ylim((0, max2))
    plt.xlabel(xlabel2)
    plt.ylabel(ylabel2)
    sub2.set_title(figtitle + '\n' + title2)
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

        # plt.xlim(0, 360)
        # plt.ylim(0, 360)
        plt.xlim(min3, max3)
        plt.ylim(min3, max3)

    plt.xlabel(xlabel3)
    plt.ylabel(ylabel3)
    sub3.set_title(title3)
    # saving and closing plot
    plt.tight_layout()
    plt.savefig(path + fname)
    plt.close()

def plotTS(plotpacket1, plotpacket2, plotpacket3):
    """this is a function that plots 3 timeseries comparison data
        title, path and file name are defined in plotpacket1
    """
    # DEFINE plot variables
    # first plot packet
    xdata1 = np.array(plotpacket1['xdata1'])
    ydata1 = np.array(plotpacket1['ydata1'])
    xdata11 = np.array(plotpacket1['xdata2'])
    ydata11 = np.array(plotpacket1['ydata2'])
    label1 = plotpacket1['label1']  # first legend label
    label11 = plotpacket1['label2']  # second legend label
    title = plotpacket1['title']
    ylabel1 = plotpacket1['ylabel']
    path = plotpacket1['path']
    fname = plotpacket1['fname']
    # 2nd plot packet
    xdata2 = np.array(plotpacket2['xdata1'])
    ydata2 = np.array(plotpacket2['ydata1'])
    xdata22 = np.array(plotpacket2['xdata2'])
    ydata22 = np.array(plotpacket2['ydata2'])
    ylabel2 = plotpacket2['ylabel']
    ymax2 = max(np.max(ydata2), np.max(ydata22)) * 1.15
    ymin2 = min(np.min(ydata2), np.min(ydata22)) * .85
    try:
        xx = len(xdata2) / 2  # find the middle index for word placement
        xmid = xdata2[xx]

    except TypeError:
        xmid = 0.5
    # 3rd plotpacket
    xdata3 = np.array(plotpacket3['xdata1'])
    ydata3 = np.array(plotpacket3['ydata1'])
    xdata33 = np.array(plotpacket3['xdata2'])
    ydata33 = np.array(plotpacket3['ydata2'])
    ylabel3 = plotpacket3['ylabel']
    ymin3 = max(np.max(ydata3), np.max(ydata33))
    ymax3 = min(np.min(ydata3), np.min(ydata33))

    # plot the defined variables
    ts = plt.figure(figsize=(12, 6), dpi=80)
    # subplot 1
    sub1 = plt.subplot(3, 1, 1)
    if (xdata1 == 0).all() and (xdata11 == 0).all():
        sub1.text(0.5, 0.5, 'NO data', fontsize=20,
                  color='gray', ha='center', va='center', alpha=0.5)
        plt.xlim(0, 1)
        plt.ylim(0, 1)
    elif (xdata1 == 0).all():
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
    # plt.ylim(0, 360)

    plt.ylabel(ylabel3)
    plt.gcf().autofmt_xdate()
    ts.tight_layout()

    ts.savefig(path + fname)
    plt.close()

# these are all the ones that were formerly in gridTools!
def plotBathyInterp(ofname2, dataDict, title):
    """This is a quick plot of the bathy interp, Not sure if its used in any work flow or was part of a quality check
    This can probably be moved to a plotting library maybe be a more generic

    designed to QA newly inserted bathy into background

    Args:
      ofname2: file output name
      dataDict: a dictionary with keys:
        'newXfrf'  new x coords (1d)
        'newYfrf'  new y coords (1d)
        'newZfrf'  new Z values (2D of shape newXfrf, newYfrf)
        'newBathyGrid'  2 d array (newly interpolated values ) wth dimensions modelGridX, modelgridY
        'goodOldBathy'
        'modelGridX' 1 d array of model domain
        'modelGridY 1 d array of model domain


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
    plt.savefig(ofname2)
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
      a plot with file name baseGridFname

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

# these are some new ones I made for the .tel file
def bathyEdgeHist(ofname, pDict, prox=None):
    """this function takes in bathy data, pulls out all the values along the edges of the new surface and plots them to
     see how far off they are from the original surface. if you hand it only 1 surface it will assume that it is a
     differenced surface

        ofname: complete filepath where the output will be stored, including extension!!!!!
        pDict: input plotting dictionary with keys
            ptitle - plot title
            x - x-positions
            y - y-positions
            hUnits - units of the x and y positions (m or ft)
            z1 - this can be any value that you want to compare, but for our applications mainly depth or elevation
            z2 - this can be any value that you want to compare, but for our applications mainly depth or elevation
            zUnits - units of the z stuff (m or ft)
            xHistLabel - label for hist x-axis
            yHistLabel - label for hist y-axis
            xcLabel - label for x-axis
            ycLabel - label for y-axis
            cbarLabel - label for the color bar
            cbarMin - minumum value to show on colorbar
            cbarMax - maximum value to show on colorbar
            cbarColor - type of colorbar you want to use
            ncLev - number of contour "levels" you want to have.
                    defaults to 100 to make it look like a continuous colorbar

    Returns
        histogram plot of the differences (z1 - z2) in the EDGES of the surface!!!!

    """
    # check for dictionary keys
    assert 'x' in pDict.keys(), "Error: x must be specified"
    assert 'y' in pDict.keys(), "Error: y must be specified"
    assert 'z1' in pDict.keys(), "Error: z1 must be specified"

    # make assumptions if optional keys are blank
    if 'xHistLabel' not in pDict.keys():
        pDict['xHistLabel'] = 'bins'
    if 'yHistLabel' not in pDict.keys():
        pDict['yHistLabel'] = 'Number'
    if 'xcLabel' not in pDict.keys():
        pDict['xcLabel'] = 'x'
    if 'ycLabel' not in pDict.keys():
        pDict['ycLabel'] = 'y'
    if 'cbarLabel' not in pDict.keys():
        pDict['cbarLabel'] = 'z'
    if 'cbarColor' not in pDict.keys():
        pDict['cbarColor'] = 'RdYlBu'
    if 'ncLev' not in pDict.keys():
        pDict['ncLev'] = 100
    if 'hUnits' not in pDict.keys():
        pDict['hUnits'] = 'm'
    if 'zUnits' not in pDict.keys():
        pDict['zUnits'] = 'm'

    # get differenced surface
    if 'z2' in pDict.keys():
        assert np.shape(pDict['z2']) == np.shape(pDict['z1']), 'Error: z2 and z1 must be same shape.'
        dz = pDict['z1'] - pDict['z2']
        dz = pDict['z1']

    # check shape of everything.
    dz_sz = np.shape(dz)
    if len(dz_sz) > 1:
        # you have a 2D grid, check the sizes of x and y
        dz_v = dz.reshape((1, dz.shape[0] * dz.shape[1]))[0]
        if dz_sz == np.shape(pDict['x']) and dz_sz == np.shape(pDict['y']):
            # reshape into list of points
            x_v = pDict['x'].reshape((1, pDict['x'].shape[0] * pDict['x'].shape[1]))[0]
            y_v = pDict['y'].reshape((1, pDict['y'].shape[0] * pDict['y'].shape[1]))[0]
        else:
            # turn x and y points into meshgrid
            tx, ty = np.meshgrid(pDict['x'], pDict['y'])
            # reshape into list of points
            x_v = tx.reshape((1, tx.shape[0] * tx.shape[1]))[0]
            y_v = ty.reshape((1, ty.shape[0] * ty.shape[1]))[0]
    else:
        # you already have lists of points
        dz_v = dz
        x_v = pDict['x']
        y_v = pDict['y']

    if 'cbarMin' not in pDict.keys():
        pDict['cbarMin'] = np.nanmin(dz_v)
    if 'cbarMax' not in pDict.keys():
        pDict['cbarMax'] = np.nanmax(dz_v)

    # now that I have a list of all points, I need to find the edges
    points = np.column_stack((x_v, y_v))
    hull = ConvexHull(points)
    hullPts = points[hull.vertices, :]
    # repeat the first point at the end so the below code checks the line between the last point and the first as well
    hullPts2 = np.concatenate((hullPts, [hullPts[0,:]]), axis=0)

    # how far from the edges are each of these points?
    hullDist = []
    for j in range(0, np.shape(points)[0]):
        dists = []
        p = points[j, :]
        for i in range(len(hullPts2) - 1):
            dists.append(sb.dist(hullPts2[i][0], hullPts2[i][1], hullPts2[i + 1][0], hullPts2[i + 1][1], p[0], p[1]))
        hullDist.append(min(dists))

    # show me the points within prox m of the edge
    if prox is None:
        # compute average nearest neighbor distance and use that
        kdt = scipy.spatial.cKDTree(points)
        k = 1  # number of nearest neighbors
        dists, neighs = kdt.query(points, k + 1)
        prox = np.mean(dists[:, 1])
    ind = np.array(hullDist) <= prox
    edgePts = points[ind, :]
    edgeDiffs = dz_v[ind]

    # make a histogram and a contourf plot of the original surface with the hull bounds and edge points overlaid
    # in a panel to the right - this is going to be a cool figure.

    # show time
    # check to see the x vs y extents of my data.  If x is >> y you are better off with a horizontal plot
    yR = max(y_v) - min(y_v)
    xR = max(x_v) - min(x_v)
    if xR >= 1.5*yR:
        sp1 = 211
        sp2 = 223
        sp3 = 224
    else:
        sp1 = 121
        sp2 = 222
        sp3 = 224

    # if I have colorbar ranges, force the data to be within the min/max bounds
    dz_v[dz_v < pDict['cbarMin']] = pDict['cbarMin']
    dz_v[dz_v > pDict['cbarMax']] = pDict['cbarMax']

    # figure out how to force my colorbar ticks through zero
    cbpts = 5
    if pDict['cbarMin'] > 0 or pDict['cbarMax'] < 0:
        v = np.linspace(pDict['cbarMin'], pDict['cbarMax'], cbpts, endpoint=True)
    else:
        # first guess at spacing
        s1 = (pDict['cbarMax'] - pDict['cbarMin']) / float(cbpts)
        cnt = 0
        if s1 > 1:
            while s1 > 1:
                cnt = cnt + 1
                s1 = s1 / float(10)
        elif s1 < 0.1:
            while s1 < 0.1:
                cnt = cnt - 1
                s1 = s1 * float(10)
        # round to nearest quarter
        s1n = round(s1 * 4) / 4
        if s1n == 0:
            s1n = round(s1, 1)
    
        # get it to the same decimal place it was before
        s1n = s1n * 10 ** cnt
    
        # build stuff out of it....
        rL = np.arange(0, pDict['cbarMax'], s1n)
        lL = -1 * np.arange(s1n, abs(pDict['cbarMin']), s1n)
        v = np.concatenate([lL, rL])

    # perform triangulation
    triang = tri.Triangulation(x_v, y_v)


    # figure time?
    fig = plt.figure(figsize=(10, 10))
    if 'ptitle' in pDict.keys():
        fig.suptitle(pDict['ptitle'], fontsize=18, fontweight='bold', verticalalignment='top')

    # colour contour plot...
    ax1 = plt.subplot(sp1)
    ax1.set_aspect('equal')
    clev = np.arange(dz_v.min(), dz_v.max(), 1 / float(pDict['ncLev']))
    tmp = ax1.tricontourf(triang, dz_v, clev, cmap=plt.get_cmap(pDict['cbarColor']))
    cb1 = plt.colorbar(tmp, orientation='horizontal', ticks=v)
    # set some other labels
    ax1.set_ylabel(pDict['ycLabel'], fontsize=12)
    ax1.set_xlabel(pDict['xcLabel'], fontsize=12)
    # overlay the hull points
    ax1.plot(hullPts2[:, 0], hullPts2[:, 1], 'k--')
    # overlay the "edge points"
    ax1.scatter(edgePts[:, 0], edgePts[:, 1], c=edgeDiffs, marker='o', zorder=1, cmap=pDict['cbarColor'])
    # ax1.scatter(edgePts[:, 0], edgePts[:, 1], c='r', marker='o', zorder=1)

    # doctor up the x-ticks
    M = 4
    xticks = ticker.MaxNLocator(M)
    ax1.xaxis.set_major_locator(xticks)

    # edge depth histogram...
    ax2 = plt.subplot(sp2)
    # want an average of 10 in each bin
    nbins = int(round(len(edgeDiffs)/float(10)))
    if nbins < 5:
        nbins = int(5)
    n, bins, patches = ax2.hist(edgeDiffs, nbins, facecolor='green', alpha=0.75)
    ax2.grid(True, linestyle='dotted')
    # set some other labels
    ax2.set_ylabel(pDict['yHistLabel'], fontsize=12)
    ax2.set_xlabel(pDict['xHistLabel'], fontsize=12)

    # some basic stats about this plot
    header_str = 'STATISTICS:'
    # edge threshold
    edgeThresh_str = '\n edge threshold $=%s$ $%s$' % ("{0:.2f}".format(prox), pDict['hUnits'])
    # how many edge points
    edgeNum_str = '\n number of edge points $=%s$' % (str(len(edgeDiffs)))
    # average and std of the depth difference
    meanDiff_str = '\n mean difference $=%s$ $%s$' % ("{0:.2f}".format(np.mean(edgeDiffs)), pDict['zUnits'])
    sDev_str = '\n s.dev of difference $=%s$ $%s$' % ("{0:.2f}".format(np.std(edgeDiffs)), pDict['zUnits'])
    plot_str = edgeThresh_str + edgeNum_str + meanDiff_str + sDev_str
    ax3 = plt.subplot(sp3)
    ax3.axis('off')
    ax3.text(0.01, 0.99, header_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=18,
             fontweight='bold')
    ax3.text(0.01, 0.95, plot_str, verticalalignment='top', horizontalalignment='left', color='black', fontsize=16)

    fig.subplots_adjust(wspace=0.4, hspace=0.1)
    fig.tight_layout(pad=1, h_pad=2.5, w_pad=1, rect=[0.0, 0.0, 1.0, 0.925])
    # save this?
    plt.savefig(ofname, dpi=300)

def plot_scatterAndQQ(fname, time,  model, observations, **kwargs):
    """
    This will make a time-series, a binned scatter plot and a QQ plot for models and observations

    Args:
        fname (str): save filename
        time (list): datetime objects matched to model and observations
        model (list): plottable values from the model (plotted on the y axis)
        observations (list): plottable values from the observations (plotted on the x axis)

    Keyword Args:
        ** title (str): the title for the plot
        ** units (str): used as axis label (not implemented)

    """
    ### imports
    from statsmodels.graphics import gofplots
    from testbedutils import sblib as sb

    if 'title' in kwargs:
        title = kwargs['title']
    else:
        title = 'Observations and model comparisons'
    if 'units' in kwargs:
        units = kwargs['units']
    else:
        units=None
    ###########
    # calculate statistics
    if np.ma.isMaskedArray(model) and model.mask.any():
        raise NotImplementedError('These are not fixed, check binned_xshoreSkillStat_generic for ideas')
    else:
        model=np.array(model)
    if np.ma.isMaskedArray(observations) and observations.mask.any():
        raise NotImplementedError ('These are not fixed, check binned_xshoreSkillStat_generic for ideas')
    else:
        observations = np.array(observations)
    stats_dict = sb.statsBryant(observations, model)

    ## generate string for plot
    statString1 =  "Statistics\n\nModel to Observations:\n\nBias: {0:.2f}\nRMSE: {1:.2f}\n".format(stats_dict['bias'], stats_dict['RMSE'])
    statString2 = "Scatter Index: {0:.2f}\nSymmetric Slope: {1:.2f}\n".format(stats_dict['scatterIndex'], stats_dict['symSlope'])
    statString3 = "$R^2$: {0:.2f}\nsample Count: {1}".format(stats_dict['corr']**2, len(stats_dict['residuals']))
    statString = statString1 + statString2 + statString3
    #############################################
    #  # # prep for plot
    nbins = 100
    H, xedges, yedges = np.histogram2d(observations, model, bins=nbins)
    H = np.rot90(H)
    H = np.flipud(H)
    # Mask zeros
    Hmasked = np.ma.masked_where(H==0,H)
    # find data lims
    ax1max = np.ceil(max(xedges.max(), yedges.max()))
    ax1min = np.floor(min(xedges.min(), yedges.min()))

    ########## make plot ########################
    fig = plt.figure(figsize=(12,7))
    fig.suptitle(title)

    ax0 = plt.subplot2grid((2,3),(0,0), colspan=3)
    ax0.plot(time, model, 'b.', label='model')
    ax0.plot(time, observations, 'r.', ms=1, label='observation')
    ax0.legend()

    ax1 = plt.subplot2grid((2,3),(1,0))
    ax1.plot([ax1min, ax1max], [ax1min, ax1max], 'k--', lw=1)
    ax1.set_ylim([ax1min, ax1max])
    ax1.set_xlim([ax1min, ax1max])
    histo = ax1.pcolormesh(xedges, yedges, Hmasked)
    cbar = plt.colorbar(histo)
    cbar.ax.set_ylabel('Counts')
    ax1.set_xlabel('observations')
    ax1.set_ylabel('model')

    ax2 = plt.subplot2grid((2,3),(1,1), sharey=ax1)
    gofplots.qqplot_2samples(model, observations,  xlabel='observations', ylabel='model', line='45', ax=ax2)

    ax3 = plt.subplot2grid((2,3), (1,2))
    ax3.text(0, 0, statString, fontsize=12 )
    ax3.set_axis_off()

    plt.tight_layout(rect=[0, 0, 1, .95])
    plt.savefig(fname); plt.close()

def halfPlanePolarPlot(spectra, frequencies, directions, lims=[-18, 162], **kwargs):
    """ creates single polar plot for spectra, taken in part from CDIP

    Args:
        spectra (array): 2D array only
        frequencies (array): 1 d array of corresponding frequencies to spectra
        directions(array): directions associated with spectra
        lims (list): default is half plane for Duck (incident energy only), will NOT truncate spectra
            set to None if looking to plot whole 360 polar plot

    Keyword Args:
        'contour_levels'(list): a list of contour levels to color
        'figsize' (tup): a tuple of figure size eg. (12, 10)
        'fname' (str): file path save name
    Returns:
        Axis object

    """
    # begin by checking inputs
    assert np.array(spectra).ndim == 2, 'spectra needs to be 2 dimensional'
    assert np.array(spectra).shape[0] == np.array(frequencies).shape[
        0], 'spectra should be shaped by freq then direction'
    assert np.array(spectra).shape[1] == np.array(directions).shape[
        0], 'spectra should be shaped by freq then direction'
    # pre-processing spectra
    Edarray = np.asarray(spectra, dtype=object)  # make spectra an array (if not already )
    Ednew = np.append(spectra, spectra[:, 0:1], axis=1)  # add extra directionalWaveGaugeList band to get it to wrap
    Dmean_rad = np.deg2rad(np.append(directions, directions[0]))  # convert input directions to radian
    ## set Color-scale
    if 'contour_levels' in kwargs:  # manually set contours
        contour_levels = kwargs['contour_levels']
    else:  # automatically set contours
        Edmax = float(np.max(spectra))  # take max for colorbars
        contourNumber = 50  # set default number of contour levels
        minlevel = Edmax / contourNumber  # calculate min level
        maxlevel = Edmax  # calculate max level
        step = (maxlevel - minlevel) / contourNumber  # associated step
        contour_levels = np.arange(minlevel, maxlevel, step)  # create list/array of contour levels for plot
    if 'figsize' in kwargs:
        figSize = kwargs['figsize']
    else:
        figSize = (11, 11)
    ########################################################################
    fig = plt.figure(figsize=figSize)  # create figure
    thetas = Dmean_rad[:]  # in radian NOT DEGREES

    ax = plt.subplot(111, polar=True)  # create polar axis object
    ax.set_theta_direction(-1)  # set to counter clock-wise plot
    ax.set_theta_zero_location("N")  # set zero as up
    colorax = ax.contourf(thetas, frequencies, Ednew, contour_levels)  # make plot

    ## Set titles and colorbar
    plt.suptitle('Polar Spectrum ', fontsize=22, y=0.95, x=0.45)
    cbar = fig.colorbar(colorax)
    cbar.set_label('Energy Density ($m^2/Hz/deg$)', rotation=270, fontsize=16)
    cbar.ax.get_yaxis().labelpad = 30

    #     degrange = range(0,360,30)
    #     lines, labels = plt.thetagrids(degrange, labels=None, frac = 1.07)
    if lims is not None:
        ax.set_thetalim(np.deg2rad(lims))
    if 'fname' in kwargs:
        plt.savefig(kwargs['fname']);
        plt.close()

    return ax
