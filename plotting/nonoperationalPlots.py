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

def pltFRFgrid(xyzDict, save=False):
    """
    This function plots a dictionary of values with keys x, y, z

    :param save:
    :return:
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

    """
    This function plots a 2d spectra showing the 1d direction and 1d frequency spectra on both sides, idea and base function
    was taken from the below website from
    http://www.astrobetter.com/blog/2014/02/10/visualization-fun-with-python-2d-histogram-with-1d-histograms-on-axes/
    :param fname: outpufile name
    :param spec2D: 2 dimensional spectrum (single)
    :param freqBin: associated freuqncy bins
    :param dirBin: associated direction bins
    :param pathCHLlogo: defaults to None, but will put a logo at the top right if path is given

    :return: saves a plot to the fname location

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
    """
    this plots a single spectra
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
    if show == 1:
        plt.show()

def plot121(plotpacket1, plotpacket2, plotpacket3):
    """
    this is a plot fuction that will plot 2 dataList and plot them 1-1 as a means
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
    """
    this is a function that plots 3 timeseries comparison data
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
def plotBathyInterp(ofname, dataDict, title):
    """
    This is a quick plot of the bathy interp, Not sure if its used in any work flow or was part of a quality check
    This can probably be moved to a plotting library maybe be a more generic

    designed to QA newly inserted bathy into background

    :param ofname: file output name

    :param dataDict: a dictionary with keys:
        'newXfrf'  new x coords (1d)
        'newYfrf'  new y coords (1d)
        'newZfrf'  new Z values (2D of shape newXfrf, newYfrf)
        'newBathyGrid'  2 d array (newly interpolated values ) wth dimensions modelGridX, modelgridY
        'goodOldBathy'
        'modelGridX' 1 d array of model domain
        'modelGridY 1 d array of model domain


    :param title:  Plot title
    :return:
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
    """
    This function creates a plot of bathymetry grid.  The axis labels assume FRF coordinates

    :param outi:  a 2 by x array of grid nodes with the locations of interest in [:, 0]
    :param outj:  a 2 by y array of grid nodes with the locations of interest in [:, 1]
    :param spacings: a dictionary with keys dx/dy and ni/nj
            where dx dy are the (constant) grid spacing in x and y
            where ni/nj are the number of cells in x and y
    :return: a plot with file name fname
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
def plotUnstructBathy(ofname, pDict):
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

def bathyEdgeHist(ofname, pDict, prox=None):
    """
    Okay the point of this function is to take in some bathy data, pull out all the values along the edges of the
    new surface and plot them to see how far off they are from the original surface.
    if you hand it only 1 surface it will assume that it is a DIFFERENCED surface!!!!
    :param ofname: complete filepath where the output will be stored, including extension!!!!!
    :param pDict:
        Keys:
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

    :return:
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
    else:
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
    else:
        pass
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
        else:
            pass
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

# cool anotation functions
def get_text_positions(x_data, y_data, txt_width, txt_height):
    a = zip(y_data, x_data)
    text_positions = y_data.copy()
    for index, (y, x) in enumerate(a):
        local_text_positions = [i for i in a if i[0] > (y - txt_height)
                            and (abs(i[1] - x) < txt_width * 2) and i != (y,x)]
        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)
            if abs(sorted_ltp[0][0] - y) < txt_height: #True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height
                for k, (j, m) in enumerate(differ):
                    #j is the vertical distance between words
                    if j > txt_height * 2: #if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break
    return text_positions

def text_plotter(x_data, y_data, text_positions, axis,txt_width,txt_height):
    for x,y,t in zip(x_data, y_data, text_positions):
        axis.text(x - txt_width, 1.01*t, '%d'%int(y),rotation=0, color='blue')
        if y != t:
            axis.arrow(x, t,0,y-t, color='red',alpha=0.3, width=txt_width*0.1,
                       head_width=txt_width, head_length=txt_height*0.5,
                       zorder=0,length_includes_head=True)
















