# *- coding: utf-8 -*-
"""
Created on Thu Aug 20 16:45:37 2015
This is to analyze and visualize the data for the test bed
@author: Spicer Bak

"""
import numpy as np
from matplotlib import pyplot as plt
from prepdata import prepDataLib
import matplotlib.colors as mc
from matplotlib.ticker import NullFormatter, MaxNLocator
from testbedutils import waveLib

# from matplotlib import cm

# THIS ENTIRE .PY FILE IS DEPRECATED.
# ALL THESE FUNCTIONS HAVE BEEN MOVED TO nonoperationalPlots.py OR operationalPlots.py!!!!!!!!

class plotLib():

    def gradient_fill(self, x, y, fill_color=None, ax=None, zfunc=False, **kwargs):
        """
        This is a function that plots a gradient fill found here
        http://stackoverflow.com/questions/29321835/is-it-possible
        -to-get-color-gradients-under-curve-in-matplotlb?noredirect=1&lq=1
        :param x:
        :param y:
        :param fill_color:
        :param ax:  The axis from the plot
        :param zfunc:
        :param kwargs:
        :return:
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
            z[:,:,:3] = rgb
            z[:,:,-1] = np.linspace(0, alpha, h)[:,None]

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

    def pltFRFgrid(self, xyzDict, save=False):
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

    def plot2DcontourSpec(self, spec2D, freqBin, dirBin, fname, pathCHLlogo=None, **kwargs):

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

    def plotTripleSpectra(self, fnameOut, time, Hs, raw, rot, interp, full=False):
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

        nlines = 15  # number of lines to span across Half planed spectra
        lw = 3 # this is the line width factor for showing the non shore perpendicular value
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

        # pltrawdWED = np.zeros([rawspec.shape[1], np.size(rawDirBin)])
        # pltrotdWED = np.zeros([np.size(rotDirbin), np.size(rawDirBin)])
        # pltintdWED = np.zeros([np.size(rotDirbin), np.size(rawDirbin)])

        # %%%% plotting loop %%%%%
        # for zz in range(0, raw.shape[0]):
        # prep formatting for plt
        pltrawdWED = rawdwed  # [zz, :, :]
        pltrotdWED = rot_dWED  # [zz, :, :]
        pltintdWED = interp_dWED  # [zz, :, :]
        # now set the interpd dwed based oon full or half plane


        # getting proper colorbars and labels forthe contour plots
        # cbar_min = np.min(rawspec['dWED']) # holding constant over entire run
        # cbar_max = np.max(rawspec['dWED']) # holding constant over entire run
        cbar_min = np.min(pltrawdWED)
        cbar_max = np.max(pltrawdWED)
        levels = np.linspace(cbar_min, cbar_max, 35)  # the established levels to be plotted
        # levels = np.logspace(cbar_min, cbar_max**(1/cbar_max),num=35, endpoint=True, base=10)
        from matplotlib import colors
        norm = colors.LogNorm() #mc.BoundaryNorm(levels, 256)  # color palate for contourplots

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
        aaa = sub1.contourf(rawFreqBin, rawDirBin, zip(*pltrawdWED),
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
        bbb = sub2.contourf(rotFreqBin, rotDirBin, zip(*pltrotdWED),
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
        ccc = sub3.contourf(interpFreqBin, interpDirBin, zip(*pltintdWED),
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

    def plotSpatialFieldData(self, contourpacket, fieldpacket, namebase='/file', prefix='', nested=1, *args):
        """
            This function plots a 2D field of data

        :param fieldpacket:  field packet contains
                field:  field of data type: numpy array of [time, x coords, ycoords]
                title:  title for the plot
                xlabel: label for the x axis
                ylabel: label for the y axis
                xcoord: array of xcoordinates = number of cells in x direction
                ycoord: array of y coordinates = number of cells in y direction
                cblabel: label for the colorbar, the value being plotted
        :param prefix: prefix to savefile (path
        :param namebase: a base to create filenames with, datetime will be appended
        :return:  a plot to file
        """
        # aplace to manipulate axes - not manipulated now
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
            print "spatial plotting function cannot currently handle dx != dy"
            raise NotImplementedError
        # applying colorbar labels
        cbar_max = np.ceil(fieldpacket['field'].max())
        cbar_min = np.floor(fieldpacket['field'].min())
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

        #
        # __LOOPING THROUGH PLOTS___
        for tt in range(0, numrecs):
            # print '\ntitle: %s plot \nsize: %s \ntime %s \ncbar_min %d cbar_max %d' %(title, fgsize, time[tt], cbar_min, cbar_max)

            plt.figure(figsize=fgsize, dpi=80, tight_layout=True)
            plt.title(title + '\n%s' % time[tt])
            try:
                plt.contourf(xcoord, ycoord, fieldpacket['field'][tt, :, :], levels, vmin=cbar_min, vmax=cbar_max,
                         cmap='coolwarm', levels=levels, norm=norm)
            except TypeError:
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

    def pltspec(self, dirbin, freqbin, spec, name, bounds=[161.8, 341.8], nlines=15, show=1):
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

    def plot121(self, plotpacket1, plotpacket2, plotpacket3):
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

    def plotTS(self, plotpacket1, plotpacket2, plotpacket3):
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

    def plotWaveProfile(self, x, waveHs, bathyToPlot, fname):
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


