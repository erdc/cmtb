import numpy as np
import matplotlib.colors as colors

# set the colormap and centre the colorbar
class MidpointNormalize(colors.Normalize):
	"""	Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)

	e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))

	Examples:
        elev_min=-1000
        elev_max=3000
        mid_val=0

        plt.imshow(ras, cmap=cmap, clim=(elev_min, elev_max), norm=MidpointNormalize(midpoint=mid_val,vmin=elev_min, vmax=elev_max))
        plt.colorbar()
        plt.show()
	"""
	def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
		self.midpoint = midpoint
		colors.Normalize.__init__(self, vmin, vmax, clip)

	def __call__(self, value, clip=None):
		# I'm ignoring masked values and all kinds of edge cases to make a
		# simple example...
		x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
		return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def placeSubFigurePosition(text, ax, position, percentSpace=0.05, fontsize=12, fontcolor='black'):
    """ Function will insert a subplot label into the subplot axis eg A, B, C for multiple plots (for publication)

    Args:
        text (str): Text to label the subplot
        ax (subplot axis object):  matplotlib subplot axis object
        position (str): where in the subplot to put the text available locations 'upper right', 'upper left'
        percentSpace (float):  decimal percent (default = 0.05)
        fontsize (int): font size (Default = 12)
        fontcolor (string): a text object color string (default='black') see below link for more detials
                        https://matplotlib.org/users/text_intro.html
    Returns:
        None

    """
    import matplotlib as mpl
    axLims_x = ax.get_xlim()
    axLims_y = ax.get_ylim()
    placeDiff_x = np.diff(axLims_x) * percentSpace
    placeDiff_y = np.diff(axLims_y) * percentSpace
    if position == 'upper right':
        va = 'top'
        ha = 'left'
        if type(ax.yaxis._scale) == mpl.scale.LogScale:
            placeDiff_y = placeDiff_y * 5
        placePos_x = float(axLims_x[-1] - placeDiff_x)
        placePos_y = float(axLims_y[-1] - placeDiff_y)
    elif position == 'upper left':
        va = 'top'
        ha = 'right'
        placePos_y = float(axLims_y[-1] - placeDiff_y)
        placePos_x = float(axLims_x[0] + placeDiff_x)

    ax.text(placePos_x, placePos_y, text, fontsize=fontsize, verticalalignment=va, horizontalalignment=ha, color=fontcolor)

def gradient_fill(x, y, fill_color=None, ax=None, zfunc=False, **kwargs):
    """This is a function that plots a gradient fill found here
    http://stackoverflow.com/questions/29321835/is-it-possible
    -to-get-color-gradients-under-curve-in-matplotlb?noredirect=1&lq=1

    Args:
      x:
      y:
      fill_color:
      param ax:  The axis from the plot (Default value = None)
      zfunc: param kwargs: (Default value = False)


    Keyword Args:


    Returns:

    """

    from matplotlib.patches import Polygon
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    from PIL import Image
    from PIL import ImageDraw
    from PIL import ImageFilter

    def zfunc(x, y, fill_color='k', alpha=1.0):
        """

        Args:
          x:
          y:
          fill_color:  (Default value = 'k')
          alpha:  (Default value = 1.0)

        Returns:

        """
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