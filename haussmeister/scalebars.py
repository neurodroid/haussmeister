# -*- coding: utf-8 -*-
# -*- mode: python -*-
# Adapted from mpl_toolkits.axes_grid2
# LICENSE: Python Software Foundation (http://docs.python.org/license.html)
# https://gist.github.com/matejak/4100881

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, size=0, label=None, horizontal = True, style = 'dark', loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width / height of the bar, in data units.
        - label: label for bars; None to omit
        - horizontal: Whether the bar is horizontal (True) or vertical (False)
        - style: Whether the bar is dark ('dark') or bright (anything else)
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea
        import matplotlib.patches as mpatches
        if style == 'dark':
            textcol = 'k'
        else:
            textcol = 'w'
        bars = AuxTransformBox(transform)
        endpt = (size, 0) if horizontal else (0, size)
        art = mpatches.FancyArrowPatch((0, 0), endpt, color = textcol,
                                       arrowstyle = "|-|")
        # This doesn't work; why?
        #                              arrowstyle = mpatches.ArrowStyle.BarAB(1, 60, 1))
        bars.add_artist(art)

        packer = VPacker if horizontal else HPacker
        bars = packer(children=[bars, TextArea(label, dict(color = textcol), minimumdescent=False)],
                       align="center", pad=0, sep=sep)

        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb
