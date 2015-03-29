"""
Module for generating movies from 2p images

(c) 2015 C. Schmidt-Hieber
GPLv3
"""

import os
import sys
try:
    import subprocess32 as sp
except ImportError:
    sys.stdout.write("Couldn't find subprocess32; using subprocess instead\n")
    import subprocess as sp
import shlex
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from base64 import b64encode

FFMPEG = "ffmpeg"

def get_normbright(arr):
    """
    Create 3-point calibration curve. Returns 3 points of input brightness
    that will be mapped to 0, 0.5, and 1.0 output

    Parameters
    ----------
    arr : numpy.ndarray
        Frame that will be used for calibration

    Returns
    -------
    normcurve : 3-tuple of floats
        3 points of input brightness that will be mapped to
        0, 0.5, and 1.0 output
    """
    return(float(np.min(arr))/2**16,
           8.0*float(np.median(arr)-np.min(arr))/2**16,
           float(np.max(arr))/2**16)

def save_scale_bar(png_name, scale_length_int, scale_length_px, xpx, ypx):
    """
    Save a png with transparent background showing a white scale bar
    with white label that can be added to a movie.

    Parameters
    ----------
    png_name : str
        Full path to png file name
    scale_length_int : int
        Scale bar length in specimen dimensions (e.g. um)
    scale_length_px : int
        Scale bar length in pixels
    xpx : int
        Image width in pixels
    ypx : int
        Image height in pixels
    """
    aa = 4

    margin_x = xpx/50.0

    scale_text = u"{0:0d} \u03BCm".format(int(scale_length_int))

    im = Image.new("RGBA", (xpx*aa, ypx*aa))
    draw = ImageDraw.Draw(im)

    draw.line([((xpx-margin_x-scale_length_px)*aa, (ypx*0.9)*aa), 
               ((xpx-margin_x)*aa, (ypx*0.9)*aa)], 
              width=int(ypx/125.0)*aa, fill="white")

    # get a font
    fnt = ImageFont.truetype(
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     'data/FreeSansBold.ttf'), 24*aa)
    w, h = draw.textsize(scale_text, font=fnt)
    draw.text(((xpx - margin_x - scale_length_px/2.0)*aa - w/2.0, (ypx*0.91)*aa), 
              scale_text, fill="white", font=fnt)

    im = im.resize((xpx, ypx), Image.ANTIALIAS)

    im.save(png_name)

def make_movie(tiff_trunk, out_file, fps, normbright=None, scalebarframe=None, verbose=False, scale=None):
    """
    Produce a movie from a directory with individual tiffs
    at given frame rate

    Parameters
    ----------
    tiff_trunk : str
        File name filter (full path) used as input to ffmpeg 
        (e.g. ``/home/cs/data/ChanA_0001_%04d.tif``)
    out_file : str
        Full path to movie file name
    fps : float
        Acquisition rate of tiffs in frames per second 
    normbright : 3-tuple of floats, optional
        Brightness adjustment curve; see get_normbright() for format
    scalebarframe : str, optional
        Full path to png with scale bar
    verbose : bool, optional
        Print progress of ffmpeg
    scale : 2-tuple of ints
        Rescale movie to given width and height in pixels

    Returns
    -------
    html_movie : str
        An html tag containing the complete movie
    """

    addin = ""
    if scale is not None or normbright is not None or scalebarframe is not None:
        sfilter = "-filter_complex "
    else:
        sfilter = ""

    prev = "[0:v]"
    if scale is not None:
        sfilter += prev + "scale={0}:{1}".format(scale[0], scale[1])
        prev = "[scale];[scale]"

    if normbright is not None:
        sfilter += prev + "curves=all='{0}/0 {1}/0.5 {2}/1'".format(
            normbright[0], normbright[1], normbright[2])
        prev = "[bright];[bright]"
        
    if scalebarframe is not None:
        addin = "-i {0}".format(scalebarframe)
        sfilter +=  prev + "[1:v]overlay=0:0"

    tiff_input = tiff_trunk

    cmd = "{0} -y -r {1} -i {2} {3} {4} ".format(
        FFMPEG, fps, tiff_input, addin, sfilter)
    cmd += "-an -vcodec libx264 -preset slow -crf 27 -pix_fmt yuv420p "
    cmd += "-metadata author=\"(c) 2015 Christoph Schmidt-Hieber\" {0}".format(
        out_file)
    
    sys.stdout.write(cmd)
    cmd_split = shlex.split(cmd)
    sys.stdout.write("\nCreating movie...")
    sys.stdout.flush()
    if verbose:
        stdout = sp.PIPE
    else:
        stdout = None

    P = sp.Popen(cmd_split, stdout=stdout, stderr=stdout, bufsize=4094*4094*16)

    if verbose:
        for line in iter(P.stdout.readline, b''):
            sys.stdout.write(line)
            sys.stdout.flush()

    P.wait()

    sys.stdout.write(" done\n")

    return html_movie(out_file)

def html_movie(fn):
    """
    Converts an mp4 movie to an html tag containing 
    the complete encoded movie

    Parameters
    ----------
    fn : str
        Full path to movie file name

    Returns
    -------
    html_movie : str
        An html tag containing the complete movie
    """
    
    video = open(fn, "rb").read()
    video_encoded = b64encode(video)
    video_tag = '<video controls alt="test" src="data:video/mp4;base64,{0}">'.format(video_encoded)

    return video_tag
