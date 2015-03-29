"""
Module for importing and exporting 2p imaging datasets

(c) 2015 C. Schmidt-Hieber
GPLv3
"""

import os
import sys
import abc
import glob
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image

import sima
from sima.misc import tifffile

import movies

class HaussIO(object):
    """
    Base class for objects representing 2p imaging data.

    Attributes
    ----------
    xsize : float
        Width of frames in specimen dimensions (e.g. um)
    ysize : float
        Height of frames in specimen dimensions (e.g. um)
    xpx : int
        Width of frames in pixels
    ypx : int
        Height of frames in pixels
    timing : numpy.ndarray
        Time points of frame acquisitions
    fps : float
        Acquisition rate in frames per seconds
    fps : float
        Acquisition interval in seconds
    movie_fn : str
        File path (full path) for exported movie
    scale_png : str
        File path (full path) for png showing scale bar
    sima_dir : str
        Full path to directory for sima exports
    basefile : str
        File name trunk for individual tiffs (without path)
        (e.g. ``ChanA_0001_``)
    filetrunk : str
        Full path and file name trunk for individual tiffs
        (e.g. ``/home/cs/data/ChanA_0001_``)
    ffmpeg_fn : str
        File name filter (full path) used as input to ffmpeg 
        (e.g. ``/home/cs/data/ChanA_0001_%04d.tif``)
    filenames : list of str
        List of file paths (full paths) of individual tiffs
    """
    def __init__(self, dirname, chan='A', xml_path=None):

        self.dirname = dirname
        self.chan = chan

        self._get_filenames(xml_path)

        sys.stdout.write("Reading experiment settings... ")
        sys.stdout.flush()
        self.xml_root = ET.parse(self.xml_name).getroot()
        self._get_dimensions()
        self._get_timing()
        self.dt = np.mean(np.diff(self.timing))
        self.fps = 1.0/self.dt
        self.nframes = len(self.timing)
        sys.stdout.write("done\n")

        assert(len(self.filenames) == self.nframes)

    @abc.abstractmethod
    def _get_dimensions(self):
        return

    @abc.abstractmethod
    def _get_timing(self):
        return

    def _get_filenames(self, xml_path):
        self.movie_fn = self.dirname + ".mp4"
        self.scale_png = self.dirname + "_scale.png"
        self.sima_dir = self.dirname + ".sima"
        self.basefile = "Chan" + self.chan + "_0001_0001_0001_"
        self.filetrunk = self.dirname + '/' + self.basefile
        self.ffmpeg_fn = self.filetrunk + "%04d.tif"
        self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))

    def get_normframe(self):
        """
        Return a representative frame that will be used to normalize
        the brightness in movies

        Returns
        -------
        arr : numpy.ndarray
            Frame converted to numpy.ndarray
        """
        normframe = self.filetrunk + "{0:04d}.tif".format(int(len(self.filenames)/2))
        sample = Image.open(normframe)
        arr = np.asarray(sample, dtype=np.float)
        return arr

    def tosima(self, startIdx=0, stopIdx=None):
        """
        Convert the experiment to a sima.ImagingDataset

        Parameters
        ----------
        startIdx : int, optional
            Starting index (inclusive) for conversion.
        stopIdx : int
            Last index (exclusive) for conversion.

        Returns
        -------
        dataset : sima.ImagingDataset
            A sima.ImagingDataset
        """
        # The string paths[i][j] is a unix style expression for the
        # filenames for plane i and channel j
        if stopIdx is None:
            stopIdx = self.nframes
        sequences = [sima.Sequence.create(
            'TIFFs', [[self.filetrunk + "????.tif"]])]
        if stopIdx is not None:
            sequences = [seq[:stopIdx,:,:,:,:] for seq in sequences]
        if startIdx != 0:
            sequences = [seq[startIdx:,:,:,:,:] for seq in sequences]
        return sima.ImagingDataset(
            sequences, self.sima_dir, channel_names=[self.chan,])

    def make_movie(self, norm=True, scalebar=True):
        """
        Produce a movie of the experiment

        Parameters
        ----------
        norm : bool, optional
            Normalize min, median and max brightness to 0, 0.5 and 1.0
        scalebar : bool, optional
            Show a scale bar in the movie.

        Returns
        -------
        html_movie : str
            An html tag containing the complete movie
        """

        if norm:
            normbright = movies.get_normbright(self.get_normframe())
        else:
            normbright = None

        if scalebar:
            self.save_scale_bar()
            scalebarframe = self.scale_png
        else:
            scalebarframe = None

        return movies.make_movie(self.ffmpeg_fn, self.movie_fn, self.fps, normbright,
                                 scalebarframe)

    def make_movie_extern(self, path_extern, norm=True, scalebar=True):
        """
        Produce a movie from a directory with individual tiffs, using
        the present experimental settings for scale bar and frame rate

        Parameters
        ----------
        path_extern : str
            Full path to the directory that contains the individual tiffs
        norm : bool, optional
            Normalize min, median and max brightness to 0, 0.5 and 1.0
        scalebar : bool, optional
            Show a scale bar in the movie.

        Returns
        -------
        html_movie : str
            An html tag containing the complete movie
        """
        if norm:
            normbright = movies.get_normbright(np.asarray(Image.open(
                path_extern + "/" + self.basefile + "{0:04d}.tif".format(
                    int(self.nframes/2)))))
        else:
            normbright = None

        if scalebar:
            self.save_scale_bar()
            scalebarframe = self.scale_png
        else:
            scalebarframe = None

        return movies.make_movie(
            path_extern + "/" + self.basefile + "%04d.tif",
            path_extern + ".mp4",
            self.fps,
            normbright,
            scalebarframe,
            scale=(self.xpx, self.ypx))

    def get_scale_bar(self, prop=1/8.0):
        """
        Returns lengths in specimen dimensions (e.g. um) and in pixels
        of a scale bar that fills the given fraction of the width of the
        image

        Parameters
        ----------
        prop : float, optional
            Length of scale bar expressed as fraction of image width

        Returns
        -------
        scale_length_int : int
            Scale bar length in specimen dimensions (e.g. um)
        scale_length_px : int
            Scale bar length in pixels
        """
        
        # Reasonable scale bar length (in specimen dimensions, e.g. um)
        # given the width of the image:
        scale_length_float = self.xsize * prop

        # Find closest integer that looks pretty as a scale bar label:
        nzeros = int(np.log10(scale_length_float))
        closest_int = np.round(scale_length_float/10**nzeros)

        if closest_int <= 5:
            scale_length_int = closest_int * 10**nzeros
        else:
            # Closer to 5 or closer to 10?
            if 10-closest_int < closest_int-5:
                scale_length_int = 1 * 10**(nzeros+1)
            else:
                scale_length_int = 5 * 10**(nzeros)

        scale_length_px = scale_length_int * self.xpx/self.xsize

        return scale_length_int, scale_length_px

    def save_scale_bar(self):
        """
        Save scale bar as png (using file name stored in self.scale_png)
        so that it can be used in a movie
        """
        scale_length_int, scale_length_px =self.get_scale_bar()
        movies.save_scale_bar(self.scale_png, scale_length_int, scale_length_px,
                              self.xpx, self.ypx)

    def plot_scale_bar(self, ax):
        """
        Add scale bar to a matplotlib axis

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            matplotlib axes on which to plot the scale bar, as
            e.g. returned by 'fig.add_suplot(1,1,1)'
        """
        sb_int, sb_px = self.get_scale_bar()
        scale_text = u"{0:0d} \u03BCm".format(int(sb_int))
        ax.plot([self.xpx/10.0,
                 self.xpx/10.0+sb_px],
                [self.ypx/20.0,
                 self.ypx/20.0],
                lw=self.xpx/125.0, color='w')
        ax.text(self.xpx/10.0+sb_px/2.0,
                self.ypx/15.0,
                scale_text,
                va='top', ha='center', color='w')

class ThorHaussIO(HaussIO):
    """
    Object representing 2p imaging data acquired with ThorImageLS
    """
    def _get_filenames(self, xml_path):
        super(ThorHaussIO, self)._get_filenames(xml_path)
        if xml_path is None:
            self.xml_name = self.dirname + "/Experiment.xml"
        else:
            self.xml_name = xml_path

    def _get_dimensions(self):
        self.xsize, self.ysize = None, None
        self.xpx, self.ypx = None, None
        for child in self.xml_root:
            if child.tag=="LSM":
                self.xpx = int(child.attrib['pixelX'])
                self.ypx = int(child.attrib['pixelY'])
            if child.tag=="Sample":
                for grandchild in child:
                    if grandchild.tag=="Wells":
                        for ggrandchild in grandchild:
                            self.xsize = float(ggrandchild.attrib['subOffsetXMM'])*1e3
                            self.ysize = float(ggrandchild.attrib['subOffsetYMM'])*1e3

    def _get_timing(self):
        self.timing = np.loadtxt(self.dirname + "/timing.txt")

class PrairieHaussIO(HaussIO):
    """
    Object representing 2p imaging data acquired with Prairie scopes
    """

    def _get_filenames(self, xml_path):
        super(PrairieHaussIO, self)._get_filenames(xml_path)
        if not os.path.exists(self.filetrunk + "{0:04d}.tif".format(1)):
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            sys.stdout.write("Converting to individual tiffs... ")
            sys.stdout.flush()
            self.mptif = tifffile.TiffFile(self.dirname + ".tif")
            for nf,frame in enumerate(self.mptif.pages):
                tifffile.imsave(self.filetrunk + "{0:04d}.tif".format(nf+1),
                                frame.asarray())
            sys.stdout.write("done\n")
        if xml_path is None:
            self.xml_name = self.dirname + ".xml"
        else:
            self.xml_name = xml_path

    def _get_dimensions(self):
        self.xsize, self.ysize = None, None
        self.xpx, self.ypx = None, None
        for fi in self.xml_root.find("PVStateShard").findall("PVStateValue"):
            if fi.attrib['key'] == "linesPerFrame":
                self.ypx = int(fi.attrib['value'])
            elif fi.attrib['key'] == "pixelsPerLine":
                self.xpx = int(fi.attrib['value'])
            elif fi.attrib['key'] == "micronsPerPixel":
                for child in fi:
                    if child.attrib['index'] == 'XAxis':
                        self.xsize = float(child.attrib['value'])
                    if child.attrib['index'] == 'YAxis':
                        self.ysize = float(child.attrib['value'])

        self.xsize *= self.xpx
        self.ysize *= self.ypx

    def _get_timing(self):
        self.timing = np.array([float(fi.attrib['relativeTime'])
                                for fi in self.xml_root.find(
                                        "Sequence").findall("Frame")])

def sima_export_frames(dataset, path, filenames):
    """
    Export a sima.ImagingDataset to individual tiffs.
    Works around sima only producing multipage tiffs.

    Parameters
    ----------
    dataset : sima.ImagingDataset
        The sima.ImagingDataset to be exported
    path : Full path to target directory for exported tiffs
    filenames : Filenames to be used for the export. While these
        can be full paths, only the file name part will be used.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    output_filenames = [
        [[path + "/tmp" + os.path.basename(fn)]]
        for fn in filenames]
    dataset.export_frames(output_filenames, fill_gaps=True)

    mptif_fn = path + "/tmp" + os.path.basename(filenames[0])
    mptif = tifffile.TiffFile(mptif_fn)
    for nf,frame in enumerate(mptif.pages):
        tifffile.imsave(
            path + "/" + \
            os.path.basename(filenames[nf]),
            frame.asarray())

    os.unlink(mptif_fn)

        
