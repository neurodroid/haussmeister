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
import tables

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
    def __init__(self, dirname, chan='A', xml_path=None, sync_path=None):

        self.dirname = os.path.abspath(dirname)
        self.chan = chan

        self._get_filenames(xml_path, sync_path)
        if self.sync_path is None:
            self.sync_episodes = None
            self.sync_xml = None

        sys.stdout.write("Reading experiment settings... ")
        sys.stdout.flush()
        self.xml_root = ET.parse(self.xml_name).getroot()
        self._get_dimensions()
        self._get_timing()
        self._get_sync()
        self.dt = np.mean(np.diff(self.timing))
        self.fps = 1.0/self.dt
        self.nframes = len(self.timing)
        sys.stdout.write("done\n")

        assert(len(self.filenames) <= self.nframes)

    @abc.abstractmethod
    def _get_dimensions(self):
        return

    @abc.abstractmethod
    def _get_timing(self):
        return

    @abc.abstractmethod
    def _get_sync(self):
        return

    @abc.abstractmethod
    def read_sync(self):
        return

    def _get_filenames(self, xml_path, sync_path):
        self.movie_fn = self.dirname + ".mp4"
        self.scale_png = self.dirname + "_scale.png"
        self.sima_dir = self.dirname + ".sima"
        self.basefile = "Chan" + self.chan + "_0001_0001_0001_"
        self.filetrunk = self.dirname + '/' + self.basefile
        self.ffmpeg_fn = self.filetrunk + "%04d.tif"
        self.sync_path = sync_path

    def get_normframe(self):
        """
        Return a representative frame that will be used to normalize
        the brightness in movies

        Returns
        -------
        arr : numpy.ndarray
            Frame converted to numpy.ndarray
        """
        normframe = self.filetrunk + "{0:04d}.tif".format(
            int(len(self.filenames)/2))
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
            sequences = [seq[:stopIdx, :, :, :, :] for seq in sequences]
        if startIdx != 0:
            sequences = [seq[startIdx:, :, :, :, :] for seq in sequences]
        return sima.ImagingDataset(
            sequences, self.sima_dir, channel_names=[self.chan, ])

    def make_movie(self, norm=16.0, scalebar=True, crf=28.0):
        """
        Produce a movie of the experiment

        Parameters
        ----------
        norm : float, optional
            Normalize min, norm*median and max brightness to 0, 0.5 and 1.0
            None for no normalization. Default: 16.0
        scalebar : bool, optional
            Show a scale bar in the movie.
        crf : int, optional
            crf value to be passed to ffmpeg. Default: 28

        Returns
        -------
        html_movie : str
            An html tag containing the complete movie
        """

        if norm is not None:
            normbright = movies.get_normbright(
                self.get_normframe(), mid=norm)
        else:
            normbright = None

        if scalebar:
            self.save_scale_bar()
            scalebarframe = self.scale_png
        else:
            scalebarframe = None

        return movies.make_movie(self.ffmpeg_fn, self.movie_fn, self.fps,
                                 normbright, scalebarframe, crf=crf)

    def make_movie_extern(self, path_extern, norm=16.0, scalebar=True,
                          crf=28):
        """
        Produce a movie from a directory with individual tiffs, using
        the present experimental settings for scale bar and frame rate

        Parameters
        ----------
        path_extern : str
            Full path to the directory that contains the individual tiffs
        norm : float, optional
            Normalize min, norm*median and max brightness to 0, 0.5 and 1.0
            None for no normalization. Default: 16.0
        scalebar : bool, optional
            Show a scale bar in the movie.
        crf : int, optional
            crf value to be passed to ffmpeg. Default: 28

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
            scale=(self.xpx, self.ypx),
            crf=crf)

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
        scale_length_int, scale_length_px = self.get_scale_bar()
        movies.save_scale_bar(self.scale_png, scale_length_int,
                              scale_length_px, self.xpx, self.ypx)

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
        scale_text = u"{0:0d} $\mu$m".format(int(sb_int))
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
    def _get_filenames(self, xml_path, sync_path):
        super(ThorHaussIO, self)._get_filenames(xml_path, sync_path)
        if xml_path is None:
            self.xml_name = self.dirname + "/Experiment.xml"
        else:
            self.xml_name = xml_path
        self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))

    def _get_dimensions(self):
        self.xsize, self.ysize = None, None
        self.xpx, self.ypx = None, None
        for child in self.xml_root:
            if child.tag == "LSM":
                self.xpx = int(child.attrib['pixelX'])
                self.ypx = int(child.attrib['pixelY'])
            if child.tag == "Sample":
                for grandchild in child:
                    if grandchild.tag == "Wells":
                        for ggrandchild in grandchild:
                            self.xsize = float(
                                ggrandchild.attrib['subOffsetXMM'])*1e3
                            self.ysize = float(
                                ggrandchild.attrib['subOffsetYMM'])*1e3

    def _get_timing(self):
        self.timing = np.loadtxt(
            os.path.dirname(self.xml_name) + "/timing.txt")

    def _get_sync(self):
        if self.sync_path is None:
            return

        self.sync_episodes = sorted(
            glob.glob(self.sync_path + "/Episode*.h5"))
        self.sync_xml = self.sync_path + "/ThorRealTimeDataSettings.xml"

    def _find_dt(self, name):
        self.sync_root = ET.parse(self.sync_xml).getroot()
        for child in self.sync_root:
            if child.tag == "DaqDevices":
                for cchild in child:
                    if cchild.tag == "AcquireBoard":
                        for ccchild in cchild:
                            if ccchild.tag == "DataChannel":
                                if ccchild.attrib['alias'] == name:
                                    board = cchild
        for cboard in board:
            if cboard.tag == "SampleRate":
                if cboard.attrib['enable'] == "1":
                    return 1.0/float(cboard.attrib['rate']) * 1e3

    def read_sync(self):
        if self.sync_path is None:
            return None

        sync_data = []
        sync_dt = []
        for episode in self.sync_episodes:
            sync_data.append({})
            sync_dt.append({})
            h5 = tables.open_file(episode)
            for el in h5.root.DI:
                sync_data[-1][el.name] = np.squeeze(el)
                sync_dt[-1][el.name] = self._find_dt(el.name)
            h5.close()

        return sync_data, sync_dt


class PrairieHaussIO(HaussIO):
    """
    Object representing 2p imaging data acquired with Prairie scopes
    """

    def _get_filenames(self, xml_path, sync_path):
        super(PrairieHaussIO, self)._get_filenames(xml_path, sync_path)
        if not os.path.exists(self.filetrunk + "{0:04d}.tif".format(1)):
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            sys.stdout.write("Converting to individual tiffs... ")
            sys.stdout.flush()
            self.mptif = tifffile.TiffFile(self.dirname + ".tif")
            for nf, frame in enumerate(self.mptif.pages):
                tifffile.imsave(self.filetrunk + "{0:04d}.tif".format(nf+1),
                                frame.asarray())
            sys.stdout.write("done\n")
        if xml_path is None:
            self.xml_name = self.dirname + ".xml"
        else:
            self.xml_name = xml_path
        # This needs to be done *after* the individual tiffs have been written
        self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))

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

    def _get_sync(self):
        if self.sync_path is None:
            return

    def read_sync(self):
        raise NotImplementedError(
            "Synchronization readout not implemented for Prairie files yet")


def sima_export_frames(dataset, path, filenames, startIdx=0, stopIdx=None):
    """
    Export a sima.ImagingDataset to individual tiffs.
    Works around sima only producing multipage tiffs.

    Parameters
    ----------
    dataset : sima.ImagingDataset
        The sima.ImagingDataset to be exported
    path : string
        Full path to target directory for exported tiffs
    filenames : list of strings
        Filenames to be used for the export. While these can be
        full paths, only the file name part will be used.
    startIdx : int, optional
        Index of first frame to be exported (inclusive)
    stopIdx : int, optional
        Index of last frame to be exported (exclusive)
    """
    if not os.path.exists(path):
        os.makedirs(path)
    output_filenames = [
        [[path + "/tmp" + os.path.basename(fn)]]
        for fn in filenames]
    dataset.export_frames(output_filenames, fill_gaps=True)

    mptif_fn = path + "/tmp" + os.path.basename(filenames[0])
    mptif = tifffile.TiffFile(mptif_fn)
    if stopIdx is None:
        stopIdx = len(mptif.pages)
    for nf, frame in enumerate(mptif.pages[startIdx:stopIdx]):
        tifffile.imsave(
            path + "/" + os.path.basename(filenames[nf]),
            frame.asarray())

    os.unlink(mptif_fn)
