"""
Module for importing and exporting 2p imaging datasets

(c) 2015 C. Schmidt-Hieber
GPLv3
"""

from __future__ import absolute_import

import os
import sys
import abc
import glob
import time
try:
    import lzma
except ImportError:
    import backports.lzma as lzma
try:
    import subprocess32 as sp
except ImportError:
    sys.stdout.write("Couldn't find subprocess32; using subprocess instead\n")
    import subprocess as sp
import shlex
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import tables

import sima
import tifffile

try:
    from . import movies
except (SystemError, ValueError):
    import movies

THOR_RAW_FN = "Image_0001_0001.raw"
XZ_BIN = "/usr/local/bin/xz"


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
    width_idx : str
        Width of index string in file names
    maxtime : float
        Limit data to maxtime
    """
    def __init__(self, dirname, chan='A', xml_path=None, sync_path=None,
                 width_idx=4, maxtime=None):

        self.raw_array = None
        self.maxtime = maxtime

        self.dirname = os.path.abspath(dirname)
        self.chan = chan
        self.width_idx = width_idx

        self._get_filenames(xml_path, sync_path)
        if self.sync_path is None:
            self.sync_episodes = None
            self.sync_xml = None

        sys.stdout.write("Reading experiment settings... ")
        sys.stdout.flush()
        if self.xml_name is not None:
            self.xml_root = ET.parse(self.xml_name).getroot()
        self._get_dimensions()
        self._get_timing()
        self._get_sync()
        self.dt = np.mean(np.diff(self.timing))
        self.fps = 1.0/self.dt
        self.nframes = len(self.timing)
        sys.stdout.write("done\n")

        if self.maxtime is not None:
            self.iend = np.where(self.timing >= self.maxtime)[0][0]
            self.filenames = self.filenames[:self.iend]
        else:
            self.iend = None

        if xml_path is None:
            if self.rawfile is None or not os.path.exists(self.rawfile):
                try:
                    assert(len(self.filenames) <= self.nframes)
                except AssertionError as err:
                    print(len(self.filenames), self.nframes)
                    raise err
        else:
            if self.rawfile is None or not os.path.exists(self.rawfile):
                if len(self.filenames) != self.nframes:
                    self.nframes = len(self.filenames)

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

    @abc.abstractmethod
    def read_raw(self):
        return

    def raw2tiff(self, mp=False):
        arr = self.read_raw()
        if not mp:
            for ni, img in enumerate(arr):
                sys.stdout.write(
                    "\r{0:6.2f}%".format(float(ni)/arr.shape[0] * 100.0))
                sys.stdout.flush()
                tifffile.imsave(os.path.join(
                    self.dirname_comp,
                    self.basefile + self.format_index(ni+1)) + ".tif", img)
                sys.stdout.write("\n")
        else:
            tifffile.imsave(os.path.join(
                self.dirname_comp, self.basefile + "mp.tif"), arr)

    def tiff2raw(self, path=None, compress=True):
        if path is None:
            path_f = self.dirname_comp
        else:
            path_f = path
        rawfn = os.path.join(path_f, THOR_RAW_FN)
        assert(not os.path.exists(rawfn))
        compressfn = rawfn + ".xz"
        assert(not os.path.exists(compressfn))

        if not os.path.exists(rawfn):
            sys.stdout.write("Reading files...")
            sys.stdout.flush()
            t0 = time.time()
            arr = self.asarray_uint16()
            assert(len(arr.shape) == 3)
            sys.stdout.write(" done in {0:.2f}s\n".format(time.time()-t0))
            compress_np(arr, path_f, THOR_RAW_FN, compress=compress)

    def _get_filenames(self, xml_path, sync_path):
        self.dirname_comp = self.dirname.replace("?", "n")
        self.movie_fn = self.dirname_comp + ".mp4"
        self.scale_png = self.dirname_comp + "_scale.png"
        self.sima_dir = self.dirname_comp + ".sima"
        self.basefile = "Chan" + self.chan + "_0001_0001_0001_"
        self.rawfile = None
        self.filetrunk = os.path.join(self.dirname, self.basefile)
        self.sync_path = sync_path
        if "?" in self.filetrunk:
            self.dirnames = sorted(glob.glob(self.dirname))
            self.ffmpeg_fn = "'" + self.filetrunk + self.format_index(
                "?") + ".tif'"
        else:
            self.dirnames = [self.dirname]
            self.ffmpeg_fn = self.filetrunk + self.format_index("%") + ".tif"

    def get_normframe(self):
        """
        Return a representative frame that will be used to normalize
        the brightness in movies

        Returns
        -------
        arr : numpy.ndarray
            Frame converted to numpy.ndarray
        """
        if not os.path.exists(self.rawfile):
            if "?" in self.dirname:
                normdir = self.dirnames[int(np.round(len(self.dirnames)/2.0))]
                normtrunk = self.filetrunk.replace(
                    self.dirname, normdir)
                nframes = len(
                    glob.glob(os.path.join(normdir, self.basefile + "*.tif")))
                normframe = normtrunk + self.format_index(int(nframes/2)) + ".tif"
            else:
                normframe = self.filetrunk + self.format_index(
                    int(len(self.filenames)/2)) + ".tif"
            sample = Image.open(normframe)
            arr = np.asarray(sample, dtype=np.float)
        else:
            arr = self.read_raw()[int(len(self.filenames)/2)]

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
        if os.path.exists(self.sima_dir):
            try:
                return sima.ImagingDataset.load(self.sima_dir)
            except EOFError as err:
                sys.stderr.write("Could not read from " + self.sima_dir +
                                 "regenerating sima files: " + err + "\n")
            except IndexError as err:
                sys.stderr.write("Could not read from " + self.sima_dir +
                                 "regenerating sima files: " + err + "\n")

        if self.rawfile is None or not os.path.exists(self.rawfile):
            # The string paths[i][j] is a unix style expression for the
            # filenames for plane i and channel j
            sequences = [sima.Sequence.create(
                'TIFFs', [[self.filetrunk + self.format_index("?") + ".tif"]])]
        else:
            sequences = [sima.Sequence.create(
                'ndarray', self.read_raw()[:, np.newaxis, :, :, np.newaxis])]

        if stopIdx is None:
            stopIdx = self.nframes
        else:
            sequences = [seq[:stopIdx, :, :, :, :] for seq in sequences]
        if startIdx != 0:
            sequences = [seq[startIdx:, :, :, :, :] for seq in sequences]
        return sima.ImagingDataset(
            sequences, self.sima_dir, channel_names=[self.chan, ])

    def asarray(self):
        return np.array(self.tosima().sequences[0]).squeeze()

    def asarray_uint16(self):
        arr = np.array([
            np.array(Image.open(fn, 'r'), dtype=np.uint16)
            for fn in self.filenames])
        try:
            assert(arr.dtype == np.uint16)
            assert(arr.shape[0] == self.nframes)
        except AssertionError as err:
            print(arr.dtype)
            print(arr.shape)
            print(self.nframes, self.xpx, self.ypx)
            raise err

        return arr

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
            if self.rawfile is not None and os.path.exists(self.rawfile):
                normbright = movies.get_normbright(
                    self.read_raw(), mid=norm)
            else:
                normbright = movies.get_normbright(
                    self.get_normframe(), mid=norm)
        else:
            normbright = None

        if scalebar:
            self.save_scale_bar()
            scalebarframe = self.scale_png
        else:
            scalebarframe = None

        if os.path.exists(self.rawfile):
            movie_input = self.read_raw()
        else:
            movie_input = self.ffmpeg_fn

        return movies.make_movie(
            movie_input, self.movie_fn, self.fps, normbright, scalebarframe,
            crf=crf)

    def make_movie_extern(self, path_extern, norm=16.0, scalebar=True,
                          crf=28, width_idx=None):
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
        width_idx : int, optional
            Override default index string width. Default: None

        Returns
        -------
        html_movie : str
            An html tag containing the complete movie
        """
        rawfile = os.path.join(path_extern, THOR_RAW_FN)
        if not os.path.exists(rawfile):
            rawfile += ".xz"
            if not os.path.exists(rawfile):
                rawfile = None

        if rawfile is not None:
            shapefn = os.path.join(path_extern, THOR_RAW_FN[:-3] + "shape.npy")
            shape = np.load(shapefn)
            movie_input = raw2np(rawfile, (shape[0], shape[2], shape[3]))
        else:
            movie_input = os.path.join(
                path_extern, self.basefile + self.format_index(
                    "%", width_idx=width_idx) + ".tif"),

        if norm:
            if rawfile is not None:
                normbright = movies.get_normbright(movie_input)
            else:
                normbright = movies.get_normbright(np.asarray(Image.open(
                    os.path.join(
                        path_extern, self.basefile + self.format_index(
                            int(self.nframes/2), width_idx=width_idx) +
                        ".tif"))))
        else:
            normbright = None

        if scalebar:
            self.save_scale_bar()
            scalebarframe = self.scale_png
        else:
            scalebarframe = None

        return movies.make_movie(
            movie_input,
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

    def format_index(self, n, width_idx=None):
        """
        Return formatted index string

        Parameters
        ----------
        n : int or str
            Index as int, or "?" (returns series of "?"), or "%" (returns
            old-school formatter)
        width_idx : int, optional
            Override default width of index string. Default: None

        Returns
        -------
        format : string
            Formatted index string, or series of "?", or old-school formatter
        """
        if width_idx is None:
            width_idx = self.width_idx

        if isinstance(n, str):
            if n == "?":
                ret = "?"
                for nq in range(width_idx-1):
                    ret += "?"
                return ret
            elif n == "%":
                return "%0{0:01d}d".format(width_idx)
        else:
            return "{0:0{width}d}".format(n, width=width_idx)


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
        if "?" in self.xml_name:
            self.xml_name = sorted(glob.glob(self.xml_name))[0]
        if "?" in self.dirname:
            self.filenames = []
            for dirname in self.dirnames:
                filenames_orig = sorted(glob.glob(os.path.join(
                    dirname, self.basefile + "*.tif")))
                nf = len(self.filenames)
                self.filenames += [os.path.join(
                    self.dirname_comp, self.basefile +
                    self.format_index(nf+nfno) + ".tif")
                    for nfno, fno in enumerate(filenames_orig)]
        else:
            self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))

        self.rawfile = os.path.join(self.dirname_comp, THOR_RAW_FN)
        if os.path.exists(self.rawfile + ".xz"):
            self.rawfile = self.rawfile + ".xz"

    def _get_dimensions(self):
        self.xsize, self.ysize = None, None
        self.xpx, self.ypx = None, None
        for child in self.xml_root:
            if child.tag == "LSM":
                self.xpx = int(child.attrib['pixelX'])
                self.ypx = int(child.attrib['pixelY'])
                if int(child.attrib['averageMode']) == 1:
                    self.naverage = int(child.attrib['averageNum'])
                else:
                    self.naverage = None
            elif child.tag == "Sample":
                for grandchild in child:
                    if grandchild.tag == "Wells":
                        for ggrandchild in grandchild:
                            self.xsize = float(
                                ggrandchild.attrib['subOffsetXMM'])*1e3
                            self.ysize = float(
                                ggrandchild.attrib['subOffsetYMM'])*1e3

    def _get_timing(self):
        if "?" in self.dirname:
            dirname_wildcard = self.dirname[
                :len(os.path.dirname(self.xml_name))] + "/timing.txt"
            timings = sorted(glob.glob(dirname_wildcard))
            self.timing = np.loadtxt(timings[0])
            for timing in timings[1:]:
                self.timing = np.concatenate([
                    self.timing, np.loadtxt(timing)+self.timing[-1]])
        else:
            self.timing = np.loadtxt(
                os.path.dirname(self.xml_name) + "/timing.txt")

    def _get_sync(self):
        if self.sync_path is None:
            return

        self.sync_paths = sorted(glob.glob(self.sync_path))
        self.sync_episodes = [sorted(glob.glob(sync_path + "/Episode*.h5"))
                              for sync_path in self.sync_paths]
        self.sync_xml = [sync_path + "/ThorRealTimeDataSettings.xml"
                         for sync_path in self.sync_paths]

    def _find_dt(self, name, nsync=0):
        self.sync_root = ET.parse(self.sync_xml[nsync]).getroot()
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
        for epi_files in self.sync_episodes:
            for episode in epi_files:
                sync_data.append({})
                sync_dt.append({})
                print(episode)
                h5 = tables.open_file(episode)
                for el in h5.root.DI:
                    sync_data[-1][el.name] = np.squeeze(el)
                    sync_dt[-1][el.name] = self._find_dt(
                        el.name, len(sync_dt)-1)
                h5.close()

        return sync_data, sync_dt

    def read_raw(self):
        if self.raw_array is None:
            self.raw_array = raw2np(
                self.rawfile, (self.nframes, self.xpx, self.ypx))[:self.iend]

        return self.raw_array


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

        if "?" in self.filetrunk:
            self.ffmpeg_fn = "'" + self.filetrunk + self.format_index(
                "?") + ".tif'"
        else:
            self.ffmpeg_fn = self.filetrunk + self.format_index("%") + ".tif"

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
        self.naverage = None

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

    def read_raw(self):
        raise NotImplementedError(
            "Raw file reading not implemented for Prairie files yet")


class MovieHaussIO(HaussIO):
    """
    Object representing 2p imaging data for which only a movie is available
    """
    def __init__(self, dirname, dx, dt, chan='A', xml_path=None,
                 sync_path=None, width_idx=4):
        self.dx = dx
        self.dt = dt
        print(dirname + ".mp4")
        self.movie = movies.numpy_movie(dirname + ".mp4")
        super(MovieHaussIO, self).__init__(
            dirname, chan, xml_path, sync_path, width_idx)

    def _get_dimensions(self):
        self.xpx = self.movie.shape[2]
        self.ypx = self.movie.shape[1]
        self.xsize = self.xpx*self.dx
        self.ysize = self.ypx*self.dx
        self.naverage = None

    def _get_filenames(self, xml_path, sync_path):
        self.dirname_comp = self.dirname.replace("?", "n")
        self.movie_fn = self.dirname_comp + ".mp4"
        self.scale_png = self.dirname_comp + "_scale.png"
        self.sima_dir = self.dirname_comp + ".sima"
        self.basefile = "Chan" + self.chan + "_0001_0001_0001_"
        self.rawfile = self.movie_fn
        self.filetrunk = os.path.join(self.dirname, self.basefile)
        self.sync_path = sync_path
        if "?" in self.filetrunk:
            self.dirnames = sorted(glob.glob(self.dirname))
            self.ffmpeg_fn = "'" + self.filetrunk + self.format_index(
                "?") + ".tif'"
        else:
            self.dirnames = [self.dirname]
            self.ffmpeg_fn = self.filetrunk + self.format_index("%") + ".tif"

        self.xml_name = None
        self.filenames = sorted(glob.glob(self.filetrunk + "*.tif"))

    def _get_timing(self):
        self.timing = np.arange(self.movie.shape[0]) * self.dt

    def _get_sync(self):
        if self.sync_path is None:
            return

    def read_sync(self):
        raise NotImplementedError(
            "Synchronization readout not implemented for Prairie files yet")

    def read_raw(self):
        return self.movie


def sima_export_frames(dataset, path, filenames, startIdx=0, stopIdx=None,
                       ftype="tiff"):
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
        Index of first frame to be exported (inclusive). Default: 0
    stopIdx : int, optional
        Index of last frame to be exported (exclusive). Default: None
    ftype : stf, optional
        file type, one of "tiff" or "raw". Default: "tiff"
    """
    try:
        assert(len(filenames) == dataset.sequences[0].shape[0])
    except AssertionError as err:
        print(len(filenames), dataset.sequences[0].shape[0])
        raise err
    assert(ftype in ["tiff", "raw"])

    if ftype == "raw":
        if startIdx != 0 or stopIdx is not None:
            raise RuntimeError(
                "Can only export complete dataset in raw format")

    if not os.path.exists(path):
        os.makedirs(path)

    if stopIdx is None or stopIdx > len(filenames):
        stopIdx = len(filenames)
    save_frames = sima.sequence._fill_gaps(
        iter(dataset.sequences[0]), iter(dataset.sequences[0]))
    if ftype == "tiff":
        for nf, frame in enumerate(save_frames):
            if nf >= startIdx and nf < stopIdx:
                tifffile.imsave(
                    os.path.join(path, os.path.basename(filenames[nf])),
                    np.array(frame[0]).squeeze().astype(
                        np.uint16))
    elif ftype == "raw":
        sys.stdout.write("Reading files...")
        sys.stdout.flush()
        t0 = time.time()
        arr = np.array(
            [frame for frame in save_frames]).squeeze().astype(
                np.uint16)
        sys.stdout.write(" done in {0:.2f}s\n".format(time.time()-t0))
        compress_np(
            arr, path, THOR_RAW_FN, dataset.sequences[0].shape,
            compress=True)


def compress_np(arr, path, rawfn, shape=None, compress=True):
    if shape is None:
        shape = arr.shape

    shapefn = os.path.join(path, THOR_RAW_FN[:-3] + "shape.npy")
    np.save(shapefn, shape)

    rawfn = os.path.join(path, rawfn)

    sys.stdout.write("Writing raw file...")
    sys.stdout.flush()
    t0 = time.time()
    arr.tofile(rawfn)
    sys.stdout.write(" done in {0:.2f}s\n".format(time.time()-t0))

    if compress:
        cmd = shlex.split(XZ_BIN + " -T 0")
        cmd.append(rawfn)
        sys.stdout.write("Compressing file...")
        sys.stdout.flush()
        t0 = time.time()
        P = sp.Popen(cmd)
        P.wait()
        sys.stdout.write(" done in {0:.2f}s\n".format(time.time()-t0))


def raw2np(filename, shape):
    if filename[-3:] == ".xz":
        sys.stdout.write("Decompressing data...\n")
        sys.stdout.flush()
        with lzma.open(filename) as decompf:
            return np.fromstring(
                decompf.read(), dtype=np.uint16).reshape(shape)
    else:
        return np.fromfile(filename, dtype=np.uint16).reshape(shape)
