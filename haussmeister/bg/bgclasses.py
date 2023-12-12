import sys
import os
sys.path.append("../vr/")
sys.path.append(os.path.expanduser("~/Trillian/code/NICE/"))
sys.path.append(os.path.expanduser("~/Trillian/code/py2p/"))

# sys.path.append(os.path.expanduser("~/../cs/py2p/tools"))
import numpy as np
import bottleneck as bn
from haussmeister import pipeline2p as p2p
import matplotlib.pyplot as plt
from math import ceil, floor
import maps
import laps
import training
import functools
import move
import syncfiles
import nice
try:
    from stfio import plot as stfio_plot
except ImportError:
    print("stfio not available")
from scipy.interpolate import interp1d
from itertools import combinations
from haussmeister import spectral as hspectral
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import pickle
from scipy.stats import ttest_ind as ttest
from scipy.stats import wilcoxon as wilcoxon
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from bgvisualize import labels_novel
import bgvisualize as visualize
from utils import chunk_shuffle_index
from copy import deepcopy
import multiprocessing
import time
from numpy.random import RandomState, shuffle, seed
import hashlib
import base64

from sklearn.linear_model import LinearRegression, Ridge, BayesianRidge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from caiman.source_extraction.cnmf.deconvolution import constrained_foopsi

DEBUG=False

def get_all_halves(length):
    arr = np.arange(length)
    return [
        (set1, tuple(arr[~np.in1d(range(length), set1)]))
        for set1 in combinations(
                arr, int(np.ceil((length/2.0))))][:int(np.ceil((length/2.0)))]

class Roi:
    def __init__(self, activations, dF_F, S, times, position, speed, running_mask, settings):
        # --- input data requirements
        assert len(dF_F) == len(times), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and times must have the same size'
        assert len(dF_F) == len(position), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and position must have the same size'
        assert len(dF_F) == len(speed), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and speed must have the same size'
        assert len(dF_F) == len(S), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and S must have the same size'

        # --- compulsory variable assignment
        self.dF_F = dF_F
        self.activations = activations
        self.S = S
        self.position = position
        self.speed = speed
        self.settings = settings
        self.times = times
        self.running_mask = running_mask

        # --- additional firing events computations
        self.position_of_activations = np.asarray([self.position[ev] for ev in self.activations])
        self.time_of_activations = np.asarray([self.times[ev] for ev in self.activations])
        self.n_activations = len(self.activations)
        self.activation_raster = self.discretize_activations()
        self.rate = self.n_activations/(np.sum(self.running_mask) * np.median(np.diff(self.times)) * 1e-3)
        self.F_ratemap = []

    # --- constructor analysis functions ---

    def discretize_activations(self):
        relative_time = self.times - self.times[0]
        T = int(ceil(relative_time[-1] / self.settings['discretization_timescale']))
        raster = np.zeros(T)

        if self.n_activations:
            for t in self.activations:
                raster[int(floor(relative_time[t] / self.settings['discretization_timescale']))] = 1.0

        return raster

    def compute_spatial_map_F(self, bins, occupancy, running_mask):
        if len(self.position_of_activations)==0:
            Frm = np.zeros(len(bins)-1)
        else:
        # events are already thresholded out by speed in the find_events() function at Session level
            dt = np.median(np.diff(self.times)) * 1e-3
            activations_map = np.histogram(self.position_of_activations, bins)[0] * dt
            activations_map[self.real_occupancy==0] = np.nan
            activations_map = maps.nanfilter1d(
                activations_map, sigma=self.settings["mapfilter"]/self.settings["binsize"],
                return_nan=self.settings["return_nan_filter"])
            Frm = maps.ratemap(activations_map, occupancy, min_occupancy=1e-6)

        Frm[self.real_occupancy==0] = np.nan
        self.F_ratemap = Frm

    def compute_spatial_map_S(self, bins, occupancy, running_mask, position, times):
        #implemented by Hsin-Lun, only tested on BG's data, not yet tested on Manu's pipeline!!!
        pass

class Lap:
    def __init__(self, index, laptype, dF_F, S, activations, times, position, speed, events, settings):

        # --- input data requirements
        assert len(dF_F[0]) == len(times), '----- ! ERROR ! ----- (roi constructor) ----- single roi F and times must have the same size'
        assert len(times) == len(position), '----- ! ERROR ! ----- (roi constructor) ----- times and position must have the same size'
        assert len(times) == len(speed), '----- ! ERROR ! ----- (roi constructor) ----- times and speed must have the same size'
        assert np.shape(dF_F) == np.shape(S), '----- ! ERROR ! ----- (roi constructor) ----- F and S must have the same shape'
        # --- compulsory variable assignment
        N, T = np.shape(dF_F)
        self.index = index
        self.laptype = laptype
        self.n_roi = N
        self.dF_F = dF_F
        self.S = S
        self.activations = activations
        self.times = times
        self.position = position
        self.speed = speed
        self.events = events
        self.settings = settings
        self.tot_running_time, self.running_mask = compute_running_mask(self)
        self.mean_speed = np.mean(self.speed[self.running_mask])
        self.raster = []
        self.rate_maps = []
        self.s_maps = []

        # --- single roi construction
        self.rois = []
        for i in range(N):
            newroi = Roi(activations[i], dF_F[i], S[i], times, position, speed, self.running_mask, settings)
            self.rois.append(newroi)

        # --- online computations
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.lick_stats = []
        self.compute_lick_stats()
        self.discretize_activations()
        self.compute_hits()
        self.reward = self.lick_stats['number_hits'] > 0

    def compute_spatial_maps(self, bins):
        dt = np.median(np.diff(self.times)) * 1e-3
        run_periods, rest_periods = move.run_rest_periods(hspectral.Timeseries(self.speed, dt), self.settings["thr_rest"],
                                                          self.settings["thr_run"], self.settings["run_min_dur"])

        running_mask = laps.in_period(np.arange(len(self.times)), run_periods)
        occupancy = np.histogram(self.position[running_mask], bins)[0] * dt
        self.occupancy = maps.nanfilter1d(
            occupancy, sigma=self.settings["mapfilter"]/self.settings["binsize"], return_nan=self.settings["return_nan_filter"]) # filtering occupancy vector
        for roi in self.rois:
            roi.compute_spatial_map_F(bins, occupancy=self.occupancy, running_mask=running_mask)
            #roi.compute_spatial_map_S(bins, occupancy=self.occupancy, running_mask=running_mask, position=self.position, times=self.times)

        self.rate_maps = np.asarray([roi.F_ratemap for roi in self.rois])
        #self.s_maps = np.asarray([roi.S_ratemap for roi in self.rois])

    def compute_lick_stats(self):
        licktimes = np.asarray([e.time for e in self.events if e.evtype=='lick'])
        reward_times = np.asarray([e.time for e in self.events if e.evtype=='reward'])

        reward_distances = training.reward_distance(trainingdata=[], licktimes=licktimes, reward_times=reward_times)
        stats = training.lick_stats(reward_times, reward_distances, licktimes, self.settings)
        self.lick_stats = stats
        lick_array = np.zeros(len(self.times))
        for t in licktimes:
            t_lick = np.argmin(np.abs(self.times*1e-3 - t))
            lick_array[t_lick] = 1
        self.lick_array = lick_array


    def compute_hits(self):
        # TODO: don't hard code the reward zone
        # rewarded_env = training.read_rewarded_env(filetrunk + "_settings.py")
        # reward_zones = training.read_reward_zones(filetrunk + "_settings.py")
        # https://github.com/neurodroid/py2p/blob/master/tools/syncfiles.py#L2139
        if self.laptype=='vertical':
            false_reward_zone = [0.3375, 0.675]
            true_reward_zone = [0.810, 1.1475]
        elif self.laptype=='oblique':
            false_reward_zone = [0.810, 1.1475]
            true_reward_zone = [0.3375, 0.675]

        frz_mask = (self.position < false_reward_zone[1]) & (self.position > false_reward_zone[0])
        trz_mask = (self.position < true_reward_zone[1]) & (self.position > true_reward_zone[0])
        no_rz_mask = (self.position < 0.3375) | (self.position > 1.1475) | ((self.position < 0.810) & (self.position > 0.675))

        stop_bins_in_frz = np.sum(frz_mask & (self.running_mask==0))
        stop_bins_in_trz = np.sum(trz_mask & (self.running_mask==0))
        stop_bins_in_norz = np.sum(no_rz_mask & (self.running_mask==0))

        licks_in_frz = np.sum(self.lick_array[frz_mask])
        licks_in_trz = np.sum(self.lick_array[trz_mask])
        licks_in_norz = np.sum(self.lick_array[no_rz_mask])

        mean_speed_in_frz = np.mean(self.speed[frz_mask])
        mean_speed_in_trz = np.mean(self.speed[trz_mask])
        mean_speed_in_norz = np.mean(self.speed[no_rz_mask])

        # saving false statistics in self
        if stop_bins_in_frz==0 and stop_bins_in_norz==0:
            self.false_stop_ratio = 1.0
        elif stop_bins_in_norz==0:
            self.false_stop_ratio = 9.9 ### thresholding to 2.0 when it only stops at the false reward zone
        else:
            self.false_stop_ratio = float(stop_bins_in_frz)/float(stop_bins_in_norz) * 1.555 # so that chance level is 1

        if licks_in_frz==0 and licks_in_norz==0:
            self.false_lick_ratio = 1.0
        elif licks_in_norz==0:
            self.false_lick_ratio = 9.9
        else:
            self.false_lick_ratio = float(licks_in_frz)/float(licks_in_norz) * 1.555 # so that chance level is 1

        self.false_speed_ratio = float(mean_speed_in_norz)/float(mean_speed_in_frz)

        # saving true statistics in self
        if stop_bins_in_trz==0 and stop_bins_in_norz==0:
            self.true_stop_ratio = 1.0
        elif stop_bins_in_norz==0:
            self.true_stop_ratio = 2.0 ### thresholding to 2.0 when it only stops at the false reward zone
        else:
            self.true_stop_ratio = float(stop_bins_in_trz)/float(stop_bins_in_norz) * 1.555 # so that chance level is 1

        if licks_in_trz ==0 and licks_in_norz == 0:
            self.true_lick_ratio = 1.0
        elif licks_in_norz==0:
            self.true_lick_ratio = 9.9
        else:
            self.true_lick_ratio = float(licks_in_trz)/float(licks_in_norz) * 1.555 # so that chance level is 1

        self.true_speed_ratio = float(mean_speed_in_norz)/float(mean_speed_in_trz)


    def discretize_activations(self):
        self.raster = np.transpose(np.vstack([roi.discretize_activations() for roi in self.rois])) # first dimension is time, second is rois

class Session(object):
    def __init__(self, exp, settings):
        # --- loading data
        self.settings = settings
        vrdict, calciumdict = syncfiles.load_2p_data(exp, settings)

        self.vrdict = vrdict
        self.events = vrdict['evlist']

        # --- synchronizing times
        Tmax = min([vrdict['framet2p'][-1], vrdict['frametvr'][-1]])
        Tmin = max([vrdict['framet2p'][0], vrdict['frametvr'][0]])
        keep_2p = (vrdict['framet2p'] < Tmax) & (vrdict['framet2p'] > Tmin)
        keep_vr = (vrdict['frametvr'] < Tmax) & (vrdict['frametvr'] > Tmin)

        # --- assigning variables
        self.session_number = exp.session_number
        data_hio = exp.to_haussio(mc=True)
        self.session_name = os.path.basename(data_hio.dirname_comp)
        self.times = vrdict['framet2p'][keep_2p]
        self.speed = vrdict['speed2p_lo'][keep_2p]
        pos_func = interp1d(vrdict['frametvr'], vrdict['posy'])
        self.position = pos_func(self.times)

        self.dF_F = self.process_dF_F(calciumdict, vrdict, keep_2p)
        self.n_roi, T = np.shape(self.dF_F)
        self.name_roi = np.arange(self.n_roi)
        self.S = calciumdict['S'][:, keep_2p]
        self.tot_running_time, self.running_mask = compute_running_mask(self)
        self.mean_speed = np.mean(self.speed[self.running_mask])

        # --- finding events on a session scale
        self.activations = self.find_activations()

        # --- eliminating silent and bad rois
        self.kept_cells = self.eliminate_silent_rois(exp.rois_eliminate)
        self.spatial_cells = []
        self.spatial_cells_F = [] #add by Hsin-Lun
        self.spatial_cells_N = [] #add by Hsin-Lun
        self.C_FF = []; self.C_NN = []; self.C_FN = []

        # --- building laps
        self.laps = []
        self.nlaps = 0
        self.incompletelaps = [] #add by Hsin-Lun
        self.divide_in_laps(vrdict)

        self.familiar_laps = [l for l in self.laps if l.laptype=='vertical']
        self.novel_laps = [l for l in self.laps if l.laptype=='oblique']

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            newroi = Roi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
            self.rois.append(newroi)

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.reward_rate = np.sum([l.reward for l in self.laps])/float(self.nlaps)
        self.reward_rate_N = np.sum([l.reward for l in self.novel_laps])/float(len(self.novel_laps))
        self.reward_rate_F = np.sum([l.reward for l in self.familiar_laps])/float(len(self.familiar_laps))

        self.compute_lick_stats()
        self.compute_spatial_maps()
        self.compute_spatial_zscores(VR=True)
        self.behavior_F, self.behavior_N, self.behavior_F_false, self.behavior_N_false = compute_reward_stats(exp, self.settings)

        self.false_stop_F = np.asarray([l.false_stop_ratio for l in self.familiar_laps])
        self.false_stop_N = np.asarray([l.false_stop_ratio for l in self.novel_laps])
        self.false_lick_F = np.asarray([l.false_lick_ratio for l in self.familiar_laps])
        self.false_lick_N = np.asarray([l.false_lick_ratio for l in self.novel_laps])
        self.false_speed_ratio_F = np.asarray([l.false_speed_ratio for l in self.familiar_laps])
        self.false_speed_ratio_N = np.asarray([l.false_speed_ratio for l in self.novel_laps])

        self.true_stop_F = np.asarray([l.true_stop_ratio for l in self.familiar_laps])
        self.true_stop_N = np.asarray([l.true_stop_ratio for l in self.novel_laps])
        self.true_lick_F = np.asarray([l.true_lick_ratio for l in self.familiar_laps])
        self.true_lick_N = np.asarray([l.true_lick_ratio for l in self.novel_laps])
        self.true_speed_ratio_F = np.asarray([l.true_speed_ratio for l in self.familiar_laps])
        self.true_speed_ratio_N = np.asarray([l.true_speed_ratio for l in self.novel_laps])

        # / --- end of constructor --- / #

    # --- segmentation functions --- #

    def divide_in_laps(self, vrdict) :

        gratings, teleport_times = laps.split_laps_2p(self.times * 1e-3, vrdict['evlist'], allow_extra_events=True)
        if DEBUG:
            print(gratings)
        teleport_times *= 1e3
        boundary_times = teleport_times[1:]
        gratings = gratings[1:] # discarding the first and the last lap (uncomplete)

        self.nlaps = len(boundary_times) - 1
        fam_index = 0
        nov_index = 0
        for i in range(self.nlaps):
            time_mask = (self.times > boundary_times[i] + self.settings["lap_split_offset"]) & (self.times < boundary_times[i+1] - self.settings["lap_split_offset"])
            lap_events = [e for e in vrdict['evlist'] if boundary_times[i] + self.settings["lap_split_offset"] < (e.time * 1000.) < boundary_times[i+1] - self.settings["lap_split_offset"]]
            lap_activations = self.find_lap_activations(time_mask)
            laptype = gratings[i]

            if laptype=='vertical':
                index = fam_index
                fam_index += 1
            if laptype=='oblique':
                index = nov_index
                nov_index += 1

            new_lap = Lap(
                index=index,
                laptype=laptype,
                dF_F=self.dF_F[:, time_mask],
                S=self.S[:, time_mask],
                activations=lap_activations,
                times=self.times[time_mask],
                position=self.position[time_mask],
                speed=self.speed[time_mask],
                events=lap_events,
                settings=self.settings)

            self.laps.append(new_lap)

    # --- activations functions --- #

    def process_dF_F(self, calciumdict, vrdict, keep_2p):
        dF_F = calciumdict['Fraw'][:, keep_2p]-self.settings['Fneu_factor']*calciumdict['Fneu'][:, keep_2p]
        n_roi, T = np.shape(dF_F)
        dt = np.median(np.diff(vrdict['framet2p']))*1e-3
        for i in range(n_roi):
            dF_F[i] = hspectral.lowpass(hspectral.highpass(hspectral.Timeseries(dF_F[i].astype(np.float), dt), 0.002, verbose=False), self.settings["F_filter"], verbose=False).data
            dF_F[i] -= np.mean(dF_F[i])
        return dF_F

    def find_activations(self):
        all_activations = []
        for i in range(self.n_roi):
            events, amplitudes = p2p.find_events(self.dF_F[i], self.speed, self.settings['min_speed'], self.settings['event_std_threshold'])
            events = np.asarray(events)
            amplitudes = np.asarray(amplitudes)
            events = events[amplitudes > self.settings['F_theta']]
            all_activations.append(events)

        return all_activations

    def find_lap_activations(self, time_mask):
        t0 = np.nonzero(time_mask)[0][0]  # first non-zero index of time mask = start of the lap
        lap_activations = []
        for n in range(self.n_roi):
            a = []
            for t in self.activations[n]:
                if time_mask[t]:
                    a.append(t - t0)
            lap_activations.append(a)

        return lap_activations

    def eliminate_silent_rois(self, rois_eliminate):
        kept_cells = np.zeros(self.n_roi, dtype=bool)
        for n in range(self.n_roi):
            if rois_eliminate:
                kept_cells[n] = (len(self.activations[n])>0) and (n not in rois_eliminate)
            else:
                kept_cells[n] = (len(self.activations[n])>0)

        self.dF_F = self.dF_F[kept_cells] # eliminating rows from arrays
        self.S = self.S[kept_cells]
        self.S_noisy = self.S_noisy[kept_cells]
        self.activations = [self.activations[i] for i in range(len(self.activations)) if kept_cells[i]]
        self.n_roi = int(np.sum(kept_cells))
        self.name_roi = self.name_roi[kept_cells]

        return kept_cells

    # --- behavioral analysis --- #

    def compute_lick_stats(self):
        licktimes = np.asarray([e.time for e in self.events if e.evtype=='lick'])
        reward_times = np.asarray([e.time for e in self.events if e.evtype=='reward'])

        reward_distances = training.reward_distance(trainingdata=[], licktimes=licktimes, reward_times=reward_times)
        stats = training.lick_stats(reward_times, reward_distances, licktimes, self.settings)
        self.lick_stats = stats
        lick_array = np.zeros(len(self.times))
        for t in licktimes:
            t_lick = np.argmin(np.abs(self.times*1e-3 - t))
            lick_array[t_lick] = 1
        self.lick_array = lick_array

    # --- spatial maps functions --- #

    def compute_spatial_maps(self):
        print("---- O> Computing spatial maps"); t0 = time.time()
        maxs = [np.max(l.position) for l in self.laps]
        mins = [np.min(l.position) for l in self.laps]
        spatial_bins = self.settings["spatial_bins"]
        bins = np.linspace(np.max(mins), np.min(maxs), spatial_bins + 1)
        self.bins = bins
        for l in self.laps:
            l.compute_spatial_maps(bins)
        print("---- <O Computed spatial maps, time elapsed = %.1f s" % (time.time() - t0))

    def compute_spatial_zscores(self, VR=True):
        if len(self.spatial_cells)==0:
            C_FF, C_NN, C_FN, C_FF_B, C_NN_B, C_FN_B = compute_spatial_bootstrap(
                self, chunk_size = self.settings["bootstrap_chunk_size"], n_bootstrap=self.settings["nbootstrap"], VR=VR)
            self.C_FF = C_FF
            self.C_NN = C_NN
            self.C_FN = C_FN
            if VR:
                mode = 'cells'
            else:
                mode = 'cells'
            self.spatial_Z_FF = bootstrap_zscores(C_FF, C_FF_B, mode=mode)
            self.spatial_Z_NN = bootstrap_zscores(C_NN, C_NN_B, mode=mode)
            self.spatial_Z_FN = bootstrap_zscores(C_FN, C_FN_B, mode=mode)
            self.spatial_cells = (self.spatial_Z_FF > self.settings['bootstrap_std_threshold']) | (self.spatial_Z_NN > self.settings['bootstrap_std_threshold'])
            self.spatial_cells_F = self.spatial_Z_FF > self.settings['bootstrap_std_threshold'] #add by Hsin-Lun
            self.spatial_cells_N = self.spatial_Z_NN > self.settings['bootstrap_std_threshold'] #add by Hsin-Lun

    def compute_spatial_correlations(self, force = False, pairwise=True):
        print("DEBUG HALVES")
        if len(self.C_FF) == 0 or force:
            print("DEBUG_HALVES")
            print("---- O> Computing spatial correlations"); t0 = time.time()
            if pairwise:
                C_FF = [[] for i in range(self.n_roi)]
                C_NN = [[] for i in range(self.n_roi)]
                C_FN = [[] for i in range(self.n_roi)]

                for n in range(self.n_roi):
                    for i in range(len(self.familiar_laps)):
                        for j in range(i+1, len(self.familiar_laps)):
                            c = ratemap_correlation(self.familiar_laps[i].rois[n].F_ratemap, self.familiar_laps[j].rois[n].F_ratemap, self.settings)
                            C_FF[n].append(c)

                    for i in range(len(self.novel_laps)):
                        for j in range(i+1, len(self.novel_laps)):
                            c = ratemap_correlation(self.novel_laps[i].rois[n].F_ratemap, self.novel_laps[j].rois[n].F_ratemap, self.settings)
                            C_NN[n].append(c)


                    for i in range(len(self.familiar_laps)):
                        for j in range(len(self.novel_laps)):
                            c = ratemap_correlation(self.familiar_laps[i].rois[n].F_ratemap, self.novel_laps[j].rois[n].F_ratemap, self.settings)
                            C_FN[n].append(c)
                self.C_FF = np.array(C_FF)
                self.C_NN = np.array(C_NN)
                self.C_FN = np.array(C_FN)
            else:
                print("DEBUG HALVES_1")
                ihalves = get_all_halves(len(self.familiar_laps))
                ifirst = [ihalf[0] for ihalf in ihalves]
                isecond = [ihalf[1] for ihalf in ihalves]
                # even_familiar_maps = bn.nanmean([
                #     self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                even_familiar_maps = bn.nanmean([
                    bn.nanmean([
                        self.familiar_laps[i].rate_maps for i in ifirstset], axis=0)
                for ifirstset in ifirst], axis=0)
                # odd_familiar_maps = bn.nanmean([
                #     self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                odd_familiar_maps = bn.nanmean([
                    bn.nanmean([
                        self.familiar_laps[i].rate_maps for i in isecondset], axis=0)
                    for isecondset in isecond], axis=0)

                # even_novel_maps = bn.nanmean([
                #     self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                even_novel_maps = bn.nanmean([
                    bn.nanmean([
                        self.novel_laps[i].rate_maps for i in ifirstset], axis=0)
                    for ifirstset in ifirst], axis=0)
                # odd_novel_maps = bn.nanmean([
                #     self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                odd_novel_maps = bn.nanmean([
                    bn.nanmean([
                        self.novel_laps[i].rate_maps for i in isecondset], axis=0)
                    for isecondset in isecond], axis=0)

                familiar_maps = bn.nanmean([l.rate_maps for l in self.familiar_laps], axis=0)
                novel_maps = bn.nanmean([l.rate_maps for l in self.novel_laps], axis=0)

                self.C_FF = np.array([[pov,] for pov in visualize.pov(even_familiar_maps, odd_familiar_maps, percell=True, VR=False, nmin=2)])
                self.C_NN = np.array([[pov,] for pov in visualize.pov(even_novel_maps, odd_novel_maps, percell=True, VR=False, nmin=2)])
                self.C_FN = np.array([[pov,] for pov in visualize.pov(familiar_maps, novel_maps, percell=True, VR=False, nmin=2)])

            print("---- <O Computed spatial correlations, time elapsed = %.1f s" % (time.time() - t0))

        return self.C_FF, self.C_NN, self.C_FN

    # --- bootstrap randomizations --- #

    def shuffle_F(self, chunk_size=100, randomstate = 'none'):
        # chunk shuffle self.dF_F
        T = len(self.dF_F[0])
        shuffled_index = chunk_shuffle_index(T, chunk_size, randomstate = randomstate)
        self.dF_F = self.dF_F[:, shuffled_index]

        # --- re-finding events on a session scale
        self.activations = self.find_activations()

        # --- re-building laps
        self.laps = []
        self.nlaps = 0
        self.divide_in_laps(self.vrdict)

        self.familiar_laps = [l for l in self.laps if l.laptype=='vertical']
        self.novel_laps = [l for l in self.laps if l.laptype=='oblique']

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            newroi = Roi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
            self.rois.append(newroi)

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.compute_spatial_maps()


### GENERAL CLASS FUNCTIONS ###

def compute_running_mask(object, VR=True):
    if VR:
        dt = np.median(np.diff(object.times)) * 1e-3
    else:
        dt = np.median(np.diff(object.times))
    run_periods, rest_periods = move.run_rest_periods(hspectral.Timeseries(object.speed, dt), object.settings["thr_rest"],
                                                      object.settings["thr_run"], object.settings["run_min_dur"])
    running_mask = laps.in_period(np.arange(len(object.times)), run_periods)
    tot_running_time = np.sum(running_mask) * dt
    return tot_running_time, running_mask

def compute_rate_vector_laps(laps):
    rate = np.zeros(laps[0].n_roi)
    running_time = 0

    for l in laps:
        rate += l.rate_vector * l.tot_running_time
        running_time += l.tot_running_time

    rate /= running_time
    return rate

def ratemap_correlation(rate1, rate2, settings, periodic=True):
    ibins_peaksearch = int(len(rate1)*settings["ccpeak_fraction"])
    mask = (np.isnan(rate1) == 0) & (np.isnan(rate2) == 0)
    map1 = rate1[mask]
    map2 = rate2[mask]
    if np.sum(map1) == 0 and np.sum(map2) != 0:
        return 0
    elif np.sum(map1) != 0 and np.sum(map2) == 0:
        return 0
    elif np.sum(map1) == 0 and np.sum(map2) == 0:
        return np.nan
    else:
        if periodic:
            cc = maps.periodic_corr(map1, map2)
            peak_cc_at_center = np.max((np.max(cc[:ibins_peaksearch]), np.max(cc[-ibins_peaksearch:])))
            return peak_cc_at_center
        else:
            #cc = hspectral.xcorr(map1-map1.mean(), map2-map2.mean(), normed=True)
            #peak_cc_at_center = np.max(cc[int(np.round(cc.shape[0]/2.0))-ibins_peaksearch:int(np.round(cc.shape[0]/2.0))+ibins_peaksearch])
            #peak_cc_at_center = pearsonr(map1, map2)[0]

            #return peak_cc_at_center
            if len(map1) > 1 and len(map2) > 1:
                return pearsonr(map1, map2) [0]
            else:
                return np.nan

def compute_reward_stats(exp, settings):
    filetrunk = exp.vr_path + '.'
    mousecal = syncfiles.read_mousecal(filetrunk + "    ")
    training_results, training_results_false = training.process_session(filetrunk, mousecal, settings, conditions=['vertical', 'oblique'], teleport_distance=(-0.5, 1.0), hashcode='', results={}, resultsfn=None, force=False, plot=False, lickthreshold=None)

    for key, result in list(training_results.items()):
        if 'oblique' in key:
            behavioral_oblique = result
        elif 'vertical' in key:
            behavioral_vertical = result
        else:
            print('ERROR')

    for key, result in list(training_results_false.items()):
        if 'oblique' in key:
            behavioral_oblique_false = result
        elif 'vertical' in key:
            behavioral_vertical_false = result
        else:
            print('ERROR')

    return behavioral_vertical, behavioral_oblique, behavioral_vertical_false, behavioral_oblique_false


### BOOTSTRAP COMPARISON functions

class shuffler(object): # necessary for parallelization of bootstrap
    def __init__(self, session, chunk_size, maptype='F', VR=True, shift=True):
        self.session = deepcopy(session)
        self.chunk_size = chunk_size
        self.maptype = maptype
        self.VR = VR
        self.shift = shift

    def __call__(self, x):
        randomstate = RandomState(x)
        if self.shift:
            self.session.shuffle_F(self.chunk_size, randomstate = randomstate)
        else:
            self.session.permutate_labels(self.chunk_size, randomstate = randomstate)
        if self.VR:
            return self.session.compute_spatial_correlations(force=True, pairwise=self.VR)
        else:
            return self.session.compute_spatial_correlations(force=True,maptype=self.maptype, pairwise=self.VR)

def compute_spatial_bootstrap(session, chunk_size, n_bootstrap = 10, maptype='F', VR=True):
    t0 = time.time()
    if VR:
        C_FF, C_NN, C_FN = session.compute_spatial_correlations(pairwise=VR)
    else:
        C_FF, C_NN, C_FN, \
        selectivity, selectivity_discrete, selectivity_continuous, selectivity_manu, selectivity_manu_div, selectivity_manu_nodiv = session.compute_spatial_correlations(maptype=maptype, pairwise=VR)
    print("( '-) time for one spatial maps computation: %.2f s" % (time.time() - t0))

    #move.printA('C_FF',C_FF)
    #move.printA('C_NN',C_NN)
    #move.printA('C_FN',C_FN)

    C_FF_B = np.zeros((n_bootstrap, C_FF.shape[0], C_FF.shape[1]))
    C_NN_B = np.zeros((n_bootstrap, C_NN.shape[0], C_NN.shape[1]))
    C_FN_B = np.zeros((n_bootstrap, C_FN.shape[0], C_FN.shape[1]))
    if not VR:
        selectivity_B = np.zeros((n_bootstrap, len(selectivity)))
        selectivity_discrete_B = np.zeros((n_bootstrap, len(selectivity_discrete)))
        selectivity_continuous_B = np.zeros((n_bootstrap, len(selectivity_continuous)))
        selectivity_manu_B = np.zeros((n_bootstrap, len(selectivity_manu)))
        selectivity_manu_div_B = np.zeros((n_bootstrap, len(selectivity_manu)))
        selectivity_manu_nodiv_B = np.zeros((n_bootstrap, len(selectivity_manu)))

    t0 = time.time()
    pool = multiprocessing.Pool(20)
    if VR:
        C_FF_B, C_NN_B, C_FN_B = list(zip(*pool.map(shuffler(session, chunk_size, maptype, VR, VR), list(range(n_bootstrap)))))
    else:
        C_FF_B, C_NN_B, C_FN_B, selectivity_B, selectivity_discrete_B, selectivity_continuous_B, selectivity_manu_B, selectivity_manu_div_B, selectivity_manu_nodiv_B = list(zip(*pool.map(shuffler(session, chunk_size, maptype, VR, VR), list(range(n_bootstrap)))))
        session.selectivity_B = selectivity_B
        session.selectivity_discrete_B = selectivity_discrete_B
        session.selectivity_continuous_B = selectivity_continuous_B
        session.selectivity_manu_B = selectivity_manu_B
        session.selectivity_manu_div_B = selectivity_manu_div_B
        session.selectivity_manu_nodiv_B = selectivity_manu_nodiv_B
    print("( '-) time for %u spatial maps computations: %.2f s" % (n_bootstrap, time.time() - t0))
    pool.close()

    return C_FF, C_NN, C_FN, np.array(C_FF_B), np.array(C_NN_B), np.array(C_FN_B)

def bootstrap_zscores(C, C_B, mode='cells'):
    if mode == 'cells':
        mean_corrs = bn.nanmean(C, 1)
        mean_corrs_bs = bn.nanmean(bn.nanmean(C_B, 2), 0)
        std_corrs_bs = np.nanstd(bn.nanmean(C_B, 2), 0)
        zscores = (mean_corrs - mean_corrs_bs)/std_corrs_bs

    if mode == 'laps':
        mean_corrs = bn.nanmean(C, 0)
        mean_corrs_bs = bn.nanmean(bn.nanmean(C_B, 1), 0)
        std_corrs_bs = np.nanstd(bn.nanmean(C_B, 1), 0)
        zscores = (mean_corrs - mean_corrs_bs)/std_corrs_bs

    if mode == 'all':
        zscores = (C - bn.nanmean(C_B, 0))/np.nanstd(C_B, 0)

    zscores[np.isinf(zscores)] = np.nan
    zscores[np.abs(zscores) > 20] = np.nan # for numerical errors
    return zscores
"""
class shuffler_selectivity(object): # necessary for parallelization of bootstrap
    def __init__(self, session, chunk_size, maptype='F', VR=True):
        self.session = deepcopy(session)
        self.chunk_size = chunk_size
        self.maptype = maptype
        self.VR = VR
    def __call__(self, x):
        randomstate = RandomState(x)
        self.session.shuffle_F(self.chunk_size, randomstate = randomstate)
        result = self.session.compute_selectivity()
        return result[2]

def compute_bootstrap_selectivity(session, chunk_size, n_bootstrap = 10):
    print("start compute bootstrap selectivity ("+str(n_bootstrap)+")")
    t0 = time.time()
    pool = multiprocessing.Pool(20)
    selectivity_B = list(zip(*pool.map(shuffler_selectivity(session, chunk_size, maptype='S', VR=False), list(range(n_bootstrap)))))
    print("( '-) time for %u bootstrap selectivity computation: %.2f s" % (n_bootstrap, time.time() - t0))
    pool.close()
    return selectivity_B
"""

### OTHER FUNCTIONS ###

def get_session_names(exp):
    # data_hio = exp.to_haussio(mc=True)
    savepath = exp.data_path.split('/')
    savepath = "%s_%s" % (savepath[-2], savepath[-1])
    topo_name = savepath[:6]
    dirname_comp = os.path.join(exp.root_path, exp.fn2p)
    session_name = os.path.basename(dirname_comp)
    return savepath, topo_name, session_name

def make_hash_sha256(o, len=10):
    hasher = hashlib.sha256()
    hasher.update(repr(make_hashable(o)).encode())
    h = base64.b64encode(hasher.digest()).decode()
    h = h.replace("/", "")
    return h[:len]

def make_hashable(o):
    if isinstance(o, (tuple, list)):
        return tuple((make_hashable(e) for e in o))
    if isinstance(o, dict):
        return tuple(sorted((k,make_hashable(v)) for k,v in o.items()))
    if isinstance(o, (set, frozenset)):
        return tuple(sorted(make_hashable(e) for e in o))
    return o

def cache_conditional_load(cache_folder, savepath, exp, settings, overwrite=False, VR=True, YMAZE=False):
    if VR:
        h = make_hash_sha256(settings)
    else:
        h = 'bgCache'

    cachepath = cache_folder + '/%s' % h
    if os.path.exists(cachepath)==0:
        os.mkdir(cachepath)

    cachename = cachepath + '/' + savepath
    if os.path.exists(cachename) and not overwrite:
        print(('---- Loading cached session %s' % cachename))
        with open(cachename, 'rb') as f:
            if sys.version_info.major < 3:
                session = pickle.load(f)
            else:
                session = pickle.load(f, encoding='latin1')
    else:
        print("--- Writing cached session %s" % cachename)
        if VR:
            session = Session(exp, settings)
        else:
            if YMAZE:
                session = ymSession(exp, settings)
            else:
                session = bgSession(exp, settings)
        with open(cachename, 'wb') as f:
            pickle.dump(session, f, protocol=2)
    return session

### Inherited Class for BG ###

class bgRoi(Roi):
    def __init__(self, activations, dF_F, S, times, position, speed, running_mask, settings):
        # --- input data requirements
        assert len(dF_F) == len(times), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and times must have the same size'
        assert len(dF_F) == len(position), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and position must have the same size'
        assert len(dF_F) == len(speed), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and speed must have the same size'
        assert len(dF_F) == len(S), '----- ! ERROR ! ----- (roi constructor) ----- dF_F and S must have the same size'

        # --- compulsory variable assignment
        self.dF_F = dF_F
        self.activations = activations
        self.S = S
        self.position = position
        self.speed = speed
        self.settings = settings
        self.times = times
        self.running_mask = running_mask

        # --- additional firing events computations
        self.position_of_activations = np.asarray([self.position[ev] for ev in self.activations])
        self.time_of_activations = np.asarray([self.times[ev] for ev in self.activations])
        self.n_activations = len(self.activations)
        self.activation_raster = self.discretize_activations()
        self.rate = self.n_activations/(np.sum(self.running_mask) * np.median(np.diff(self.times))) #* 1e-3
        self.F_ratemap = []
        self.S_ratemap = []

    def compute_spatial_map_F(self, bins, occupancy, real_occupancy, running_mask):
        if len(self.position_of_activations)==0:
            self.F_ratemap = np.zeros(len(bins)-1)
            #move.printA("occupancy",occupancy)
            self.F_ratemap[real_occupancy==0] = np.nan
        else:
        # events are already thresholded out by speed in the find_events() function at Session level
            dt = np.median(np.diff(self.times)) * 1e-3
            activations_map = np.histogram(self.position_of_activations, bins)[0] * dt
            activations_map[real_occupancy==0] = np.nan
            activations_map = maps.nanfilter1d(
                activations_map, sigma=self.settings["mapfilter"]/self.settings["binsize"],
                return_nan=self.settings["return_nan_filter"])
            self.F_ratemap = maps.ratemap(activations_map, occupancy, min_occupancy=1e-6)
            #move.printA("occupancy",occupancy)
            self.F_ratemap[real_occupancy==0] = np.nan

    def compute_spatial_map_S(self, bins, occupancy, real_occupancy, running_mask, vrdict, axiskey='posx',):
        if len(self.position_of_activations)<0:
            self.S_ratemap = np.zeros(len(bins)-1)
            self.S_ratemap[real_occupancy==0] = np.nan
            #move.printA("occupancy",occupancy)
            #self.S_ratemap[:] = np.nan
        else:
            #events are already thresholded out by speed in the find_events() function at Session level
            #dt = np.median(np.diff(self.times)) * 1e-3
            #activations_map = np.histogram(self.position_of_activations, bins)[0] * dt
            #activations_map = maps.nanfilter1d(activations_map, sigma=self.settings["mapfilter"]/self.settings["binsize"])
            #self.S_ratemap = maps.ratemap(activations_map, occupancy, min_occupancy=1e-6)

            self.S_ratemap = np.zeros(len(bins)-1)
            #move.printA("occupancy",occupancy)
            #self.S_ratemap[:] = np.nan

            dt = np.mean(np.diff(self.times))
            irun_periods, irest_periods = move.run_rest_periods(
                  hspectral.Timeseries(self.speed, dt), self.settings["thr_rest"], self.settings["thr_run"], self.settings["run_min_dur"])
            assert(len(irun_periods))
            run_periods = np.array(irun_periods) * dt + self.times[0]
            rest_periods = np.array(irest_periods) * dt + self.times[0]
            #times_run = times[hkao_in_period(times, run_periods)]
            #times_rest = times[hkao_in_period(times, rest_periods)]

            #move.printA("vrdict['tracktimes']*1e3",vrdict['tracktimes']*1e3)
            #move.printA("self.times",self.times)
            #move.printA("occupancy",occupancy)

            postimes_crossing, ypos_crossing = cut_array(
                vrdict['tracktimes'], vrdict[axiskey], self.times[0], self.times[-1])
            postimes_in_period = laps.in_period(postimes_crossing, run_periods)
            if not(np.any(postimes_in_period)):
                assert(np.abs(ypos_crossing[-1]-ypos_crossing.min()) < 0.05)
                return
            postimes_crossing_run = postimes_crossing[postimes_in_period]
            ypos_crossing_run = ypos_crossing[postimes_in_period]

            #binstart = np.floor(vrdict[axiskey].min())-1e-3
            #binend = vrdict[axiskey].max() + self.settings["binsize"]
            binstart = bins[0]
            binend = bins[-1]
            #posdt = np.mean(np.diff(vrdict['frametvr']*1e-3))

            Stimes_crossing, S_crossing = self.times, self.S
            Stimes_in_period = laps.in_period(Stimes_crossing, run_periods)
            #Stimes_out_period = in_period(Stimes_crossing, rest_periods)
            Stimes_crossing_run = Stimes_crossing[Stimes_in_period]
            Sy_crossing_run = S_crossing[Stimes_in_period]
            #Sy_crossing_rest = S_crossing[Stimes_out_period]
            diglistS = np.digitize(Stimes_crossing_run*1e3, vrdict['tracktimes']*1e3, right=True)
            S_pos_crossing_run = vrdict[axiskey][diglistS]

            binsize_S = bins[1] - bins[0]

            #print("binsize_S",binsize_S)
            #print stop

            bins_crossing_run3, Smap_crossing_run = maps.ymap(Sy_crossing_run, S_pos_crossing_run, ypos_crossing_run, binsize=binsize_S, binstart=bins[0]-binsize_S, binend=bins[-1]+binsize_S)

            #move.printA("BEFORE self.S_ratemap",self.S_ratemap)
            #move.printA("Smap_crossing_run",Smap_crossing_run)
            #move.printA("occupancy",occupancy)

            Smap_crossing_run[:len(real_occupancy)][real_occupancy==0] = np.nan
            self.S_ratemap = maps.nanfilter1d(
                Smap_crossing_run, sigma=self.settings["mapfilter"]/self.settings["binsize"],
                return_nan=self.settings["return_nan_filter"])[:len(occupancy)] #[occupancy!=0]   self.settings["binsize"]
            #move.printA("AFTER  self.S_ratemap",self.S_ratemap)
            self.S_ratemap[real_occupancy==0] = np.nan

class bgLap(Lap):
    def __init__(self, index, laptype, dF_F, S, activations, times, position, speed, events, settings):

        # --- input data requirements
        #assert len(dF_F[0]) == len(times), '----- ! ERROR ! ----- (roi constructor) ----- single roi F and times must have the same size'
        assert len(times) == len(position), '----- ! ERROR ! ----- (roi constructor) ----- times and position must have the same size'
        assert len(times) == len(speed), '----- ! ERROR ! ----- (roi constructor) ----- times and speed must have the same size'
        assert np.shape(dF_F) == np.shape(S), '----- ! ERROR ! ----- (roi constructor) ----- F and S must have the same shape'
        # --- compulsory variable assignment
        N, T = np.shape(dF_F)
        self.index = index
        self.laptype = laptype
        self.n_roi = N
        self.dF_F = dF_F
        self.S = S
        self.event_length = [len(s[s>0])*1.0 for s in S]
        self.lap_length = len(S[0])*1.0
        self.activations = activations
        self.event_count = [len(a) for a in activations]
        self.times = times
        self.duration = times[-1]-times[0]
        self.position = position
        self.speed = speed
        self.events = events
        self.settings = settings
        self.tot_running_time, self.running_mask = compute_running_mask(self, VR=False)
        self.mean_speed = np.mean(self.speed[self.running_mask])
        self.raster = []
        self.rate_maps = []
        self.s_maps = []

        # --- single roi construction
        self.rois = []
        for i in range(N):
            assert(np.sum(self.running_mask) != 0)
            newroi = bgRoi(activations[i], dF_F[i], S[i], times, position, speed, self.running_mask, settings)
            self.rois.append(newroi)

        # --- online computations
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.lick_stats = []
        self.compute_lick_stats()
        self.discretize_activations()
        #self.compute_hits()
        self.reward = False#self.lick_stats['number_hits'] > 0

    def compute_spatial_maps(self, bins, vrdict):
        #print("start lap.compute_spatial_maps")
        dt = np.median(np.diff(self.times))
        run_periods, rest_periods = move.run_rest_periods(hspectral.Timeseries(self.speed, dt), self.settings["thr_rest"],
                                                          self.settings["thr_run"], self.settings["run_min_dur"])

        running_mask = laps.in_period(np.arange(len(self.times)), run_periods)
        self.real_occupancy = np.histogram(self.position[running_mask], bins)[0] * dt
        self.occupancy = maps.nanfilter1d(
            self.real_occupancy, sigma=self.settings["mapfilter"]/self.settings["binsize"],
            return_nan=self.settings["return_nan_filter"]) # filtering occupancy vector
        if DEBUG:
            move.printA('self.occupancy',self.occupancy)
            move.printA('self.times',self.times)

        ts = hspectral.Timeseries(self.speed, dt)

        for roi in self.rois:
            roi.compute_spatial_map_F(bins, occupancy=self.occupancy, real_occupancy=self.real_occupancy, running_mask=running_mask)
            roi.compute_spatial_map_S(bins, occupancy=self.occupancy, real_occupancy=self.real_occupancy, running_mask=running_mask, vrdict=vrdict)

        self.rate_maps = np.asarray([roi.F_ratemap for roi in self.rois])
        self.s_maps = np.asarray([roi.S_ratemap for roi in self.rois])

class bgSession(Session):
    def __init__(self, exp, settings):
        # --- loading data
        self.settings = settings
        vrdict, calciumdict = syncfiles.load_2p_data(exp, settings, isvr=False)
        #print(("vrdict.keys:"+str(list(vrdict.keys()))))

        #move.printA("vrdict['tracktimes']",vrdict['tracktimes'])
        #move.printA("vrdict['frametimes']",vrdict['frametimes'])
        #move.printA("vrdict['posx']",vrdict['posx'])

        speed_lowpass = hspectral.lowpass( hspectral.Timeseries( vrdict['trackspeed'], np.mean( np.diff( vrdict['tracktimes']))), 0.03)
        speed_thres = np.min( speed_lowpass.data) + 0.2 * ( np.max( speed_lowpass.data) - np.min( speed_lowpass.data))

        vrdict['speed_lowpass'] = speed_lowpass.data
        vrdict['speed_thres']   = speed_thres
        vrdict['posx_lowpass']   = hspectral.lowpass(hspectral.Timeseries(vrdict['posx_frames'], np.median(np.diff(vrdict['frametimes']))), settings['pos_lowpass_para']).data

        self.cut_periods = exp.cut_periods
        self.vrdict = vrdict
        #move.printA("vrdict",vrdict)
        #move.printA("vrdict.keys()",list(vrdict.keys()))

        laptypes, lapboundtimes = laps.split_laps_1p(
            vrdict['posx_lowpass'], np.median(np.diff(vrdict['frametimes'])),True)
        self.events = lapboundtimes

        # --- synchronizing times
        Tmax = min([vrdict['frametimes'][-1], vrdict['tracktimes'][-1]])
        Tmin = max([vrdict['frametimes'][0], vrdict['tracktimes'][0]])
        keep_1p = (vrdict['frametimes'] <= Tmax) & (vrdict['frametimes'] >= Tmin)
        keep_vr = (vrdict['tracktimes'] <= Tmax) & (vrdict['tracktimes'] >= Tmin)

        # --- assigning variables
        self.session_number = exp.session_number
        # data_hio = exp.to_haussio(mc=True)
        dirname_comp = os.path.join(exp.root_path, exp.fn2p)
        self.session_name = os.path.basename(dirname_comp)
        self.times = vrdict['frametimes'][keep_1p]
        self.speed = vrdict['trackspeed'][keep_1p]
        pos_func = interp1d(vrdict['tracktimes'], vrdict['posx'])
        self.position = pos_func(self.times)

        self.dF_F = self.process_dF_F(calciumdict, vrdict, keep_1p)
        self.n_roi, T = np.shape(self.dF_F)
        self.name_roi = np.arange(self.n_roi)
        self.process_S(calciumdict,settings,keep_1p)
        self.tot_running_time, self.running_mask = compute_running_mask(self)
        self.mean_speed = np.mean(self.speed[self.running_mask])
        if DEBUG:
            print("check_S",len(self.dF_F),len(self.S),len(self.S_noisy))

        # --- finding events on a session scale
        self.activations = self.find_activations()
        #self.activations = self.find_activations_old(usemap=settings['activation_map'])

        # --- eliminating silent and bad rois
        if self.settings['std'] == 0.0:
            self.kept_cells = self.eliminate_silent_rois(exp.rois_eliminate)
        #print("exp.rois_eliminate: ",exp.rois_eliminate)
        #print("check_S "," kept_cells ",self.kept_cells)
        self.spatial_cells_S = []
        self.spatial_cells_F = []
        self.spatial_cells_S_IN = []
        self.spatial_cells_S_OUT = []
        self.spatial_cells_S_DUO = []
        self.spatial_cells_F_IN = []
        self.spatial_cells_F_OUT = []
        self.spatial_cells_F_DUO = []
        self.C_FF_F = []; self.C_NN_F = []; self.C_FN_F = []; self.C_Flip_F = []
        self.C_FF_B_F = []; self.C_NN_B_F = []; self.C_FN_B_F = []
        self.C_FF_S = []; self.C_NN_S = []; self.C_FN_S = []; self.C_Flip_S = []
        self.C_FF_B_S = []; self.C_NN_B_S = []; self.C_FN_B_S = []
        # --- building laps
        self.laps = []
        self.nlaps = 0
        self.laptypes = [] #add by Hsin-Lun
        self.lapboundtimes = [] #add by Hsin-Lun
        self.incompletelaps = [] #add by Hsin-Lun
        self.silentrois = [] #add by Hsin-Lun
        if self.settings['std'] == 0.0:
            self.divide_in_laps(self.vrdict,discard_roi=False)
            assert len(self.silentrois) == 0, \
            "silentrois should be empty, now it is "+str(self.silentrois)+" for "+self.session_name
        else:
            self.divide_in_laps(self.vrdict,discard_roi=True)

        self.familiar_laps = [l for l in self.laps if l.laptype=='top'] #inbound
        self.novel_laps = [l for l in self.laps if l.laptype=='bottom'] #outbound

        if DEBUG:
            print("check_S",len(self.dF_F),len(self.S),len(self.S_noisy))

        self.compute_selectivity(division=self.settings['selectivity_division'])

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            if i not in self.silentrois:
                newroi = bgRoi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
                self.rois.append(newroi)

        #print("check_S",len(self.dF_F),len(self.S),len(self.S_noisy))
        #print("self.settings['std'] == ",self.settings['std'])

        # eliminate no-activity ROI
        #print("self.rois",len(self.rois))
        self.n_roi_all = self.n_roi
        self.n_roi = len(self.rois)
        #print(self.n_roi_all, "=>" , self.n_roi)
        #print(len(self.dF_F),len(self.S),len(self.S_noisy),len(self.activations))
        self.dF_F = self.dF_F[self.roi_mask]
        self.S_noisy = self.S_noisy[self.roi_mask]
        self.S = self.S[self.roi_mask]
        self.activations = [self.activations[r] for r in range(self.n_roi_all) if r not in self.silentrois]
        #print("=>",len(self.dF_F),len(self.S),len(self.S_noisy),len(self.activations))

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        #self.reward_rate = np.sum([l.reward for l in self.laps])/float(self.nlaps)
        #self.reward_rate_N = np.sum([l.reward for l in self.novel_laps])/float(len(self.novel_laps))
        #self.reward_rate_F = np.sum([l.reward for l in self.familiar_laps])/float(len(self.familiar_laps))

        self.compute_lick_stats()
        self.compute_spatial_maps()
        try:
            self.compute_spatial_zscores(VR=False)
        except:
            print('Could not run compute_spatial_zscores.')

        #self.behavior_F, self.behavior_N, self.behavior_F_false, self.behavior_N_false = compute_reward_stats(exp, self.settings)

        #self.false_stop_F = np.asarray([l.false_stop_ratio for l in self.familiar_laps])
        #self.false_stop_N = np.asarray([l.false_stop_ratio for l in self.novel_laps])
        #self.false_lick_F = np.asarray([l.false_lick_ratio for l in self.familiar_laps])
        #self.false_lick_N = np.asarray([l.false_lick_ratio for l in self.novel_laps])
        #self.false_speed_ratio_F = np.asarray([l.false_speed_ratio for l in self.familiar_laps])
        #self.false_speed_ratio_N = np.asarray([l.false_speed_ratio for l in self.novel_laps])

        #self.true_stop_F = np.asarray([l.true_stop_ratio for l in self.familiar_laps])
        #self.true_stop_N = np.asarray([l.true_stop_ratio for l in self.novel_laps])
        #self.true_lick_F = np.asarray([l.true_lick_ratio for l in self.familiar_laps])
        #self.true_lick_N = np.asarray([l.true_lick_ratio for l in self.novel_laps])
        #self.true_speed_ratio_F = np.asarray([l.true_speed_ratio for l in self.familiar_laps])
        #self.true_speed_ratio_N = np.asarray([l.true_speed_ratio for l in self.novel_laps])

        # / --- end of constructor --- / #

    def compute_spatial_maps(self):
        #print "---- O> Computing spatial maps"; t0 = time.time()

        #maxs = [np.max(l.position) for l in self.laps]
        #mins = [np.min(l.position) for l in self.laps]
        spatial_bins = self.settings["spatial_bins"]
        #bins = np.linspace(np.min(mins), np.max(maxs), spatial_bins + 1)
        #bins = np.linspace(np.min(self.position), np.max(self.position), spatial_bins + 1)
        bins = np.linspace(0, self.settings['position_max'], spatial_bins + 1)
        assert self.settings['position_max'] > np.max(self.position), "settings[position_max]: "+ self.settings['position_max']+" should larger then np.max(position): "+ np.max(self.position)
        self.bins = bins
        #print('maxs',maxs)
        #print('mins',mins)
        if DEBUG:
            move.printA("bins",bins)
        for l in self.laps:
            l.compute_spatial_maps(bins,self.vrdict)
        #print "---- <O Computed spatial maps, time elapsed = %.1f s" % (time.time() - t0)

    def process_S(self, calciumdict, settings, keep_1p):
        if settings["caiman"]: #True
            self.S_noisy = calciumdict['S'][:, keep_1p]
            self.S = np.zeros(shape=self.S_noisy.shape, dtype=float)
            for nf, df in enumerate(self.dF_F):
                gra = np.gradient(df)
                c, bl, c1, g, sn, sp_noisy, lam = constrained_foopsi(df,p=1,s_min=settings["smin"])
                self.S_noisy[nf] = sp_noisy
                sp = deepcopy(self.S_noisy[nf])
                pulse = np.arange(len(sp))[sp>0]
                splits = np.append(np.where(np.diff(pulse)>5)[0],len(pulse))
                for i, split in enumerate(splits):
                    if i == 0:
                        single = pulse[:split+1]
                    elif i == len(splits) - 1:
                        single = pulse[splits[i-1]+1:]
                    else:
                        single = pulse[splits[i-1]+1:splits[i]+1]
                    rise = gra[max(single[0]-2,0):single[-1]]
                    decay = gra[single[-1]:single[-1]+1+(single[-1]-single[0])]
                    mean_rise = bn.nanmean(rise)
                    mean_decay = bn.nanmean(decay[decay<0])
                    peak = max(gra[max(single[0]-2,0):single[-1]+1+(single[-1]-single[0])])
                    if (mean_rise > abs(mean_decay) and mean_rise > 0.08) or peak > 0.17:
                        #print("REAL PULSE")
                        pass
                    else:
                        sp[single[0]-1:single[-1]+1] = 0.0
                self.S[nf] = sp
        elif settings["caiman_simple"]: #True
            self.S_noisy = calciumdict['S'][:, keep_1p]
            self.S = np.zeros(shape=self.S_noisy.shape, dtype=float)
            for nf, df in enumerate(self.dF_F):
                c, bl, c1, g, sn, sp_noisy, lam = constrained_foopsi(df,p=1,s_min=settings["smin"])
                self.S_noisy[nf] = sp_noisy
                sp = deepcopy(self.S_noisy[nf])
                self.S[nf] = sp
        elif settings["thr_inference"]: #True
            self.S_noisy = calciumdict['S'][:, keep_1p]
            self.S = np.zeros(shape=self.S_noisy.shape, dtype=float)
            if DEBUG:
                print("check_S",len(self.dF_F),len(self.S),len(self.S_noisy))
                print("check_S shape",self.dF_F.shape,self.S.shape,self.S_noisy.shape)
            for ns, sp in enumerate(self.S_noisy):
                norm_sp = sp-np.min(sp)
                norm_sp /= np.max(norm_sp)
                norm_sp *= sigmoid(norm_sp, settings['std_slope'], norm_sp.mean()+np.std(norm_sp)*settings['std_scale'])
                if settings["SMART_FILTER"]:
                    smart_FILTER = (sp.mean()+np.std(sp)*settings["SM_filter"])/max(sp) #norm_sp.mean()+np.std(norm_sp)*7.0
                    norm_sp[norm_sp < smart_FILTER] = 0.0
                elif settings["STRONG_FILTER"]:
                    norm_sp[norm_sp < settings["ST_filter"]] = 0.0
                self.S[ns] = norm_sp
        else: ##settings["thr_inference"]==False AND settings["caiman"]==False
            self.S_noisy = calciumdict['S'][:, keep_1p]
            self.S = calciumdict['S'][:, keep_1p]

    def process_dF_F(self, calciumdict, vrdict, keep_1p):
        dF_F = calciumdict['Fraw'][:, keep_1p]-self.settings['Fneu_factor']*calciumdict['Fneu'][:, keep_1p]
        n_roi, T = np.shape(dF_F)
        dt = np.median(np.diff(vrdict['frametimes']))#*1e-3
        """
        for i in range(n_roi):
            dF_F[i] = hspectral.lowpass(hspectral.highpass(hspectral.Timeseries(dF_F[i].astype(np.float), dt), 0.002, verbose=False), self.settings["F_filter"], verbose=False).data
            dF_F[i] -= np.mean(dF_F[i])
        """
        dF_F = np.array([
            hspectral.lowpass(
                hspectral.highpass(
                    hspectral.Timeseries(df.astype(np.float), dt), 0.002, verbose=False),
                1.000, verbose=False).data
            for df in dF_F])

        if self.settings["detrend"]:
            dF_F_detrend = np.zeros(dF_F.shape,dtype=float) #deepcopy(dF_F)
            for j, df in enumerate(dF_F):
                X = [i for i in range(0, len(df))]
                X = np.reshape(X, (len(X), 1))
                y = df
                model = make_pipeline(PolynomialFeatures(5), Ridge())
                model.fit(X, y)
                # calculate trend
                df_trend = model.predict(X)
                dF_F_detrend[j] = df-df_trend
            return dF_F_detrend
        else:
            return dF_F

    def divide_in_laps(self, vrdict, discard_roi=False) :
        if not self.laptypes and not self.lapboundtimes:
            if DEBUG:
                print("run split_laps_1p")
            gratings, boundary_times = laps.split_laps_1p(
                vrdict['posx_lowpass'], np.median(np.diff(vrdict['frametimes'])),
                True)
            """
            print("vrdict['tracktimes']",len(vrdict['tracktimes']))
            print(vrdict['tracktimes'])
            print("vrdict['frametimes']",len(vrdict['frametimes']))
            print(vrdict['frametimes'])
            print("vrdict['posx']",len(vrdict['posx']))
            print(vrdict['posx'])
            for bt in boundary_times:
                idx = np.where(vrdict['tracktimes'] > bt)[0][0]
                print("time:",bt,"new_time",vrdict['tracktimes'][idx],
                      ",index:",idx,"=>pos:",vrdict['posx'][idx])
            """
            if DEBUG:
                print("record timearray")
                print(self.times)
            #teleport_times *= 1e3
            self.laptypes = gratings
            self.lapboundtimes = boundary_times
        track_length = np.max(vrdict['posx'].squeeze())-np.min(vrdict['posx'].squeeze())

        for lap_trace, (lapstart, lapend, stimulus) in enumerate(zip(self.lapboundtimes[:-1], self.lapboundtimes[1:], self.laptypes)):
            istart = np.where(vrdict['tracktimes'].squeeze() >= lapstart)[0][0]
            iend = np.where(vrdict['tracktimes'].squeeze() > lapend)[0]
            if len(iend) == 0:
                iend = None
            else:
                iend = iend[0]
            lapspeed_lowpass = vrdict['speed_lowpass'][istart:iend]
            lapspeed = vrdict['trackspeed'][istart:iend]
            lappostimes = vrdict['tracktimes'].squeeze()[istart:iend]
            lapposx = vrdict['posx'].squeeze()[istart:iend]
            if len(lapposx):
                track_fraction = (np.max(lapposx)-np.min(lapposx)) / track_length
            else:
                track_fraction = 0
            #print("max(lapposx)",np.max(lapposx),"min(lapposx)",np.min(lapposx),"track_length",track_length)
            if track_fraction < self.settings['min_lap_fraction']:
                if DEBUG:
                    print("Lap {0:d} :Skipping partial lap crossing, fraction {1:.2f}".format(lap_trace,track_fraction))
                self.incompletelaps.append(lap_trace)
            elif max(lappostimes)-min(lappostimes) <= 1:
                if DEBUG:
                    print("Lap {0:d} :Skipping lap crossing that took less than 1s, might be concatenated?".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            elif len(np.where(lapspeed == 0)[0])>1:
                if DEBUG:
                    print("Lap {0:d} :Skipping lap with zero speed".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            elif stimulus == 'none':
                if DEBUG:
                    print("Lap {0:d}: Just skip it".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            #elif len(np.where(lapspeed_lowpass < vrdict['speed_thres'])[0])>1:
            #    print("Skipping lap with unpassionate running")
            #    self.incompletelaps.append(lap_trace)
        self.incompletelaps.append(len(self.lapboundtimes) - 1)
        if DEBUG:
            print("incomplete laps")
            print(self.incompletelaps)

        #boundary_times = boundary_times[1:]
        #gratings = gratings[1:] # discarding the first and the last lap (incomplete)

        self.nlaps = len(boundary_times) - 1
        out_index = 0
        in_index = 0

        if discard_roi and (self.settings['std'] > 0.0):
            for r in range(self.n_roi):
                silent = True
                #print("roi(",r,")")
                for i in range(self.nlaps):
                    #print("lap(",i,")")
                    if i not in self.incompletelaps:
                        time_mask = (self.times > boundary_times[i] + self.settings["lap_split_offset"]) & (self.times < boundary_times[i+1] - self.settings["lap_split_offset"])
                        lap_activations = self.find_lap_activations(time_mask)
                        #print("lap_activations(r=",r,",i=",i,"):",lap_activations)
                        if lap_activations[r]:
                            silent = False
                            break
                if silent:
                    self.silentrois.append(r)

            if DEBUG:
                print("silentrois",self.silentrois)

        for i in range(self.nlaps):
            if i not in self.incompletelaps:
                time_mask = (self.times > boundary_times[i] + self.settings["lap_split_offset"]) & (self.times < boundary_times[i+1] - self.settings["lap_split_offset"])
                #lap_events = [e for e in vrdict['evlist'] if boundary_times[i] + self.settings["lap_split_offset"] < (e.time * 1000.) < boundary_times[i+1] - self.settings["lap_split_offset"]]
                lap_activations = self.find_lap_activations(time_mask)
                laptype = gratings[i]

                if laptype=='bottom':#noval
                    index = out_index
                    out_index += 1
                if laptype=='top':#familiar
                    index = in_index
                    in_index += 1

                #print("self.dF_F[:, time_mask]",self.dF_F[:, time_mask])
                array_index = np.arange(self.n_roi)
                #print("array_index",array_index)
                roi_mask = [a not in self.silentrois for a in array_index]
                #print("roi_mask",len(roi_mask),roi_mask)
                #print("self.dF_F.shape",self.dF_F.shape)
                #print("time_mask",len(time_mask),time_mask)
                #print("self.dF_F[roi_mask]",self.dF_F[roi_mask])
                #print("self.dF_F[roi_mask][:,time_mask]",self.dF_F[roi_mask][:,time_mask])
                #print("lap_activations",lap_activations)
                #print("lap_activations[roi_mask]",[lap_activations[a] for a in array_index if a not in self.silentrois])
                self.roi_mask = roi_mask

                #print(stop)
                if discard_roi:
                    new_lap = bgLap(
                        index=index,
                        laptype=laptype,
                        dF_F=self.dF_F[roi_mask][:, time_mask],
                        S=self.S[roi_mask][:, time_mask],
                        activations=[lap_activations[a] for a in array_index if a not in self.silentrois], #lap_activations,
                        times=self.times[time_mask],
                        position=self.position[time_mask],
                        speed=self.speed[time_mask],
                        events=[],#lap_events,
                        settings=self.settings)
                else:
                    new_lap = bgLap(
                        index=index,
                        laptype=laptype,
                        dF_F=self.dF_F[:, time_mask],
                        S=self.S[:, time_mask],
                        activations=lap_activations,
                        times=self.times[time_mask],
                        position=self.position[time_mask],
                        speed=self.speed[time_mask],
                        events=[],#lap_events,
                        settings=self.settings)

                self.laps.append(new_lap)
            else:
                self.nlaps -= 1


    def find_activations(self):
        all_activations = []
        for i in range(self.n_roi):
            events, amplitudes = p2p.find_events(self.dF_F[i], self.speed, self.settings['min_speed'], self.settings['event_std_threshold'])
            events = np.asarray(events)
            amplitudes = np.asarray(amplitudes)
            events = events[amplitudes > self.settings['F_theta']]
            all_activations.append(events)

        return all_activations

    def find_activations_new(self):
        all_activations = []
        for ns, sp in enumerate(self.S):
            activations = []
            pulse = np.arange(len(sp))[sp>0]
            splits = np.append(np.where(np.diff(pulse)>5)[0],len(pulse))
            for i, split in enumerate(splits):
                if i == 0:
                    single = pulse[:split+1]
                elif i == len(splits) - 1:
                    single = pulse[splits[i-1]+1:]
                else:
                    single = pulse[splits[i-1]+1:splits[i]+1]
                activations.append(single[0])
            all_activations.append(activations)
        if DEBUG:
            print("all_activations",all_activations)
        #print stop
        return all_activations


    def find_activations_old(self, usemap='F'):
        all_activations = []
        ns = 1
        fig = plt.figure(figsize=(8, 48))
        ax0 = fig.add_subplot(self.n_roi+1, 1, ns)
        ns += 1
        ax0.plot(self.times, self.position)
        if usemap == 'F':
            use = self.dF_F
        else: #usemap == 'S':
            use = self.S
        for i in range(self.n_roi):
            if self.settings['std']==0.0:
                events, amplitudes = p2p.find_events(use[i], self.speed, self.settings['min_speed'], self.settings['event_std_threshold'])
                events = np.asarray(events)
                amplitudes = np.asarray(amplitudes)
                events = events[amplitudes > self.settings['F_theta']]
                all_activations.append(events)

                #print("events",events)
            else:
                new_std = np.std(use[i])
                new_mean = np.mean(use[i])
                new_events = np.where(use[i]>new_mean+new_std*self.settings['std'])[0]
                #print("new_events",new_events)
                #print("new_diff",np.diff(np.append([0],new_events)))
                new_starts = new_events[np.diff(np.append([0],new_events))>1]
                #print("new_starts", new_starts)

                all_activations.append(new_starts)

            ax = fig.add_subplot(self.n_roi+1, 1, ns, sharex=ax0)
            ax.plot(self.times, use[i], '-g', alpha=0.5)
            #ax.plot(self.times, self.S[i], '-r', alpha=0.5)
            #ax.plot(self.times[events], np.ones((len(events)))*self.dF_F[i].min(), '|b', ms=20)
            #ax.plot(self.times[new_starts], np.ones((len(new_starts)))*self.dF_F[i].min(), '|b', ms=20)
            ax.text(
                1.0, 0.5, "{0}".format(ns-1),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax.transAxes)
            ns += 1
        if DEBUG:
            print("all_activations: ",all_activations)
        #print stop
        return all_activations

    def compute_lick_stats(self):
        licktimes = self.events#np.asarray([e.time for e in self.events])
        #reward_times = np.asarray([e.time for e in self.events])

        #reward_distances = training.reward_distance(trainingdata=[], licktimes=licktimes, reward_times=reward_times)
        #stats = training.lick_stats(reward_times, reward_distances, licktimes, self.settings)
        #self.lick_stats = stats
        lick_array = np.zeros(len(self.times))
        for t in licktimes:
            t_lick = np.argmin(np.abs(self.times - t))
            lick_array[t_lick] = 1
        self.lick_array = lick_array

    def compute_spatial_correlations(self, force=False, maptype='F', pairwise=True):
        if DEBUG:
            print('self:',self.session_name)
            print("maptype == "+maptype)
            move.printA('self.C_FF_F',self.C_FF_F)
            move.printA('self.C_FF_S',self.C_FF_S)
        if len(self.C_FF_F) == 0 or len(self.C_FF_S) == 0 or force:
            print("---- O> Computing spatial correlations"); t0 = time.time()
            if pairwise:
                C_FF = [[] for i in range(self.n_roi)]
                C_NN = [[] for i in range(self.n_roi)]
                C_FN = [[] for i in range(self.n_roi)]
                C_Flip = [[] for i in range(self.n_roi)]


                for n in range(self.n_roi):
                    for i in range(len(self.familiar_laps)):
                        for j in range(i+1, len(self.familiar_laps)):
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.familiar_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.familiar_laps[j].s_maps[n], self.settings, periodic=False)
                            C_FF[n].append(c)

                    for i in range(len(self.novel_laps)):
                        for j in range(i+1, len(self.novel_laps)):
                            if maptype == 'F':
                                c = ratemap_correlation(self.novel_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.novel_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                            C_NN[n].append(c)


                    for i in range(len(self.familiar_laps)):
                        for j in range(len(self.novel_laps)):
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                                #print("inbound lap("+str(i)+").roi("+str(n)+").smap vs outbound lap("+str(j)+").roi("+str(n)+").smap => pearson="+str(c))
                            C_FN[n].append(c)

                    for i in range(len(self.familiar_laps)):
                        for j in range(len(self.novel_laps)):
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], np.flip(self.novel_laps[j].rate_maps[n]), self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], np.flip(self.novel_laps[j].s_maps[n]), self.settings, periodic=False)
                            C_Flip[n].append(c)
                if maptype == 'F':
                    self.C_FF_F = np.array(C_FF)
                    self.C_NN_F = np.array(C_NN)
                    self.C_FN_F = np.array(C_FN)
                    self.C_Flip_F = np.array(C_Flip)
                elif maptype == 'S':
                    self.C_FF_S = np.array(C_FF)
                    self.C_NN_S = np.array(C_NN)
                    self.C_FN_S = np.array(C_FN)
                    self.C_Flip_S = np.array(C_Flip)
            else:
                print("DEBUG HALVES_1")
                if maptype == 'F':
                    even_familiar_maps = bn.nanmean([self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                    odd_familiar_maps = bn.nanmean([self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                    even_novel_maps = bn.nanmean([self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                    odd_novel_maps = bn.nanmean([self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                    familiar_maps = bn.nanmean([l.rate_maps for l in self.familiar_laps], axis=0)
                    novel_maps = bn.nanmean([l.rate_maps for l in self.novel_laps], axis=0)

                #    ihalves = get_all_halves(len(self.familiar_laps))
                #    ifirst = [ihalf[0] for ihalf in ihalves]
                #    isecond = [ihalf[1] for ihalf in ihalves]
                #    even_familiar_maps = bn.nanmean([self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                 #   even_familiar_maps = bn.nanmean([bn.nanmean([self.familiar_laps[i].rate_maps for i in ifirstset], axis=0) for ifirstset in ifirst], axis=0)
                  #  odd_familiar_maps = bn.nanmean([self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                   # odd_familiar_maps = bn.nanmean([bn.nanmean([self.familiar_laps[i].rate_maps for i in isecondset], axis=0) for isecondset in isecond], axis=0)

                    #even_novel_maps = bn.nanmean([self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                    #even_novel_maps = bn.nanmean([bn.nanmean([self.novel_laps[i].rate_maps for i in ifirstset], axis=0) for ifirstset in ifirst], axis=0)
                    #odd_novel_maps = bn.nanmean([self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                    #odd_novel_maps = bn.nanmean([bn.nanmean([self.novel_laps[i].rate_maps for i in isecondset], axis=0) for isecondset in isecond], axis=0)
                    #familiar_maps = bn.nanmean([
                    #    l.rate_maps for l in self.familiar_laps], axis=0)
                    #novel_maps = bn.nanmean([
                    #    l.rate_maps for l in self.novel_laps], axis=0)

                else:
                    even_familiar_maps = bn.nanmean([self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                    odd_familiar_maps = bn.nanmean([self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                    even_novel_maps = bn.nanmean([self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                    odd_novel_maps = bn.nanmean([self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                    familiar_maps = bn.nanmean([l.s_maps for l in self.familiar_laps], axis=0)
                    novel_maps = bn.nanmean([l.s_maps for l in self.novel_laps], axis=0)

                  #  ihalves = get_all_halves(len(self.familiar_laps))
                  #  ifirst = [ihalf[0] for ihalf in ihalves]
                  #  isecond = [ihalf[1] for ihalf in ihalves]
                  #  even_familiar_maps = bn.nanmean([self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                   # even_familiar_maps = bn.nanmean([bn.nanmean([self.familiar_laps[i].s_maps for i in ifirstset], axis=0) for ifirstset in ifirst], axis=0)
                    #odd_familiar_maps = bn.nanmean([self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                    #odd_familiar_maps = bn.nanmean([bn.nanmean([self.familiar_laps[i].s_maps for i in isecondset], axis=0) for isecondset in isecond], axis=0)

                    #even_novel_maps = bn.nanmean([self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                    #even_novel_maps = bn.nanmean([bn.nanmean([self.novel_laps[i].s_maps for i in ifirstset], axis=0) for ifirstset in ifirst], axis=0)
                    #odd_novel_maps = bn.nanmean([self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                 #   odd_novel_maps = bn.nanmean([bn.nanmean([self.novel_laps[i].s_maps for i in isecondset], axis=0) for isecondset in isecond], axis=0)
                  #  familiar_maps = bn.nanmean([
                  #      l.s_maps for l in self.familiar_laps], axis=0)
                  #  novel_maps = bn.nanmean([
                   #     l.s_maps for l in self.novel_laps], axis=0)

                    #familiar_maps =bn.nanmean([l.s_maps for l in self.familiar_laps], axis=0)
                    #novel_maps = bn.nanmean([l.s_maps for l in self.novel_laps], axis=0)

                C_FF = np.array([[pov,] for pov in visualize.pov(even_familiar_maps, odd_familiar_maps, percell=True, VR=False, nmin=2)])
                C_NN = np.array([[pov,] for pov in visualize.pov(even_novel_maps, odd_novel_maps, percell=True, VR=False, nmin=2)])
                C_FN = np.array([[pov,] for pov in visualize.pov(familiar_maps, novel_maps, percell=True, VR=False, nmin=2)])
                C_Flip = np.array([[pov,] for pov in visualize.pov(familiar_maps, novel_maps, percell=True, VR=False, nmin=2, flip=True)])

                print("---- <O Computed spatial correlations, time elapsed = %.1f s" % (time.time() - t0))

                if maptype == 'F':
                    self.C_FF_F = C_FF
                    self.C_NN_F = C_NN
                    self.C_FN_F = C_FN
                    self.C_Flip_F = C_Flip

                    return self.C_FF_F, self.C_NN_F, self.C_FN_F, self.selectivity, self.selectivity_discrete, self.selectivity_continuous, self.selectivity_manu, self.selectivity_manu_div, self.selectivity_manu_nodiv
                elif maptype == 'S':
                    self.C_FF_S = C_FF
                    self.C_NN_S = C_NN
                    self.C_FN_S = C_FN
                    self.C_Flip_S = C_Flip

                    return self.C_FF_S, self.C_NN_S, self.C_FN_S, self.selectivity, self.selectivity_discrete, self.selectivity_continuous, self.selectivity_manu, self.selectivity_manu_div, self.selectivity_manu_nodiv


            #print('self:',self.session_name)
            #print('C_FN_S:',self.C_FN_S)
            #print('np.mean(C_FN_S,0):',np.mean(self.C_FN_S,0))
            #print('np.mean(C_FN_S,1):',np.mean(self.C_FN_S,1))


        else:
            print("DEBUG!")
            if maptype == 'F':
                return self.C_FF_F, self.C_NN_F, self.C_FN_F, self.selectivity, self.selectivity_discrete, self.selectivity_continuous, self.selectivity_manu, self.selectivity_manu_div, self.selectivity_manu_nodiv
            elif maptype == 'S':
                return self.C_FF_S, self.C_NN_S, self.C_FN_S, self.selectivity, self.selectivity_discrete, self.selectivity_continuous, self.selectivity_manu, self.selectivity_manu_div, self.selectivity_manu_nodiv

                       #if maptype == 'F':
                #    even_familiar_maps = bn.nanmean([
                #        self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                #    odd_familiar_maps = bn.nanmean([
                #        self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                #    even_novel_maps = bn.nanmean([
                #        self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                #    odd_novel_maps = bn.nanmean([
                #        self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                #    familiar_maps = bn.nanmean([
                #        l.rate_maps for l in self.familiar_laps], axis=0)
                #    novel_maps = bn.nanmean([
                #        l.rate_maps for l in self.novel_laps], axis=0)
#  even_familiar_maps = bn.nanmean([
                  #      self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==0], axis=0)
                  #  odd_familiar_maps = bn.nanmean([
                  #      self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==1], axis=0)
                  #  even_novel_maps = bn.nanmean([
                  #      self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==0], axis=0)
                  #  odd_novel_maps = bn.nanmean([
                  #      self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==1], axis=0)
                  #  familiar_maps = bn.nanmean([
                  #      l.s_maps for l in self.familiar_laps], axis=0)
                  #  novel_maps = bn.nanmean([
                  #      l.s_maps for l in self.novel_laps], axis=0)



    def compute_spatial_correlations_conditions(self, force = False, maptype='F', pairwise=False):
        try:
            conds = self.vrdict['conditions']
        except:
            conds = 'none'
        assert(conds != 'none')
        assert(conds == ['A', 'B', 'C'])
        ends = self.vrdict['tracktimes_ends']
        if DEBUG:
            print(('condition A:['+str(ends[0])+','+str(ends[1])+']'))
            print(('condition B:['+str(ends[1])+','+str(ends[2])+']'))
            print(('condition C:['+str(ends[2])+','+str(ends[3])+']'))

        print("---- O> Computing spatial correlations"); t0 = time.time()
        print("maptype == "+maptype)

        if pairwise:
            #move.printA('C_FF cond',C_FF)
            #move.printA('C_NN cond',C_NN)
            #move.printA('C_FN cond',C_FN)
            #move.printA('C_Flip cond',C_Flip)

            C_FF_A = [[] for i in range(self.n_roi)]
            C_NN_A = [[] for i in range(self.n_roi)]
            C_FN_A = [[] for i in range(self.n_roi)]
            C_Flip_A = [[] for i in range(self.n_roi)]
            C_FF_B = [[] for i in range(self.n_roi)]
            C_NN_B = [[] for i in range(self.n_roi)]
            C_FN_B = [[] for i in range(self.n_roi)]
            C_Flip_B = [[] for i in range(self.n_roi)]
            C_FF_C = [[] for i in range(self.n_roi)]
            C_NN_C = [[] for i in range(self.n_roi)]
            C_FN_C = [[] for i in range(self.n_roi)]
            C_Flip_C = [[] for i in range(self.n_roi)]

            C_FF = [C_FF_A,C_FF_B,C_FF_C]
            C_NN = [C_NN_A,C_NN_B,C_NN_C]
            C_FN = [C_FN_A,C_FN_B,C_FN_C]
            C_Flip = [C_Flip_A,C_Flip_B,C_Flip_C]

            for n in range(self.n_roi):
                for i in range(len(self.familiar_laps)):
                    for j in range(i+1, len(self.familiar_laps)):
                        cond_i = in_conditions(self.familiar_laps[i].times,ends)
                        cond_j = in_conditions(self.familiar_laps[j].times,ends)
                        if cond_i==cond_j and cond_i != -1:
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.familiar_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.familiar_laps[j].s_maps[n], self.settings, periodic=False)
                            C_FF[cond_i][n].append(c)

                for i in range(len(self.novel_laps)):
                     for j in range(i+1, len(self.novel_laps)):
                        cond_i = in_conditions(self.novel_laps[i].times,ends)
                        cond_j = in_conditions(self.novel_laps[j].times,ends)
                        if cond_i==cond_j and cond_i != -1:
                            if maptype == 'F':
                                c = ratemap_correlation(self.novel_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.novel_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                            C_NN[cond_i][n].append(c)


                for i in range(len(self.familiar_laps)):
                    for j in range(len(self.novel_laps)):
                        cond_i = in_conditions(self.familiar_laps[i].times,ends)
                        cond_j = in_conditions(self.novel_laps[j].times,ends)
                        if cond_i==cond_j and cond_i != -1:
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                            C_FN[cond_i][n].append(c)

                for i in range(len(self.familiar_laps)):
                    for j in range(len(self.novel_laps)):
                        cond_i = in_conditions(self.familiar_laps[i].times,ends)
                        cond_j = in_conditions(self.novel_laps[j].times,ends)
                        if cond_i==cond_j and cond_i != -1:
                            if maptype == 'F':
                                c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], np.flip(self.novel_laps[j].rate_maps[n]), self.settings, periodic=False)
                            elif maptype == 'S':
                                c = ratemap_correlation(self.familiar_laps[i].s_maps[n], np.flip(self.novel_laps[j].s_maps[n]), self.settings, periodic=False)
                            C_Flip[cond_i][n].append(c)
        else:
            C_FF = [[], [], []]
            C_NN = [[], [], []]
            C_FN = [[], [], []]
            C_Flip = [[], [], []]
            familiar_idcs = [in_conditions(self.familiar_laps[i].times,ends) for i in range(len(self.familiar_laps))]
            novel_idcs = [in_conditions(self.novel_laps[i].times,ends) for i in range(len(self.novel_laps))]
            for ncond in range(len(C_FF)):
                if maptype == 'F':
                    even_familiar_maps = bn.nanmean([
                        self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==0 and familiar_idcs[i]==ncond], axis=0)
                    odd_familiar_maps = bn.nanmean([
                        self.familiar_laps[i].rate_maps for i in range(len(self.familiar_laps)) if i%2==1 and familiar_idcs[i]==ncond], axis=0)
                    even_novel_maps = bn.nanmean([
                        self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==0 and novel_idcs[i]==ncond], axis=0)
                    odd_novel_maps = bn.nanmean([
                        self.novel_laps[i].rate_maps for i in range(len(self.novel_laps)) if i%2==1 and novel_idcs[i]==ncond], axis=0)
                    familiar_maps = bn.nanmean([
                        l.rate_maps for i, l in enumerate(self.familiar_laps) if familiar_idcs[i]==ncond], axis=0)
                    novel_maps = bn.nanmean([
                        l.rate_maps for i, l in enumerate(self.novel_laps) if novel_idcs[i]==ncond], axis=0)
                else:
                    even_familiar_maps = bn.nanmean([
                        self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==0 and familiar_idcs[i]==ncond], axis=0)
                    odd_familiar_maps = bn.nanmean([
                        self.familiar_laps[i].s_maps for i in range(len(self.familiar_laps)) if i%2==1 and familiar_idcs[i]==ncond], axis=0)
                    even_novel_maps = bn.nanmean([
                        self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==0 and novel_idcs[i]==ncond], axis=0)
                    odd_novel_maps = bn.nanmean([
                        self.novel_laps[i].s_maps for i in range(len(self.novel_laps)) if i%2==1 and novel_idcs[i]==ncond], axis=0)
                    familiar_maps = bn.nanmean([
                        l.s_maps for i, l in enumerate(self.familiar_laps) if familiar_idcs[i]==ncond], axis=0)
                    novel_maps = bn.nanmean([
                        l.s_maps for i, l in enumerate(self.novel_laps) if novel_idcs[i]==ncond], axis=0)
                C_FF[ncond] = np.array([[pov,] for pov in visualize.pov(even_familiar_maps, odd_familiar_maps, percell=True, VR=False, nmin=2)])
                C_NN[ncond] = np.array([[pov,] for pov in visualize.pov(even_novel_maps, odd_novel_maps, percell=True, VR=False, nmin=2)])
                C_FN[ncond] = np.array([[pov,] for pov in visualize.pov(familiar_maps, novel_maps, percell=True, VR=False, nmin=2)])
                C_Flip[ncond] = np.array([[pov,] for pov in visualize.pov(familiar_maps, novel_maps, percell=True, VR=False, nmin=2, flip=True)])
        print("---- <O Computed spatial correlations, time elapsed = %.1f s" % (time.time() - t0))
        return C_FF, C_NN, C_FN, C_Flip


    def compute_spatial_zscores(self, force=False, VR=True):
        if len(self.spatial_cells_S)==0 or force:
            C_FF, C_NN, C_FN, C_FF_B, C_NN_B, C_FN_B = compute_spatial_bootstrap(
                self, chunk_size = self.settings["bootstrap_chunk_size"], n_bootstrap=self.settings["nbootstrap"], maptype='S', VR=VR)
            self.C_FF_S = C_FF
            self.C_NN_S = C_NN
            self.C_FN_S = C_FN
            self.C_FF_B_S = C_FF_B
            self.C_NN_B_S = C_NN_B
            self.C_FN_B_S = C_FN_B
            if VR:
                mode = 'cells'
            else:
                mode = 'cells'
            self.spatial_Z_FF_S = bootstrap_zscores(C_FF, C_FF_B, mode=mode)
            self.spatial_Z_NN_S = bootstrap_zscores(C_NN, C_NN_B, mode=mode)
            self.spatial_Z_FN_S = bootstrap_zscores(C_FN, C_FN_B, mode=mode)
            self.spatial_cells_S_IN = self.spatial_Z_FF_S > self.settings['bootstrap_std_threshold']
            self.spatial_cells_S_OUT = self.spatial_Z_NN_S > self.settings['bootstrap_std_threshold']
            self.spatial_cells_S = self.spatial_cells_S_IN | self.spatial_cells_S_OUT
            self.spatial_cells_S_DUO = self.spatial_cells_S_IN & self.spatial_cells_S_OUT
        elif force:
            self.spatial_cells_S_IN = self.spatial_Z_FF_S > self.settings['bootstrap_std_threshold']
            self.spatial_cells_S_OUT = self.spatial_Z_NN_S > self.settings['bootstrap_std_threshold']
            self.spatial_cells_S = self.spatial_cells_S_IN | self.spatial_cells_S_OUT
            self.spatial_cells_S_DUO = self.spatial_cells_S_IN & self.spatial_cells_S_OUT


        if len(self.spatial_cells_F)==0 or force:
            C_FF, C_NN, C_FN, C_FF_B, C_NN_B, C_FN_B = compute_spatial_bootstrap(
                self, chunk_size = self.settings["bootstrap_chunk_size"], n_bootstrap=self.settings["nbootstrap"], maptype='F', VR=VR)
            self.C_FF_F = C_FF
            self.C_NN_F = C_NN
            self.C_FN_F = C_FN
            self.C_FF_B_F = C_FF_B
            self.C_NN_B_F = C_NN_B
            self.C_FN_B_F = C_FN_B
            if VR:
                mode = 'cells'
            else:
                mode = 'cells'
            self.spatial_Z_FF_F = bootstrap_zscores(C_FF, C_FF_B, mode=mode)
            self.spatial_Z_NN_F = bootstrap_zscores(C_NN, C_NN_B, mode=mode)
            self.spatial_Z_FN_F = bootstrap_zscores(C_FN, C_FN_B, mode=mode)
            self.spatial_cells_F_IN = self.spatial_Z_FF_F > self.settings['bootstrap_std_threshold']
            self.spatial_cells_F_OUT = self.spatial_Z_NN_F > self.settings['bootstrap_std_threshold']
            self.spatial_cells_F = self.spatial_cells_F_IN | self.spatial_cells_F_OUT
            self.spatial_cells_F_DUO = self.spatial_cells_F_IN & self.spatial_cells_F_OUT
        elif force:
            self.spatial_cells_F_IN = self.spatial_Z_FF_F > self.settings['bootstrap_std_threshold']
            self.spatial_cells_F_OUT = self.spatial_Z_NN_F > self.settings['bootstrap_std_threshold']
            self.spatial_cells_F = self.spatial_cells_F_IN | self.spatial_cells_F_OUT
            self.spatial_cells_F_DUO = self.spatial_cells_F_IN & self.spatial_cells_F_OUT

        #move.printA("force=",force)
        #move.printA("self.spatial_cells_S",self.spatial_cells_S)
        #move.printA("self.spatial_cells_F",self.spatial_cells_F)

    # --- bootstrap randomizations --- #

    def shuffle_F(self, chunk_size=100, randomstate = 'none'):
        # chunk shuffle self.dF_F
        T = len(self.dF_F[0])
        shuffled_index = chunk_shuffle_index(T, chunk_size, randomstate = randomstate)
        self.dF_F = self.dF_F[:, shuffled_index]
        self.S = self.S[:, shuffled_index]

        # --- re-finding events on a session scale
        self.activations = self.find_activations()

        # --- re-building laps
        self.laps = []
        self.nlaps = 0
        self.laptypes = [] #add by Hsin-Lun
        self.lapboundtimes = [] #add by Hsin-Lun
        self.incompletelaps = [] #add by Hsin-Lun
        self.silentrois = [] #add by Hsin-Lun
        self.divide_in_laps(self.vrdict, discard_roi=False)

        self.familiar_laps = [l for l in self.laps if l.laptype=='top']
        self.novel_laps = [l for l in self.laps if l.laptype=='bottom']

        self.compute_selectivity(division=self.settings['selectivity_division'])

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            newroi = Roi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
            self.rois.append(newroi)

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.compute_spatial_maps()

    def permutate_labels(self, chunk_size=100, randomstate = 'none'):

        # --- re-building laps
        self.laps = []
        self.nlaps = 0
        self.laptypes = [] #add by Hsin-Lun
        self.lapboundtimes = [] #add by Hsin-Lun
        self.incompletelaps = [] #add by Hsin-Lun
        self.silentrois = [] #add by Hsin-Lun
        self.divide_in_laps(self.vrdict, discard_roi=False)

        bag_of_laps = [l.laptype for l in self.laps]
        if randomstate=='none':
            np.random.shuffle(bag_of_laps)
        else:
            randomstate.shuffle(bag_of_laps)
        self.familiar_laps = [lap for lap, ltype in zip(self.laps, bag_of_laps) if ltype=='top']
        self.novel_laps = [lap for lap, ltype in zip(self.laps, bag_of_laps) if ltype=='bottom']

        self.compute_selectivity(division=self.settings['selectivity_division'])

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            newroi = Roi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
            self.rois.append(newroi)

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.compute_spatial_maps()

    def compute_selectivity(self, division=True):
        self.rate_f = np.sum([l.event_length for l in self.familiar_laps], 0)/np.sum([l.lap_length for l in self.familiar_laps])
        self.rate_n = np.sum([l.event_length for l in self.novel_laps], 0)/np.sum([l.lap_length for l in self.novel_laps])
        if division:
            self.selectivity = [ abs(rf - rn)/(rf+rn) for rf, rn in zip(self.rate_f,self.rate_n)]
        else:
            self.selectivity = [ abs(rf - rn) for rf, rn in zip(self.rate_f,self.rate_n)]
        #return [self.rate_f,self.rate_n,self.selectivity]

        self.rate_f_discrete = np.sum([ l.event_count for l in self.familiar_laps],0) / np.sum([ l.duration for l in self.familiar_laps])
        self.rate_n_discrete = np.sum([ l.event_count for l in self.novel_laps],0) / np.sum([ l.duration for l in self.novel_laps])
        if division:
            self.selectivity_discrete = [ abs(rf - rn)/(rf+rn) for rf, rn in zip(self.rate_f_discrete,self.rate_n_discrete)]
        else:
            self.selectivity_discrete = [ abs(rf - rn) for rf, rn in zip(self.rate_f_discrete,self.rate_n_discrete)]

        self.rate_f_continuous = np.empty((self.n_roi))
        self.rate_n_continuous = np.empty((self.n_roi))
        self.selectivity_continuous = np.empty((self.n_roi))
        t_familiar = np.sum([ l.duration for l in self.familiar_laps])
        t_novel = np.sum([ l.duration for l in self.novel_laps])
        for nroi in range(self.n_roi):
            # Do we have more than 1 event at all?
            nevents = np.sum(np.sum((l.S[nroi]>0).astype(np.int)) for l in self.familiar_laps) + \
                      np.sum(np.sum((l.S[nroi]>0).astype(np.int)) for l in self.novel_laps)
            self.rate_f_continuous[nroi] = np.sum(np.sum(l.S[nroi]) for l in self.familiar_laps) / t_familiar
            self.rate_n_continuous[nroi] = np.sum(np.sum(l.S[nroi]) for l in self.novel_laps) / t_novel
            if nevents > 1:
                rf = self.rate_f_continuous[nroi]
                rn = self.rate_n_continuous[nroi]
                if division:
                    self.selectivity_continuous[nroi] = np.abs(rf - rn)/(rf+rn)
                else:
                    self.selectivity_continuous[nroi] = np.abs(rf - rn)
            else:
                self.selectivity_continuous[nroi] = np.nan

        mfr_f = np.mean(np.vstack([l.raster for l in self.familiar_laps]), 0) / (self.settings['discretization_timescale']/1000.)
        mfr_n = np.mean(np.vstack([l.raster for l in self.novel_laps]), 0) / (self.settings['discretization_timescale']/1000.)
        if division:
            self.selectivity_manu = np.abs(mfr_f - mfr_n)/(mfr_f + mfr_n)
        else:
            self.selectivity_manu = np.abs(mfr_f - mfr_n)
        self.selectivity_manu_div = np.abs(mfr_f - mfr_n)/(mfr_f + mfr_n)
        self.selectivity_manu_nodiv = np.abs(mfr_f - mfr_n)
    """
    def compute_baseline_selectivity(self):
        print("start compute baseline selectivity bootstrap("+str(self.settings["bootstrap_chunk_size"])+")")
        self.selectivity_B = compute_bootstrap_selectivity(self, chunk_size = self.settings["bootstrap_chunk_size"], n_bootstrap=self.settings["nbootstrap"])
        return self.selectivity_B
    """

def cut_array(times, data, tstart, tend):
    return times[(times >= tstart) & (times < tend)], data[(times >= tstart) & (times < tend)]

def flip(maps):
    print("### flip maps ###")
    #move.printA("maps",maps)
    #flip_maps = np.zeros((len(maps),len(maps[0])))
    #for flip_map,single_map in zip(flip_maps,maps):
    #    flip_map = np.flip(single_map,0)
    flip_maps = np.flip(maps,1)
    #move.printA("flip_maps",flip_maps)
    return flip_maps

def shuffle_seed(maps,s_seed=10):
    seed(s_seed)
    print("### shuffle maps ###")
    #move.printA("maps",maps)
    shuffle_maps = np.copy(maps)
    #flip_maps = np.zeros((len(maps),len(maps[0])))
    #for flip_map,single_map in zip(flip_maps,maps):
    #    flip_map = np.flip(single_map,0)
    shuffle(shuffle_maps)
    #move.printA("shuffle_maps",shuffle_maps)
    return shuffle_maps

def in_conditions(times, ends):
    for i, (start, end) in enumerate(zip(ends[:-1],ends[1:])):
        if in_period(times[0],start,end) and in_period(times[-1],start,end):
            return i
    return -1

def in_period(time, start, end):
    if DEBUG:
        print("is "+str(time)+" between "+str(start)+" and "+str(end))
    if time < start:
        if DEBUG:
            print("False")
        return False
    elif time > end:
        if DEBUG:
            print("False")
        return False
    else:
        if DEBUG:
            print("True")
        return True

def sigmoid(x, k, x0):
    return 1.0/(1.0+np.exp(-k*(x-x0)))

class ymLap(bgLap):
    def __init__(self, index, laptype, arm, dF_F, S, activations, times, position, speed, events, settings):

        # --- input data requirements
        #assert len(dF_F[0]) == len(times), '----- ! ERROR ! ----- (roi constructor) ----- single roi F and times must have the same size'
        assert len(times) == len(position), '----- ! ERROR ! ----- (roi constructor) ----- times and position must have the same size'
        assert len(times) == len(speed), '----- ! ERROR ! ----- (roi constructor) ----- times and speed must have the same size'
        assert np.shape(dF_F) == np.shape(S), '----- ! ERROR ! ----- (roi constructor) ----- F and S must have the same shape'
        # --- compulsory variable assignment
        N, T = np.shape(dF_F)
        self.index = index
        self.laptype = laptype
        self.arm = arm
        self.n_roi = N
        self.dF_F = dF_F
        self.S = S
        self.event_length = [len(s[s>0])*1.0 for s in S]
        self.lap_length = len(S[0])*1.0
        self.activations = activations
        self.event_count = [len(a) for a in activations]
        self.times = times
        self.duration = times[-1]-times[0]
        self.position = position
        self.speed = speed
        self.events = events
        self.settings = settings
        self.tot_running_time, self.running_mask = compute_running_mask(self, VR=False)
        self.mean_speed = np.mean(self.speed[self.running_mask])
        self.raster = []
        self.rate_maps = []
        self.s_maps = []

        # --- single roi construction
        self.rois = []
        for i in range(N):
            assert(np.sum(self.running_mask) != 0)
            newroi = bgRoi(activations[i], dF_F[i], S[i], times, position, speed, self.running_mask, settings)
            self.rois.append(newroi)

        # --- online computations
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        self.lick_stats = []
        self.compute_lick_stats()
        self.discretize_activations()
        #self.compute_hits()
        self.reward = False#self.lick_stats['number_hits'] > 0
    def compute_spatial_maps(self, bins, vrdict):
        #print("start lap.compute_spatial_maps")
        dt = np.median(np.diff(self.times))
        run_periods, rest_periods = move.run_rest_periods(hspectral.Timeseries(self.speed, dt), self.settings["thr_rest"],
                                                          self.settings["thr_run"], self.settings["run_min_dur"])

        running_mask = laps.in_period(np.arange(len(self.times)), run_periods)
        occupancy = np.histogram(self.position[running_mask], bins)[0] * dt
        occupancy[occupancy==0] = np.nan
        self.occupancy = maps.nanfilter1d(
            occupancy, sigma=self.settings["mapfilter"]/self.settings["binsize"],
            return_nan=self.settings["return_nan_filter"]) # filtering occupancy vector

        ts = hspectral.Timeseries(self.speed, dt)

        for roi in self.rois:
            roi.compute_spatial_map_F(bins, occupancy=self.occupancy, running_mask=running_mask)
            roi.compute_spatial_map_S(bins, occupancy=self.occupancy, running_mask=running_mask, vrdict=vrdict, axiskey='posd')

        self.rate_maps = np.asarray([roi.F_ratemap for roi in self.rois])
        self.s_maps = np.asarray([roi.S_ratemap for roi in self.rois])


class ymSession(bgSession):
    def __init__(self, exp, settings):
        # --- loading data
        self.settings = settings
        vrdict, calciumdict = syncfiles.load_2p_data(exp, settings, isvr=False)
        #print(("vrdict.keys:"+str(list(vrdict.keys()))))

        #move.printA("vrdict['tracktimes']",vrdict['tracktimes'])
        #move.printA("vrdict['frametimes']",vrdict['frametimes'])
        #move.printA("vrdict['posx']",vrdict['posx'])

        speed_lowpass = hspectral.lowpass( hspectral.Timeseries( vrdict['trackspeed'], np.mean( np.diff( vrdict['tracktimes']))), 0.03)
        speed_thres = np.min( speed_lowpass.data) + 0.2 * ( np.max( speed_lowpass.data) - np.min( speed_lowpass.data))

        vrdict['speed_lowpass'] = speed_lowpass.data
        vrdict['speed_thres']   = speed_thres

        vrdict['conditions'] = ['A', 'B', 'C'] #ymaze

        self.cut_periods = exp.cut_periods
        self.vrdict = vrdict
        #move.printA("vrdict",vrdict)
        #move.printA("vrdict.keys()",list(vrdict.keys()))

        #laptypes, lapboundtimes = laps.split_laps_1p(
        #    vrdict['posx_lowpass'], np.median(np.diff(vrdict['frametimes'])),True)
        laptypes, lapboundtimes = laps.split_laps_ymaze(
            vrdict['posx_frames'], vrdict['posy_frames'],np.median(np.diff(vrdict['frametimes'])),cut_center=False)
        self.events = lapboundtimes

        #ymaze findcenter
        center,distance0tomouse, ARM_B = laps.findcenter(vrdict['posx_frames'], vrdict['posy_frames']) #ymaze

        self.center = center #ymaze
        self.vrdict['posd'] = distance0tomouse #ymaze
        self.ARM_B = ARM_B #ymaze

        #print('posx',type(vrdict['posx']))
        #move.printA('posx',vrdict['posx'])
        #print('posd',type(vrdict['posd']))
        #move.printA('posd',vrdict['posd'])

        # --- synchronizing times
        Tmax = min([vrdict['frametimes'][-1], vrdict['tracktimes'][-1]])
        Tmin = max([vrdict['frametimes'][0], vrdict['tracktimes'][0]])
        keep_1p = (vrdict['frametimes'] <= Tmax) & (vrdict['frametimes'] >= Tmin)
        keep_vr = (vrdict['tracktimes'] <= Tmax) & (vrdict['tracktimes'] >= Tmin)

        # --- assigning variables
        self.session_number = exp.session_number
        # data_hio = exp.to_haussio(mc=True)
        dirname_comp = os.path.join(exp.root_path, exp.fn2p)
        self.session_name = os.path.basename(dirname_comp)
        self.times = vrdict['frametimes'][keep_1p]
        self.speed = vrdict['trackspeed'][keep_1p]
        pos_func = interp1d(vrdict['tracktimes'], self.vrdict['posd']) #ymaze
        self.position = pos_func(self.times)

        self.dF_F = self.process_dF_F(calciumdict, vrdict, keep_1p)
        self.n_roi, T = np.shape(self.dF_F)
        self.name_roi = np.arange(self.n_roi)
        self.process_S(calciumdict,settings,keep_1p)
        self.tot_running_time, self.running_mask = compute_running_mask(self)
        self.mean_speed = np.mean(self.speed[self.running_mask])

        # --- finding events on a session scale
        self.activations = self.find_activations()
        #self.activations = self.find_activations_old(usemap=settings['activation_map'])


        # --- eliminating silent and bad rois
        if self.settings['std'] == 0.0:
            self.kept_cells = self.eliminate_silent_rois(exp.rois_eliminate)
        #print("exp.rois_eliminate: ",exp.rois_eliminate)
        #print("check_S "," kept_cells ",self.kept_cells)
        self.spatial_cells_S = []
        self.spatial_cells_F = []
        self.spatial_cells_S_IN = []
        self.spatial_cells_S_OUT = []
        self.spatial_cells_S_DUO = []
        self.spatial_cells_F_IN = []
        self.spatial_cells_F_OUT = []
        self.spatial_cells_F_DUO = []
        self.C_FF_F = []; self.C_NN_F = []; self.C_FN_F = []; self.C_Flip_F = []
        self.C_FF_B_F = []; self.C_NN_B_F = []; self.C_FN_B_F = []
        self.C_FF_S = []; self.C_NN_S = []; self.C_FN_S = []; self.C_Flip_S = []
        self.C_FF_B_S = []; self.C_NN_B_S = []; self.C_FN_B_S = []
        # --- building laps
        self.laps = []
        self.nlaps = 0
        self.laptypes = [] #add by Hsin-Lun
        self.lapboundtimes = [] #add by Hsin-Lun
        self.incompletelaps = [] #add by Hsin-Lun
        self.silentrois = [] #add by Hsin-Lun
        if self.settings['std'] == 0.0:
            self.divide_in_laps(self.vrdict,discard_roi=False)
            assert len(self.silentrois) == 0, \
            "silentrois should be empty, now it is "+str(self.silentrois)+" for "+self.session_name
        else:
            self.divide_in_laps(self.vrdict,discard_roi=True)

        self.familiar_laps = [l for l in self.laps if l.laptype=='top'] #inbound
        self.novel_laps = [l for l in self.laps if l.laptype=='bottom'] #outbound

        self.compute_selectivity(division=self.settings['selectivity_division'])

        # --- building rois
        self.rois = []
        for i in range(self.n_roi):
            if i not in self.silentrois:
                newroi = bgRoi(self.activations[i], self.dF_F[i], self.S[i], self.times, self.position, self.speed, self.running_mask, self.settings)
                self.rois.append(newroi)

        if DEBUG:
            print("check_S",len(self.dF_F),len(self.S),len(self.S_noisy))
            print("self.settings['std'] == ",self.settings['std'])

        # eliminate no-activity ROI
        if DEBUG:
            print("self.rois",len(self.rois))
        self.n_roi_all = self.n_roi
        self.n_roi = len(self.rois)
        if DEBUG:
            print(self.n_roi_all, "=>" , self.n_roi)
            print(len(self.dF_F),len(self.S),len(self.S_noisy),len(self.activations))
        self.dF_F = self.dF_F[self.roi_mask]
        self.S_noisy = self.S_noisy[self.roi_mask]
        self.S = self.S[self.roi_mask]
        self.activations = [self.activations[r] for r in range(self.n_roi_all) if r not in self.silentrois]
        if DEBUG:
            print("=>",len(self.dF_F),len(self.S),len(self.S_noisy),len(self.activations))

        # --- computing variables
        self.rate_vector = np.asarray([r.rate for r in self.rois])
        #self.reward_rate = np.sum([l.reward for l in self.laps])/float(self.nlaps)
        #self.reward_rate_N = np.sum([l.reward for l in self.novel_laps])/float(len(self.novel_laps))
        #self.reward_rate_F = np.sum([l.reward for l in self.familiar_laps])/float(len(self.familiar_laps))

        self.compute_lick_stats()
        self.compute_spatial_maps()
        self.compute_spatial_zscores(VR=False)

        #self.behavior_F, self.behavior_N, self.behavior_F_false, self.behavior_N_false = compute_reward_stats(exp, self.settings)

        #self.false_stop_F = np.asarray([l.false_stop_ratio for l in self.familiar_laps])
        #self.false_stop_N = np.asarray([l.false_stop_ratio for l in self.novel_laps])
        #self.false_lick_F = np.asarray([l.false_lick_ratio for l in self.familiar_laps])
        #self.false_lick_N = np.asarray([l.false_lick_ratio for l in self.novel_laps])
        #self.false_speed_ratio_F = np.asarray([l.false_speed_ratio for l in self.familiar_laps])
        #self.false_speed_ratio_N = np.asarray([l.false_speed_ratio for l in self.novel_laps])

        #self.true_stop_F = np.asarray([l.true_stop_ratio for l in self.familiar_laps])
        #self.true_stop_N = np.asarray([l.true_stop_ratio for l in self.novel_laps])
        #self.true_lick_F = np.asarray([l.true_lick_ratio for l in self.familiar_laps])
        #self.true_lick_N = np.asarray([l.true_lick_ratio for l in self.novel_laps])
        #self.true_speed_ratio_F = np.asarray([l.true_speed_ratio for l in self.familiar_laps])
        #self.true_speed_ratio_N = np.asarray([l.true_speed_ratio for l in self.novel_laps])

        # / --- end of constructor --- / #

    def divide_in_laps(self, vrdict, discard_roi=False) :
        if not self.laptypes and not self.lapboundtimes:
            if DEBUG:
                print("run split_laps_ymaze")
            gratings, boundary_times = laps.split_laps_ymaze(
                vrdict['posx_frames'], vrdict['posy_frames'],np.median(np.diff(vrdict['frametimes'])),cut_center=False)

            if DEBUG:
                move.printA('gratings:',gratings)
            lp = np.array([ g[0] for g in gratings ])
            if DEBUG:
                move.printA('laptypes(bound-wise):',lp)
            lp[lp=='outbound']='bottom'
            lp[lp=='inbound']='top'
            self.laptypes = lp.tolist()
            self.arms = [ g[1] for g in gratings ]
            self.infos = [ g[2:] for g in gratings ]
            self.lapboundtimes = boundary_times
            if DEBUG:
                move.printA('laptypes(bottome/top):',self.laptypes)
                move.printA('arms:',self.arms)
                move.printA('infos:',self.infos)
                move.printA('boundtimes',self.lapboundtimes)

            #print stop
        track_length = np.max(vrdict['posd'])-np.min(vrdict['posd'])

        for lap_trace, (lapstart, lapend, stimulus, arm, info) in enumerate(zip(self.lapboundtimes[:-1], self.lapboundtimes[1:], self.laptypes, self.arms, self.infos)):
            istart = np.where(vrdict['tracktimes'].squeeze() >= lapstart)[0][0]
            iend = np.where(vrdict['tracktimes'].squeeze() > lapend)[0]
            if len(iend) == 0:
                iend = None
            else:
                iend = iend[0]
            lapspeed_lowpass = vrdict['speed_lowpass'][istart:iend]
            lapspeed = vrdict['trackspeed'][istart:iend]
            lappostimes = vrdict['tracktimes'].squeeze()[istart:iend]
            lapposd = vrdict['posd'][istart:iend]
            if len(lapposd):
                track_fraction = (np.max(lapposd)-np.min(lapposd)) / track_length
            else:
                track_fraction = 0
            #print("max(lapposx)",np.max(lapposx),"min(lapposx)",np.min(lapposx),"track_length",track_length)
            if track_fraction < self.settings['min_lap_fraction_ymaze']:
                if DEBUG:
                    print("Lap {0:d} :Skipping partial lap crossing, fraction {1:.2f}".format(lap_trace,track_fraction))
                self.incompletelaps.append(lap_trace)
            elif max(lappostimes)-min(lappostimes) <= 1:
                if DEBUG:
                    print("Lap {0:d} :Skipping lap crossing that took less than 1s, might be concatenated?".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            elif len(np.where(lapspeed == 0)[0])>1:
                if DEBUG:
                    print("Lap {0:d} :Skipping lap with zero speed".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            elif arm == 'arm not found':
                if DEBUG:
                    print("Lap {0:d}: Arm not found. Just skip it".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            elif 'overrun' in info:
                if DEBUG:
                    print("Lap {0:d}: Overrun. Just skip it".format(lap_trace))
                self.incompletelaps.append(lap_trace)
            #elif len(np.where(lapspeed_lowpass < vrdict['speed_thres'])[0])>1:
            #    print("Skipping lap with unpassionate running")
            #    self.incompletelaps.append(lap_trace)
        self.incompletelaps.append(len(self.lapboundtimes) - 1)
        if DEBUG:
            print("incomplete laps")
            print(self.incompletelaps)

        #boundary_times = boundary_times[1:]
        #gratings = gratings[1:] # discarding the first and the last lap (incomplete)

        self.nlaps = len(boundary_times) - 1
        out_index = 0
        in_index = 0

        if discard_roi and (self.settings['std'] > 0.0):
            for r in range(self.n_roi):
                silent = True
                #print("roi(",r,")")
                for i in range(self.nlaps):
                    #print("lap(",i,")")
                    if i not in self.incompletelaps:
                        time_mask = (self.times > boundary_times[i] + self.settings["lap_split_offset"]) & (self.times < boundary_times[i+1] - self.settings["lap_split_offset"])
                        lap_activations = self.find_lap_activations(time_mask)
                        #print("lap_activations(r=",r,",i=",i,"):",lap_activations)
                        if lap_activations[r]:
                            silent = False
                            break
                if silent:
                    self.silentrois.append(r)

            if DEBUG:
                print("silentrois",self.silentrois)

        for i in range(self.nlaps):
            if i not in self.incompletelaps:
                time_mask = (self.times > boundary_times[i] + self.settings["lap_split_offset"]) & (self.times < boundary_times[i+1] - self.settings["lap_split_offset"])
                #lap_events = [e for e in vrdict['evlist'] if boundary_times[i] + self.settings["lap_split_offset"] < (e.time * 1000.) < boundary_times[i+1] - self.settings["lap_split_offset"]]
                lap_activations = self.find_lap_activations(time_mask)
                laptype = self.laptypes[i]
                arm = self.arms[i]
                info = self.infos[i]

                if laptype=='bottom':#noval
                    index = out_index
                    out_index += 1
                if laptype=='top':#familiar
                    index = in_index
                    in_index += 1

                array_index = np.arange(self.n_roi)

                roi_mask = [a not in self.silentrois for a in array_index]

                self.roi_mask = roi_mask

                #print(stop)
                if discard_roi:
                    new_lap = ymLap(
                        index=index,
                        laptype=laptype,
                        arm = arm,
                        dF_F=self.dF_F[roi_mask][:, time_mask],
                        S=self.S[roi_mask][:, time_mask],
                        activations=[lap_activations[a] for a in array_index if a not in self.silentrois], #lap_activations,
                        times=self.times[time_mask],
                        position=self.position[time_mask],
                        speed=self.speed[time_mask],
                        events=[],#lap_events,
                        settings=self.settings)
                else:
                    new_lap = ymLap(
                        index=index,
                        laptype=laptype,
                        arm = arm,
                        dF_F=self.dF_F[:, time_mask],
                        S=self.S[:, time_mask],
                        activations=lap_activations,
                        times=self.times[time_mask],
                        position=self.position[time_mask],
                        speed=self.speed[time_mask],
                        events=[],#lap_events,
                        settings=self.settings)

                self.laps.append(new_lap)
            else:
                self.nlaps -= 1

    def compute_spatial_correlations_conditions(self, force = False, maptype='F'):

        conds = ['A', 'B', 'C']

        C_FF_A = [[] for i in range(self.n_roi)]
        C_NN_A = [[] for i in range(self.n_roi)]
        C_FN_A = [[] for i in range(self.n_roi)]
        C_Flip_A = [[] for i in range(self.n_roi)]
        C_FF_B = [[] for i in range(self.n_roi)]
        C_NN_B = [[] for i in range(self.n_roi)]
        C_FN_B = [[] for i in range(self.n_roi)]
        C_Flip_B = [[] for i in range(self.n_roi)]
        C_FF_C = [[] for i in range(self.n_roi)]
        C_NN_C = [[] for i in range(self.n_roi)]
        C_FN_C = [[] for i in range(self.n_roi)]
        C_Flip_C = [[] for i in range(self.n_roi)]

        C_FF = [C_FF_A,C_FF_B,C_FF_C]
        C_NN = [C_NN_A,C_NN_B,C_NN_C]
        C_FN = [C_FN_A,C_FN_B,C_FN_C]
        C_Flip = [C_Flip_A,C_Flip_B,C_Flip_C]

        #move.printA('C_FF cond',C_FF)
        #move.printA('C_NN cond',C_NN)
        #move.printA('C_FN cond',C_FN)
        #move.printA('C_Flip cond',C_Flip)

        print("---- O> Computing spatial correlations"); t0 = time.time()
        print("maptype == "+maptype)

        for n in range(self.n_roi):
            for i in range(len(self.familiar_laps)):
                for j in range(i+1, len(self.familiar_laps)):
                    cond_i = conds.index(self.familiar_laps[i].arm[-1])
                    cond_j = conds.index(self.familiar_laps[j].arm[-1])
                    if cond_i==cond_j and cond_i != -1:
                        if maptype == 'F':
                            c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.familiar_laps[j].rate_maps[n], self.settings, periodic=False)
                        elif maptype == 'S':
                            c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.familiar_laps[j].s_maps[n], self.settings, periodic=False)
                        C_FF[cond_i][n].append(c)

            for i in range(len(self.novel_laps)):
                 for j in range(i+1, len(self.novel_laps)):
                    cond_i = conds.index(self.novel_laps[i].arm[-1])
                    cond_j = conds.index(self.novel_laps[j].arm[-1])
                    if cond_i==cond_j and cond_i != -1:
                        if maptype == 'F':
                            c = ratemap_correlation(self.novel_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                        elif maptype == 'S':
                            c = ratemap_correlation(self.novel_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                        C_NN[cond_i][n].append(c)


            for i in range(len(self.familiar_laps)):
                for j in range(len(self.novel_laps)):
                    cond_i = conds.index(self.familiar_laps[i].arm[-1])
                    cond_j = conds.index(self.novel_laps[j].arm[-1])
                    if cond_i==cond_j and cond_i != -1:
                        if maptype == 'F':
                            c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], self.novel_laps[j].rate_maps[n], self.settings, periodic=False)
                        elif maptype == 'S':
                            c = ratemap_correlation(self.familiar_laps[i].s_maps[n], self.novel_laps[j].s_maps[n], self.settings, periodic=False)
                        C_FN[cond_i][n].append(c)

            for i in range(len(self.familiar_laps)):
                for j in range(len(self.novel_laps)):
                    cond_i = conds.index(self.familiar_laps[i].arm[-1])
                    cond_j = conds.index(self.novel_laps[j].arm[-1])
                    if cond_i==cond_j and cond_i != -1:
                        if maptype == 'F':
                            c = ratemap_correlation(self.familiar_laps[i].rate_maps[n], np.flip(self.novel_laps[j].rate_maps[n]), self.settings, periodic=False)
                        elif maptype == 'S':
                            c = ratemap_correlation(self.familiar_laps[i].s_maps[n], np.flip(self.novel_laps[j].s_maps[n]), self.settings, periodic=False)
                        C_Flip[cond_i][n].append(c)
            print("---- <O Computed spatial correlations, time elapsed = %.1f s" % (time.time() - t0))
        return C_FF, C_NN, C_FN, C_Flip
