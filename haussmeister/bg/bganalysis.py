import numpy as np
import matplotlib.pyplot as plt
import nice
from scipy.stats import pearsonr
import seaborn as sns
import pandas as pd
import pickle
from math import ceil, floor
from bgclasses import ratemap_correlation
from bgclasses import flip
from scipy.stats import ttest_ind as ttest
from scipy.stats import wilcoxon as wilcoxon
import bottleneck as bn
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from bgvisualize import labels_novel
from bgvisualize import pov
import bgvisualize as visualize
import move

DEBUG=False

def nanpearsonr(x, y, nmin=2):
    mask = (np.isnan(x) == 0) & (np.isnan(y) == 0)
    if len(x[mask]) < nmin and len(y[mask]) < nmin:
        return [np.nan,]
    else:
        return pearsonr(x[mask], y[mask])

# --- PoV analysis --- #

def PoV_analysis(map1, map2, region1, region2, n_bootstrap):
    assert(map1.shape[0] == map2.shape[0])
    nbins = map1.shape[0]
    PoV = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            mask = (np.isnan(map1[i]) == 0) & (np.isnan(map2[j]) == 0)
            PoV[i, j] = pearsonr(map1[i][mask], map2[j][mask])[0]
    PoV_bs = np.copy(PoV)
    correlation = bn.nanmean(PoV[region1, :][:, region2])
    correlations_bs = []
    for n in range(n_bootstrap):
        idx = np.random.permutation(np.arange(nbins))
        PoV_bs = PoV[idx, :]
        correlations_bs.append(PoV_bs[region1, :][:, region2])
    correlations_bs = np.asarray(correlations_bs)
    p = bn.nanmean(correlations_bs > correlation)
    return p


def PoV_matrix(map1, map2, nmin=2):
    assert(map1.shape[0] == map2.shape[0])
    nbins = map1.shape[0]
    PoV = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            mask = (np.isnan(map1[i]) == 0) & (np.isnan(map2[j]) == 0)
            if len(map1[i][mask]) < nmin and len(map2[j][mask]) < nmin:
                PoV[i, j] = np.nan
            else:
                PoV[i, j] = pearsonr(map1[i][mask], map2[j][mask])[0]
    return PoV


def plot_PoV_matrix(ax, PoV, title, xlabel, ylabel):
    pPoV = ax.pcolor(PoV)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(pPoV, ax=ax)
    ax.set_aspect(1)


def PoV_matrix_session(session, only_spatial=False, savepath='none', plot=True, nmin=2):#changed only_spatial to FALSE (SS) to plot all cells; change to TRUE for PC cells
    familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    familiar_maps_odd = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    familiar_maps_even = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    nfamiliar = len(session.familiar_laps)
    nfamiliar_odd = int(nfamiliar/2) + nfamiliar%2
    nfamiliar_even = nfamiliar-nfamiliar_odd
    novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    novel_maps_odd = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    novel_maps_even = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    #flip_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    nnovel = len(session.novel_laps)
    nnovel_odd = int(nnovel/2) + nnovel%2
    nnovel_even = nnovel-nnovel_odd

    for i,l in enumerate(session.familiar_laps):
        familiar_maps += l.rate_maps / float(nfamiliar)
        if i%2 == 0:
            familiar_maps_even += l.rate_maps / float(nfamiliar_even)
        else:
            familiar_maps_odd += l.rate_maps / float(nfamiliar_odd)

    for i,l in enumerate(session.novel_laps):
        novel_maps += l.rate_maps / float(len(session.novel_laps))
        if i%2 == 0:
            novel_maps_even += l.rate_maps / float(nnovel_even)
        else:
            novel_maps_odd += l.rate_maps / float(nnovel_odd)

    if only_spatial:
        familiar_maps = familiar_maps[session.spatial_cells]
        familiar_maps_odd = familiar_maps_odd[session.spatial_cells]
        familiar_maps_even = familiar_maps_even[session.spatial_cells]
        novel_maps = novel_maps[session.spatial_cells]
        novel_maps_odd = novel_maps_odd[session.spatial_cells]
        novel_maps_even = novel_maps_even[session.spatial_cells]

    familiar_maps = np.transpose(familiar_maps)
    familiar_maps_odd = np.transpose(familiar_maps_odd)
    familiar_maps_even = np.transpose(familiar_maps_even)
    novel_maps = np.transpose(novel_maps)
    novel_maps_odd = np.transpose(novel_maps_odd)
    novel_maps_even = np.transpose(novel_maps_even)
    #flip_maps = flip(novel_maps)##
    #flip_maps = np.transpose(flip_maps)

    FN = PoV_matrix(familiar_maps, novel_maps, nmin=nmin)
    FF_odd_even = PoV_matrix(familiar_maps_odd, familiar_maps_even, nmin=nmin)
    NN_odd_even = PoV_matrix(novel_maps_odd, novel_maps_even, nmin)
    FF = PoV_matrix(familiar_maps, familiar_maps, nmin)
    NN = PoV_matrix(novel_maps, novel_maps, nmin)
    #FLIP = PoV_matrix(familiar_maps, flip_maps, nmin=nmin)##


    if savepath != 'none':
        if savepath.find(".pdf") != -1:
            savepath_trunk = savepath[:savepath.find(".pdf")]
        else:
            savepath_trunk = savepath

        if plot:
            f, ax = plt.subplots(3, 2, figsize=(9, 11))
            plot_PoV_matrix(ax[0, 0], FN, 'PoV matrix FN', 'Position in N', 'Position in F')
            plot_PoV_matrix(ax[1, 0], FF_odd_even, 'PoV matrix FF even-odd', 'Position in F, even', 'Position in F, odd')
            plot_PoV_matrix(ax[1, 1], NN_odd_even, 'PoV matrix NN even-odd', 'Position in N, even', 'Position in N, odd')
            plot_PoV_matrix(ax[2, 0], FF, 'PoV matrix FF all', 'Position in F', 'Position in F')
            plot_PoV_matrix(ax[2, 1], NN, 'PoV matrix NN all', 'Position in N', 'Position in N')
            #plot_PoV_matrix(ax[2, 2], FLIP, 'PoV matrix FLIP all', 'Position in N', 'Position in N')##

            f.savefig(savepath_trunk + ".pdf")

    return FN, FF_odd_even, NN_odd_even, FF, NN


# --- place cells analysis --- #

def identify_place_fields(session, min_persistence=0.1, debug=False, savename=''):

    familiar_fields = []
    novel_fields = []
    # for i in range(session.n_roi):
    #     for lap in session.familiar_laps:
    #         np.random.shuffle(lap.rois[i].F_ratemap)

    for i in range(session.n_roi):
        if debug:
            print('\n\n------------ cell %u -----------\n' % i)
        f_maps = np.asarray([lap.rois[i].F_ratemap for lap in session.familiar_laps])
        n_maps = np.asarray([lap.rois[i].F_ratemap for lap in session.novel_laps])

        f_fields = tank_place_fields(f_maps, min_persistence=min_persistence, debug=debug)
        n_fields = tank_place_fields(n_maps, min_persistence=min_persistence, debug=debug)

        # p_f = 0
        # p_n = 0
        # for n in range(100):
        #     if len(tank_place_fields(f_maps, min_persistence=min_persistence, shuffle=True)):
        #         p_f += 1./100
        #     if len(tank_place_fields(n_maps, min_persistence=min_persistence, shuffle=True)):
        #         p_n += 1./100
        #
        # if debug:
        #     print("\nCell %u, p_familiar: %.3f, p_novel: %.3f\n" % (i, p_f, p_n))
        # if p_f > 0.05:
        #     f_fields = []
        # if p_n > 0.05:
        #     n_fields = []

        familiar_fields.append(f_fields)
        novel_fields.append(n_fields)

    if debug:
        f, axs = plt.subplots(session.n_roi, 1, figsize=(5, 1.5*session.n_roi), sharex=True)
        for i in range(session.n_roi):
            ax = axs[i]
            for lap in session.familiar_laps:
                ax.plot(lap.rois[i].F_ratemap, linewidth=1, color=pltcolors[0], alpha=0.5)

            for lap in session.novel_laps:
                ax.plot(lap.rois[i].F_ratemap, linewidth=1, color=pltcolors[1], alpha=0.5)

            ys = ax.get_ylim()

            mean_fam = bn.nanmean(np.asarray([lap.rois[i].F_ratemap for lap in session.familiar_laps]), 0)
            mean_nov = bn.nanmean(np.asarray([lap.rois[i].F_ratemap for lap in session.novel_laps]), 0)
            #ax.plot(mean_fam * ys[1]/np.nanmax(mean_fam)/2, linewidth=2, color=pltcolors[0], alpha=0.5, linestyle='-')
            #ax.plot(mean_nov * ys[1]/np.nanmax(mean_nov)/2, linewidth=2, color=pltcolors[1], alpha=0.5, linestyle='-')

            for field in familiar_fields[i]:
                ax.fill_between(field, ys[1]*np.ones(len(field)), color=pltcolors[0], alpha=0.1)
            for field in novel_fields[i]:
                ax.fill_between(field, ys[1]*np.ones(len(field)), color=pltcolors[1], alpha=0.1)
            sns.despine(ax=ax)
            ax.set_title('Cell %u, is_spatial = %u' % (i, session.spatial_cells[i]))
        f.savefig('plots/review/place_fields/%s_.pdf' % savename)
        plt.close(f)


    return familiar_fields, novel_fields


def tank_place_fields(all_maps, place_field_min_size=3, min_persistence = 0.1, debug=False, shuffle=False):
    # From Dombeck, Tank 2010:
    # Potential place fields were first identified as contiguous regions of this plot
    # in which all of the points were greater than 25% of the difference between the ----------> for us 50%
    # peak dF/F value (for all 80 bins) and the baseline value (mean of the lowest 20 out of 80 dF/F values).
    lap_maps = np.copy(all_maps)
    if shuffle:
        for m in lap_maps:
            np.random.shuffle(m)

    map = np.mean(lap_maps, 0)
    peak = np.nanmax(map)
    baseline = bn.nanmean(map[map <= np.percentile(map[np.isnan(map)==0], 50)])

    if debug:
        print("peak: %.5f, baseline: %.5f, percentile: %.3f" % (peak, baseline, np.percentile(map[np.isnan(map)==0], 50)))

    threshold = baseline + (peak - baseline)/2.
    active_region = map > threshold

    if debug:
        print("fraction of active regions: %.1f" % (np.mean(active_region)))


    # These potential place field regions then had to satisfy the following criteria:
    # 1. The field must be >18cm in width -> [for us a minimum of 3 bins]
    fields = find_contiguous_regions(active_region, place_field_min_size)

    if debug:
        print("1. Fields:", fields)


    # 2. The field must have one value of at least 10% mean dF/F
    if len(fields):
        remove_fields = []
        threshold = bn.nanmean(map) * 0.1
        for field in fields:
            if np.sum(map[field] > threshold) == 0:
                remove_fields.append(field)
        for field in remove_fields:
            fields.remove(field)

    if debug:
        print("2. Fields:", fields)

    # 3. The mean in field dF/F value must be >3 times the mean out of field dF/F value
    if len(fields):
        remove_fields = []
        in_field = np.hstack(fields)
        out_of_field = []
        for x in range(len(map)):
            if (x in in_field)==False:
                out_of_field.append(x)

        threshold = 3 * bn.nanmean(map[out_of_field])

        for field in fields:
            if debug:
                print(field, bn.nanmean(map[field]), threshold)
            if bn.nanmean(map[field]) < threshold:
                remove_fields.append(field)
        for field in remove_fields:
            fields.remove(field)

    if debug:
        print("3. Fields:", fields)

    # 4. Significant calcium transients must be present >30% of the time the mouse spent in the place field
    min_persistence = max([3./len(all_maps), min_persistence])
    if len(fields):
        remove_fields = []
        in_field = np.hstack(fields)
        out_of_field = []
        for x in range(len(map)):
            if (x in in_field)==False:
                out_of_field.append(x)

        for field in fields:
            persistence = 0
            for lap_map in lap_maps:
                threshold = 3 * bn.nanmean(lap_map[out_of_field])
                if np.mean(lap_map[field]) > threshold:
                    persistence += 1./len(lap_maps)
            if persistence < min_persistence:
                remove_fields.append(field)
            if debug:
                print("persistence: %.2f" % persistence)
        for field in remove_fields:
            fields.remove(field)
    if debug:
        print("4. Fields:", fields)

    return fields


def find_contiguous_regions(map, min_size=3):
    fields = []
    # find first active region
    start = -1
    for x in range(len(map)):
        if map[x]:
            start = x
            break
    if start > -1:
        size = 0
        for x in range(start, len(map)):
            if map[x]:
                size +=1
            else:
                region = range(start, x)
                if len(region) >= min_size:
                    fields.append(region)
                start = x+1
        if map[-1]:
            if len(map)-start >= min_size:
                fields.append(list(range(start, len(map))))
    return fields



# --- rate analysis functions --- #

def running_mean_std(array, array_sort, window):
    index = np.argsort(array_sort)
    datay = array[index]
    datax = array_sort[index]
    x = []
    y = []
    yerr = []
    for i in range(len(array)-window):
        x.append(bn.nanmean(datax[i:i+window]))
        y.append(bn.nanmean(datay[i:i+window]))
        yerr.append(np.nanstd(datay[i:i+window])/np.sqrt(window-1))
    return x, y, yerr



def firing_variances(session):
    N = session.n_roi
    variances_f = []
    variances_n = []
    for i in range(N):
        rates = []
        for l in session.familiar_laps:
            rates.append(l.rate_vector[i])
        variances_f.append(np.sum(np.asarray(rates)>0) / float(len(rates)))
    for i in range(N):
        rates = []
        for l in session.novel_laps:
            rates.append(l.rate_vector[i])
        variances_n.append(np.sum(np.asarray(rates)>0) / float(len(rates)))
    return variances_f, variances_n

def compare_laps_rate_vectors(session, savepath='none', region = 'DG'):
    FF = []
    for i in range(len(session.familiar_laps)):
        for j in range(i+1, len(session.familiar_laps)):
            FF.append(nanpearsonr(session.familiar_laps[i].rate_vector, session.familiar_laps[j].rate_vector)[0])

    NN = []
    for i in range(len(session.novel_laps)):
        for j in range(i+1, len(session.novel_laps)):
            NN.append(nanpearsonr(session.novel_laps[i].rate_vector, session.novel_laps[j].rate_vector)[0])

    FN = []
    for i in range(len(session.familiar_laps)):
        for j in range(len(session.novel_laps)):
            FN.append(nanpearsonr(session.familiar_laps[i].rate_vector, session.novel_laps[j].rate_vector)[0])
    visualize.box_comparison_three(FF, FN, NN, '||| - |||', '||| - %s' % labels_novel[region], '%s - %s' % (labels_novel[region], labels_novel[region]), 'lap $\\vec{r}$ correlation')
    sns.despine()
    plt.ylim([-0.3, 1.0])
    plt.axhline([0], color = 'k', linestyle='--')
    if savepath != 'none':
        plt.savefig(savepath)
        plt.close()
    return FF, NN, FN

def compare_laps_rate_vectors_time(session, savepath, region = 'DG', maxdist=8):
    FF = {i : [] for i in range(1, len(session.familiar_laps))}

    for i in range(len(session.familiar_laps)):
        for j in range(i+1, len(session.familiar_laps)):
            d = np.abs(session.familiar_laps[i].index - session.familiar_laps[j].index)
            FF[d].append(nanpearsonr(session.familiar_laps[i].rate_vector, session.familiar_laps[j].rate_vector)[0])

    NN = {i : [] for i in range(1, len(session.novel_laps))}
    for i in range(len(session.novel_laps)):
        for j in range(i+1, len(session.novel_laps)):
            d = np.abs(session.novel_laps[i].index - session.novel_laps[j].index)
            NN[d].append(nanpearsonr(session.novel_laps[i].rate_vector, session.novel_laps[j].rate_vector)[0])

    FN = {i : [] for i in range(1, len(session.laps))}
    for i in range(len(session.laps)):
        for j in range(i+1, len(session.laps)):
            if session.laps[i].laptype != session.laps[j].laptype:
                d = np.abs(j-i)
                FN[d].append(nanpearsonr(session.laps[i].rate_vector, session.laps[j].rate_vector)[0])

    means_FF, means_NN, means_FN = visualize.plot_corr_timedist(FF, NN, FN, maxdist, region, savepath=savepath)

    return means_FF, means_NN, means_FN

def compare_laps_rate_vectors_timearrow(session, savepath, region = 'DG', width = 1):

    FF = []
    NN = []
    FN = []

    for i in range(session.nlaps-1):
        if session.laps[i].laptype != session.laps[i+1].laptype:
            FN.append([i, nanpearsonr(session.laps[i].rate_vector, session.laps[i+1].rate_vector)[0]])
        if session.laps[i].laptype == 'vertical' and session.laps[i+1].laptype == 'vertical':
            FF.append([i, nanpearsonr(session.laps[i].rate_vector, session.laps[i+1].rate_vector)[0]])
        if session.laps[i].laptype == 'oblique' and session.laps[i+1].laptype == 'oblique':
            NN.append([i, nanpearsonr(session.laps[i].rate_vector, session.laps[i+1].rate_vector)[0]])

    return FF, NN, FN

def compare_laps_reference_vector(session):
    half_F = int(len(session.familiar_laps)/2)
    half_N = int(len(session.novel_laps)/2)
    reference_vector_F = np.zeros(session.n_roi)
    reference_vector_N = np.zeros(session.n_roi)


    for n in range(session.n_roi):
        tot_activations = 0
        tot_time = 0
        for i in range(half_F):
            tot_activations += session.familiar_laps[i].rois[n].n_activations
            tot_time += session.familiar_laps[i].tot_running_time
        if tot_time == 0:
            tot_time = np.nan
        reference_vector_F[n] = tot_activations/tot_time

        tot_activations = 0
        tot_time = 0
        for i in range(half_N):
            tot_activations += session.novel_laps[i].rois[n].n_activations
            tot_time += session.novel_laps[i].tot_running_time
        if tot_time == 0:
            tot_time = np.nan
        reference_vector_N[n] = tot_activations/tot_time

    FF = []
    FN = []
    for i in range(half_F+1, len(session.familiar_laps)):
        assert len(session.familiar_laps[i].rate_vector) == len(reference_vector_F), \
              "some problem here with "+session.session_name \
              +"\n => i.rate_vector("+str(len(session.familiar_laps[i].rate_vector))+")\n" \
              +str(session.familiar_laps[i].rate_vector) \
              +"\n => refe_vector_F("+str(len(reference_vector_F))+")\n" \
              +str(reference_vector_F)+"\nn_roi:"+str(session.n_roi) \
              +"\nlen of rois:"+str(len(session.rois))
        FF.append(nanpearsonr(session.familiar_laps[i].rate_vector, reference_vector_F)[0])
        FN.append(nanpearsonr(session.familiar_laps[i].rate_vector, reference_vector_N)[0])

    NN = []
    NF = []
    for i in range(half_N+1, len(session.novel_laps)):
        NF.append(nanpearsonr(session.novel_laps[i].rate_vector, reference_vector_F)[0])
        NN.append(nanpearsonr(session.novel_laps[i].rate_vector, reference_vector_N)[0])

    return FF, FN, NN, NF

def compare_laps_reward(session, savepath, min_data, region = 'DG'):
    reward_familiar = [l for l in session.laps if l.reward and l.laptype == "vertical"]
    nonreward_familiar = [l for l in session.laps if l.reward==0 and l.laptype == "vertical"]
    reward_novel = [l for l in session.laps if l.reward and l.laptype == "oblique"]
    nonreward_novel = [l for l in session.laps if l.reward==0 and l.laptype == "oblique"]

    corr_rew_fam_nov = np.asarray([nanpearsonr(l1.rate_vector, l2.rate_vector)[0] for l1 in reward_familiar for l2 in reward_novel])
    corr_nonrew_fam_nov = np.asarray([nanpearsonr(l1.rate_vector, l2.rate_vector)[0] for l1 in nonreward_familiar for l2 in nonreward_novel])

    corr_rew_fam_fam = [nanpearsonr(reward_familiar[i].rate_vector, reward_familiar[j].rate_vector)[0] for i in range(len(reward_familiar)) for j in range(i+1, len(reward_familiar))]
    corr_nonrew_fam_fam = [nanpearsonr(nonreward_familiar[i].rate_vector, nonreward_familiar[j].rate_vector)[0] for i in range(len(nonreward_familiar)) for j in range(i+1, len(nonreward_familiar))]

    corr_rew_nov_nov = [nanpearsonr(reward_novel[i].rate_vector, reward_novel[j].rate_vector)[0] for i in range(len(reward_novel)) for j in range(i+1, len(reward_novel))]
    corr_nonrew_nov_nov = [nanpearsonr(nonreward_novel[i].rate_vector, nonreward_novel[j].rate_vector)[0] for i in range(len(nonreward_novel)) for j in range(i+1, len(nonreward_novel))]

    enough_data_fam = min([len(nonreward_familiar), len(reward_familiar)]) >= min_data
    enough_data_nov = min([len(nonreward_novel), len(reward_novel)]) >= min_data

    if enough_data_fam and enough_data_nov:
        labels = ['||| - %s NO' % labels_novel[region], '||| - %s REW' % labels_novel[region], '|||-||| NO', '|||-||| REW', '%s - %s NO' % (labels_novel[region], labels_novel[region]), '%s - %s REW' % (labels_novel[region], labels_novel[region])]
        datas = [corr_nonrew_fam_nov, corr_rew_fam_nov, corr_nonrew_fam_fam,  corr_rew_fam_fam, corr_nonrew_nov_nov, corr_rew_nov_nov]
        f, ax = visualize.box_comparison(datas, labels, 'rate vector correlation (per lap couple)')
        visualize.annotate_ttest_p(corr_rew_fam_nov, corr_nonrew_fam_nov, 0, 1, ax)
        visualize.annotate_ttest_p(corr_rew_fam_fam, corr_nonrew_fam_fam, 2, 3, ax)
        visualize.annotate_ttest_p(corr_rew_nov_nov, corr_nonrew_nov_nov, 4, 5, ax)
        plt.savefig(savepath + '.pdf')
        plt.close('all')

        return np.mean(corr_rew_fam_nov), np.mean(corr_nonrew_fam_nov), np.mean(corr_rew_fam_fam), np.mean(corr_nonrew_fam_fam), np.mean(corr_rew_nov_nov), np.mean(corr_nonrew_nov_nov)

    elif enough_data_fam:
        return np.nan, np.nan, np.mean(corr_rew_fam_fam), np.mean(corr_nonrew_fam_fam), np.nan, np.nan
    elif enough_data_nov:
        return np.nan, np.nan, np.nan, np.nan, np.mean(corr_rew_nov_nov), np.mean(corr_nonrew_nov_nov)
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def compare_laps_rate_correlation_speed(session, savepath, region, min_data, fast_threshold):
    fast_laps_F = [l for l in session.familiar_laps if l.mean_speed > fast_threshold]
    fast_laps_N = [l for l in session.novel_laps if l.mean_speed > fast_threshold]
    slow_laps_F = [l for l in session.familiar_laps if l.mean_speed < fast_threshold]
    slow_laps_N = [l for l in session.novel_laps if l.mean_speed < fast_threshold]

    corr_ff_fast = np.asarray([nanpearsonr(fast_laps_F[i].rate_vector, fast_laps_F[j].rate_vector)[0] for i in range(len(fast_laps_F)) for j in range(i+1, len(fast_laps_F))])
    corr_nn_fast = np.asarray([nanpearsonr(fast_laps_N[i].rate_vector, fast_laps_N[j].rate_vector)[0] for i in range(len(fast_laps_N)) for j in range(i+1, len(fast_laps_N))])
    corr_fn_fast = np.asarray([nanpearsonr(l1.rate_vector, l2.rate_vector)[0] for l1 in fast_laps_F for l2 in fast_laps_N])

    corr_ff_slow = np.asarray([nanpearsonr(slow_laps_F[i].rate_vector, slow_laps_F[j].rate_vector)[0] for i in range(len(slow_laps_F)) for j in range(i+1, len(slow_laps_F))])
    corr_nn_slow = np.asarray([nanpearsonr(slow_laps_N[i].rate_vector, slow_laps_N[j].rate_vector)[0] for i in range(len(slow_laps_N)) for j in range(i+1, len(slow_laps_N))])
    corr_fn_slow = np.asarray([nanpearsonr(l1.rate_vector, l2.rate_vector)[0] for l1 in slow_laps_F for l2 in slow_laps_N])

    enough_fast = min([len(fast_laps_F), len(fast_laps_N)]) >= min_data
    enough_slow = min([len(slow_laps_F), len(slow_laps_N)]) >= min_data

    # plot results
    labels = ['||| - %s slow' % labels_novel[region], '||| - %s fast' % labels_novel[region], '|||-||| slow', '|||-||| fast', '%s - %s slow' % (labels_novel[region], labels_novel[region]), '%s - %s fast' % (labels_novel[region], labels_novel[region])]
    datas = [corr_fn_slow, corr_fn_fast, corr_ff_slow,  corr_ff_fast, corr_nn_slow, corr_nn_fast]
    f, ax = visualize.box_comparison(datas, labels, 'rate vector correlation (per lap couple)')
    visualize.annotate_ttest_p(datas[0], datas[1], 0, 1, ax)
    visualize.annotate_ttest_p(datas[2], datas[3], 2, 3, ax)
    visualize.annotate_ttest_p(datas[4], datas[5], 4, 5, ax)
    plt.savefig(savepath + '.pdf')
    plt.close('all')

    if enough_fast and enough_slow:
        return np.mean(corr_fn_slow), np.mean(corr_fn_fast), np.mean(corr_ff_slow), np.mean(corr_ff_fast), np.mean(corr_nn_slow), np.mean(corr_nn_fast)
    elif enough_fast:
        return np.nan, np.mean(corr_fn_fast), np.nan, np.mean(corr_ff_fast), np.nan, np.mean(corr_nn_fast)
    elif enough_slow:
        return np.mean(corr_fn_slow), np.nan, np.mean(corr_ff_slow), np.nan, np.mean(corr_nn_slow), np.nan
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def compare_activity_reward(session, savepath, min_data, region = 'DG'):
    fr = 0; fnr = 0; nr = 0; nnr = 0;

    F_reward_rate = np.zeros(session.n_roi)
    F_no_reward_rate = np.zeros(session.n_roi)

    F_reward_running_time = 0
    F_no_reward_running_time = 0

    for l in session.familiar_laps:
        if l.reward:
            F_reward_rate += l.rate_vector * l.tot_running_time
            F_reward_running_time += l.tot_running_time
            fr +=1
        if l.reward == 0:
            F_no_reward_rate += l.rate_vector * l.tot_running_time
            F_no_reward_running_time += l.tot_running_time
            fnr +=1

    F_reward_rate /= F_reward_running_time
    F_no_reward_rate /= F_no_reward_running_time

    N_reward_rate = np.zeros(session.n_roi)
    N_no_reward_rate = np.zeros(session.n_roi)

    N_reward_running_time = 0
    N_no_reward_running_time = 0

    for l in session.novel_laps:
        if l.reward:
            N_reward_rate += l.rate_vector * l.tot_running_time
            N_reward_running_time += l.tot_running_time
            nr +=1

        if l.reward == 0:
            N_no_reward_rate += l.rate_vector * l.tot_running_time
            N_no_reward_running_time += l.tot_running_time
            nnr+=1

    N_reward_rate /= N_reward_running_time
    N_no_reward_rate /= N_no_reward_running_time

    labels = ['|||', '%s' % labels_novel[region]]
    datas = [F_no_reward_rate-F_reward_rate, N_no_reward_rate-N_reward_rate]
    f, ax = visualize.box_comparison(datas, labels, '$r^{rew} - r^{no rew}$ (ev/s) - per cell', box =False, swarm=True)
    ys = ax.get_ylim()
    ax.set_ylim([-np.max(np.abs(ys)), np.max(np.abs(ys))])
    ax.axhline([0], color='k')
    visualize.annotate_wilcoxon_p_single(F_no_reward_rate-F_reward_rate, -0.3, 0.3, ax)
    visualize.annotate_wilcoxon_p_single(N_no_reward_rate-N_reward_rate, 0.7, 1.3, ax)
    plt.savefig(savepath + '.pdf')
    plt.close(f)

    if np.min([fr, fnr, nr, nnr]) >= min_data:
        return np.mean(F_reward_rate), np.mean(F_no_reward_rate), np.mean(N_reward_rate), np.mean(N_no_reward_rate)
    elif np.min([fr, fnr]) >= min_data:
        return np.mean(F_reward_rate), np.mean(F_no_reward_rate), np.nan, np.nan
    elif np.min([nr, nnr]) >= min_data:
        return np.nan, np.nan, np.mean(N_reward_rate), np.mean(N_no_reward_rate)
    else:
        return np.nan, np.nan, np.nan, np.nan

def mfr_analysis_around_teleportation(session, time_around, type1='vertical', type2='oblique', signal='dF'):
    pre = []
    post = []

    for i in range(1, len(session.laps)):
        if (session.laps[i].laptype == type1) and (session.laps[i-1].laptype == type2):
            mfr_pre = np.zeros(len(session.laps[i].rois))
            mfr_post = np.zeros(len(session.laps[i].rois))
            for k in range(len(session.laps[i].rois)):
                if signal=='dF':
                    signal_pre = session.laps[i-1].rois[k].dF_F
                    signal_post = session.laps[i].rois[k].dF_F
                    signal_pre = signal_pre[-time_around*10:]
                    signal_post = signal_post[:time_around*10]
                elif signal=='S':
                    signal_pre = session.laps[i-1].rois[k].S
                    signal_post = session.laps[i].rois[k].S
                    signal_pre = signal_pre[-time_around*10:]
                    signal_post = signal_post[:time_around*10]
                elif signal=='events':
                    nbins = int(time_around/120.)
                    signal_pre = session.laps[i-1].rois[k].activation_raster[-nbins:]
                    signal_post = session.laps[i].rois[k].activation_raster[:nbins]
                mfr_pre[k] = np.mean(signal_pre)
                mfr_post[k] = np.mean(signal_post)

            pre.append(mfr_pre)
            post.append(mfr_post)
    return np.asarray(pre), np.asarray(post)

def compare_novelty_rate(session, time_around, savepath, signal='dF'):
    # time_around in ms

    AB_pre, AB_post = mfr_analysis_around_teleportation(session, time_around, type1='vertical', type2='oblique', signal=signal)
    AA_pre, AA_post = mfr_analysis_around_teleportation(session, time_around, type1='vertical', type2='vertical', signal=signal)
    AB_pre, AB_post = mfr_analysis_around_teleportation(session, time_around, type1='oblique', type2='vertical', signal=signal)
    BB_pre, BB_post = mfr_analysis_around_teleportation(session, time_around, type1='oblique', type2='oblique', signal=signal)

    # compare first AB with all AAs
    AB_delta = (AB_post - AB_pre)
    AA_delta = (AA_post - AA_pre)

    if len(AA_delta) and len(AB_delta):
        if DEBUG:
            print(np.shape(AA_delta), np.shape(AB_delta))
        AB = AB_delta[0]
        AA = np.mean(AA_delta, 0)
        f, ax = visualize.box_comparison_two(AA, AB, 'AA', 'AB', 'Activation Difference (post-pre)', swarm=False, paired=True, bar=True)
    else:
        return  np.nan, np.nan

    savepath = savepath + '_%s' % signal
    f.savefig(savepath+'.pdf')
    return np.mean(AB), np.mean(AA)


# --- decoding functions --- #

def decode_conditions(training_A, training_B, test_A, test_B, min_activations=1, pseudo_count=0.5):
    # select only cells that are active in the training rasters
    active_cells = (np.sum(training_A, 0) + np.sum(training_B, 0)) >= min_activations

    training_A = training_A[:, active_cells]
    training_B = training_B[:, active_cells]
    test_A = test_A[:, active_cells]
    test_B = test_B[:, active_cells]

    # train indpendent model
    model_A = nice.independent_model()
    model_B = nice.independent_model()

    model_A.train(training_A, pseudo_count=pseudo_count/len(training_A))
    model_B.train(training_B, pseudo_count=pseudo_count/len(training_B))

    # selecting only non-silent test bins for performance computation
    test_A = test_A[np.sum(test_A, 1) >= 1]
    test_B = test_B[np.sum(test_B, 1) >= 1]

    # computing delta likelihood
    DL_A = model_A.score(test_A) - model_B.score(test_A)
    DL_B = model_A.score(test_B) - model_B.score(test_B)

    return DL_A, DL_B

def decode_raster(training_A, training_B, test, min_activations=1):
    # select only cells that are active in the training rasters
    active_cells = (np.sum(training_A, 0) + np.sum(training_B, 0)) >= min_activations

    training_A = training_A[:, active_cells]
    training_B = training_B[:, active_cells]
    test = test[:, active_cells]

    # train indpendent model
    model_A = nice.independent_model()
    model_B = nice.independent_model()

    model_A.train(training_A, pseudo_count=0.1/len(training_A))
    model_B.train(training_B, pseudo_count=0.1/len(training_B))

    # selecting only non-silent test bins for performance computation
    test = test[np.sum(test, 1) >= min_activations]

    # computing delta likelihood
    DL = model_A.score(test) - model_B.score(test)

    return DL

def selectivity_manu(session, division=True):
    mfr_f = bn.nanmean(np.vstack([l.raster for l in session.familiar_laps]), 0) / (session.settings['discretization_timescale']/1000.)
    mfr_n = bn.nanmean(np.vstack([l.raster for l in session.novel_laps]), 0) / (session.settings['discretization_timescale']/1000.)
    #return [mfr_f, mfr_n]
    if division:
        return np.abs(mfr_f - mfr_n)/(mfr_f + mfr_n)
    else:
        return np.abs(mfr_f - mfr_n)

def selectivity(session, division=True): #_continuous
    #move.printA("event_length",[l.event_length for l in session.familiar_laps])
    #move.printA("sum(event_length of familiar,0)",np.sum([l.event_length for l in session.familiar_laps], 0))
    #move.printA("sum(event_length of novel,0)",np.sum([l.event_length for l in session.novel_laps], 0))
    #move.printA("sum(lap_length of familiar,0)",np.sum([l.lap_length for l in session.familiar_laps]))
    #move.printA("sum(lap_length of novel,0)",np.sum([l.lap_length for l in session.novel_laps]))
    rate_f = bn.nansum([l.event_length for l in session.familiar_laps], 0)/bn.nansum([l.lap_length for l in session.familiar_laps])
    rate_n = bn.nansum([l.event_length for l in session.novel_laps], 0)/bn.nansum([l.lap_length for l in session.novel_laps])
    #move.printA('rate_f',rate_f)
    #move.printA('rate_n',rate_n)
    if division:
        selectivity = [ abs(rf - rn)/(rf+rn) for rf, rn in zip(rate_f,rate_n)]
    else:
        selectivity = [ abs(rf - rn) for rf, rn in zip(rate_f,rate_n)]
    #for rf, rn in zip(rate_f,rate_n):
    #    print(rf,rn,'->', abs(rf - rn)/(rf+rn))
    #move.printA('selectivity',selectivity)
    return [rate_f,rate_n,selectivity]

def selectivity_discrete(session, division=True):
    #move.printA("event_count", [ l.event_count for l in session.familiar_laps])
    #move.printA("sum(event_count,0)", np.sum([ l.event_count for l in session.familiar_laps],0))
    #move.printA("times", [ l.times for l in session.familiar_laps])
    #move.printA("duration", [ l.duration for l in session.familiar_laps])
    #move.printA("sum(duration)", np.sum([ l.duration for l in session.familiar_laps]))
    rate_f = bn.nansum([ l.event_count for l in session.familiar_laps],0) / bn.nansum([ l.duration for l in session.familiar_laps])
    rate_n = bn.nansum([ l.event_count for l in session.novel_laps],0) / bn.nansum([ l.duration for l in session.novel_laps])
    #move.printA('rate_f',rate_f)
    #move.printA('rate_n',rate_n)
    if division:
        selectivity = [ abs(rf - rn)/(rf+rn) for rf, rn in zip(rate_f,rate_n)]
    else:
        selectivity = [ abs(rf - rn) for rf, rn in zip(rate_f,rate_n)]
    #for rf, rn in zip(rate_f,rate_n):
    #    print(rf,rn,'->', abs(rf - rn)/(rf+rn))
    #move.printA('selectivity',selectivity)
    return [rate_f,rate_n,selectivity]

def selectivity_continuous(session, division=True):
    #move.printA("event_count", [ l.event_count for l in session.familiar_laps])
    #move.printA("sum(event_count,0)", np.sum([ l.event_count for l in session.familiar_laps],0))
    #move.printA("times", [ l.times for l in session.familiar_laps])
    #move.printA("duration", [ l.duration for l in session.familiar_laps])
    #move.printA("sum(duration)", np.sum([ l.duration for l in session.familiar_laps]))
    #move.printA('rate_f',rate_f)
    #move.printA('rate_n',rate_n)
    rate_f = np.empty((session.n_roi))
    rate_n = np.empty((session.n_roi))
    selectivity = np.empty((session.n_roi))
    t_familiar = bn.nansum([ l.duration for l in session.familiar_laps])
    t_novel = bn.nansum([ l.duration for l in session.novel_laps])
    for nroi in range(session.n_roi):
        # Do we have more than 1 event at all?
        nevents = bn.nansum([bn.nansum((l.S[nroi]>0).astype(np.int)) for l in session.familiar_laps]) + bn.nansum([bn.nansum((l.S[nroi]>0).astype(np.int)) for l in session.novel_laps])
        rate_f[nroi] = bn.nansum([bn.nansum(l.S[nroi]) for l in session.familiar_laps]) / t_familiar
        rate_n[nroi] = bn.nansum([bn.nansum(l.S[nroi]) for l in session.novel_laps]) / t_novel
        if nevents > 1:
            rf = rate_f[nroi]
            rn = rate_n[nroi]
            if division:
                selectivity[nroi] = np.abs(rf - rn)/(rf+rn)
            else:
                selectivity[nroi] = np.abs(rf - rn)
        else:
            selectivity[nroi] = np.nan
    #for rf, rn in zip(rate_f,rate_n):
    #    print(rf,rn,'->', abs(rf - rn)/(rf+rn))
    #move.printA('selectivity',selectivity)
    return [rate_f,rate_n,selectivity]


def raster_position(lap): ### TODO: incorporate this in vrclasses
    dt = np.median(np.diff(lap.times))
    index = np.floor(np.arange(len(lap.position)) * dt/120.)
    raster_pos = np.zeros(len(lap.raster))*np.nan
    for i in range(len(raster_pos)):
        raster_pos[i] = np.mean(lap.position[index==i])
    return raster_pos

def visualize_session_FP(session, type='speed', threshold=1.25, ax='auto', name='none', VR=True):
    t0 = np.min(session.laps[0].times) * 1e-3
    tmax = np.max(session.laps[-1].times) * 1e-3
    if ax=='auto':
        f, ax = plt.subplots(figsize=(8,3))
    ax.plot(session.times * 1e-3, session.position)
    times = np.copy(session.times) * 1e-3
    position = np.copy(session.position)
    position[session.running_mask] = np.nan
    ax.plot(times, position, color='r')
    ax.scatter(session.times[session.lick_array==1]*1.e-3, session.position[session.lick_array==1], marker='|', color='k', s=9)
    ax.set_ylabel('x (m)')
    ax.set_yticks([0, 1.2])
    ax.set_ylim([0, 1.2])
    ax.yaxis.set_tick_params(labelsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if VR:
        type_one = 'oblique'
        type_two = 'vertical'
    else:
        type_one = 'bottom'
        type_two = 'top'

    for i,l in enumerate(session.laps):
        if type=='speed':
            if VR:
                ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.18, 'T %.1f' % l.true_speed_ratio, fontsize=4)
                ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.25, 'F %.1f' % l.false_speed_ratio, fontsize=4)

            if l.laptype==type_one and (not VR or (l.false_speed_ratio > threshold and l.true_speed_ratio > threshold)):
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[3])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[2])
            elif VR and l.false_speed_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[3])
            elif VR and l.true_speed_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[2])

            if l.laptype==type_two and (not VR or (l.false_speed_ratio > threshold and l.true_speed_ratio > threshold)):
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[6])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[9])
            elif VR and l.false_speed_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[6])
            elif VR and l.true_speed_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[9])

        elif type=='stop':
            ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.18, 'T %.1f' % l.true_stop_ratio, fontsize=4)
            ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.25, 'F %.1f' % l.false_stop_ratio, fontsize=4)

            if l.laptype=='oblique' and l.false_stop_ratio > threshold and l.true_stop_ratio > threshold:
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[3])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[2])
            elif l.false_stop_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[3])
            elif l.true_stop_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[2])

            if l.laptype=='vertical' and l.false_stop_ratio > threshold and l.true_stop_ratio > threshold:
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[6])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[9])
            elif l.false_stop_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[6])
            elif l.true_stop_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[9])

        elif type=='lick':
            ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.18, 'T %.1f' % l.true_lick_ratio, fontsize=4)
            ax.text(np.min(l.times)*1e-3 + 1, session.position.max()-0.25, 'F %.1f' % l.false_lick_ratio, fontsize=4)

            if l.laptype=='oblique' and l.false_lick_ratio > threshold and l.true_lick_ratio > threshold:
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[3])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[2])
            elif l.false_lick_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[3])
            elif l.true_lick_ratio > threshold and l.laptype=='oblique':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[2])

            if l.laptype=='vertical' and l.false_lick_ratio > threshold and l.true_lick_ratio > threshold:
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.6, 0.6], 0, alpha=0.3, color=pltcolors[6])
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0.6, alpha=0.3, color=pltcolors[9])
            elif l.false_lick_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[6])
            elif l.true_lick_ratio > threshold and l.laptype=='vertical':
                ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.2, 1.2], 0, alpha=0.3, color=pltcolors[9])

        if l.laptype == type_two:
            ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.675, 0.675] , 0.3375, alpha=0.05, color='k')
            ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.1475, 1.1475] , 0.810, alpha=0.15, color='k')
        if l.laptype == type_one:
            ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [0.675, 0.675] , 0.3375, alpha=0.15, color='k')
            ax.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [1.1475, 1.1475] , 0.810, alpha=0.05, color='k')

    if VR:
        for ev in session.vrdict['evlist']:
            if ev.evtype == 'vertical' and t0 <= ev.time < tmax-1:
                ax.text(ev.time+1, session.position.max()-0.1, '|||', fontsize=4)
            if ev.evtype == 'oblique' and t0 <= ev.time < tmax-1:
                ax.text(ev.time+1, session.position.max()-0.1, '///', fontsize=4)

    if name!='none':
        ax.text(np.min(session.times)*1e-3 - 20, session.position.max()+0.05, name, fontsize=6)

def visualize_session_decoded(session, n_roi_shown=30, n_laps=100, savepath='none', only_spatial_cells = False, only_nonspatial_cells = False):
    n_laps = min([n_laps, session.nlaps-1])
    n_roi_shown = min([n_roi_shown, session.n_roi])

    if only_spatial_cells:
        n_roi_shown = min([n_roi_shown, np.sum(session.spatial_cells)])
    if only_nonspatial_cells:
        n_roi_shown = min([n_roi_shown, np.sum(1-session.spatial_cells)])

    odd_familiar_activity = np.vstack([l.raster for i,l in enumerate(session.familiar_laps) if i%2==1])
    even_familiar_activity = np.vstack([l.raster for i,l in enumerate(session.familiar_laps) if i%2==0])
    odd_novel_activity = np.vstack([l.raster for i,l in enumerate(session.novel_laps) if i%2==1])
    even_novel_activity = np.vstack([l.raster for i,l in enumerate(session.novel_laps) if i%2==0])
    session_activity = np.vstack([l.raster for l in session.laps])

    index = np.argsort(-np.mean(session_activity, 0))
    mfr_f = np.mean(np.vstack([l.raster for l in session.familiar_laps]), 0) / 0.12
    mfr_n = np.mean(np.vstack([l.raster for l in session.novel_laps]), 0) / 0.12
    sorted_diff = mfr_f[index[:n_roi_shown]] - mfr_n[index[:n_roi_shown]]
    index[:n_roi_shown] = index[:n_roi_shown][np.argsort(-sorted_diff)]

    if only_spatial_cells:
        odd_familiar_activity = odd_familiar_activity[:, session.spatial_cells]
        even_familiar_activity = even_familiar_activity[:, session.spatial_cells]
        odd_novel_activity = odd_novel_activity[:, session.spatial_cells]
        even_novel_activity = even_novel_activity[:, session.spatial_cells]
        session_activity = session_activity[:, session.spatial_cells]

    if only_nonspatial_cells:
        odd_familiar_activity = odd_familiar_activity[:, session.spatial_cells==0]
        even_familiar_activity = even_familiar_activity[:, session.spatial_cells==0]
        odd_novel_activity = odd_novel_activity[:, session.spatial_cells==0]
        even_novel_activity = even_novel_activity[:, session.spatial_cells==0]
        session_activity = session_activity[:, session.spatial_cells==0]

    plt.figure(figsize=(10, 3))
    plt.bar(np.arange(n_roi_shown), mfr_f[index[:n_roi_shown]], width = 0.5, color=pltcolors[2])
    plt.bar(np.arange(n_roi_shown), -mfr_n[index[:n_roi_shown]], width = 0.5, color=pltcolors[1])
    plt.savefig(savepath+'_mfr')
    plt.close()

    t0 = np.min(session.laps[0].times) * 1e-3
    tmax = np.max(session.laps[-1].times) * 1e-3
    tmaxplot = np.max(session.laps[n_laps].times) * 1e-3 -1

    DL = decode_raster(even_familiar_activity, even_novel_activity, session_activity, min_activations=0)

    import matplotlib as mpl
    import matplotlib.gridspec as gridspec
    mpl.rcParams.update({'figure.autolayout': False})
    mpl.rcParams['axes.labelsize'] = 8
    offset = 11
    nrows = n_roi_shown + offset

    ns = 1
    fig = plt.figure(figsize=(7, nrows/6))
    gs = gridspec.GridSpec(nrows, 2, figure=fig)
    ax0 = fig.add_subplot(gs[0:6, :])
    DL_m = int(np.percentile(np.abs(DL[DL!=0]), 99))

    for i,l in enumerate(session.familiar_laps):
        if i%2==1:
            ax0.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [DL_m, DL_m], 0, alpha=0.6, color=pltcolors[2])
        else:
            ax0.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [DL_m, DL_m], 0, alpha=0.2, color=pltcolors[2])
    for i,l in enumerate(session.novel_laps):
        if i%2==1:
            ax0.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [-DL_m, -DL_m], 0, alpha=0.6, color=pltcolors[1])
        else:
            ax0.fill_between([np.min(l.times)*1e-3, np.max(l.times)*1e-3], [-DL_m, -DL_m], 0, alpha=0.2, color=pltcolors[1])

    bartimes = np.linspace(t0, tmax, len(DL))

    nonzeroDL = DL[DL!=0]
    nonzerotimes = bartimes[DL!=0]
    # pt = ax0.plot(nonzerotimes, nonzeroDL, color='k', linewidth=0.5)
    ax0.fill_between(nonzerotimes, nonzeroDL, 0, alpha=0.8, zorder=2)
    ax0.bar(bartimes, DL, width=0.5, color=[0.2, 0.2, 0.2], zorder=3)

    ax0.set_ylim([-DL_m, DL_m])
    ax0.set_yticks([-DL_m, 0, DL_m])
    ax0.axhline([0], color='k', zorder=4)
    ax0.set_xticks([])
    ax0.yaxis.set_tick_params(labelsize=8)
    ax0.set_ylabel('$\\Delta\\mathcal{L}$')
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)

    ax1 = fig.add_subplot(gs[7:10, :], sharex=ax0)
    ax1.plot(session.times * 1e-3, session.position)
    ax1.set_ylabel('x (m)')
    ax1.set_yticks([0, 1.2])
    ax1.set_ylim([0, 1.2])
    ax1.yaxis.set_tick_params(labelsize=8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    for ev in session.vrdict['evlist']:
        if ev.evtype == 'vertical' and t0 <= ev.time < tmaxplot-1:
            ax1.text(ev.time+1, session.position.max()-0.1, '|||', fontsize=4)
        if ev.evtype == 'oblique' and t0 <= ev.time < tmaxplot-1:
            ax1.text(ev.time+1, session.position.max()-0.1, '///', fontsize=4)


    # for n in range(n_roi_start, n_roi_start + n_roi_shown-1):
    n = 0
    nplotted = 0
    while nplotted < n_roi_shown-1:
        if (only_spatial_cells) and (session.spatial_cells[index[n]]==0):
            n+=1
            continue
        if (only_nonspatial_cells) and (session.spatial_cells[index[n]]):
            n+=1
            continue
        else:
            ax = fig.add_subplot(gs[nplotted+offset, :], sharex=ax0)
            ax.plot(session.times * 1e-3, session.dF_F[index[n]], '-k', alpha=0.5)
            if len(session.activations[index[n]]):
                ax.bar(session.times[session.activations[index[n]]] * 1e-3, np.max(session.dF_F[index[n]]) * np.ones(len(session.activations[index[n]])), color='r', width=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            nplotted+=1
            n+=1

    ax = fig.add_subplot(gs[nplotted+offset, :])
    ax.plot(session.times * 1e-3, session.dF_F[index[n]], '-k', alpha=0.5)
    if len(session.activations[index[n]]):
        ax.bar(session.times[session.activations[index[n]]] * 1e-3, np.max(session.dF_F[index[n]]) * np.ones(len(session.activations[index[n]])), color='r', width=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.set_xticks([ceil(t0), floor(tmaxplot/2), floor(tmaxplot)-10, floor(tmaxplot)])
    ax.set_xlabel('time (s)')
    ax.set_xlim(t0, tmaxplot)
    ax.xaxis.set_tick_params(labelsize=8)

    ax0.set_xlim(t0, tmaxplot)
    if savepath != 'none':
        plt.savefig(savepath)
        plt.close()
    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams['axes.labelsize'] = 12



### NEW DECODING FUNCTIONS


def decode_laps(laps_A, laps_B, n_shuffles='auto', mode='AUC', cell_subset = 'none', training_fraction=0.5, min_data=1, min_activations=1, track_start = 0.0, track_end = 2.0, debug=False):
    aucs = []
    if len(laps_A)>= min_data*2 and len(laps_B) >= min_data*2:
        n_A = len(laps_A)
        n_B = len(laps_B)
        if n_shuffles == 'auto':
            n_shuffles = int(n_A/2 * n_B/2)
        for n in range(n_shuffles):
            if debug:
                print(("[decode_laps]\t Decoding trial %u with mode %s" % (n, mode)))
            idxf = np.random.permutation(n_A)
            set1_A = idxf[:int(n_A*training_fraction)]
            set2_A = idxf[int(n_A*training_fraction):]
            if len(set2_A)==0:
                set1_A = idxf[:-1]
                set2_A = [idxf[-1]]

            training_laps_A = [l for i,l in enumerate(laps_A) if i in set1_A]
            testing_laps_A = [l for i,l in enumerate(laps_A) if i in set2_A]

            idxf = np.random.permutation(n_B)
            set1_B = idxf[:int(n_B*training_fraction)]
            set2_B = idxf[int(n_B*training_fraction):]
            if len(set2_B)==0:
                set1_B = idxf[:-1]
                set2_B = [idxf[-1]]

            training_laps_B = [l for i,l in enumerate(laps_B) if i in set1_B]
            testing_laps_B = [l for i,l in enumerate(laps_B) if i in set2_B]

            training_A = np.vstack([l.raster[(raster_position(l) > track_start) & (raster_position(l) < track_end)] for l in training_laps_A])
            training_B = np.vstack([l.raster[(raster_position(l) > track_start) & (raster_position(l) < track_end)] for l in training_laps_B])

            if cell_subset != 'none':
                training_A = training_A[:, cell_subset]
                training_B = training_B[:, cell_subset]

            if mode == 'AUC':
                testing_A = np.vstack([l.raster[(raster_position(l) > track_start) & (raster_position(l) < track_end)] for l in testing_laps_A])
                testing_B = np.vstack([l.raster[(raster_position(l) > track_start) & (raster_position(l) < track_end)] for l in testing_laps_B])

                if cell_subset != 'none':
                    testing_A = testing_A[:, cell_subset]
                    testing_B = testing_B[:, cell_subset]

                DL_A, DL_B = decode_conditions(training_A, training_B, testing_A, testing_B)

                if (len(DL_A) >= min_activations) and (len(DL_B) >= min_activations):

                    auc = nice.AUC(DL_A, DL_B)
                    aucs.append(auc)
                    if debug:
                        print(("[decode_laps]\t Number of data points\t training A: %u\t training B: %u\t test A: %u\t test B: %u" % (np.sum(training_A), np.sum(training_B), np.sum(testing_A), np.sum(testing_B))))
                        print(("[decode_laps]\t Number of activations\t training A: %u\t training B: %u\t test A: %u\t test B: %u" % (len(training_A), len(training_B), len(testing_A), len(testing_B))))
                        print(("[decode_laps]\t Decoding performance: %.3f" % auc))
                        print("---------------------------------------------")
            if mode == 'laps':
                count = 0
                guess = 0
                for l_A in testing_laps_A:
                    for l_B in testing_laps_B:
                        testing_A = l_A.raster[(raster_position(l_A) > track_start) & (raster_position(l_A) < track_end)]
                        testing_B = l_B.raster[(raster_position(l_B) > track_start) & (raster_position(l_B) < track_end)]
                        if cell_subset != 'none':
                            testing_A = testing_A[:, cell_subset]
                            testing_B = testing_B[:, cell_subset]
                        DL_A, DL_B = decode_conditions(training_A, training_B, testing_A, testing_B)
                        if (len(DL_A) >= min_activations) and (len(DL_B) >= min_activations):
                            if np.sum(DL_A[DL_A !=0]) > np.sum(DL_B[DL_B !=0]):
                                guess += 1
                            count +=1
                            if debug:
                                print(("[decode_laps]\t DL_A sum: %.3f, DL_B sum: %.3f, guess: %u" % (np.sum(DL_A[DL_A !=0]), np.sum(DL_B[DL_B !=0]), np.sum(DL_A[DL_A !=0]) > np.sum(DL_B[DL_B !=0]))))
                if count:
                    if debug:
                        print(("[decode_laps]\t Decoding performance: %.3f, corresponding to %u guess over %u pair trials" % (float(guess)/count, guess, count)))
                        print("---------------------------------------------")
                    aucs.append(float(guess)/count)
    return aucs


def decode_reward(session, min_data=1, min_activations=1, training_fraction=0.5, n_shuffles=50, mode='AUC', track_boundaries=[0, 10]):
    # select reward and non reward laps
    reward_familiar = [l for l in session.laps if l.reward and l.laptype == "vertical"]
    nonreward_familiar = [l for l in session.laps if l.reward==0 and l.laptype == "vertical"]
    reward_novel = [l for l in session.laps if l.reward and l.laptype == "oblique"]
    nonreward_novel = [l for l in session.laps if l.reward==0 and l.laptype == "oblique"]

    aucs_F = decode_laps(reward_familiar, nonreward_familiar, n_shuffles, mode=mode,
                            training_fraction=training_fraction, min_data=min_data,
                            min_activations=min_activations, track_start=track_boundaries[0],
                            track_end=track_boundaries[1])

    aucs_N = decode_laps(reward_novel, nonreward_novel, n_shuffles, mode=mode,
                            training_fraction=training_fraction, min_data=min_data,
                            min_activations=min_activations, track_start=track_boundaries[0],
                            track_end=track_boundaries[1])

    aucs_any = decode_laps(reward_novel+reward_familiar, nonreward_novel+nonreward_familiar, n_shuffles, mode=mode,
                            training_fraction=training_fraction, min_data=min_data,
                            min_activations=min_activations, track_start=track_boundaries[0],
                            track_end=track_boundaries[1])

    return np.mean(aucs_F), bn.nanmean(aucs_N), bn.nanmean(aucs_any)


def decode_environment(session, min_data=1, min_activations=1, reward='any', place_cells='any', training_fraction=0.5, n_shuffles='auto', mode='AUC', track_boundaries=[0, 10], debug=False, VR=True):
    # select reward and non reward laps
    laps_A = []
    laps_B = []

    if VR:
        typeone = 'vertical'
        typetwo = 'oblique'
    else:
        typeone = 'top'
        typetwo = 'bottom'

    if reward=='any':
        for i in range(len(session.laps)):
            if session.laps[i].laptype == typeone:
                laps_A.append(session.laps[i])
            if session.laps[i].laptype == typetwo:
                laps_B.append(session.laps[i])

    if reward=='no':
        for i in range(len(session.laps)):
            if session.laps[i].laptype == 'vertical' and session.laps[i].reward == False:
                laps_A.append(session.laps[i])
            if session.laps[i].laptype == 'oblique' and session.laps[i].reward == False:
                laps_B.append(session.laps[i])

    if reward=='yes':
        for i in range(len(session.laps)):
            if session.laps[i].laptype == 'vertical' and session.laps[i].reward:
                laps_A.append(session.laps[i])
            if session.laps[i].laptype == 'oblique' and session.laps[i].reward:
                laps_B.append(session.laps[i])

    if place_cells=='any':
        cell_subset = np.ones(session.n_roi) > 0
    if place_cells=='yes':
        cell_subset = (session.place_cells_F) | (session.place_cells_N)
    if place_cells == 'no':
        cell_subset = (np.ones(session.n_roi) - ((session.place_cells_F) | (session.place_cells_N))) > 0

    if debug:
        print(("[decode_environment]\t laps condition A: %u\t laps condition B: %u" % (len(laps_A), len(laps_B))))

    aucs = decode_laps(laps_A, laps_B, n_shuffles, mode=mode, cell_subset=cell_subset,
                        training_fraction=training_fraction, min_data=min_data,
                        min_activations=min_activations, track_start=track_boundaries[0],
                        track_end=track_boundaries[1], debug=debug)

    return np.mean(aucs)


def decode_previous_environment(session, min_data=1, min_activations=1, reward='any', training_fraction=0.5, n_shuffles=50, mode='AUC'):
    # select reward and non reward laps
    laps_AA = []
    laps_BA = []
    if reward=='any':
        for i in range(1, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical':
                    laps_AA.append(session.laps[i])
                if session.laps[i-1].laptype == 'oblique':
                    laps_BA.append(session.laps[i])
    if reward=='no':
        for i in range(1, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical' and session.laps[i-1].reward==False:
                    laps_AA.append(session.laps[i])
                if session.laps[i-1].laptype == 'oblique' and session.laps[i-1].reward==False:
                    laps_BA.append(session.laps[i])
    if reward=='yes':
        for i in range(1, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical' and session.laps[i-1].reward==True:
                    laps_AA.append(session.laps[i])
                if session.laps[i-1].laptype == 'oblique' and session.laps[i-1].reward==True:
                    laps_BA.append(session.laps[i])

    aucs = decode_laps(laps_AA, laps_BA, n_shuffles, mode=mode, training_fraction=training_fraction, min_data=min_data, min_activations=min_activations)

    return np.mean(aucs)


def decode_previous_previous_environment(session, min_data=1, min_activations=1, reward='any', training_fraction=0.5, n_shuffles=50, mode='AUC'):

    # select reward and non reward laps
    laps_AAA = []
    laps_BAA = []
    if reward=='any':
        for i in range(2, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical':
                    if session.laps[i-2].laptype == 'vertical':
                        laps_AAA.append(session.laps[i])
                    if session.laps[i-2].laptype == 'oblique':
                        laps_BAA.append(session.laps[i])
    if reward=='yes':
        for i in range(2, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical':
                    if session.laps[i-2].laptype == 'vertical' and session.laps[i-2].reward:
                        laps_AAA.append(session.laps[i])
                    if session.laps[i-2].laptype == 'oblique' and session.laps[i-2].reward:
                        laps_BAA.append(session.laps[i])

    if reward=='no':
        for i in range(2, len(session.laps)):
            if session.laps[i].laptype == 'vertical':
                if session.laps[i-1].laptype == 'vertical':
                    if session.laps[i-2].laptype == 'vertical' and session.laps[i-2].reward==False:
                        laps_AAA.append(session.laps[i])
                    if session.laps[i-2].laptype == 'oblique' and session.laps[i-2].reward==False:
                        laps_BAA.append(session.laps[i])

    aucs = decode_laps(laps_AAA, laps_BAA, n_shuffles, mode=mode, training_fraction=training_fraction, min_data=min_data, min_activations=min_activations)
    return bn.nanmean(aucs)


def decode_future_environment(session, savepath='none', min_data=1, min_activations=1, reward='any',training_fraction=0.5, n_shuffles=50, mode='AUC'):
    # select reward and non reward laps
    laps_AA = []
    laps_BA = []
    if reward=='any':
        for i in range(1, len(session.laps)):
            if session.laps[i-1].laptype == 'vertical':
                if session.laps[i].laptype == 'vertical':
                    laps_AA.append(session.laps[i-1])
                if session.laps[i].laptype == 'oblique':
                    laps_BA.append(session.laps[i-1])
    if reward=='no':
        for i in range(1, len(session.laps)):
            if session.laps[i-1].laptype == 'vertical':
                if session.laps[i].laptype == 'vertical' and session.laps[i].reward==False:
                    laps_AA.append(session.laps[i-1])
                if session.laps[i].laptype == 'oblique' and session.laps[i].reward==False:
                    laps_BA.append(session.laps[i-1])
    if reward=='yes':
        for i in range(1, len(session.laps)):
            if session.laps[i-1].laptype == 'vertical':
                if session.laps[i].laptype == 'vertical' and session.laps[i].reward==True:
                    laps_AA.append(session.laps[i-1])
                if session.laps[i].laptype == 'oblique' and session.laps[i].reward==True:
                    laps_BA.append(session.laps[i-1])

    aucs = decode_laps(laps_AA, laps_BA, n_shuffles, mode=mode, training_fraction=training_fraction, min_data=min_data, min_activations=min_activations)

    return bn.nanmean(aucs)


# old decoding function: only takes even and odd laps to divide training and test
def decode_odd_even(session, savepath='none', min_data=3, min_activations=1, region='DG', spatial=False, anti_spatial=False, min_spatial=6, megaplot_ax = 0, exclude_reward=0, only_reward=0, pseudo_count=0.5, VR=True, maptype='F'):

    if VR:
        session_spatial_cells = session.spatial_cells
    else:
        if maptype=='F':
            session_spatial_cells = session.spatial_cells_F
        else:
            session_spatial_cells = session.spatial_cells_S

    # condition only if novel and familiar laps are enough
    if (len(session.familiar_laps) >= 2*min_data) and (len(session.novel_laps) >= 2*min_data):

        # define training (even) and test (odd) activity rasters
        if exclude_reward==0 and only_reward == 0:
            odd_familiar_activity = np.vstack([l.raster for i,l in enumerate(session.familiar_laps) if i%2==1])
            even_familiar_activity = np.vstack([l.raster for i,l in enumerate(session.familiar_laps) if i%2==0])
            odd_novel_activity = np.vstack([l.raster for i,l in enumerate(session.novel_laps) if i%2==1])
            even_novel_activity = np.vstack([l.raster for i,l in enumerate(session.novel_laps) if i%2==0])

        elif exclude_reward:
            norew_F_laps = [l for l in session.familiar_laps if l.reward==0]
            norew_N_laps = [l for l in session.novel_laps if l.reward==0]

            if len(norew_F_laps)>1 and len(norew_N_laps)>1:
                odd_familiar_activity = np.vstack([l.raster for i,l in enumerate(norew_F_laps) if (i%2==1 and l.reward==0)])
                even_familiar_activity = np.vstack([l.raster for i,l in enumerate(norew_F_laps) if (i%2==0 and l.reward==0)])
                odd_novel_activity = np.vstack([l.raster for i,l in enumerate(norew_N_laps) if (i%2==1 and l.reward==0)])
                even_novel_activity = np.vstack([l.raster for i,l in enumerate(norew_N_laps) if (i%2==0 and l.reward==0)])
            else:
                return np.nan, np.nan

        elif only_reward:
            rew_F_laps = [l for l in session.familiar_laps if l.reward]
            rew_N_laps = [l for l in session.novel_laps if l.reward]

            if len(rew_F_laps)>1 and len(rew_N_laps)>1:
                odd_familiar_activity = np.vstack([l.raster for i,l in enumerate(rew_F_laps) if (i%2==1 and l.reward==1)])
                even_familiar_activity = np.vstack([l.raster for i,l in enumerate(rew_F_laps) if (i%2==0 and l.reward==1)])
                odd_novel_activity = np.vstack([l.raster for i,l in enumerate(rew_N_laps) if (i%2==1 and l.reward==1)])
                even_novel_activity = np.vstack([l.raster for i,l in enumerate(rew_N_laps) if (i%2==0 and l.reward==1)])
            else:
                return np.nan, np.nan

        if anti_spatial and np.sum(1-session_spatial_cells) < min_spatial:
            return np.nan, np.nan
        if anti_spatial:
            odd_familiar_activity = odd_familiar_activity[:, session_spatial_cells==0]
            even_familiar_activity = even_familiar_activity[:, session_spatial_cells==0]
            odd_novel_activity = odd_novel_activity[:, session_spatial_cells==0]
            even_novel_activity = even_novel_activity[:, session_spatial_cells==0]

        if spatial and np.sum(session_spatial_cells) < min_spatial:
            return np.nan, np.nan
        if spatial:
            odd_familiar_activity = odd_familiar_activity[:, session_spatial_cells]
            even_familiar_activity = even_familiar_activity[:, session_spatial_cells]
            odd_novel_activity = odd_novel_activity[:, session_spatial_cells]
            even_novel_activity = even_novel_activity[:, session_spatial_cells]

        DL_familiar, DL_novel = decode_conditions(even_familiar_activity, even_novel_activity, odd_familiar_activity, odd_novel_activity, min_activations=min_activations, pseudo_count=pseudo_count)

        # for i,l in enumerate(session.familiar_laps):
        #     if i%2 == 1:
        #         dl_f, nope = decode_conditions(even_familiar_activity, even_novel_activity, l.raster, l.raster)
        #         l.DL = dl_f
        # for i,l in enumerate(session.novel_laps):
        #     if i%2 == 1:
        #         nope, dl_n = decode_conditions(even_familiar_activity, even_novel_activity, l.raster, l.raster)
        #         l.DL = dl_n

        # print DL_familiar, DL_novel
        if len(DL_familiar) < 2 or len(DL_novel) < 2:
            return np.nan, np.nan
        if VR:
            auc, pvalue = nice.compare_populations(DL_familiar, DL_novel, 'familiar', 'novel',
                            '$\log \\frac{P(s \ | \ familiar)}{P(s \ | \ novel)}$',
                            timescale=session.settings['discretization_timescale'],
                            savepath=savepath)
        else:
            if DEBUG:
                move.printA('DL_inbound',DL_familiar)
                move.printA('DL_outbound',DL_novel)
            if np.sum(DL_familiar.astype(int)) > 0 and np.sum(DL_novel.astype(int)) > 0:
                auc, pvalue = nice.compare_populations(DL_familiar, DL_novel, 'inbound', 'outbound',
                            '$\log \\frac{P(s \ | \ inbound)}{P(s \ | \ outbound)}$',
                            timescale=session.settings['discretization_timescale'],
                            savepath=savepath)
            else:
                return np.nan, np.nan
        if megaplot_ax:
            nice.AUC(DL_familiar, DL_novel, ax = megaplot_ax)
        if savepath != 'none':
            if (spatial + anti_spatial == 0) or (spatial and np.sum(session_spatial_cells) >= min_spatial) or (anti_spatial and np.sum(1-session_spatial_cells) >= min_spatial):
                if VR:
                    visualize_session_decoded(session, n_roi_shown = 40, n_laps = 99, savepath = savepath+'_time', only_spatial_cells = spatial, only_nonspatial_cells = anti_spatial)
        return auc, pvalue
    else:
        return np.nan, np.nan



# --- spatial analysis functions --- #

def compute_spatial_decorrelation(session, savepath='none', mode='cells', maptype='F', VR=True):
    if VR:
        C_FF, C_NN, C_FN = session.compute_spatial_correlations(pairwise=VR)
    else:
        C_FF, C_NN, C_FN , _ , _, _, _, _, _ = session.compute_spatial_correlations(maptype=maptype, pairwise=VR)

    if mode=='cells':
        ff = bn.nanmean(C_FF, 1)
        nn = bn.nanmean(C_NN, 1)
        fn = bn.nanmean(C_FN, 1)
        if not VR:
            if maptype == 'F':
                flip = bn.nanmean(session.C_Flip_F, 1)
            elif maptype == 'S':
                flip = bn.nanmean(session.C_Flip_S, 1)

    if mode=='laps':
        ff = bn.nanmean(C_FF, 0)
        nn = bn.nanmean(C_NN, 0)
        fn = bn.nanmean(C_FN, 0)
        if not VR:
            if maptype == 'F':
                flip = bn.nanmean(session.C_Flip_F, 0)
            elif maptype == 'S':
                flip = bn.nanmean(session.C_Flip_S, 0)

    decorrelation = (bn.nanmean(ff) + bn.nanmean(nn) - 2*bn.nanmean(fn))/2.
    if VR:
        f, ax = visualize.box_comparison_three(ff, fn, nn, 'FF', 'FN', 'NN', 'spatial correlation (%s)' % mode)
    else:
        f, ax = visualize.box_comparison_three(ff, fn, nn, 'IN-IN', 'IN-OUT', 'OUT-OUT', 'spatial correlation (%s)' % mode)
    ax.set_title('Decorrelation = %.2f' % decorrelation)
    ax.axhline([0.0], color='k', linestyle='-')
    if savepath != 'none':
        f.savefig(savepath)
        plt.close(f)
    if VR:
        return ff, nn, fn, decorrelation
    else:
        return ff, nn, fn, decorrelation, flip

def compute_spatial_decorrelation_zscore(session, savepath='none',VR=True, maptype='F'):
    if VR:
        session.compute_spatial_zscores()
        decorrelation = (bn.nanmean(session.spatial_Z_FF) + bn.nanmean(session.spatial_Z_NN) - 2*bn.nanmean(session.spatial_Z_FN))/2.

        mask_or = (session.spatial_Z_FF > 2.0) | (session.spatial_Z_NN > 2.0)
        mask_and = (session.spatial_Z_FF > 2.0) & (session.spatial_Z_NN > 2.0)

        decorrelation_spatial_or = (bn.nanmean(session.spatial_Z_FF[mask_or]) + bn.nanmean(session.spatial_Z_NN[mask_or]) - 2*bn.nanmean(session.spatial_Z_FN[mask_or]))/2.
        decorrelation_spatial_and = (bn.nanmean(session.spatial_Z_FF[mask_and]) + bn.nanmean(session.spatial_Z_NN[mask_and]) - 2*bn.nanmean(session.spatial_Z_FN[mask_and]))/2.

        f, ax = visualize.box_comparison_three(session.spatial_Z_FF, session.spatial_Z_FN, session.spatial_Z_NN, 'FF', 'FN', 'NN', 'spatial z-score (cells)')
        ax.set_title('Decorrelation = %.2f' % decorrelation)
        ax.axhline([2.0], color='0.5', linestyle='--')
        ax.axhline([0.0], color='k', linestyle='-')
        if savepath != 'none':
            f.savefig(savepath)
            plt.close(f)

        return decorrelation, decorrelation_spatial_or, decorrelation_spatial_and
    else:
        if maptype=='F':
            session.compute_spatial_zscores()
            decorrelation_F = (bn.nanmean(session.spatial_Z_FF_F) + bn.nanmean(session.spatial_Z_NN_F) - 2*bn.nanmean(session.spatial_Z_FN_F))/2.

            mask_or_F = (session.spatial_Z_FF_F > 2.0) | (session.spatial_Z_NN_F > 2.0)
            mask_and_F = (session.spatial_Z_FF_F > 2.0) & (session.spatial_Z_NN_F > 2.0)

            decorrelation_spatial_or_F = (bn.nanmean(session.spatial_Z_FF_F[mask_or_F]) + bn.nanmean(session.spatial_Z_NN_F[mask_or_F]) - 2*bn.nanmean(session.spatial_Z_FN_F[mask_or_F]))/2.
            decorrelation_spatial_and_F = (bn.nanmean(session.spatial_Z_FF_F[mask_and_F]) + bn.nanmean(session.spatial_Z_NN_F[mask_and_F]) - 2*bn.nanmean(session.spatial_Z_FN_F[mask_and_F]))/2.

            f, ax = visualize.box_comparison_three(session.spatial_Z_FF_F, session.spatial_Z_FN_F, session.spatial_Z_NN_F, 'FF', 'FN', 'NN', 'spatial z-score (cells)')
            ax.set_title('Decorrelation = %.2f' % decorrelation_F)
            ax.axhline([2.0], color='0.5', linestyle='--')
            ax.axhline([0.0], color='k', linestyle='-')
            if savepath != 'none':
                f.savefig(savepath)
                plt.close(f)
            return decorrelation_F, decorrelation_spatial_or_F, decorrelation_spatial_and_F
        elif maptype=='S':
            decorrelation_S = (bn.nanmean(session.spatial_Z_FF_S) + bn.nanmean(session.spatial_Z_NN_S) - 2*bn.nanmean(session.spatial_Z_FN_S))/2.

            mask_or_S = (session.spatial_Z_FF_S > 2.0) | (session.spatial_Z_NN_S > 2.0)
            mask_and_S = (session.spatial_Z_FF_S > 2.0) & (session.spatial_Z_NN_S > 2.0)

            decorrelation_spatial_or_S = (bn.nanmean(session.spatial_Z_FF_S[mask_or_S]) + bn.nanmean(session.spatial_Z_NN_S[mask_or_S]) - 2*bn.nanmean(session.spatial_Z_FN_S[mask_or_S]))/2.
            decorrelation_spatial_and_S = (bn.nanmean(session.spatial_Z_FF_S[mask_and_S]) + bn.nanmean(session.spatial_Z_NN_S[mask_and_S]) - 2*bn.nanmean(session.spatial_Z_FN_S[mask_and_S]))/2.

            f, ax = visualize.box_comparison_three(session.spatial_Z_FF_S, session.spatial_Z_FN_S, session.spatial_Z_NN_S, 'FF', 'FN', 'NN', 'spatial z-score (cells)')
            ax.set_title('Decorrelation = %.2f' % decorrelation_S)
            ax.axhline([2.0], color='0.5', linestyle='--')
            ax.axhline([0.0], color='k', linestyle='-')
            if savepath != 'none':
                f.savefig(savepath)
                plt.close(f)

            return decorrelation_S, decorrelation_spatial_or_S, decorrelation_spatial_and_S

def compare_spatial_correlations_reward(session, savepath, min_data=3, region = 'DG', mode='cells'):
    reward_familiar = [l for l in session.laps if l.reward and l.laptype == "vertical"]
    nonreward_familiar = [l for l in session.laps if l.reward==0 and l.laptype == "vertical"]
    reward_novel = [l for l in session.laps if l.reward and l.laptype == "oblique"]
    nonreward_novel = [l for l in session.laps if l.reward==0 and l.laptype == "oblique"]

    C_FF_rew = [[] for i in range(session.n_roi)]
    C_NN_rew = [[] for i in range(session.n_roi)]
    C_FN_rew = [[] for i in range(session.n_roi)]

    C_FF_nonrew = [[] for i in range(session.n_roi)]
    C_NN_nonrew = [[] for i in range(session.n_roi)]
    C_FN_nonrew = [[] for i in range(session.n_roi)]

    for n in range(session.n_roi):
        for i in range(len(reward_familiar)):
            for j in range(i+1, len(reward_familiar)):
                c = ratemap_correlation(reward_familiar[i].rois[n].F_ratemap, reward_familiar[j].rois[n].F_ratemap, session.settings)
                C_FF_rew[n].append(c)

        for i in range(len(reward_novel)):
            for j in range(i+1, len(reward_novel)):
                c = ratemap_correlation(reward_novel[i].rois[n].F_ratemap, reward_novel[j].rois[n].F_ratemap, session.settings)
                C_NN_rew[n].append(c)

        for i in range(len(reward_familiar)):
            for j in range(len(reward_novel)):
                c = ratemap_correlation(reward_familiar[i].rois[n].F_ratemap, reward_novel[j].rois[n].F_ratemap, session.settings)
                C_FN_rew[n].append(c)

        for i in range(len(nonreward_familiar)):
            for j in range(i+1, len(nonreward_familiar)):
                c = ratemap_correlation(nonreward_familiar[i].rois[n].F_ratemap, nonreward_familiar[j].rois[n].F_ratemap, session.settings)
                C_FF_nonrew[n].append(c)

        for i in range(len(nonreward_novel)):
            for j in range(i+1, len(nonreward_novel)):
                c = ratemap_correlation(nonreward_novel[i].rois[n].F_ratemap, nonreward_novel[j].rois[n].F_ratemap, session.settings)
                C_NN_nonrew[n].append(c)


        for i in range(len(nonreward_familiar)):
            for j in range(len(nonreward_novel)):
                c = ratemap_correlation(nonreward_familiar[i].rois[n].F_ratemap, nonreward_novel[j].rois[n].F_ratemap, session.settings)
                C_FN_nonrew[n].append(c)

    C_FF_rew = np.asarray(C_FF_rew)
    C_NN_rew = np.asarray(C_NN_rew)
    C_FN_rew = np.asarray(C_FN_rew)

    C_FF_nonrew = np.asarray(C_FF_nonrew)
    C_NN_nonrew = np.asarray(C_NN_nonrew)
    C_FN_nonrew = np.asarray(C_FN_nonrew)

    enough_data_fam = min([len(nonreward_familiar), len(reward_familiar)]) >= min_data
    enough_data_nov = min([len(nonreward_novel), len(reward_novel)]) >= min_data

    if mode=='cells':
        avax = 1
    elif mode=='laps':
        avax = 0

    corr_rew_fam_nov = bn.nanmean(C_FN_rew, avax)
    corr_nonrew_fam_nov = bn.nanmean(C_FN_nonrew, avax)

    corr_rew_fam_fam = bn.nanmean(C_FF_rew, avax)
    corr_nonrew_fam_fam = bn.nanmean(C_FF_nonrew, avax)

    corr_rew_nov_nov = bn.nanmean(C_NN_rew, avax)
    corr_nonrew_nov_nov = bn.nanmean(C_NN_nonrew, avax)

    if enough_data_fam and enough_data_nov:
        labels = ['||| - %s NO' % labels_novel[region], '||| - %s REW' % labels_novel[region], '|||-||| NO', '|||-||| REW', '%s - %s NO' % (labels_novel[region], labels_novel[region]), '%s - %s REW' % (labels_novel[region], labels_novel[region])]
        datas = [corr_nonrew_fam_nov, corr_rew_fam_nov, corr_nonrew_fam_fam,  corr_rew_fam_fam, corr_nonrew_nov_nov, corr_rew_nov_nov]
        f, ax = visualize.box_comparison(datas, labels, 'spatial correlation (per %s)' % mode)
        visualize.annotate_ttest_p(corr_rew_fam_nov, corr_nonrew_fam_nov, 0, 1, ax)
        visualize.annotate_ttest_p(corr_rew_fam_fam, corr_nonrew_fam_fam, 2, 3, ax)
        visualize.annotate_ttest_p(corr_rew_nov_nov, corr_nonrew_nov_nov, 4, 5, ax)
        plt.savefig(savepath + '.pdf')
        plt.close('all')

        return bn.nanmean(corr_rew_fam_nov), bn.nanmean(corr_nonrew_fam_nov), bn.nanmean(corr_rew_fam_fam), bn.nanmean(corr_nonrew_fam_fam), bn.nanmean(corr_rew_nov_nov), bn.nanmean(corr_nonrew_nov_nov)

    elif enough_data_fam:
        return np.nan, np.nan, bn.nanmean(corr_rew_fam_fam), bn.nanmean(corr_nonrew_fam_fam), np.nan, np.nan
    elif enough_data_nov:
        return np.nan, np.nan, np.nan, np.nan, bn.nanmean(corr_rew_nov_nov), bn.nanmean(corr_nonrew_nov_nov)
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

# --- visualization functions --- #

def plot_laps(session, savepath, plotrois='all'):
    if plotrois=='all':
        plotrois = session.n_roi

    f, axs = plt.subplots(plotrois+2, session.nlaps, figsize=(3*session.nlaps, 1.5*(plotrois+2)), sharey='row')
    for i in range(session.nlaps):
        axs[0, i].plot(session.laps[i].times * 1.e-3, session.laps[i].position, 'k')
        axs[0, i].set_title("%s %u" % (session.laps[i].laptype, session.laps[i].index))
        axs[1, i].plot(session.laps[i].times * 1.e-3, session.laps[i].speed)
        for n in range(plotrois):
            axs[n+2, i].plot(session.laps[i].rois[n].times * 1e-3, session.laps[i].rois[n].dF_F, 'g')
            for x in session.laps[i].rois[n].time_of_activations:
                axs[n+2, i].axvline([x * 1e-3], color='r', linewidth=3, alpha=0.7)
    plt.savefig(savepath)
    plt.close()

def plot_ratemaps(session, savepath):
    a=0

def plot_spatial_maps_odd_even(session, region, savepath='none', HB=False, maptype='F', VR=False, nmin_corr=2):
    session.compute_spatial_zscores()
    if 'CA1' in region:
        min_cells = 4 #5
    elif 'DG' in region:
        min_cells = 2 #2
    else:
        min_cells = 0 #0
    if DEBUG:
        print(region, min_cells)

    if VR:
        even_familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        odd_familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        even_novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        odd_novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    else:
        print("DEBUG_1")
        if maptype == 'F':
            spatial_bins_new = len(session.familiar_laps[0].rate_maps[0])
        elif maptype == 'S':
            spatial_bins_new = len(session.familiar_laps[0].s_maps[0])

        familiar_maps = [] #np.zeros((session.n_roi, spatial_bins_new))
        novel_maps =  [] #np.zeros((session.n_roi, spatial_bins_new))
        even_familiar_maps =  [] #np.zeros((session.n_roi, spatial_bins_new))
        odd_familiar_maps =  [] #np.zeros((session.n_roi, spatial_bins_new))
        even_novel_maps =  [] #np.zeros((session.n_roi, spatial_bins_new))
        odd_novel_maps =  [] #np.zeros((session.n_roi, spatial_bins_new))

        # move.printA("inbound_maps ZERO",familiar_maps)
        # move.printA("outbound_maps ZERO",novel_maps)

    # building familiar maps

    for i,l in enumerate(session.familiar_laps):
        if maptype == 'F':
            update_maps = np.copy(l.rate_maps)
        elif maptype == 'S':
            update_maps = np.copy(l.s_maps)
        if not VR:
            if DEBUG:
                move.printA("UPDATE_MAPs NAN", update_maps)
            #update_maps = move.change_nan_zero(update_maps)
            if DEBUG:
                move.printA("UPDATE_MAPs ZERO", update_maps)
        if i%2 == 0:
            even_familiar_maps.append(update_maps)
        if i%2 == 1:
            odd_familiar_maps.append(update_maps)
        familiar_maps.append(update_maps)
        #move.printA("inbound_ratemaps UPDATE",familiar_maps)

    even_familiar_maps = bn.nanmean(even_familiar_maps, axis=0)
    odd_familiar_maps = bn.nanmean(odd_familiar_maps, axis=0)
    familiar_maps = bn.nanmean(familiar_maps, axis=0)

    for i,l in enumerate(session.novel_laps):
        if maptype == 'F':
            update_maps = np.copy(l.rate_maps)
        elif maptype == 'S':
            update_maps = np.copy(l.s_maps)
        if not VR:
            if DEBUG:
                move.printA("UPDATE_MAPs NAN", update_maps)
            #update_maps = move.change_nan_zero(update_maps)
            if DEBUG:
                move.printA("UPDATE_MAPs ZERO", update_maps)
        if i%2 == 0:
            even_novel_maps.append(update_maps)
        if i%2 == 1:
            odd_novel_maps.append(update_maps)
        novel_maps.append(update_maps)

    even_novel_maps = bn.nanmean(even_novel_maps, axis=0)
    if len(odd_novel_maps):
        odd_novel_maps = bn.nanmean(odd_novel_maps, axis=0)
    else:
        odd_novel_maps = np.empty((session.n_roi, spatial_bins_new))
        odd_novel_maps[:] = np.nan

    novel_maps = bn.nanmean(novel_maps, axis=0)

    if not VR and DEBUG:
        if maptype == 'F':
            move.printA("inbound_ratemaps",familiar_maps)
            move.printA("outbound_ratemaps",novel_maps)
            move.printA("even_inbound_maps",even_familiar_maps)
            move.printA("odd_inbound_maps",odd_familiar_maps)
            move.printA("even_outbound_maps",even_novel_maps)
            move.printA("odd_outbound_maps",odd_novel_maps)
        elif maptype == 'S':
            move.printA("inbound_smaps",familiar_maps)
            move.printA("outbound_smaps",novel_maps)
            move.printA("even_inbound_maps",even_familiar_maps)
            move.printA("odd_inbound_maps",odd_familiar_maps)
            move.printA("even_outbound_maps",even_novel_maps)
            move.printA("odd_outbound_maps",odd_novel_maps)
        #print(stop)

    if len(familiar_maps) and len(novel_maps) and len(session.familiar_laps)>1 and len(session.novel_laps)>1:
        # print(savepath+'_AC')
        pov_FF_allcells, pov_NN_allcells, pov_FN_allcells, pov_FF_allcells_percell, pov_NN_allcells_percell, pov_FN_allcells_percell, pov_FLIP_allcells, pov_FLIP_allcells_percell, nF_allcells, nN_allcells = \
            visualize.visualize_spatial_maps(
                even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_AC_F', HB=HB, min_cells=min_cells, VR=VR, nmin_corr=nmin_corr)
        pov_FF_allcells, pov_NN_allcells, pov_FN_allcells, pov_FF_allcells_percell, pov_NN_allcells_percell, pov_FN_allcells_percell, pov_FLIP_allcells, pov_FLIP_allcells_percell, nF_allcells, nN_allcells = \
            visualize.visualize_spatial_maps(
                even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_AC_F', HB=HB, mode='F', min_cells=min_cells, VR=VR, nmin_corr=nmin_corr)
    else:
        pov_FF_allcells = np.nan
        pov_NN_allcells = np.nan
        pov_FN_allcells = np.nan
        pov_FLIP_allcells = np.nan
        pov_FF_allcells_percell = np.nan
        pov_NN_allcells_percell = np.nan
        pov_FN_allcells_percell = np.nan
        pov_FLIP_allcells_percell = np.nan
        nF_allcells = 0
        nN_allcells = 0

    povs_allcells = [
        pov_FF_allcells, pov_NN_allcells, pov_FN_allcells, pov_FF_allcells_percell, pov_NN_allcells_percell, pov_FN_allcells_percell, pov_FLIP_allcells, pov_FLIP_allcells_percell,
        nF_allcells, nN_allcells]

    # selecting only spatial cells
    if VR:

        p_even_familiar_maps = even_familiar_maps[session.spatial_cells]
        p_odd_familiar_maps = odd_familiar_maps[session.spatial_cells]
        p_familiar_maps = familiar_maps[session.spatial_cells]

        p_even_novel_maps = even_novel_maps[session.spatial_cells]
        p_odd_novel_maps = odd_novel_maps[session.spatial_cells]
        p_novel_maps = novel_maps[session.spatial_cells]
    else:
        print("DEBUG_1")
        if maptype == 'F':
            p_even_familiar_maps = even_familiar_maps[session.spatial_cells_F]
            p_odd_familiar_maps = odd_familiar_maps[session.spatial_cells_F]
            p_familiar_maps = familiar_maps[session.spatial_cells_F]

            p_even_novel_maps = even_novel_maps[session.spatial_cells_F]
            p_odd_novel_maps = odd_novel_maps[session.spatial_cells_F]
            p_novel_maps = novel_maps[session.spatial_cells_F]
        else:
            p_even_familiar_maps = even_familiar_maps[session.spatial_cells_S]
            p_odd_familiar_maps = odd_familiar_maps[session.spatial_cells_S]
            p_familiar_maps = familiar_maps[session.spatial_cells_S]

            p_even_novel_maps = even_novel_maps[session.spatial_cells_S]
            p_odd_novel_maps = odd_novel_maps[session.spatial_cells_S]
            p_novel_maps = novel_maps[session.spatial_cells_S]


    #print("# spatial cells ="+ len(p_even_familiar_maps))
    #print("# all cells ="+ len(even_familiar_maps))
    #if (len(p_familiar_maps) and len(p_novel_maps) and len(session.familiar_laps)>1 and len(session.novel_laps)>1) or not VR:
    #new if statement added by SS date: 20.07.2021
    if ((len(p_familiar_maps) and len(p_novel_maps) and len(session.familiar_laps)>1 and len(session.novel_laps)>1)) or (not VR):
        # print(savepath+'_PC')
        pov_FF, pov_NN, pov_FN, pov_FF_percell, pov_NN_percell, pov_FN_percell, pov_FLIP, pov_FLIP_percell, nF, nN = visualize.visualize_spatial_maps(
            p_even_familiar_maps, p_odd_familiar_maps, p_even_novel_maps, p_odd_novel_maps, p_familiar_maps, p_novel_maps, savepath=savepath+'_PC', HB=HB, min_cells=min_cells, VR=VR, nmin_corr=nmin_corr)
        # print(savepath+'_PC_F')
        pov_FF_f, pov_NN_f, pov_FN_f, pov_FF_f_percell, pov_NN_f_percell, pov_FN_f_percell, pov_FLIP_f, pov_FLIP_f_percell, nF_f, nN_f = visualize.visualize_spatial_maps(
            p_even_familiar_maps, p_odd_familiar_maps, p_even_novel_maps, p_odd_novel_maps, p_familiar_maps, p_novel_maps, savepath=savepath+'_PC_F', HB=HB, mode='F', min_cells=min_cells, VR=VR, nmin_corr=nmin_corr)
    else:
        pov_FF = np.nan
        pov_NN = np.nan
        pov_FN = np.nan
        pov_FLIP = np.nan
        pov_FF_percell = np.nan
        pov_NN_percell = np.nan
        pov_FN_percell = np.nan
        pov_FLIP_percell = np.nan
        nF = 0
        nN = 0
        pov_FF_f = np.nan
        pov_NN_f = np.nan
        pov_FN_f = np.nan
        pov_FLIP_f = np.nan
        pov_FF_f_percell = np.nan
        pov_NN_f_percell = np.nan
        pov_FN_f_percell = np.nan
        pov_FLIP_f_percell = np.nan
        nF_f = 0
        nN_f = 0

    povs = [pov_FF, pov_NN, pov_FN, pov_FF_percell, pov_NN_percell, pov_FN_percell, pov_FLIP, pov_FLIP_percell, nF, nN]
    povs_f = [pov_FF_f, pov_NN_f, pov_FN_f, pov_FF_f_percell, pov_NN_f_percell, pov_FN_f_percell, pov_FLIP_f, pov_FLIP_f_percell, nF_f, nN_f]
    if VR:
        return p_even_familiar_maps, p_odd_familiar_maps, p_even_novel_maps, p_odd_novel_maps, p_familiar_maps, p_novel_maps, povs, povs_f, povs_allcells
    else:
        return p_even_familiar_maps, p_odd_familiar_maps, p_even_novel_maps, p_odd_novel_maps, p_familiar_maps, p_novel_maps, povs, povs_f, povs_allcells, even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps

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

def plot_spatial_maps_conditions(session, region, savepath='none', HB=False, maptype='F', VR=True, YMAZE=False, nmin_corr=2):
    try:
        conds = session.vrdict['conditions']
    except:
        conds = 'none'
    assert(conds != 'none')
    assert(conds == ['A', 'B', 'C'])
    if not YMAZE:
        ends = session.vrdict['tracktimes_ends']
        if DEBUG:
            print(('condition A:['+str(ends[0])+','+str(ends[1])+']'))
            print(('condition B:['+str(ends[1])+','+str(ends[2])+']'))
            print(('condition C:['+str(ends[2])+','+str(ends[3])+']'))

    session.compute_spatial_zscores()
    if 'CA1' in region:
        min_cells = 4
    elif 'DG' in region:
        min_cells = 4
    else:
        min_cells = 4
    if DEBUG:
        print(region, min_cells)

    if VR:
        familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        even_familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        odd_familiar_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        even_novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
        odd_novel_maps = np.zeros((session.n_roi, session.settings["spatial_bins"]))
    else:
        if maptype == 'F':
            spatial_bins_new = len(session.familiar_laps[0].rate_maps[0])
        elif maptype == 'S':
            spatial_bins_new = len(session.familiar_laps[0].s_maps[0])

        familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        A_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        B_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        C_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        A_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        B_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        C_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))

        even_A_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_A_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        even_B_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_B_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        even_C_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_C_familiar_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        even_A_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_A_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        even_B_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_B_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        even_C_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))
        odd_C_novel_maps = [] # np.zeros((session.n_roi, spatial_bins_new))

        if DEBUG:
            move.printA("inbound_maps ZERO",familiar_maps)
            move.printA("outbound_maps ZERO",novel_maps)

    # building familiar maps

    for i,l in enumerate(session.familiar_laps):

        if maptype == 'F':
            update_maps = np.copy(l.rate_maps)
        elif maptype == 'S':
            update_maps = np.copy(l.s_maps)
        if not VR:
            if DEBUG:
                move.printA("UPDATE_MAPs NAN", update_maps)
            #update_maps = move.change_nan_zero(update_maps)
            if DEBUG:
                move.printA("UPDATE_MAPs ZERO", update_maps)
        if YMAZE:
            if l.arm == 'arm A':
                if i%2 == 0:
                    even_A_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_A_familiar_maps.append(update_maps)
                A_familiar_maps.append(update_maps)
            elif l.arm == 'arm B':
                if i%2 == 0:
                    even_B_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_B_familiar_maps.append(update_maps)
                B_familiar_maps.append(update_maps)
            elif l.arm == 'arm C':
                if i%2 == 0:
                    even_C_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_C_familiar_maps.append(update_maps)
                C_familiar_maps.append(update_maps)
            else:
                print(("The "+str(i)+"th inbound lap belongs to neither A nor B nor C: ["+str(l.times[0])+","+str(l.times[-1])+"]"))
                assert(False)
        else:
            if in_period(l.times[0],ends[0],ends[1]) and in_period(l.times[-1],ends[0],ends[1]):
                if i%2 == 0:
                    even_A_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_A_familiar_maps.append(update_maps)
                A_familiar_maps.append(update_maps)
            elif in_period(l.times[0],ends[1],ends[2]) and in_period(l.times[-1],ends[1],ends[2]):
                if i%2 == 0:
                    even_B_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_B_familiar_maps.append(update_maps)
                B_familiar_maps.append(update_maps)
            elif in_period(l.times[0],ends[2],ends[3]) and in_period(l.times[-1],ends[2],ends[3]):
                if i%2 == 0:
                    even_C_familiar_maps.append(update_maps)
                if i%2 == 1:
                    odd_C_familiar_maps.append(update_maps)
                C_familiar_maps.append(update_maps)
            else:
                if DEBUG:
                    print(("The "+str(i)+"th inbound lap belongs to neither A nor B nor C: ["+str(l.times[0])+","+str(l.times[-1])+"]"))
                    print(('condition A:['+str(ends[0])+','+str(ends[1])+']'))
                    print(('condition B:['+str(ends[1])+','+str(ends[2])+']'))
                    print(('condition C:['+str(ends[2])+','+str(ends[3])+']'))
                #assert(False)

        familiar_maps.append(update_maps)
        if DEBUG:
            move.printA("inbound_maps UPDATE",familiar_maps)

    A_familiar_maps = bn.nanmean(A_familiar_maps, axis=0)
    B_familiar_maps = bn.nanmean(B_familiar_maps, axis=0)
    C_familiar_maps = bn.nanmean(C_familiar_maps, axis=0)
    even_A_familiar_maps = bn.nanmean(even_A_familiar_maps, axis=0)
    even_B_familiar_maps = bn.nanmean(even_B_familiar_maps, axis=0)
    even_C_familiar_maps = bn.nanmean(even_C_familiar_maps, axis=0)
    odd_A_familiar_maps = bn.nanmean(odd_A_familiar_maps, axis=0)
    odd_B_familiar_maps = bn.nanmean(odd_B_familiar_maps, axis=0)
    odd_C_familiar_maps = bn.nanmean(odd_C_familiar_maps, axis=0)
    familiar_maps = bn.nanmean(familiar_maps, axis=0)

    for i,l in enumerate(session.novel_laps):
        if maptype == 'F':
            update_maps = np.copy(l.rate_maps)
        elif maptype == 'S':
            update_maps = np.copy(l.s_maps)
        if not VR:
            if DEBUG:
                move.printA("UPDATE_MAPs NAN", update_maps)
            #update_maps = move.change_nan_zero(update_maps)
            if DEBUG:
                move.printA("UPDATE_MAPs ZERO", update_maps)
        if YMAZE:
            if l.arm == 'arm A':
                if i%2 == 0:
                    even_A_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_A_novel_maps.append(update_maps)
                A_novel_maps.append(update_maps)
            elif l.arm == 'arm B':
                if i%2 == 0:
                    even_B_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_B_novel_maps.append(update_maps)
                B_novel_maps.append(update_maps)
            elif l.arm == 'arm C':
                if i%2 == 0:
                    even_C_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_C_novel_maps.append(update_maps)
                C_novel_maps.append(update_maps)
            else:
                print(("The "+str(i)+"th outbound lap belongs to neither A nor B nor C: ["+str(l.times[0])+","+str(l.times[-1])+"]"))
                assert(False)
        else:
            if in_period(l.times[0],ends[0],ends[1]) and in_period(l.times[-1],ends[0],ends[1]):
                if i%2 == 0:
                    even_A_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_A_novel_maps.append(update_maps)
                A_novel_maps.append(update_maps)
            elif in_period(l.times[0],ends[1],ends[2]) and in_period(l.times[-1],ends[1],ends[2]):
                if i%2 == 0:
                    even_B_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_B_novel_maps.append(update_maps)
                B_novel_maps.append(update_maps)
            elif in_period(l.times[0],ends[2],ends[3]) and in_period(l.times[-1],ends[2],ends[3]):
                if i%2 == 0:
                    even_C_novel_maps.append(update_maps)
                if i%2 == 1:
                    odd_C_novel_maps.append(update_maps)
                C_novel_maps.append(update_maps)
            else:
                if DEBUG:
                    print(("The "+str(i)+"th outbound lap belongs to neither A nor B nor C: ["+str(l.times[0])+","+str(l.times[-1])+"]"))
                    print(('condition A:['+str(ends[0])+','+str(ends[1])+']'))
                    print(('condition B:['+str(ends[1])+','+str(ends[2])+']'))
                    print(('condition C:['+str(ends[2])+','+str(ends[3])+']'))
                #assert(False)
        novel_maps.append(update_maps)

    A_novel_maps = bn.nanmean(A_novel_maps, axis=0)
    B_novel_maps = bn.nanmean(B_novel_maps, axis=0)
    C_novel_maps = bn.nanmean(C_novel_maps, axis=0)
    even_A_novel_maps = bn.nanmean(even_A_novel_maps, axis=0)
    even_B_novel_maps = bn.nanmean(even_B_novel_maps, axis=0)
    even_C_novel_maps = bn.nanmean(even_C_novel_maps, axis=0)
    odd_A_novel_maps = bn.nanmean(odd_A_novel_maps, axis=0)
    odd_B_novel_maps = bn.nanmean(odd_B_novel_maps, axis=0)
    odd_C_novel_maps = bn.nanmean(odd_C_novel_maps, axis=0)
    novel_maps = bn.nanmean(novel_maps, axis=0)

    if not VR and DEBUG:
        if maptype == 'F':
            move.printA("inbound_ratemaps",familiar_maps)
            move.printA("outbound_ratemaps",novel_maps)
            move.printA("A_inbound_ratemaps",A_familiar_maps)
            move.printA("B_inbound_ratemaps",B_familiar_maps)
            move.printA("C_inbound_ratemaps",C_familiar_maps)
            move.printA("A_outbound_ratemaps",A_novel_maps)
            move.printA("B_outbound_ratemaps",B_novel_maps)
            move.printA("C_outbound_ratemaps",C_novel_maps)
        elif maptype == 'S':
            print(np.nansum(familiar_maps))
            move.printA("inbound_smaps",familiar_maps)
            print(np.nansum(novel_maps))
            move.printA("outbound_smaps",novel_maps)
            print(np.nansum(A_familiar_maps))
            move.printA("A_inbound_smaps",A_familiar_maps)
            print(np.nansum(B_familiar_maps))
            move.printA("B_inbound_smaps",B_familiar_maps)
            print(np.nansum(C_familiar_maps))
            move.printA("C_inbound_smaps",C_familiar_maps)
            print(np.nansum(A_novel_maps))
            move.printA("A_outbound_smaps",A_novel_maps)
            print(np.nansum(B_novel_maps))
            move.printA("B_outbound_smaps",B_novel_maps)
            print(np.nansum(C_novel_maps))
            move.printA("C_outbound_smaps",C_novel_maps)


    if len(A_familiar_maps) and len(A_novel_maps) and len(B_familiar_maps) and len(B_novel_maps) and np.nansum(A_familiar_maps) and np.nansum(A_novel_maps) and np.nansum(B_familiar_maps) and np.nansum(B_novel_maps):
        if DEBUG:
            print(savepath+'_A_vs_B')
        pov_FF_allcells_AB, pov_NN_allcells_AB, pov_FN_allcells_AB, pov_FF_allcells_AB_percell, pov_NN_allcells_AB_percell, pov_FN_allcells_AB_percell, pov_FLIP_allcells_AB, pov_FLIP_allcells_AB_percell, nF_allcells_AB, nN_allcells_AB = \
            visualize.visualize_spatial_maps(
                A_familiar_maps, B_familiar_maps, A_novel_maps, B_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_A_vs_B', HB=HB, mode='F', min_cells=min_cells, VR=VR, conds=['A','B'], nmin_corr=nmin_corr)
    else:
        pov_FF_allcells_AB = np.nan
        pov_NN_allcells_AB = np.nan
        pov_FN_allcells_AB = np.nan
        nF_allcells_AB = 0
        nN_allcells_AB = 0

    if len(A_familiar_maps) and len(A_novel_maps) and len(C_familiar_maps) and len(C_novel_maps) and np.nansum(A_familiar_maps) and np.nansum(A_novel_maps) and np.nansum(C_familiar_maps) and np.nansum(C_novel_maps):
        if DEBUG:
            print(savepath+'_A_vs_C')
        pov_FF_allcells_AC, pov_NN_allcells_AC, pov_FN_allcells_AC, pov_FF_allcells_AC_percell, pov_NN_allcells_AC_percell, pov_FN_allcells_AC_percell, pov_FLIP_allcells_AC, pov_FLIP_allcells_AC_percell, nF_allcells_AC, nN_allcells_AC = \
            visualize.visualize_spatial_maps(
                A_familiar_maps, C_familiar_maps, A_novel_maps, C_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_A_vs_C', HB=HB, mode='F', min_cells=min_cells, VR=VR, conds=['A','C'], nmin_corr=nmin_corr)
    else:
        pov_FF_allcells_AC = np.nan
        pov_NN_allcells_AC = np.nan
        pov_FN_allcells_AC = np.nan
        nF_allcells_AC = 0
        nN_allcells_AC = 0

    if len(B_familiar_maps) and len(B_novel_maps) and len(C_familiar_maps) and len(C_novel_maps) and np.nansum(B_familiar_maps) and np.nansum(B_novel_maps) and np.nansum(C_familiar_maps) and np.nansum(C_novel_maps):
        if DEBUG:
            print(savepath+'_B_vs_C')
        pov_FF_allcells_BC, pov_NN_allcells_BC, pov_FN_allcells_BC, pov_FF_allcells_BC_percell, pov_NN_allcells_BC_percell, pov_FN_allcells_BC_percell, pov_FLIP_allcells_BC, pov_FLIP_allcells_BC_percell, nF_allcells_BC, nN_allcells_BC = \
            visualize.visualize_spatial_maps(B_familiar_maps, C_familiar_maps, B_novel_maps, C_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_B_vs_C', HB=HB, mode='F', min_cells=min_cells, VR=VR, conds=['B','C'], nmin_corr=nmin_corr)
    else:
        pov_FF_allcells_BC = np.nan
        pov_NN_allcells_BC = np.nan
        pov_FN_allcells_BC = np.nan
        nF_allcells_BC = 0
        nN_allcells_BC = 0

    povs_allcells_AB = [pov_FF_allcells_AB, pov_NN_allcells_AB, pov_FN_allcells_AB, nF_allcells_AB, nN_allcells_AB]
    povs_allcells_AC = [pov_FF_allcells_AC, pov_NN_allcells_AC, pov_FN_allcells_AC, nF_allcells_AC, nN_allcells_AC]
    povs_allcells_BC = [pov_FF_allcells_BC, pov_NN_allcells_BC, pov_FN_allcells_BC, nF_allcells_BC, nN_allcells_BC]

    return A_familiar_maps, B_familiar_maps, C_familiar_maps, A_novel_maps, B_novel_maps, C_novel_maps, \
even_A_familiar_maps, even_B_familiar_maps, even_C_familiar_maps, even_A_novel_maps, even_B_novel_maps, even_C_novel_maps, \
odd_A_familiar_maps, odd_B_familiar_maps, odd_C_familiar_maps, odd_A_novel_maps, odd_B_novel_maps, odd_C_novel_maps

    """
    # selecting only spatial cells
    if VR:
        even_familiar_maps = even_familiar_maps[session.spatial_cells]
        odd_familiar_maps = odd_familiar_maps[session.spatial_cells]
        familiar_maps = familiar_maps[session.spatial_cells]

        even_novel_maps = even_novel_maps[session.spatial_cells]
        odd_novel_maps = odd_novel_maps[session.spatial_cells]
        novel_maps = novel_maps[session.spatial_cells]
    else:
        pass

        if maptype == 'F':
            even_familiar_maps = even_familiar_maps[session.spatial_cells_F]
            odd_familiar_maps = odd_familiar_maps[session.spatial_cells_F]
            familiar_maps = familiar_maps[session.spatial_cells_F]

            even_novel_maps = even_novel_maps[session.spatial_cells_F]
            odd_novel_maps = odd_novel_maps[session.spatial_cells_F]
            novel_maps = novel_maps[session.spatial_cells_F]
        else:
            even_familiar_maps = even_familiar_maps[session.spatial_cells_S]
            odd_familiar_maps = odd_familiar_maps[session.spatial_cells_S]
            familiar_maps = familiar_maps[session.spatial_cells_S]

            even_novel_maps = even_novel_maps[session.spatial_cells_S]
            odd_novel_maps = odd_novel_maps[session.spatial_cells_S]
            novel_maps = novel_maps[session.spatial_cells_S]


    if (len(familiar_maps) and len(novel_maps) and len(session.familiar_laps)>1 and len(session.novel_laps)>1) or not VR:
        print savepath+'_PC'
        pov_FF, pov_NN, pov_FN, nF, nN = visualize.visualize_spatial_maps(even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_PC', HB=HB, min_cells=min_cells, VR=VR)
        print savepath+'_PC_F'
        pov_FF_f, pov_NN_f, pov_FN_f, nF_f, nN_f = visualize.visualize_spatial_maps(even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, savepath=savepath+'_PC_F', HB=HB, mode='F', min_cells=min_cells, VR=VR)
    else:
        pov_FF = np.nan
        pov_NN = np.nan
        pov_FN = np.nan
        nF = 0
        nN = 0
        pov_FF_f = np.nan
        pov_NN_f = np.nan
        pov_FN_f = np.nan
        nF_f = 0
        nN_f = 0

    povs = [pov_FF, pov_NN, pov_FN, nF, nN]
    povs_f = [pov_FF_f, pov_NN_f, pov_FN_f, nF_f, nN_f]

    return even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, povs, povs_f, povs_allcells
    """


def compute_spatial_decorrelation_conditions(session, savepath='none', mode='cells', maptype='F', VR=True):
    try:
        conds = session.vrdict['conditions']
    except:
        conds = 'none'
    assert(conds != 'none')
    assert(conds == ['A', 'B', 'C'])

    C_FF_abc, C_NN_abc, C_FN_abc, C_Flip_abc = session.compute_spatial_correlations_conditions(force=True,maptype=maptype, pairwise=VR)

    if mode=='cells':
        ff_a = bn.nanmean(C_FF_abc[0], 1)
        nn_a = bn.nanmean(C_NN_abc[0], 1)
        fn_a = bn.nanmean(C_FN_abc[0], 1)
        flip_a = bn.nanmean(C_Flip_abc[0], 1)
        ff_b = bn.nanmean(C_FF_abc[1], 1)
        nn_b = bn.nanmean(C_NN_abc[1], 1)
        fn_b = bn.nanmean(C_FN_abc[1], 1)
        flip_b = bn.nanmean(C_Flip_abc[1], 1)
        ff_c = bn.nanmean(C_FF_abc[2], 1)
        nn_c = bn.nanmean(C_NN_abc[2], 1)
        fn_c = bn.nanmean(C_FN_abc[2], 1)
        flip_c = bn.nanmean(C_Flip_abc[2], 1)

    if mode=='laps':
        ff_a = bn.nanmean(C_FF_abc[0], 0)
        nn_a = bn.nanmean(C_NN_abc[0], 0)
        fn_a = bn.nanmean(C_FN_abc[0], 0)
        flip_a = bn.nanmean(C_Flip_abc[0], 0)
        ff_b = bn.nanmean(C_FF_abc[1], 0)
        nn_b = bn.nanmean(C_NN_abc[1], 0)
        fn_b = bn.nanmean(C_FN_abc[1], 0)
        flip_b = bn.nanmean(C_Flip_abc[1], 0)
        ff_c = bn.nanmean(C_FF_abc[2], 0)
        nn_c = bn.nanmean(C_NN_abc[2], 0)
        fn_c = bn.nanmean(C_FN_abc[2], 0)
        flip_c = bn.nanmean(C_Flip_abc[2], 0)

    decorrelation_a = (bn.nanmean(ff_a) + bn.nanmean(nn_a) - 2*bn.nanmean(fn_a))/2.
    decorrelation_b = (bn.nanmean(ff_b) + bn.nanmean(nn_b) - 2*bn.nanmean(fn_b))/2.
    decorrelation_c = (bn.nanmean(ff_c) + bn.nanmean(nn_c) - 2*bn.nanmean(fn_c))/2.


    f, ax = visualize.box_comparison_three(ff_a, fn_a, nn_a, 'IN-IN', 'IN-OUT', 'OUT-OUT', 'spatial correlation (%s)' % mode)
    ax.set_title('Decorrelation = %.2f' % decorrelation_a)
    ax.axhline([0.0], color='k', linestyle='-')
    if savepath != 'none':
        f.savefig(savepath+'_a.pdf')
        plt.close(f)

    f, ax = visualize.box_comparison_three(ff_b, fn_b, nn_b, 'IN-IN', 'IN-OUT', 'OUT-OUT', 'spatial correlation (%s)' % mode)
    ax.set_title('Decorrelation = %.2f' % decorrelation_b)
    ax.axhline([0.0], color='k', linestyle='-')
    if savepath != 'none':
        f.savefig(savepath+'_b.pdf')
        plt.close(f)

    f, ax = visualize.box_comparison_three(ff_c, fn_c, nn_c, 'IN-IN', 'IN-OUT', 'OUT-OUT', 'spatial correlation (%s)' % mode)
    ax.set_title('Decorrelation = %.2f' % decorrelation_c)
    ax.axhline([0.0], color='k', linestyle='-')
    if savepath != 'none':
        f.savefig(savepath+'_c.pdf')
        plt.close(f)

    return ff_a, nn_a, fn_a, decorrelation_a, flip_a, ff_b, nn_b, fn_b, decorrelation_b, flip_b, ff_c, nn_c, fn_c, decorrelation_c, flip_c
