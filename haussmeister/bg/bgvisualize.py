# from settings import *
import os
import matplotlib as mpl
import sys
if sys.version_info.major < 3:
    if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive PDF backend')
        mpl.use('PDF')
import matplotlib.font_manager

mpl.rcParams.update({'figure.autolayout': True})
# mpl.rcParams['font.sans-serif'] = "Arial"
# mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['lines.linewidth'] = 1.0
mpl.rcParams['axes.labelsize'] = 12
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import numpy as np
import bottleneck as bn
import time
import pandas as pd
import seaborn as sns
import pickle
import maps
import scipy.stats as stats
from scipy.stats import ttest_ind as ttest
from scipy.stats import ttest_rel as ttest_p
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import linregress
from scipy.stats import kruskal
from scipy.stats import sem
pltcolors = plt.rcParams['axes.prop_cycle'].by_key()['color']
from math import ceil, floor
import move
from haussmeister import spectral as hspectral
from statsmodels.stats import multitest
from statsmodels.stats.multitest import multipletests as mlt

labels_novel = {
    'TEST' : '///',
    'HILUS' : '///',
    'YMAZE' : 'yyy',
    'DG' : '///',
    'DG_CL' : 'ooo',
    'DG_re' : '---',
    'CA1' : '///',
    'CA1_CL' : 'ooo',
    'CA1_re' : '---',
    'CA3' : '///',
    'CA3_CL' : 'ooo',
    'CA3_re' : '---',
    'CA1_CL_re' : '###',
    'CA1_nolights' : 'no',
    'CA1_CL_vert' : 'o|',
    }


def behaviour_FN(db, min_topi, savepath, quantity1='hit_rate_F', quantity2='hit_rate_N'):
    ddf_grat = db[(db['region'] == 'CA1') | (db['region'] == 'DG')]
    ddf_chun = db[(db['region'] == 'CA1_CL') | (db['region'] == 'DG_CL')]

    ddf_grat = db[(db['region'] == 'CA1')]
    ddf_chun = db[(db['region'] == 'CA1_CL')]

    # exclude data points where number of mice is smaller than min_topi - gratings
    xs = np.arange(1, np.max(ddf_grat['session_number'])+1)
    max_number_grat = np.max([n for n in xs if len(ddf_grat[ddf_grat['session_number'].values == n]) >= min_topi])
    ddf_grat = ddf_grat[ddf_grat['session_number'].values <= max_number_grat]
    xs_grat = np.arange(1, np.max(ddf_grat['session_number'])+1)

    # exclude data points where number of mice is smaller than min_topi - chunland
    xs = np.arange(1, np.max(ddf_chun['session_number'])+1)
    max_number_chun = np.max([n for n in xs if len(ddf_chun[ddf_chun['session_number'].values == n]) >= min_topi])
    ddf_chun = ddf_chun[ddf_chun['session_number'].values <= max_number_chun]
    xs_chun = np.arange(1, np.max(ddf_chun['session_number'])+1)

    # compare familiar and novel - gratings
    f, ax = plt.subplots(figsize=(4,3))
    pointplotmio(ddf_grat, xs_grat, quantity1, ax, 'Familiar')
    pointplotmio(ddf_grat, xs_grat, quantity2, ax, 'Novel')
    plt.legend()
    plt.ylabel('Hit rate (rewards/lap)')
    plt.xlabel('Session day')
    plt.savefig(savepath+'_FN_grat.pdf')

    # compare familiar and novel - chunland
    f, ax = plt.subplots(figsize=(4,3))
    pointplotmio(ddf_chun, xs_chun, quantity1, ax, 'Familiar')
    pointplotmio(ddf_chun, xs_chun, quantity2, ax, 'Novel')
    plt.legend()
    plt.ylabel('Hit rate (rewards/lap)')
    plt.xlabel('Session day')
    plt.savefig(savepath+'_FN_chun.pdf')

    # compare gratings and chunland - novel
    f, ax = plt.subplots(figsize=(4,3))
    xs = np.arange(1, np.min([np.max(xs_grat), np.max(xs_chun)])+1)
    means_grat = pointplotmio(ddf_grat, xs, quantity2, ax, 'Gratings')
    means_chun = pointplotmio(ddf_chun, xs, quantity2, ax, 'Chunland')
    plt.legend()
    plt.ylabel(quantity2)
    # plt.ylabel('Spatial decorr (cells)')
    plt.xlabel('Session day')
    plt.axhline([0.6], linestyle='--', color='k')
    plt.ylim([0, 0.7])
    plt.savefig(savepath+'_N_chun-grat.pdf')

    box_comparison_two(means_grat, means_chun, '//', 'O', quantity2, box=False, paired=True, swarm=False, force=True)
    plt.savefig(savepath+'_paired_N.pdf')

    # compare gratings and chunland - familiar
    f, ax = plt.subplots(figsize=(4,3))
    xs = np.arange(1, np.min([np.max(xs_grat), np.max(xs_chun)])+1)
    means_grat = pointplotmio(ddf_grat, xs, quantity1, ax, 'Gratings')
    means_chun = pointplotmio(ddf_chun, xs, quantity1, ax, 'Chunland')
    plt.legend()
    plt.ylabel(quantity1)
    # plt.ylabel('Spatial decorr (cells)')
    plt.xlabel('Session day')
    plt.axhline([0.6], linestyle='--', color='k')
    plt.ylim([0, 0.7])
    plt.savefig(savepath+'_F_chun-grat.pdf')

    box_comparison_two(means_grat, means_chun, '//', 'O', quantity2, box=False, paired=True, swarm=False, force=True)
    plt.savefig(savepath+'_paired_F.pdf')


def plot_correlations_reward(db, region, savepath, corr='RV'):
    data = db[(db['region'] ==  region)]
    datas = [data['%s_corr_FN_noreward' % corr].values, data['%s_corr_FN_reward' % corr].values, data['%s_corr_FF_noreward' % corr].values, data['%s_corr_FF_reward' % corr].values,  data['%s_corr_NN_noreward' % corr].values, data['%s_corr_NN_reward' % corr].values]
    labels = ['||| - %s NO' % labels_novel[region], '||| - %s REW' % labels_novel[region], '|||-||| NO', '|||-||| REW', '%s - %s NO' % (labels_novel[region], labels_novel[region]), '%s - %s REW' % (labels_novel[region], labels_novel[region])]
    f, ax = box_comparison(datas, labels, '%s correlation (per session)' % corr, swarm=False, scatter=True)
    annotate_wilcoxon_p(datas[0], datas[1], 0, 1, ax, pairplot=True, force=True)
    annotate_wilcoxon_p(datas[2], datas[3], 2, 3, ax, pairplot=True, force=True)
    annotate_wilcoxon_p(datas[4], datas[5], 4, 5, ax, pairplot=True, force=True)
    plt.savefig(savepath+'/%s_lap_%s_correlations.pdf' % (corr, region))
    plt.close(f)


def plot_rate_increase_reward(db, region, savepath):
    data = db[(db['region'] ==  region)]
    datas = [data['mean_firing_rate_F_reward'].values - data['mean_firing_rate_F_noreward'].values, data['mean_firing_rate_N_reward'].values - data['mean_firing_rate_N_noreward'].values]
    labels = ['|||', '%s' % labels_novel[region]]
    f, ax = box_comparison(datas, labels, '$<r^{rew} - r^{no rew}>$ (ev/s) - per session', box =False, swarm=True)
    annotate_wilcoxon_p_single(datas[0], -0.3, 0.3, ax)
    annotate_wilcoxon_p_single(datas[1], 0.7, 1.3, ax)
    ys = ax.get_ylim()
    ax.set_ylim([-np.max(np.abs(ys)), np.max(np.abs(ys))])
    ax.axhline([0], color='k')
    plt.savefig(savepath+'/%s_lap_rate_increase_reward.pdf' % region)
    plt.close(f)


def plot_rate_increase_novelty(db, region, savepath):
    data = db[(db['region'] ==  region)]
    #datas = [data['mean_firing_rate_N_noreward'].values, data['mean_firing_rate_F_noreward'].values, data['mean_firing_rate_N_reward'].values, data['mean_firing_rate_F_reward'].values]
    datas = [data['mean_firingrate_F'].values, data['mean_firingrate_N'].values]
    labels = ['familiar', 'novel']
    f, ax = box_comparison(datas, labels, '$<r>$ (ev/s) - per session', box = False, swarm=True)
    annotate_wilcoxon_p(datas[0], datas[1], 0, 1, ax, pairplot=True)
    # annotate_wilcoxon_p(datas[2], datas[3], 2, 3, ax)
    # ys = ax.get_ylim()
    # ax.set_ylim([-np.max(np.abs(ys)), np.max(np.abs(ys))])
    # ax.axhline([0], color='k')
    plt.savefig(savepath+'/%s_lap_rate_increase_novelty.pdf' % region)
    plt.close(f)


def plot_decorrelation_reward(db, savepath, corr='RV'):
    data_DG = db[(db['region'] ==  'DG')]
    data_DG_CL = db[(db['region'] ==  'DG_CL')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    f, ax = box_comparison_three(data_DG['%s_decorr_reward' % corr].values, data_CA1['%s_decorr_reward' % corr].values, data_CA1_CL['%s_decorr_reward' % corr].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Decorrelation %s' % corr, force=True)
    ax.set_title('only reward laps')
    #ax.set_ylim([-0.1, 0.7])
    plt.savefig(savepath+'/reward_YES_%s_decorr.pdf' % corr)
    plt.close(f)

    f, ax = box_comparison_three(data_DG['%s_decorr_noreward' % corr].values, data_CA1['%s_decorr_noreward' % corr].values, data_CA1_CL['%s_decorr_noreward' % corr].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Decorrelation %s' % corr, force=True)
    ax.set_title('only no-reward laps')
    #ax.set_ylim([-0.1, 0.7])
    plt.savefig(savepath+'/reward_NO_%s_decorr.pdf' % corr)
    plt.close(f)

    datas = [data_DG['%s_decorr_reward' % corr].values, data_DG['%s_decorr_noreward' % corr].values, data_DG_CL['%s_decorr_reward' % corr].values, data_DG_CL['%s_decorr_noreward' % corr].values, data_CA1['%s_decorr_reward' % corr].values, data_CA1['%s_decorr_noreward' % corr].values,  data_CA1_CL['%s_decorr_reward' % corr].values, data_CA1_CL['%s_decorr_noreward' % corr].values]
    labels = ['DG (rew)', 'DG (no)', 'DG CL (rew)', 'DG CL (no)', 'CA1 (rew)', 'CA1 (no)', 'CA1 CL (rew)', 'CA1 CL (no)']
    f, ax = box_comparison(datas, labels, 'Decorrelation (%s, per session)' % corr, swarm=False, scatter=True)
    annotate_ttest_p(datas[0], datas[1], 0, 1, ax, force=True)
    annotate_ttest_p(datas[2], datas[3], 2, 3, ax, force=True)
    annotate_ttest_p(datas[4], datas[5], 4, 5, ax, force=True)
    annotate_ttest_p(datas[6], datas[7], 6, 7, ax, force=True)
    plt.savefig(savepath+'/decorr_rew_norew_paired_%s.pdf' % corr)
    plt.close(f)


def plot_FN_spatial_correlation(db, savepath):
    data_DG = db[(db['region'] ==  'DG')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    f, ax = box_comparison_three(data_DG['spatial_FN_c'].values, data_CA1['spatial_FN_c'].values, data_CA1_CL['spatial_FN_c'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Spatial Correlation', force=True)
    data = [data_DG['spatial_FN_c'].values, data_CA1['spatial_FN_c'].values, data_CA1_CL['spatial_FN_c'].values]
    manufy(ax, 0.5, 0, data=data)
    plt.savefig(savepath+'/Spatial_corr_FN_c.pdf')
    plt.close(f)


def plot_PoV_correlation(db, savepath):
    data_DG = db[(db['region'] ==  'DG')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    f, ax = box_comparison_three(data_DG['spatial_FF_PoV'].values, data_CA1['spatial_FN_PoV'].values, data_CA1_CL['spatial_FN_PoV'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1_CL (|||-///)', 'PoV correlation (bins)', force=False)
    manufy(ax, 1.0, 0, data=[data_DG['spatial_FF_PoV'].values, data_CA1['spatial_FN_PoV'].values, data_CA1_CL['spatial_FN_PoV'].values])
    plt.savefig(savepath+'_FN_PoV_correlation.pdf')
    plt.close()

    f, ax = box_comparison_three(data_CA1_CL['spatial_FF_PoV'].values, data_CA1_CL['spatial_FN_PoV'].values, data_CA1_CL['spatial_NN_PoV'].values, 'CA1 (|||-|||)', 'CA1 (|||-***)', 'CA1 (***-***)', 'PoV Correlation (bins)', force=True)
    manufy(ax, 1.0, 0, data=[data_CA1_CL['spatial_FF_PoV'].values, data_CA1_CL['spatial_FN_PoV'].values])
    plt.savefig(savepath+'_PoV_correlation_CA1_Fig2.pdf')
    plt.close(f)


def plot_spatial_correlation_Fig2(db, savepath, ylim = [], yticks = []):
    data_DG = db[(db['region'] ==  'DG')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    f, ax = box_comparison_three(data_CA1['spatial_FF_c'].values, data_CA1['spatial_FN_c'].values, data_DG['spatial_FN_c'].values, 'CA1 (|||-|||)', 'CA1 (|||-///)', 'DG (|||-///)', 'Spatial correlation (cells)', force=True)
    manufy(ax, 1.0, 0.5, data = [data_CA1['spatial_FF_c'].values, data_CA1['spatial_FN_c'].values, data_DG['spatial_FN_c'].values])
    if ylim:
        plt.ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
    plt.savefig(savepath+'/_spatial_correlation_all_Fig2.pdf')

    f, ax = box_comparison_three(data_CA1_CL['spatial_FF_c'].values, data_CA1_CL['spatial_FN_c'].values, data_CA1_CL['spatial_NN_c'].values, 'CA1 (|||-|||)', 'CA1 (|||-***)', 'CA1 (***-***)', 'Spatial Correlation (cells)', force=True)
    manufy(ax, 1.0, 0.5, data = [data_CA1_CL['spatial_FF_c'].values, data_CA1_CL['spatial_FN_c'].values, data_CA1_CL['spatial_NN_c'].values])
    if ylim:
        plt.ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
    plt.savefig(savepath+'/spatial_correlation_CA1_Fig2.pdf')

### this here!
def plot_decorrelation(db, savepath, mode='auc', norm=1, per_mouse=False):
    # db = db[(db['RV_corr_FF'] > min_stability) | (db['RV_corr_NN'] > min_stability)]
    data_DG = db[(db['region'] ==  'DG')]
    data_DG_CL = db[(db['region'] ==  'DG_CL')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    if mode=='RV':
        f, ax = box_comparison_three(data_DG['RV_decorr'].values, data_CA1['RV_decorr'].values, data_CA1_CL['RV_decorr'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Rate Vector Decorrelation', force=True)
        data = [data_DG['RV_decorr'].values, data_CA1['RV_decorr'].values, data_CA1_CL['RV_decorr'].values]
        manufy(ax, 0.5, 0, data=data)

    if mode=='auc':
        # f, ax = box_comparison_three(data_DG['decoding_auc_spa'].values, data_CA1['decoding_auc_spa'].values, data_CA1_CL['decoding_auc_spa'].values, 'DG ///', 'CA1 ///', 'CA1 ***', 'Decoding AUC (spatial cells)', force=True)
        #
        # manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc_spa'].values, data_CA1['decoding_auc_spa'].values, data_CA1_CL['decoding_auc_spa'].values])
        # ax.set_ylim(0.25, 1.05)
        # plt.savefig(savepath+'/decorrelation %s spatial cells.pdf' % mode)
        # plt.close(f)
        #
        # f, ax = box_comparison_three(data_DG['decoding_auc_nonspa'].values, data_CA1['decoding_auc_nonspa'].values, data_CA1_CL['decoding_auc_nonspa'].values, 'DG ///', 'CA1 ///', 'CA1 ***', 'Decoding AUC (non-spatial cells)', force=True)
        #
        # manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc_nonspa'].values, data_CA1['decoding_auc_nonspa'].values, data_CA1_CL['decoding_auc_nonspa'].values])
        # ax.set_ylim(0.25, 1.05)
        # plt.savefig(savepath+'/decorrelation %s non-spatial cells.pdf' % mode)
        # plt.close(f)

        f, ax = box_comparison_three(data_DG['decoding_auc'].values, data_CA1['decoding_auc'].values, data_CA1_CL['decoding_auc'].values, 'DG ///', 'CA1 ///', 'CA1 ***', 'Decoding AUC', force=False, swarm = not per_mouse)
        if per_mouse:
            data = db[(db['region']=='DG') | (db['region']=='CA1') | (db['region']=='CA1_CL')]
            sns.swarmplot(data=data, x='region', y='decoding_auc', hue='topo_name', order=['DG', 'CA1', 'CA1_CL'], palette=sns.color_palette("Paired"))
        ax.get_legend().remove()

        manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc'].values, data_CA1['decoding_auc'].values, data_CA1_CL['decoding_auc'].values])
        ax.set_ylim(0.25, 1.05)

    if mode=='spatial-cells':
        norm_DG = 0.5*(data_DG['spatial_FF_c'].values + data_DG['spatial_NN_c'].values)
        norm_CA1 = 0.5*(data_CA1['spatial_FF_c'].values + data_CA1['spatial_NN_c'].values)
        norm_CA1_CL = 0.5*(data_CA1_CL['spatial_FF_c'].values + data_CA1_CL['spatial_NN_c'].values)

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_cells'].values / norm_DG, data_CA1['spatial_decorrelation_cells'].values / norm_CA1, data_CA1_CL['spatial_decorrelation_cells'].values / norm_CA1_CL, 'DG ///', 'CA1 ///', 'CA1 ***', 'Spatial Decorrelation (cells)', force=False, swarm = not per_mouse)

        if per_mouse:
            data = db[(db['region']=='DG') | (db['region']=='CA1') | (db['region']=='CA1_CL')]
            sns.swarmplot(data=data, x='region', y='spatial_decorrelation_cells_norm', hue='topo_name', order=['DG', 'CA1', 'CA1_CL'], palette=sns.color_palette("Paired"))
        ax.get_legend().remove()

        manufy(ax, 1.0, 0.0, data = [data_DG['spatial_decorrelation_cells'].values / norm_DG, data_CA1['spatial_decorrelation_cells'].values / norm_CA1, data_CA1_CL['spatial_decorrelation_cells'].values / norm_CA1_CL])

    if mode=='spatial-laps':
        if norm:
            norm_DG = 0.5*(data_DG['spatial_FF_l'].values + data_DG['spatial_NN_l'].values)
            norm_CA1 = 0.5*(data_CA1['spatial_FF_l'].values + data_CA1['spatial_NN_l'].values)
            norm_CA1_CL = 0.5*(data_CA1_CL['spatial_FF_l'].values + data_CA1_CL['spatial_NN_l'].values)
        else:
            norm_DG = np.ones(len(data_DG['spatial_FF_l'].values))
            norm_CA1 = np.ones(len(data_CA1['spatial_FF_l'].values))
            norm_CA1_CL = np.ones(len(data_CA1_CL['spatial_FF_l'].values))

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_laps'].values / norm_DG, data_CA1['spatial_decorrelation_laps'].values / norm_CA1, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL, 'DG ///', 'CA1 ///', 'CA1 ***', 'Spatial Decorrelation (laps)', force=False)
        if norm:
            manufy(ax, 1.0, 0, data=[data_DG['spatial_decorrelation_laps'].values / norm_DG, data_CA1['spatial_decorrelation_laps'].values / norm_CA1, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL])
        else:
            manufy(ax, 0.2, 0, data=[data_DG['spatial_decorrelation_laps'].values / norm_DG, data_CA1['spatial_decorrelation_laps'].values / norm_CA1, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL])

    if mode=='spatial-z':
        f, ax = box_comparison_three(data_DG['spatial_decorrelation_Z_cells'].values, data_CA1['spatial_decorrelation_Z_cells'].values, data_CA1_CL['spatial_decorrelation_Z_cells'].values, 'DG ///', 'CA1 ///', 'CA1 ***', 'Spatial Decorrelation (cells, z-score)', force=False)
        manufy(ax, 3.0, 0.0, data=[data_DG['spatial_decorrelation_Z_cells'].values, data_CA1['spatial_decorrelation_Z_cells'].values, data_CA1_CL['spatial_decorrelation_Z_cells'].values])

    if mode=='PoV':
        f, ax = box_comparison_three(data_DG['spatial_decorrelation_PoV'].values, data_CA1['spatial_decorrelation_PoV'].values, data_CA1_CL['spatial_decorrelation_PoV'].values, 'DG ///', 'CA1 ///', 'CA1 ***', 'Spatial Decorrelation (PoV)', force=False)
        manufy(ax, 2.0, 0.0, data=[data_DG['spatial_decorrelation_PoV'].values, data_CA1['spatial_decorrelation_PoV'].values, data_CA1_CL['spatial_decorrelation_PoV'].values])

        # f, ax = box_comparison_three(data_DG['spatial_decorrelation_Z_cells_spatial_or'].values, data_CA1['spatial_decorrelation_Z_cells_spatial_or'].values, data_CA1_CL['spatial_decorrelation_Z_cells_spatial_or'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Spatial Decorrelation (spatial cells || z-score)', force=False)
        # sns.despine()
        # plt.savefig(savepath+'/decorrelation %s spatial-OR.pdf' % mode)
        # plt.close(f)
        # return 0

    # ax.set_title('all laps, min_stab = %.2f' % min_stability)

    # if mode=='auc':
    #     ax.axhline([0.5], linestyle='--', color='k')
    #     ax.set_ylim([0.4, 1.1])
    #     ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # if mode[:7]=='spatial':
    #     ax.axhline([0], linestyle='--', color='k')

    # sns.despine()

    plt.savefig(savepath+'/decorrelation %s norm = %u.pdf' % (mode, norm))
    plt.close(f)


def plot_decorrelation_CL(db, savepath, mode='auc', norm=1):
    # db = db[(db['RV_corr_FF'] > min_stability) | (db['RV_corr_NN'] > min_stability)]
    data_DG = db[(db['region'] ==  'DG')]
    data_DG_CL = db[(db['region'] ==  'DG_CL')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    if mode=='RV':
        f, ax = box_comparison_three(data_DG['RV_decorr'].values, data_DG_CL['RV_decorr'].values, data_CA1_CL['RV_decorr'].values, 'DG (|||-///)', 'DG (|||-ooo)', 'CA1 (|||-ooo)', 'Rate Vector Decorrelation', force=True)
        data = [data_DG['RV_decorr'].values, data_DG_CL['RV_decorr'].values, data_CA1_CL['RV_decorr'].values]
        manufy(ax, 0.5, 0, data=data)

    if mode=='auc':
        f, ax = box_comparison_three(data_DG['decoding_auc_spa'].values, data_DG_CL['decoding_auc_spa'].values, data_CA1_CL['decoding_auc_spa'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Decoding AUC (spatial cells)', force=True)

        manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc_spa'].values, data_DG_CL['decoding_auc_spa'].values, data_CA1_CL['decoding_auc_spa'].values])
        ax.set_ylim(0.25, 1.05)
        plt.savefig(savepath+'/CL decorrelation %s spatial cells.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_three(data_DG['decoding_auc_nonspa'].values, data_DG_CL['decoding_auc_nonspa'].values, data_CA1_CL['decoding_auc_nonspa'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Decoding AUC (non-spatial cells)', force=True)

        manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc_nonspa'].values, data_DG_CL['decoding_auc_nonspa'].values, data_CA1_CL['decoding_auc_nonspa'].values])
        ax.set_ylim(0.25, 1.05)
        plt.savefig(savepath+'/CL decorrelation %s non-spatial cells.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_three(data_DG['decoding_auc'].values, data_DG_CL['decoding_auc'].values, data_CA1_CL['decoding_auc'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Decoding AUC', force=False)
        manufy(ax, 1.0, 0.5, data=[data_DG['decoding_auc'].values, data_DG_CL['decoding_auc'].values, data_CA1_CL['decoding_auc'].values])
        ax.set_ylim(0.25, 1.05)

    if mode=='spatial-cells':
        norm_DG = 0.5*(data_DG['spatial_FF_c'].values + data_DG['spatial_NN_c'].values)
        norm_DG_CL = 0.5*(data_DG_CL['spatial_FF_c'].values + data_DG_CL['spatial_NN_c'].values)
        norm_CA1_CL = 0.5*(data_CA1_CL['spatial_FF_c'].values + data_CA1_CL['spatial_NN_c'].values)

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_cells'].values / norm_DG, data_DG_CL['spatial_decorrelation_cells'].values / norm_DG_CL, data_CA1_CL['spatial_decorrelation_cells'].values / norm_CA1_CL, 'DG ///', 'DG ooo', 'CA1 ooo', 'Spatial Decorrelation (cells)', force=False)
        manufy(ax, 1.0, 0.0, data = [data_DG['spatial_decorrelation_cells'].values / norm_DG, data_DG_CL['spatial_decorrelation_cells'].values / norm_DG_CL, data_CA1_CL['spatial_decorrelation_cells'].values / norm_CA1_CL])

    if mode=='spatial-laps':
        if norm:
            norm_DG = 0.5*(data_DG['spatial_FF_l'].values + data_DG['spatial_NN_l'].values)
            norm_DG_CL = 0.5*(data_DG_CL['spatial_FF_l'].values + data_DG_CL['spatial_NN_l'].values)
            norm_CA1_CL = 0.5*(data_CA1_CL['spatial_FF_l'].values + data_CA1_CL['spatial_NN_l'].values)
        else:
            norm_DG = np.ones(len(data_DG['spatial_FF_l'].values))
            norm_DG_CL = np.ones(len(data_DG_CL['spatial_FF_l'].values))
            norm_CA1_CL = np.ones(len(data_CA1_CL['spatial_FF_l'].values))

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_laps'].values / norm_DG, data_DG_CL['spatial_decorrelation_laps'].values / norm_DG_CL, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL, 'DG ///', 'DG ooo', 'CA1 ooo', 'Spatial Decorrelation (laps)', force=False)
        if norm:
            manufy(ax, 1.0, 0, data=[data_DG['spatial_decorrelation_laps'].values / norm_DG, data_DG_CL['spatial_decorrelation_laps'].values / norm_DG_CL, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL])
        else:
            manufy(ax, 0.2, 0, data=[data_DG['spatial_decorrelation_laps'].values / norm_DG, data_DG_CL['spatial_decorrelation_laps'].values / norm_DG_CL, data_CA1_CL['spatial_decorrelation_laps'].values / norm_CA1_CL])

    if mode=='spatial-z':

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_Z_cells_spatial_or'].values, data_DG_CL['spatial_decorrelation_Z_cells_spatial_or'].values, data_CA1_CL['spatial_decorrelation_Z_cells_spatial_or'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Spatial Decorrelation (SPA cells, z-score)', force=False)
        manufy(ax, 3.0, 0.0, data=[data_DG['spatial_decorrelation_Z_cells_spatial_or'].values, data_DG_CL['spatial_decorrelation_Z_cells_spatial_or'].values, data_CA1_CL['spatial_decorrelation_Z_cells_spatial_or'].values])
        plt.savefig(savepath+'/CL decorrelation %s norm - SPcells OR.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_three(data_DG['spatial_decorrelation_Z_cells'].values, data_DG_CL['spatial_decorrelation_Z_cells'].values, data_CA1_CL['spatial_decorrelation_Z_cells'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Spatial Decorrelation (cells, z-score)', force=False)
        manufy(ax, 3.0, 0.0, data=[data_DG['spatial_decorrelation_Z_cells'].values, data_DG_CL['spatial_decorrelation_Z_cells'].values, data_CA1_CL['spatial_decorrelation_Z_cells'].values])

    if mode=='PoV':
        f, ax = box_comparison_three(data_DG['spatial_decorrelation_PoV'].values, data_DG_CL['spatial_decorrelation_PoV'].values, data_CA1_CL['spatial_decorrelation_PoV'].values, 'DG ///', 'DG ooo', 'CA1 ooo', 'Spatial Decorrelation (PoV)', force=False)
        manufy(ax, 2.0, 0.0, data=[data_DG['spatial_decorrelation_PoV'].values, data_DG_CL['spatial_decorrelation_PoV'].values, data_CA1_CL['spatial_decorrelation_PoV'].values])

    plt.savefig(savepath+'/CL decorrelation %s norm = %u.pdf' % (mode, norm))
    plt.close(f)


def plot_correlation_spatial(db, region, savepath, mode='lap', ylims = [], yticks = []):
    s = mode[0]
    data = db[(db['region'] ==  region)]
    f, ax = box_comparison_three(data['spatial_FF_%s' % s].values, data['spatial_FN_%s' % s].values, data['spatial_NN_%s' % s].values, '|||-|||', '|||-///', '///-///', 'Spatial correlation (per %s)' % mode, force=True)

    if ylims:
        manufy(ax, ylims[1], ylims[0], data=[data['spatial_FF_%s' % s].values, data['spatial_FN_%s' % s].values, data['spatial_NN_%s' % s].values])
        plt.ylim(ylims)
    else:
        manufy(ax, 1.0, 0, data=[data['spatial_FF_%s' % s].values, data['spatial_FN_%s' % s].values, data['spatial_NN_%s' % s].values])
    if yticks:
        ax.set_yticks(yticks)
    plt.title(region)
    plt.savefig(savepath+'/correlation %s spatial %s.pdf' % (region, mode))
    plt.close(f)


def plot_compare_replay(db, savepath, mode='PoV'):
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]
    data_CA1_re = db[(db['region'] ==  'CA1_re')]
    data_CA1_CL_re = db[(db['region'] ==  'CA1_CL_re')]

    if mode== 'auc':
        f, ax = box_comparison_two(data_CA1_CL['decoding_auc_spa'].values, data_CA1_CL_re['decoding_auc_spa'].values, 'CA1 ***', 'CA1 ###', 'Decoding AUC Replay (spatial cells)')
        ax.set_ylim(0.35, 1.05)
        manufy(ax, 1.0, 0.5, data=[data_CA1_CL['decoding_auc_spa'].values, data_CA1_CL_re['decoding_auc_spa'].values])
        plt.savefig(savepath+'/decorr_replay %s spatial cells.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_two(data_CA1['decoding_auc'].values, data_CA1_re['decoding_auc'].values, 'CA1 ///', 'CA1 ---', 'Decoding AUC Replay')
        ax.set_ylim(0.35, 1.05)
        manufy(ax, 1.0, 0.5, data=[data_CA1['decoding_auc'].values, data_CA1_re['decoding_auc'].values, data_CA1_CL['decoding_auc'].values])

    if mode=='spatial-laps':
        norm_CA1 = 0.5*(data_CA1['spatial_FF_l'].values + data_CA1['spatial_NN_l'].values)
        norm_CA1_CL = 0.5*(data_CA1_CL['spatial_FF_l'].values + data_CA1_CL['spatial_NN_l'].values)
        norm_CA1_re = 0.5*(data_CA1_re['spatial_FF_l'].values + data_CA1_re['spatial_NN_l'].values)
        norm_CA1_CL_re = 0.5*(data_CA1_CL_re['spatial_FF_l'].values + data_CA1_CL_re['spatial_NN_l'].values)

        f, ax = box_comparison_two(data_CA1_CL['spatial_decorrelation_laps'].values/norm_CA1_CL, data_CA1_CL_re['spatial_decorrelation_laps'].values/norm_CA1_CL_re, 'CA1 ***', 'CA1 ###', 'Spatial Decorrelation Replay (laps)')
        manufy(ax, 1.0, 0, data=[data_CA1_CL['spatial_decorrelation_laps'].values/norm_CA1_CL, data_CA1_CL_re['spatial_decorrelation_laps'].values/norm_CA1_CL_re])
        plt.savefig(savepath+'/decorr_replay %s CL.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_two(data_CA1['spatial_decorrelation_laps'].values/norm_CA1, data_CA1_re['spatial_decorrelation_laps'].values/norm_CA1_re, 'CA1 ///', 'CA1 ---', 'Spatial Decorrelation Replay (laps)')
        manufy(ax, 1.0, 0, data=[data_CA1['spatial_decorrelation_laps'].values/norm_CA1, data_CA1_re['spatial_decorrelation_laps'].values/norm_CA1_re])

    if mode == 'PoV':
        f, ax = box_comparison_two(data_CA1_CL['spatial_decorrelation_PoV'].values, data_CA1_CL_re['spatial_decorrelation_PoV'].values, 'CA1 ***', 'CA1 ###', 'Spatial Decorrelation Replay (PoV)')
        manufy(ax, 2.0, 0.0, data=[data_CA1_CL['spatial_decorrelation_PoV'].values, data_CA1_CL_re['spatial_decorrelation_PoV'].values])
        plt.savefig(savepath+'/decorr_replay %s CL.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_two(data_CA1['spatial_decorrelation_PoV'].values, data_CA1_re['spatial_decorrelation_PoV'].values, 'CA1 ///', 'CA1 ---', 'Spatial Decorrelation Replay (PoV)')
        manufy(ax, 2.0, 0.0, data=[data_CA1['spatial_decorrelation_PoV'].values, data_CA1_re['spatial_decorrelation_PoV'].values])

    if mode=='spatial-z':
        f, ax = box_comparison_two(data_CA1_CL['spatial_decorrelation_Z_cells'].values, data_CA1_CL_re['spatial_decorrelation_Z_cells'].values, 'CA1 ***', 'CA1 ###', 'Spatial Decorrelation Replay (cells, z-score)')
        manufy(ax, 3.0, 0.0, data=[data_CA1_CL['spatial_decorrelation_Z_cells'].values, data_CA1_CL_re['spatial_decorrelation_Z_cells'].values])
        plt.savefig(savepath+'/decorr_replay %s CL.pdf' % mode)
        plt.close(f)

        f, ax = box_comparison_two(data_CA1['spatial_decorrelation_Z_cells'].values, data_CA1_re['spatial_decorrelation_Z_cells'].values, 'CA1 ///', 'CA1 ---', 'Spatial Decorrelation Replay (cells, z-score)')
        manufy(ax, 3.0, 0.0, data=[data_CA1['spatial_decorrelation_Z_cells'].values, data_CA1_re['spatial_decorrelation_Z_cells'].values])

    plt.savefig(savepath+'/decorr_replay %s.pdf' % mode)
    plt.close(f)


def plot_decorrelation_speed(db, threshold, savepath):
    #db = db[(db['RV_corr_FF'] > 0.0) & (db['RV_corr_NN'] > 0.0)]
    data_DG = db[(db['region'] ==  'DG')]
    data_DG_CL = db[(db['region'] ==  'DG_CL')]
    data_CA1 = db[(db['region'] ==  'CA1')]
    data_CA1_CL = db[(db['region'] ==  'CA1_CL')]

    f, ax = box_comparison_three(data_DG['RV_decorr_fast'].values, data_CA1['RV_decorr_fast'].values, data_CA1_CL['RV_decorr_fast'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Decorrelation', force=True)
    ax.set_title('only fast laps')
    #ax.set_ylim([-0.1, 0.7])
    plt.savefig(savepath+'/fast_rate_decorr_t=%.2f.pdf' % threshold)
    plt.close(f)

    f, ax = box_comparison_three(data_DG['RV_decorr_slow'].values, data_CA1['RV_decorr_slow'].values, data_CA1_CL['RV_decorr_slow'].values, 'DG (|||-///)', 'CA1 (|||-///)', 'CA1 (|||-***)', 'Decorrelation', force=True)
    ax.set_title('only slow laps')
    #ax.set_ylim([-0.1, 0.7])
    plt.savefig(savepath+'/slow_rate_decorr_t=%.2f.pdf' % threshold)
    plt.close(f)

    datas = [data_DG['RV_decorr_fast'].values, data_DG['RV_decorr_slow'].values, data_DG_CL['RV_decorr_fast'].values, data_DG_CL['RV_decorr_slow'].values, data_CA1['RV_decorr_fast'].values, data_CA1['RV_decorr_slow'].values,  data_CA1_CL['RV_decorr_fast'].values, data_CA1_CL['RV_decorr_slow'].values]
    labels = ['DG (fast)', 'DG (slow)', 'DG CL (fast)', 'DG CL (slow)', 'CA1 (fast)', 'CA1 (slow)', 'CA1 CL (f)', 'CA1 CL (s)']
    f, ax = box_comparison(datas, labels, 'Decorrelation (per session)', swarm=False, scatter=True)
    annotate_ttest_p(datas[0], datas[1], 0, 1, ax, force=True)
    annotate_ttest_p(datas[2], datas[3], 2, 3, ax, force=True)
    annotate_ttest_p(datas[4], datas[5], 4, 5, ax, force=True)
    annotate_ttest_p(datas[6], datas[7], 6, 7, ax, force=True)
    plt.savefig(savepath+'/decorr_fast_slow_paired_%.2f.pdf' % threshold)
    plt.close(f)


def visualize_spatial_maps(even_familiar_maps, odd_familiar_maps, even_novel_maps, odd_novel_maps, familiar_maps, novel_maps, min_cells=2, savepath='none', mode='or', HB=False, cmap='magma', plotfunc = lambda x: x, VR = True, conds=['even','odd'], verbose=False, nmin_corr=2):

    if verbose:
        print("even_familiar_maps",len(even_familiar_maps),even_familiar_maps)
        print("odd_familiar_maps",len(odd_familiar_maps),odd_familiar_maps)
        print("even_novel_maps",len(even_novel_maps),even_novel_maps)
        print("odd_novel_maps",len(odd_novel_maps),odd_novel_maps)
        print("familiar_maps",len(familiar_maps),familiar_maps)
        print("novel_maps",len(novel_maps),novel_maps)
        print("savepath",savepath)

    # if not VR:
    #     even_familiar_maps = move.change_nan_zero(even_familiar_maps)
    #     odd_familiar_maps = move.change_nan_zero(odd_familiar_maps)
    #     even_novel_maps = move.change_nan_zero(even_novel_maps)
    #     odd_novel_maps = move.change_nan_zero(odd_novel_maps)
    #     familiar_maps = move.change_nan_zero(familiar_maps)
    #     novel_maps = move.change_nan_zero(novel_maps)

    if verbose:
        print("even_familiar_maps ZERO",len(even_familiar_maps),even_familiar_maps)
        print("odd_familiar_maps ZERO",len(odd_familiar_maps),odd_familiar_maps)
        print("even_novel_maps ZERO",len(even_novel_maps),even_novel_maps)
        print("odd_novel_maps ZERO",len(odd_novel_maps),odd_novel_maps)
        print("familiar_maps ZERO",len(familiar_maps),familiar_maps)
        print("novel_maps ZERO",len(novel_maps),novel_maps)    

    # plotting familiar vs. familiar
    if mode =='and':
        alive = (bn.nansum(even_familiar_maps, 1) > 0) & (bn.nansum(odd_familiar_maps, 1) > 0)
    elif mode == 'or':
        alive = (bn.nansum(even_familiar_maps, 1) > 0) | (bn.nansum(odd_familiar_maps, 1) > 0)
    elif mode == 'F':
        alive = (bn.nansum(even_familiar_maps, 1) > 0)
    elif mode == 'all':
        alive = (bn.nansum(even_familiar_maps, 1) != np.nan)

    if len(alive):
        even_familiar_maps = even_familiar_maps[alive]
        odd_familiar_maps = odd_familiar_maps[alive]

        masscenters = bn.nanargmax(even_familiar_maps, 1)
        masscenters[bn.nansum(even_familiar_maps, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(even_familiar_maps, 1))
        even_familiar_maps = even_familiar_maps[order]
        odd_familiar_maps = odd_familiar_maps[order]

        if VR:
            for i in range(len(even_familiar_maps)):
                if bn.nansum(even_familiar_maps[i]):
                    even_familiar_maps[i] /= bn.nanmax(even_familiar_maps[i])
                if bn.nansum(odd_familiar_maps[i]):
                    odd_familiar_maps[i] /= bn.nanmax(odd_familiar_maps[i])

        # plotting novel vs. novel
        if mode =='and':
            alive = (bn.nansum(even_novel_maps, 1) > 0) & (bn.nansum(odd_novel_maps, 1) > 0)
        elif mode =='or':
            alive = (bn.nansum(even_novel_maps, 1) > 0) | (bn.nansum(odd_novel_maps, 1) > 0)
        elif mode == 'F':
            alive = (bn.nansum(even_novel_maps, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(even_novel_maps, 1) != np.nan)

        even_novel_maps = even_novel_maps[alive]
        odd_novel_maps = odd_novel_maps[alive]

        masscenters = bn.nanargmax(even_novel_maps, 1)
        masscenters[bn.nansum(even_novel_maps, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(even_novel_maps, 1))
        even_novel_maps = even_novel_maps[order]
        odd_novel_maps = odd_novel_maps[order]

        if VR:
            for i in range(len(even_novel_maps)):
                if bn.nansum(even_novel_maps[i]):
                    even_novel_maps[i] /= bn.nanmax(even_novel_maps[i])
                if bn.nansum(odd_novel_maps[i]):
                    odd_novel_maps[i] /= bn.nanmax(odd_novel_maps[i])

        # plotting familiar vs. novel
        if mode == 'and':
            alive = (bn.nansum(familiar_maps, 1) > 0) & (bn.nansum(novel_maps, 1) > 0)
        elif mode == 'or':
            alive = (bn.nansum(familiar_maps, 1) > 0) | (bn.nansum(novel_maps, 1) > 0)
        elif mode == 'F':
            alive = (bn.nansum(familiar_maps, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(familiar_maps, 1) != np.nan)

        n_place_cells_F = bn.nansum(bn.nansum(familiar_maps, 1) > 0)
        n_place_cells_N = bn.nansum(bn.nansum(novel_maps, 1) > 0)

        novel_maps = novel_maps[alive]
        familiar_maps = familiar_maps[alive]

        masscenters = bn.nanargmax(familiar_maps, 1)
        masscenters[bn.nansum(familiar_maps, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(familiar_maps, 1))
        familiar_maps = familiar_maps[order]
        novel_maps = novel_maps[order]

        if VR:
            for i in range(len(familiar_maps)):
                if bn.nansum(familiar_maps[i]):
                    familiar_maps[i] /= bn.nanmax(familiar_maps[i])
                if bn.nansum(novel_maps[i]):
                    novel_maps[i] /= bn.nanmax(novel_maps[i])
        if not VR:
            if verbose:
                print("len(e_f_maps) "+str(len(even_familiar_maps)))
                print("len(e_n_maps) "+str(len(even_novel_maps)))
                print("len(o_f_maps) "+str(len(odd_familiar_maps)))
                print("len(o_n_maps) "+str(len(odd_novel_maps)))
                print("len(f_maps)   "+str(len(familiar_maps)))
                print("len(n_maps)   "+str(len(novel_maps)))
                print("min_cells "+str(min_cells))
                move.printA("inbound_maps ",familiar_maps)
                move.printA("outbound_maps",novel_maps)


        if len(even_familiar_maps) >= min_cells and len(even_novel_maps) >= min_cells and len(familiar_maps) >=min_cells and len(novel_maps) > 0 and savepath != 'none':
            print("start plot heatmap")
            xt = np.asarray([0, int(len(novel_maps[0])/2.), len(novel_maps[0])])
            xl = ['0', '60', '120']

            f, axs = plt.subplots(2, 6, figsize=(12, 4.5), gridspec_kw = {'height_ratios' : [5, 1]})
            axs[0, 0].pcolor(plotfunc(even_familiar_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 0].set_yticks([0, even_familiar_maps.shape[0]])
            if VR:
                axs[0, 0].set_title('familiar even')
            else:
                axs[0, 0].set_title('IN ('+conds[0]+')')
            axs[0, 0].set_ylabel('cell index', fontsize=12)
            axs[0, 0].set_xticks(xt)
            axs[0, 0].set_xticklabels(xl)
            axs[0 ,0].set_xlabel('distance (cm)')

            axs[0, 1].pcolor(plotfunc(odd_familiar_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 1].set_yticks([0, even_familiar_maps.shape[0]])
            if VR:
                axs[0, 1].set_title('familiar odd')
            else:
                axs[0, 1].set_title('IN ('+conds[1]+')')
            axs[0, 1].set_xticks(xt)
            axs[0, 1].set_xticklabels(xl)
            axs[0 ,1].set_xlabel('distance (cm)')

            axs[0, 2].pcolor(plotfunc(even_novel_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 2].set_yticks([0, even_novel_maps.shape[0]])
            if VR:
                axs[0, 2].set_title('novel even')
            else:
                axs[0, 2].set_title('OUT ('+conds[0]+')')
            axs[0, 2].set_xticks(xt)
            axs[0, 2].set_xticklabels(xl)
            axs[0 ,2].set_xlabel('distance (cm)')

            axs[0, 3].pcolor(plotfunc(odd_novel_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 3].set_yticks([0, even_novel_maps.shape[0]])
            if VR:
                axs[0, 3].set_title('novel odd')
            else:
                axs[0, 3].set_title('OUT ('+conds[1]+')')
            axs[0, 3].set_xticks(xt)
            axs[0, 3].set_xticklabels(xl)
            axs[0 ,3].set_xlabel('distance (cm)')

            axs[0, 4].pcolor(plotfunc(familiar_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 4].set_yticks([0, familiar_maps.shape[0]])
            if VR:
                axs[0, 4].set_title('familiar (all)')
            else:
                axs[0, 4].set_title('IN (all)')
            axs[0, 4].set_xticks(xt)
            axs[0, 4].set_xticklabels(xl)
            axs[0 ,4].set_xlabel('distance (cm)')

            axs[0, 5].pcolor(plotfunc(novel_maps), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 5].set_yticks([0, familiar_maps.shape[0]])
            if VR:
                axs[0, 5].set_title('novel (all)')
            else:
                axs[0, 5].set_title('OUT (all)')
            axs[0, 5].set_xticks(xt)
            axs[0, 5].set_xticklabels(xl)
            axs[0 ,5].set_xlabel('distance (cm)')

            pov_FF = pov(even_familiar_maps, odd_familiar_maps, VR=VR, nmin=nmin_corr)
            axs[1, 1].plot(np.arange(len(familiar_maps[0])), pov_FF, color=pltcolors[0])
            axs[1, 1].set_ylabel('PoV corr')
            axs[1, 1].set_ylim([-0.5, 1.1])
            axs[1, 1].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 1].axhline([0], color='k')
            mean_PoV_FF = bn.nanmean(pov_FF)
            axs[1, 1].set_title('<PoV corr> = %.2f' % mean_PoV_FF)

            pov_NN = pov(even_novel_maps, odd_novel_maps, VR=VR, nmin=nmin_corr)
            axs[1, 3].plot(np.arange(len(familiar_maps[0])), pov_NN, color=pltcolors[0])
            axs[1, 3].set_ylabel('PoV corr')
            axs[1, 3].set_ylim([-0.5, 1.1])
            axs[1, 3].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 3].axhline([0], color='k')
            mean_PoV_NN = bn.nanmean(pov_NN)
            axs[1, 3].set_title('<PoV corr> = %.2f' % mean_PoV_NN)

            pov_FN = pov(familiar_maps, novel_maps, VR=VR, nmin=nmin_corr)
            pov_FLIP = pov(familiar_maps, novel_maps, VR=VR, nmin=nmin_corr, flip=True)
            axs[1, 5].plot(np.arange(len(familiar_maps[0])), pov_FN, color=pltcolors[0])
            axs[1, 5].set_ylabel('PoV corr')
            axs[1, 5].set_ylim([-0.5, 1.1])
            axs[1, 5].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 5].axhline([0], color='k')
            mean_PoV_FN = bn.nanmean(pov_FN)
            axs[1, 5].set_title('<PoV corr> = %.2f' % mean_PoV_FN)

            if savepath != 'none':
                print(savepath)
                plt.savefig(savepath+'.pdf')
                plt.close()

            pov_FF_cells = pov(even_familiar_maps, odd_familiar_maps, percell=HB, VR=VR, nmin=nmin_corr)
            pov_NN_cells = pov(even_novel_maps, odd_novel_maps, percell=HB, VR=VR, nmin=nmin_corr)
            pov_FN_cells = pov(familiar_maps, novel_maps, percell=HB, VR=VR, nmin=nmin_corr)
            pov_FLIP_cells = pov(familiar_maps, novel_maps, percell=HB, VR=VR, nmin=nmin_corr, flip=True)

            if VR:
                f, ax = box_comparison_three(pov_FF_cells, pov_FN_cells, pov_NN_cells, '|||-|||', '|||-///', '///-///', 'Spatial correlation (PCs)', force=True, swarm=True, box=False, violin=False, bar=True)
            else:
                f, ax = box_comparison_three(pov_FF_cells, pov_FN_cells, pov_NN_cells, 'IN('+conds[0]+')\n-IN('+conds[1]+')', 'IN(all)\n-OUT(all)', 'OUT('+conds[0]+')\n-OUT('+conds[1]+')', 'Spatial correlation (PCs)', force=True, swarm=True, box=False, violin=False, bar=True)
            ax.set_ylim([-0.3, 1.5])
            ax.axhline([0], color='k', linestyle='--')
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            if VR:
                ax.set_title('dec: %.2f  #pc F: %u N: %u' % (0.5*(np.mean(pov_FF_cells)+np.mean(pov_NN_cells))-np.mean(pov_FN_cells), len(even_familiar_maps), len(even_novel_maps)))
            else:
                ax.set_title('dec: %.2f  #pc IN: %u OUT: %u' % (0.5*(np.mean(pov_FF_cells)+np.mean(pov_NN_cells))-np.mean(pov_FN_cells), len(even_familiar_maps), len(even_novel_maps)))
            if savepath != 'none':
                plt.savefig(savepath+'_cell_corr.pdf')
            plt.close()

            f, ax = plt.subplots(figsize=(4,3))
            if VR:
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FF, color=pltcolors[4], label='|||-|||', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_NN, color=pltcolors[5], label='///-///', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FN, color=pltcolors[6], label='|||-///', linewidth=2.0)
            else:
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FF, color=pltcolors[4], label='IN('+conds[0]+')\n-IN('+conds[1]+')', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_NN, color=pltcolors[5], label='OUT('+conds[0]+')\n-OUT('+conds[1]+')', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FN, color=pltcolors[6], label='IN(all)\n-OUT(all)', linewidth=2.0)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('PoV Correlation')
            ax.set_xticks([0, 0.6, 1.2])
            ax.axhline([0], color='k', linestyle='--')
            ax.legend()
            ax.set_ylim([-0.1, 0.7])
            ax.set_yticks([-0.1, 0, 0.2, 0.4, 0.6])
            if savepath != 'none':
                plt.savefig(savepath+'_PoV.pdf')
            plt.close()

            if verbose:
                print("len(inbound_maps) "+str(len(familiar_maps)))
                print("len(outbound_maps) "+str(len(novel_maps)))
                print("len("+conds[0]+"_inbound_maps) "+str(len(even_familiar_maps)))
                print("len("+conds[1]+"_inbound_maps) "+str(len(odd_familiar_maps)))
                print("len("+conds[0]+"_outbound_maps) "+str(len(even_novel_maps)))
                print("len("+conds[1]+"_outbound_maps) "+str(len(odd_novel_maps)))

            return bn.nanmean(pov_FF), bn.nanmean(pov_NN), bn.nanmean(pov_FN), \
                bn.nanmean(pov_FF_cells), bn.nanmean(pov_NN_cells), bn.nanmean(pov_FN_cells), \
                bn.nanmean(pov_FLIP), bn.nanmean(pov_FLIP_cells), \
                n_place_cells_F, n_place_cells_N
        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n_place_cells_F, n_place_cells_N
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, 0#n_place_cells_F, n_place_cells_N

def visualize_session_timemasked_with_map_in_rois(session, n_roi_start=0, n_roi_shown=100, VR=True, withtick=True, usemap='F', timemask=None, savepath='none', maptype='F', plotmean=True):
    import matplotlib as mpl

    if type(timemask) is not list:
        #timemask=np.isfinite(session.times)
        mask_all = np.isnan(session.times)
        for l in session.laps:
            mask_lap = np.logical_and(session.times >= l.times[0], session.times <= l.times[-1])
            mask_all = np.logical_or(mask_all,mask_lap)
        timemask = mask_all.tolist()

    idx = [ len(l.times) for l in session.laps ]
    index_starts = np.append([0],np.cumsum(idx)[:-1])
    index_ends = np.append(index_starts[1:]-1,np.cumsum(idx)[-1])
    if VR:
        scale = 1e-3
    else:
        scale = 1
    mpl.rcParams.update({'figure.autolayout': False})
    n_roi_shown = min([n_roi_shown, session.n_roi - n_roi_start])

    nrows = n_roi_shown + 2
    if usemap == 'ALL':
        nrows += n_roi_shown
    elif usemap == 'MEAN':
        nrows = 4

    ns = 0
    fig = plt.figure(figsize=(8*2, nrows*2.))
    ax0 = fig.add_subplot(nrows, 2, ns*2+1)
    ax0.plot(session.speed[timemask])
    ns += 1
    ax1 = fig.add_subplot(nrows, 2, ns*2+1, sharex=ax0)
    ax1.plot(session.position[timemask])

    types = [l.laptype for l in session.laps]
    assert len(index_starts) == len(types)
    if not VR:
        for i, (b1,b2,t) in enumerate( zip( index_starts, index_ends, types)):
            x = np.arange(b1,b2,0.1)
            y_pos_min = np.ones(x.size)*np.min(session.position[timemask])
            y_pos_max = np.ones(x.size)*np.max(session.position[timemask])
            y_sp_min = np.ones(x.size)*np.min(session.speed[timemask])
            y_sp_max = np.ones(x.size)*np.max(session.speed[timemask])

            if t == 'top': #inbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='green', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='green', alpha=0.3)
            if t == 'bottom': #outbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='red', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='red', alpha=0.3)
    ns += 1

    mapkey1 = 'Smap_running'
    mapkey2 = 'ratemap_running'
    if maptype == 'S':
        mapkey = mapkey1
        map_len = len(session.laps[0].s_maps[0])
    elif maptype == 'F':
        mapkey = mapkey2
        map_len = len(session.laps[0].rate_maps[0])
    bin_len = len(session.bins)

    for n in range(n_roi_start, n_roi_start + n_roi_shown):
        ax = fig.add_subplot(nrows, 2, ns*2+1, sharex=ax0)
        pos_max = np.max(y_pos_max)
        pos_min = np.min(y_pos_min)
        F_max = session.dF_F[n][timemask].max()
        F_min = session.dF_F[n][timemask].min()
        ax.plot((session.position[timemask]-pos_min)/(pos_max-pos_min)*(F_max-F_min)+F_min,'-k',alpha=0.2)
        if usemap == 'F':
            use = session.dF_F
        elif usemap == 'S':
            use = session.S
        elif usemap == 'TOP':
            S_max = max(1,session.S[n][timemask].max())
            ax.plot(session.S[n][timemask]/S_max*session.dF_F[n][timemask].max(), '-r', alpha=0.5)
            use = session.dF_F
        elif usemap == 'ALL':
            ax.plot(session.S[n][timemask], '-b', alpha=0.5)
            ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns*2+1, sharex=ax0)
            use = session.dF_F
        else: #usemap == 'MEAN':
            ax.plot(np.mean(session.S,0)[timemask], '-b', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(S)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns*2+1, sharex=ax0)
            ax.plot(np.mean(session.dF_F,0)[timemask], '-g', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(dF_F)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            break
        ax.plot(use[n][timemask], '-g', alpha=0.5)
        if len(session.activations[n]) and withtick:
            for k in session.activations[n]:
                index_masked = np.arange(len(timemask))[timemask].tolist()
                if k in index_masked:
                    ax.plot(index_masked.index(k), use[n].min(), '|r', ms=10)
        if not VR:
            for i, (b1,b2,t) in enumerate( zip( index_starts, index_ends, types)):
                x = np.arange(b1,b2,0.1)
                y_min = np.ones(x.size)*np.min(use[n][timemask])
                y_max = np.ones(x.size)*np.max(use[n][timemask])
                if t == 'top': #inbound
                    ax.fill_between(x,y_min,y_max,
                            facecolor='green', alpha=0.1)
                if t == 'bottom': #outbound
                    ax.fill_between(x,y_min,y_max,
                            facecolor='red', alpha=0.1)
        ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])

        if n == 0:
            ax0obl = ax_obl = fig.add_subplot(nrows,4,ns*4+3)
            ax0vert = ax_vert = fig.add_subplot(nrows,4,ns*4+4, sharey=ax_obl)
        else:
            ax0obl = fig.add_subplot(nrows,4,ns*4+3, sharex=ax_obl)
            ax0vert = fig.add_subplot(nrows,4,ns*4+4, sharey=ax0obl, sharex=ax_vert)
        if n == 0:
            if VR:
                if mapkey == 'Smap_running':
                    ax0obl.set_title("familiar (smap)")
                    ax0vert.set_title("novel (smap)")
                else:
                    ax0obl.set_title("familiar (ratemap)")
                    ax0vert.set_title("novel (ratemap)")
            else:
                if mapkey == 'Smap_running':
                    ax0obl.set_title("inbound (smap)",color='green')
                    ax0vert.set_title("outbound (smap)",color='red')
                else:
                    ax0obl.set_title("inbound (ratemap)",color='green')
                    ax0vert.set_title("outbound (ratemap)",color='red')
        for lap_i, lap in enumerate(session.laps):
            if lap_i in session.incompletelaps:
                ax0vert.plot([30],[0], alpha=0)
                ax0obl.plot([30],[0], alpha=0)
            if lap in session.familiar_laps:
                ax1_map = ax0vert
                ax2_map = ax0obl

            elif lap in session.novel_laps:
                ax1_map = ax0obl
                ax2_map = ax0vert

            else:
                continue
            if mapkey == 'Smap_running':
                #move.printA("roi["+str(n)+"]=>laps["+str(lap_i)+"].s_maps["+str(n)+"]",lap.s_maps[n])
                ax2_map.plot(session.bins[:map_len],lap.s_maps[n][:bin_len], alpha=0.5, label='lap %d' % lap_i)
                ax1_map.plot(session.bins[:map_len],lap.s_maps[n][:bin_len], alpha=0)
            else:
                #move.printA("roi["+str(n)+"]=>laps["+str(lap_i)+"].rate_maps["+str(n)+"]",lap.rate_maps[n])
                ax2_map.plot(session.bins[:map_len],lap.rate_maps[n][:bin_len], alpha=0.5, label='lap %d' % lap_i)
                ax1_map.plot(session.bins[:map_len],lap.rate_maps[n][:bin_len], alpha=0)

        ax0obl.text(
            1.0, 0.5, "{0}".format(n),
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax0obl.transAxes)
        if mapkey == 'Smap_running':
            if n in np.arange(len(session.spatial_cells_S))[session.spatial_cells_S]:
                ax0obl.text(
                    1.1, 0.5, "PC",
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax0obl.transAxes,color='red',fontsize=15)
        else:
            if n in np.arange(len(session.spatial_cells_F))[session.spatial_cells_F]:
                ax0obl.text(
                    1.1, 0.5, "PC",
                    horizontalalignment='right',
                    verticalalignment='center',
                    transform=ax0obl.transAxes,color='red',fontsize=15)

        #if n != len(session.laps[0].s_maps)-1:
        #    ax0vert.set_xticks([])
        #    ax0obl.set_xticks([])

        if plotmean:
            if mapkey == 'Smap_running':
                meanvert = bn.nanmean([lap.s_maps[n] 
                                       for lap in session.laps 
                                       if lap in session.novel_laps], axis=0)
                meanobl = bn.nanmean([lap.s_maps[n] 
                                      for lap in session.laps 
                                      if lap in session.familiar_laps], axis=0)
                ax0vert.plot(session.bins[:map_len], meanvert[:bin_len], '-r', lw=4)
                ax0obl.plot(session.bins[:map_len], meanobl[:bin_len], '-r', lw=4)
            else:
                meanvert = bn.nanmean([lap.rate_maps[n]
                                       for lap in session.laps
                                       if lap in session.novel_laps], axis=0)
                meanobl = bn.nanmean([lap.rate_maps[n]
                                      for lap in session.laps
                                      if lap in session.familiar_laps], axis=0)
                ax0vert.plot(session.bins[:map_len], meanvert[:bin_len], '-r', lw=4)
                ax0obl.plot(session.bins[:map_len], meanobl[:bin_len], '-r', lw=4)
        ns += 1
    ax.set_xlabel('time (s)')
    ax0vert.set_xlabel('position(outbound)',color='red')
    ax0obl.set_xlabel('position(inbound)',color='green')
    sns.despine()
    mpl.rcParams.update({'figure.autolayout': True})
    if savepath != 'none':
        if plotmean:
            plt.savefig(savepath+'_w_mean.pdf')
            plt.close()
        else:
            plt.savefig(savepath+'_wo_mean.pdf')
            plt.close()
    return ax0

def visualize_session_timemasked(session, n_roi_start=0, n_roi_shown=100, VR=True, withtick=True, usemap='F', timemask=None):
    import matplotlib as mpl
    if type(timemask) is not list:
        mask_all = np.isnan(session.times)
        for l in session.laps:
            mask_lap = np.logical_and(session.times >= l.times[0], session.times <= l.times[-1])
            mask_all = np.logical_or(mask_all,mask_lap)
        timemask = mask_all.tolist()

    idx = [ len(l.times) for l in session.laps ]
    index_starts = np.append([0],np.cumsum(idx)[:-1])
    index_ends = np.append(index_starts[1:]-1,np.cumsum(idx)[-1])

    if VR:
        scale = 1e-3
    else:
        scale = 1
    mpl.rcParams.update({'figure.autolayout': False})
    n_roi_shown = min([n_roi_shown, session.n_roi - n_roi_start])

    nrows = n_roi_shown + 2
    if usemap == 'ALL':
        nrows += n_roi_shown
    elif usemap == 'MEAN':
        nrows = 4

    ns = 1
    fig = plt.figure(figsize=(8, nrows*2.))
    ax0 = fig.add_subplot(nrows, 1, ns)
    ax0.plot(session.speed[timemask])
    ns += 1
    ax1 = fig.add_subplot(nrows, 1, ns, sharex=ax0)
    ax1.plot(session.position[timemask])

    types = [l.laptype for l in session.laps]

    assert len(index_starts) == len(types)
    if not VR:
        for i, (b1,b2,t) in enumerate( zip( index_starts, index_ends, types)):
            x = np.arange(b1,b2,0.1)
            y_pos_min = np.ones(x.size)*np.min(session.position[timemask])
            y_pos_max = np.ones(x.size)*np.max(session.position[timemask])
            y_sp_min = np.ones(x.size)*np.min(session.speed[timemask])
            y_sp_max = np.ones(x.size)*np.max(session.speed[timemask])
            
            if t == 'top': #inbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='green', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='green', alpha=0.3)
            if t == 'bottom': #outbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='red', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='red', alpha=0.3)
    ns += 1
    for n in range(n_roi_start, n_roi_start + n_roi_shown):
        ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
        if usemap == 'F':
            use = session.dF_F
        elif usemap == 'S':
            use = session.S
        elif usemap == 'TOP':
            S_max = max(1,session.S[n][timemask].max())
            ax.plot(session.S[n][timemask]/S_max*session.dF_F[n][timemask].max(), '-r', alpha=0.5)
            use = session.dF_F
        elif usemap == 'ALL':
            ax.plot(session.S[n][timemask], '-b', alpha=0.5)
            ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
            use = session.dF_F
        else: #usemap == 'MEAN':
            ax.plot(np.mean(session.S,0)[timemask], '-b', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(S)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
            ax.plot(np.mean(session.dF_F,0)[timemask], '-g', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(dF_F)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            break
        ax.plot(use[n][timemask], '-g', alpha=0.5)
        if len(session.activations[n]) and withtick:
            for k in session.activations[n]:
                index_masked = np.arange(len(timemask))[timemask].tolist()
                if k in index_masked:
                    ax.plot(index_masked.index(k), use[n].min(), '|r', ms=10)
        if not VR:
            for i, (b1,b2,t) in enumerate( zip( index_starts, index_ends, types)):
                x = np.arange(b1,b2,0.1)
                y_min = np.ones(x.size)*np.min(use[n][timemask])
                y_max = np.ones(x.size)*np.max(use[n][timemask])
                if t == 'top': #inbound
                    ax.fill_between(x,y_min,y_max,
                            facecolor='green', alpha=0.1)
                if t == 'bottom': #outbound
                    ax.fill_between(x,y_min,y_max,
                            facecolor='red', alpha=0.1)
        ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ns += 1
        #print('start with ns' , ns)
    #ax.set_xticks([0, np.max(session.times) * scale / 2., np.max(session.times) * scale])
    ax.set_xlabel('time (s)')
    sns.despine()
    mpl.rcParams.update({'figure.autolayout': True})
    return ax0

def visualize_session(session, n_roi_start=0, n_roi_shown=100, VR=True, withtick=True, usemap='F'):
    import matplotlib as mpl
    if VR:
        scale = 1e-3
    else:
        scale = 1
    mpl.rcParams.update({'figure.autolayout': False})
    n_roi_shown = min([n_roi_shown, session.n_roi - n_roi_start])

    nrows = n_roi_shown + 2
    if usemap == 'ALL':
        nrows += n_roi_shown
    elif usemap == 'MEAN':
        nrows = 4
        
    ns = 1
    fig = plt.figure(figsize=(8, nrows*2.))
    ax0 = fig.add_subplot(nrows, 1, ns)
    ax0.plot(session.times * scale, session.speed)
    ns += 1
    ax1 = fig.add_subplot(nrows, 1, ns, sharex=ax0)
    ax1.plot(session.times * scale, session.position)

    if not VR:
        for i, (b1,b2,t) in enumerate( zip( session.lapboundtimes[:-1],session.lapboundtimes[1:],session.laptypes[:-1])):
            if i in session.incompletelaps:
                continue
            x = np.arange(b1,b2,0.1)
            y_pos_min = np.ones(x.size)*np.min(session.position)
            y_pos_max = np.ones(x.size)*np.max(session.position)
            y_sp_min = np.ones(x.size)*np.min(session.speed)
            y_sp_max = np.ones(x.size)*np.max(session.speed)
            
            if t == 'top': #inbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='green', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='green', alpha=0.3)
            if t == 'bottom': #outbound
                ax0.fill_between(x,y_sp_min,y_sp_max,
                        facecolor='red', alpha=0.3)
                ax1.fill_between(x,y_pos_min,y_pos_max,
                        facecolor='red', alpha=0.3)
            if session.cut_periods is not None:
                for (s,e) in session.cut_periods:
                    #s_time = session.vrdict['tracktimes'][s]
                    #e_time = session.vrdict['tracktimes'][e]
                    #x_cut = np.arange(s_time,e_time)
                    x_cut = session.vrdict['tracktimes'][s:e]
                    y_pos_min_cut = np.ones(x_cut.size)*np.min(session.position)
                    y_pos_max_cut = np.ones(x_cut.size)*np.max(session.position)
                    y_sp_min_cut = np.ones(x_cut.size)*np.min(session.speed)
                    y_sp_max_cut = np.ones(x_cut.size)*np.max(session.speed)
                    ax0.fill_between(x_cut,y_sp_min_cut,y_sp_max_cut,
                        facecolor='grey', alpha=0.1)
                    ax1.fill_between(x_cut,y_pos_min_cut,y_pos_max_cut,
                        facecolor='grey', alpha=0.1)

    if VR:
        for ev in session.vrdict['evlist']:
            if ev.evtype in ['oblique', 'vertical']:
                ax1.plot(ev.time, session.position.min()-0.05, ev.marker)
    ns += 1
    for n in range(n_roi_start, n_roi_start + n_roi_shown):
        ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
        if usemap == 'F':
            use = session.dF_F
        elif usemap == 'S':
            use = session.S
        elif usemap == 'TOP':
            S_max = max(1,session.S[n].max())
            ax.plot(session.times * scale, session.S[n]/S_max*session.dF_F[n].max(), '-r', alpha=0.5)
            use = session.dF_F
        elif usemap == 'ALL':
            ax.plot(session.times * scale, session.S[n], '-b', alpha=0.5)
            ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
            use = session.dF_F
        else: #usemap == 'MEAN':
            ax.plot(session.times * scale, np.mean(session.S,0), '-b', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(S)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            ax.set_xticks([])
            ns += 1
            ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
            ax.plot(session.times * scale, np.mean(session.dF_F,0), '-g', alpha=0.5)
            ax.text(1.0, 0.5, 'mean(dF_F)', horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
            if withtick:
                usemax = max(np.mean(session.dF_F,0))
                for n in range(n_roi_start, n_roi_start + n_roi_shown):
                    if len(session.activations[n]):
                        ax.plot(session.times[session.activations[n]] * scale, np.ones(len(session.activations[n])) * usemax, '|r', ms=10)
            break
        ax.plot(session.times * scale, use[n], '-g', alpha=0.5)
        if len(session.activations[n]) and withtick:
            ax.plot(session.times[session.activations[n]] * scale, np.ones(len(session.activations[n])) * use[n].min(), '|r', ms=10)
        if not VR:
            for i, (b1,b2,t) in enumerate( zip( session.lapboundtimes[:-1],session.lapboundtimes[1:],session.laptypes[:-1])):
                if i in session.incompletelaps:
                    continue
                x = np.arange(b1,b2,0.1)
                y_min = np.ones(x.size)*np.min(use[n])
                y_max = np.ones(x.size)*np.max(use[n])
                if t == 'top':
                    ax.fill_between(x,y_min,y_max,
                            facecolor='green', alpha=0.3)
                if t == 'bottom':
                    ax.fill_between(x,y_min,y_max,
                            facecolor='red', alpha=0.3)
                if session.cut_periods is not None:
                    for (s,e) in session.cut_periods:
                        #s_time = session.vrdict['frametimes'][s]
                        #e_time = session.vrdict['frametimes'][e]
                        #x_cut = np.arange(s_time,e_time)
                        x_cut = session.vrdict['tracktimes'][s:e]
                        y_min_cut = np.ones(x_cut.size)*np.min(use[n])
                        y_max_cut = np.ones(x_cut.size)*np.max(use[n])
                        ax.fill_between(x_cut,y_min_cut,y_max_cut,
                            facecolor='grey', alpha=0.1)

        ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ns += 1
        #print('start with ns' , ns)
    ax.set_xticks([0, np.max(session.times) * scale / 2., np.max(session.times) * scale])
    ax.set_xlabel('time (s)')
    sns.despine()
    mpl.rcParams.update({'figure.autolayout': True})
    return ax0
    
    

def visualize_session_trajectory(session, n_roi_start=0, n_roi_shown=100, VR=True):
    import matplotlib as mpl
    if VR:
        scale = 1e-3
    else:
        scale = 1
    mpl.rcParams.update({'figure.autolayout': False})
    n_roi_shown = min([n_roi_shown, session.n_roi - n_roi_start])
    nrows = n_roi_shown + 1

    ns = 1
    fig = plt.figure(figsize=(8, nrows/2.))
    ax0 = fig.add_subplot(nrows, 1, ns)
    ax0.plot(session.vrdict['posx'], session.vrdict['posy'], '-k', lw=1, alpha=0.1)
    if len(session.activations[n_roi_start]):
        ax0.plot(session.vrdict['posx'][session.activations[n_roi_start]], session.vrdict['posy'][session.activations[n_roi_start]], 'or', alpha=0.2)
    ax0.text(1.0, 0.5, '%u' % session.name_roi[n_roi_start], horizontalalignment='left', verticalalignment='center', transform=ax0.transAxes)
    ax0.set_ylabel('posy(cm)')

    ns+=1
    for n in range(n_roi_start, n_roi_start + n_roi_shown):
        ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
        ax.plot(session.vrdict['posx'], session.vrdict['posy'], '-k', lw=1, alpha=0.1)
        if len(session.activations[n]):
            ax.plot(session.vrdict['posx'][session.activations[n]], session.vrdict['posy'][session.activations[n]], 'or', alpha=0.2)
        ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ns += 1
        ax.set_ylabel('posy(cm)')
        #print('start with ns' , ns)
    ax.set_xlabel('posx(cm)')
    sns.despine()
    mpl.rcParams.update({'figure.autolayout': True})
    return ax0


def visualize_spike_inference(session, n_roi_start=0, n_roi_shown=100, VR=True):
    import matplotlib as mpl
    if VR:
        scale = 1e-3
    else:
        scale = 1
    mpl.rcParams.update({'figure.autolayout': False})
    n_roi_shown = min([n_roi_shown, session.n_roi - n_roi_start])
    nrows = n_roi_shown + 2

    ns = 1
    fig = plt.figure(figsize=(8, nrows/2.))
    ax0 = fig.add_subplot(nrows, 1, ns)
    ax0.plot(session.times * scale, session.speed)
    ns += 1
    ax1 = fig.add_subplot(nrows, 1, ns, sharex=ax0)
    ax1.plot(session.times * scale, session.position)

    ns += 1
    for n in range(n_roi_start, n_roi_start + n_roi_shown):
        ax = fig.add_subplot(nrows, 1, ns, sharex=ax0)
        ax.plot(session.times * scale, session.S[n], '-g', alpha=0.5)
        #if len(session.activations[n]):
        #    ax.plot(session.times[session.activations[n]] * scale, np.ones(len(session.activations[n])) * session.dF_F[n].min(), '|r', ms=20)
        ax.text(1.0, 0.5, '%u' % session.name_roi[n], horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
        ax.set_xticks([])
        ns += 1
    ax.set_xticks([0, np.max(session.times) * scale / 2., np.max(session.times) * scale])
    ax.set_xlabel('time (s)')
    sns.despine()
    mpl.rcParams.update({'figure.autolayout': True})


### -------------------------- utilities -------------------------- ###

def correlate_scatter(db, region, label1, label2, save_path, topo_name='any', corr='pearson', labelx='auto', labely='auto', title='auto', ylim=[], yticks=[], xlim=[]):
    ddf = db[db['region']==region]
    if topo_name != 'any':
        ddf = ddf[ddf['topo_name'] == topo_name]

    mask = (np.isnan(ddf[label1].values)==0) & (np.isnan(ddf[label2].values)==0)
    if corr=='spearman':
        (r, p) = spearmanr(ddf[label1][mask], ddf[label2][mask])
    if corr=='pearson':
        (r, p) = pearsonr(ddf[label1][mask], ddf[label2][mask])

    f, ax = plt.subplots(figsize=(3.5, 3))
    if p<0.05:
        sns.regplot(x=label1, y=label2, data=ddf, ax=ax)
    else:
        sns.scatterplot(x=label1, y=label2, data=ddf, ax=ax, color=[0.7, 0.7, 0.7])
    sns.despine()
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)
    if labelx!='auto':
        ax.set_xlabel(labelx)
    if labely!='auto':
        ax.set_ylabel(labely)
    ys = ax.get_ylim()
    xs = ax.get_xlim()

    if p<0.05:
        col = 'r'
    else:
        col = 'k'
    if r>0:
        x0 = xs[0] + 0.1 * (xs[1]-xs[0])
        y0 = ys[0] + 0.9 * (ys[1]-ys[0])
        ax.text(x0, y0, p_to_text(p), ha='left', va='bottom', color=col, fontsize=12)
    if r<0:
        x0 = xs[0] + 0.9 * (xs[1]-xs[0])
        y0 = ys[0] + 0.9 * (ys[1]-ys[0])
        ax.text(x0, y0, p_to_text(p), ha='right', va='bottom', color=col, fontsize=12)

    if title!='auto':
        ax.set_title(title)

    plt.savefig(save_path)
    plt.close()


def box_quantities(db, region, label1, label2, save_path, topo_name='any', test='ttest', xticks='auto', labely='auto', title='auto'):
    ddf = db[db['region']==region]
    if topo_name != 'any':
        ddf = ddf[ddf['topo_name'] == topo_name]

    f, ax = plt.subplots(figsize=(3.5, 3))
    sns.boxplot(data=ddf[[label1, label2]], ax=ax)
    sns.swarmplot(data=ddf[[label1, label2]], ax=ax, color="k", alpha=0.6)
    sns.despine()

    if test=='ttest':
        annotate_ttest_p(ddf[label1].values, ddf[label2].values, 0, 1, ax=ax, force=True)
    if test=='wilcoxon':
        annotate_wilcoxon_p(ddf[label1].values, ddf[label2].values, 0, 1, ax=ax, force=True)
    if xticks!='auto':
        plt.xticks([0,1], xticks)
        plt.xlabel('')
    if labely!='auto':
        ax.set_ylabel(labely)
    if title!='auto':
        ax.set_title(title)
    y = ax.get_ylim()
    ax.set_ylim([y[0], y[0]+(y[1]-y[0])*1.1])
    plt.savefig(save_path)
    plt.close()


def box_regions(db, region1, region2, quantity, save_path, topo_name='any', test='ttest', xticks='auto', labely='auto', title='auto'):
    db1 = db[db['region']==region1]
    db2 = db[db['region']==region2]
    dataA = pd.DataFrame(db1[quantity].values, columns=[region1])
    dataB = pd.DataFrame(db2[quantity].values, columns=[region2])
    data = pd.concat([dataA, dataB], sort='False')
    data = data[[region1, region2]]

    f, ax = plt.subplots(figsize=(3.5, 3))
    sns.boxplot(data=data, ax=ax)
    sns.swarmplot(data=data, ax=ax, color="k", alpha=0.6)
    sns.despine()

    if test=='ttest':
        annotate_ttest_p(db1[quantity].values, db2[quantity].values, 0, 1, ax=ax, force=True)
    if test=='wilcoxon':
        annotate_wilcoxon_p(db1[quantity].values, db2[quantity].values, 0, 1, ax=ax, force=True)
    if xticks!='auto':
        plt.xticks([0,1], xticks)
        plt.xlabel('')
    if labely!='auto':
        ax.set_ylabel(labely)
    if title!='auto':
        ax.set_title(title)
    y = ax.get_ylim()
    ax.set_ylim([y[0], y[0]+(y[1]-y[0])*1.1])
    plt.savefig(save_path)
    plt.close()


def box_environments(db, quantity, save_path, topo_name='any', test='ttest', xticks='auto', labely='auto', title='auto'):
    db1 = db[(db['region']=='CA1') | (db['region']=='DG')]
    db2 = db[db['region']=='CA1_CL']
    dataA = pd.DataFrame(db1[quantity].values, columns=['///'])
    dataB = pd.DataFrame(db2[quantity].values, columns=['***'])
    data = pd.concat([dataA, dataB], sort='False')
    data = data[['///', '***']]

    f, ax = plt.subplots(figsize=(3.5, 3))
    sns.boxplot(data=data, ax=ax)
    sns.swarmplot(data=data, ax=ax, color="k", alpha=0.6)
    sns.despine()

    if test=='ttest':
        annotate_ttest_p(db1[quantity].values, db2[quantity].values, 0, 1, ax=ax, force=True)
    if test=='wilcoxon':
        annotate_wilcoxon_p(db1[quantity].values, db2[quantity].values, 0, 1, ax=ax, force=True)
    if xticks!='auto':
        plt.xticks([0,1], xticks)
        plt.xlabel('')
    if labely!='auto':
        ax.set_ylabel(labely)
    if title!='auto':
        ax.set_title(title)
    y = ax.get_ylim()
    ax.set_ylim([y[0], y[0]+(y[1]-y[0])*1.1])
    plt.savefig(save_path)
    plt.close()


def correlate_behavior_neural(db, region, keys_behavior, keys_neural, save_path, topo_name='any', corr='pearson', stability_threshold = -np.inf):
    ddf = db[db['region']==region]

    if topo_name != 'any':
        ddf = ddf[ddf['topo_name'] == topo_name]

    nb = len(keys_behavior)
    nn = len(keys_neural)

    f, axs = plt.subplots(nb, nn, figsize=(2.5*nn, 2.5*nb))

    for i in range(nb):
        for j in range(nn):
            mask = (np.isnan(ddf[keys_neural[j]].values) == 0) & (np.isnan(ddf[keys_behavior[i]].values) == 0)
            axs[i,j].scatter(ddf[keys_neural[j]][mask], ddf[keys_behavior[i]][mask], alpha=0.6)
            if i== nb-1:
                axs[i,j].set_xlabel(keys_neural[j])
            if j==0:
                axs[i,j].set_ylabel(keys_behavior[i])
            # if j and i:
            #     axs[i,j].set_xticks([])
            #     axs[i,j].set_yticks([])
            if corr=='spearman':
                (r, p) = spearmanr(ddf[keys_neural[j]][mask], ddf[keys_behavior[i]][mask])
                if p<0.05:
                    axs[i,j].set_title('$\\rho = %.2f$  $p = %.5f$' % (r, p), color='r')
                else:
                    axs[i,j].set_title('$\\rho = %.2f$  $p = %.2f$' % (r, p), color='k')
            if corr=='pearson':
                (r, p) = pearsonr(ddf[keys_neural[j]][mask], ddf[keys_behavior[i]][mask])
                if p<0.05:
                    axs[i,j].set_title('$R = %.2f$  $p = %.5f$' % (r, p), color='r')
                else:
                    axs[i,j].set_title('$R = %.2f$  $p = %.2f$' % (r, p), color='k')

    plt.savefig(save_path)
    plt.close()


def correlate_session_number(db, region, keys, save_path, min_topi = 2, topo_name='any', corr='pearson'):
    ddf = db[db['region']==region]

    if topo_name != 'any':
        ddf = ddf[ddf['topo_name'] == topo_name]

    nb = len(keys)
    f, axs = plt.subplots(1, nb, figsize=(4*nb, 3))

    xs = np.arange(1, np.max(ddf['session_number'])+1)
    max_number = np.max([n for n in xs if len(ddf[ddf['session_number'].values == n]) >= min_topi])
    ddf = ddf[ddf['session_number'].values <= max_number]
    xs = np.arange(1, np.max(ddf['session_number'])+1)
    for i in range(nb):
        xs_scatter = []
        ys_scatter = []
        mask = (np.isnan(ddf[keys[i]].values) == 0) & (np.isnan(ddf['session_number'].values) == 0)
        ys = np.zeros(len(xs))
        ys_err = np.zeros(len(xs))

        for n in range(len(xs)):
            mask = (np.isnan(ddf[keys[i]].values) == 0) & (np.isnan(ddf['session_number'].values) == 0) & (ddf['session_number'].values == xs[n])
            ys[n] = np.mean(ddf[keys[i]][mask])
            ys_err[n] = np.std(ddf[keys[i]][mask])/np.sqrt(len(ddf[keys[i]][mask]))
            ys_scatter = ys_scatter + ddf[keys[i]][mask].values.tolist()
            xs_scatter = xs_scatter + ddf['session_number'][mask].values.tolist()

        axs[i].errorbar(xs, ys, ys_err, marker='o', linestyle='-')
        axs[i].scatter(xs_scatter, ys_scatter, alpha=0.3, color='k')
        axs[i].set_xlabel('session number')
        axs[i].set_ylabel(keys[i])

        if corr=='spearman':
            (r, p) = spearmanr(xs, ys)
            if p<0.05:
                axs[i].set_title('$\\rho = %.2f$  $p = %.5f$' % (r, p), color='r')
            else:
                axs[i].set_title('$\\rho = %.2f$  $p = %.2f$' % (r, p), color='k')
        if corr=='pearson':
            (r, p) = pearsonr(xs, ys)
            if p<0.05:
                axs[i].set_title('$R = %.2f$  $p = %.5f$' % (r, p), color='r')
            else:
                axs[i].set_title('$R = %.2f$  $p = %.2f$' % (r, p), color='k')

    plt.savefig(save_path)
    plt.close()


def box_comparison(datas, labels, quantity, box=False, swarm=True, bar=True, scatter=False, f=None, subplot=None):
    if f is None:
        f, ax = plt.subplots(figsize=(3.0, 3.5))
    else:
        if subplot is None:
            ax = f.add_subplot(111)
        else:
            ax = f.add_subplot(subplot)
    dataframes = []
    for i in range(len(datas)):
        dataframes.append(pd.DataFrame(datas[i], columns=[labels[i]]))
    data = pd.concat(dataframes, sort='False')
    data = data[labels]
    if bar:
        sns.barplot(data=data, ax=ax, capsize=.2, alpha=0.7, ci=68)
    if box:
        sns.boxplot(data=data, ax=ax, saturation=0.7, palette = 'Paired')
    if swarm:
        if box or bar:
            sns.swarmplot(data=data, ax=ax, color='k', alpha=0.5)
        else:
            sns.swarmplot(data=data, ax=ax, alpha=0.5)
            sns.pointplot(data=data, ax=ax, capsize=.2, join=False, alpha=0.9, marker='none', color='k', ci=68)
    if scatter:
        for i in range(len(datas)):
            plt.scatter(i + np.zeros(len(datas[i])), datas[i], alpha=0.6, color='k')
    ax.set_ylabel(quantity)
    sns.despine()
    return f, ax

def box_comparison_four(A, B, C, D, labelA, labelB, labelC, labelD, quantity, force=False, box = False, swarm=True, violin=False, bar=True, f=None, subplot=None, hue=None, parametric=True):
    print_stats(A, labelA)
    print_stats(B, labelB)
    print_stats(C, labelC)
    print_stats(D, labelD)
    if f is None:
        f, ax = plt.subplots(figsize=(3.0, 3.5))
    else:
        if subplot is None:
            ax = f.add_subplot(111)
        else:
            ax = f.add_subplot(subplot)

    dataA = pd.DataFrame(A, columns=[labelA])
    dataB = pd.DataFrame(B, columns=[labelB])
    dataC = pd.DataFrame(C, columns=[labelC])
    dataD = pd.DataFrame(D, columns=[labelD])
    # data = pd.concat([dataA, dataB, dataC, dataD], sort='False')
    # data = data[[labelA, labelB, labelC, labelD]]
    if hue is not None:
        hue_sns = np.concatenate((hue[0], hue[1], hue[2], hue[3]))
    else:
        hue_sns = np.concatenate((np.zeros(len(A)), np.zeros(len(B)), np.zeros(len(C)), np.zeros(len(D))))

    data = pd.DataFrame({
        'data': np.concatenate((A, B, C, D)),
        'x': [labelA for n in range(len(A))] + [labelB for n in range(len(B))] + [labelC for n in range(len(C))] + [labelD for n in range(len(D))],
        'hue': hue_sns
        })
    if bar:
        sns.barplot(data=data, x='x', y='data', ax=ax, capsize=.2, alpha=0.7, ci=68)
    if box:
        sns.boxplot(data=data, x='x', y='data', ax=ax, showfliers=False)
    if violin:
        sns.violinplot(data=data, x='x', y='data', ax=ax, showfliers=False, alpha=0.1, inner=None)
        sns.pointplot(data=data, x='x', y='data', ax=ax, capsize=.2, join=False, alpha=0.9, markers='_', color='w', ci=68)
    if swarm:
        ax_strip = sns.stripplot(data=data, x='x', y='data', hue='hue', ax=ax, size=5.0, alpha=0.5)
        ax_strip.legend_.remove()

    # plt.errorbar([0,1,2], [np.mean(A), np.mean(B), np.mean(C)], [np.std(A)/np.sqrt(len(A)), np.std(B)/np.sqrt(len(B)), np.std(C)/np.sqrt(len(C))], color='k', )
    ax.set_ylabel(quantity)
    if parametric:
        fvalue, pvalue = stats.f_oneway(dataA, dataB, dataC, dataD)
        print("ANOVA:", fvalue, pvalue)
    else:
        fvalue, pvalue = stats.kruskal(dataA, dataB, dataC, dataD)
        print("Kruskal-Wallis:", fvalue, pvalue)
        
        p1 = test_data(A, B, parametric=parametric)
        p2 = test_data(A, C, parametric=parametric)
        p3 = test_data(A, D, parametric=parametric)
        p4 = test_data(B, C, parametric=parametric)
        p5 = test_data(B, D, parametric=parametric)
        p6 = test_data(C, D, parametric=parametric)

        p_val_corr = mlt((p1, p2, p3, p4, p5, p6), alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelA, labelB, p_to_ast(p_val_corr[1][0])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelA, labelC, p_to_ast(p_val_corr[1][1])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelA, labelD, p_to_ast(p_val_corr[1][2]))) 
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelB, labelC, p_to_ast(p_val_corr[1][3])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelB, labelD, p_to_ast(p_val_corr[1][4])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelC, labelD, p_to_ast(p_val_corr[1][5])))
        p1 = annotate_ttest_p(A, B, 0, 1, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][0]) 
        p2 = annotate_ttest_p(A, C, 0, 2, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][1]) 
        p3 = annotate_ttest_p(A, D, 0, 3, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][2])
        p4 = annotate_ttest_p(B, C, 1, 2, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][3]) 
        p5 = annotate_ttest_p(B, D, 1, 3, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][4])  
        p6 = annotate_ttest_p(C, D, 2, 3, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][5])  
        sns.despine()
    return f, ax

def box_comparison_three(A, B, C, labelA, labelB, labelC, quantity, force=False, box = False, swarm=True, violin=False, bar=True, parametric=True, f=None, subplot=None, hue=None, paired=False):
    print_stats(A, labelA)
    print_stats(B, labelB)
    print_stats(C, labelC)
    if f is None:
        f, ax = plt.subplots(figsize=(3.0, 9.0))
    else:
        if subplot is None:
            ax = f.add_subplot(111)
        else:
            ax = f.add_subplot(subplot)
    dataA = pd.DataFrame(A, columns=[labelA])
    dataB = pd.DataFrame(B, columns=[labelB])
    dataC = pd.DataFrame(C, columns=[labelC])
    # data = pd.concat([dataA, dataB, dataC], sort='False')
    # data = data[[labelA, labelB, labelC]]
    if hue is not None:
        print(hue[0], hue[1], hue[2])
        hue_sns = np.concatenate((hue[0], hue[1], hue[2]))
    else:
        hue_sns = np.concatenate((np.zeros(len(A)), np.zeros(len(B)), np.zeros(len(C))))

    data = pd.DataFrame({
        'data': np.concatenate((A, B, C)),
        'x': [labelA for n in range(len(A))] + [labelB for n in range(len(B))] + [labelC for n in range(len(C))],
        'hue': hue_sns
        })
    if bar:
        sns.barplot(data=data, x='x', y='data', ax=ax, capsize=.2, alpha=0.7, ci=68)
    if box:
        sns.boxplot(data=data, x='x', y='data', ax=ax, showfliers=False)
    if violin:
        sns.violinplot(data=data, x='x', y='data', ax=ax, showfliers=False, alpha=0.1, inner=None)
        sns.pointplot(data=data, x='x', y='data', ax=ax, capsize=.2, join=False, alpha=0.9, markers='_', color='w', ci=68)
    if swarm:
        ax_strip = sns.stripplot(data=data, x='x', y='data', hue='hue', ax=ax, size=5.0, alpha=0.5)
        ax_strip.legend_.remove()

    # plt.errorbar([0,1,2], [np.mean(A), np.mean(B), np.mean(C)], [np.std(A)/np.sqrt(len(A)), np.std(B)/np.sqrt(len(B)), np.std(C)/np.sqrt(len(C))], color='k', )
    ax.set_ylabel(quantity)
    #aovrm = AnovaRM(data, within = [labelA, labelB, labelC], aggregate_func = 'mean'), fit()
    #print(aovrm)
    if parametric:
        fvalue, pvalue = stats.f_oneway(dataA, dataB, dataC)
        print("ANOVA:", fvalue, pvalue)
    else:
        fvalue, pvalue = stats.kruskal(dataA, dataB, dataC)
        print("Kruskal-Wallis:", fvalue, pvalue)
    # P_val_corr = mlt(pvalue, alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
    # print(P_val_corr)

        p1 = test_data(A, B, parametric=parametric)
        p2 = test_data(B, C, parametric=parametric)
        p3 = test_data(A, C, parametric=parametric)

        p_val_corr = mlt((p1, p2, p3), alpha=0.05, method='bonferroni', is_sorted=False, returnsorted=False)
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelA, labelB, p_to_ast(p_val_corr[1][0])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelB, labelC, p_to_ast(p_val_corr[1][1])))
        print("Corrected pval bonferroni (%s)-(%s):\t%s" % (labelA, labelC, p_to_ast(p_val_corr[1][2])))
        p1 = annotate_ttest_p(A, B, 0, 1, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][0]) 
        p2 = annotate_ttest_p(B, C, 1, 2, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][1]) 
        p3 = annotate_ttest_p(A, C, 0, 2, ax, force=force, parametric=parametric, pfixed=p_val_corr[1][2]) 
        sns.despine()
    return f, ax

def pval_correction(self, method = None):
    import statsmodels.stats.multitest as smt
    if method is None:
        method = self.multitest_method
        return smt.multitests(self_pvals_raw, method = method)[1]

def box_comparison_two(A, B, labelA, labelB, quantity, force=False, swarm=True, violin=False, box = False, paired = False, bar=True, f=None, hue=None, subplot=None):
    print_stats(A, labelA)
    print_stats(B, labelB)
    if f is None:
        f, ax = plt.subplots(figsize=(3.0, 3.5))
    else:
        if subplot is None:
            ax = f.add_subplot(111)
        else:
            ax = f.add_subplot(subplot)
    dataA = pd.DataFrame(A, columns=[labelA])
    dataB = pd.DataFrame(B, columns=[labelB])
    #data = pd.concat([dataA, dataB], sort='False')
    #data = data[[labelA, labelB]]
    if hue is not None:
        print(hue[0], hue[1])
        hue_sns = np.concatenate((hue[0], hue[1]))
    else:
        hue_sns = np.concatenate((np.zeros(len(A)), np.zeros(len(B))))

    data = pd.DataFrame({
        'data': np.concatenate((A, B)),
        'x': [labelA for n in range(len(A))] + [labelB for n in range(len(B))],
        'hue': hue_sns
        })
    
    if box:
        sns.boxplot(data=data, ax=ax, showfliers=False)
    elif bar:
        sns.barplot(data=data, x='x', y='data', ax=ax, capsize=.2, alpha=0.7, ci=68)
    elif paired==False:
        sns.pointplot(data=data, ax=ax, capsize=.2, join=False, alpha=0.9, marker='none', color='k', ci=68)

    if swarm:
        if paired==0:
            #sns.swarmplot(data=data, ax=ax, color='0.5', alpha=0.6)
            sns.stripplot(data=data, x='x', y='data', ax=ax, hue = 'hue', size=5.0, alpha=0.3)
            
    # else:
    #     plt.scatter(np.zeros(len(A)),A, color='0.5', alpha=0.6)
    #     plt.scatter(np.ones(len(B)),B, color='0.5', alpha=0.6)
    if violin:
        sns.violinplot(data=data, ax=ax, color='0.5', alpha=0.3)
    if paired:
        draw_pair_plot(np.asarray(A),np.asarray(B), 0, 1, ax, showmeans=swarm)

    # plt.errorbar([0,1,2], [np.mean(A), np.mean(B), np.mean(C)], [np.std(A)/np.sqrt(len(A)), np.std(B)/np.sqrt(len(B)), np.std(C)/np.sqrt(len(C))], color='k', )
    ax.set_ylabel(quantity)
    if paired:
        p1 = annotate_ttestpaired_p(A, B, 0, 1, ax, force=force)
        print("Paired t-test (%s)-(%s):\t%s" % (labelA, labelB, p_to_ast(p1)))
        pw = wilcoxon(A, B)[1]
        print("Paired wilcoxon (%s)-(%s):\t%s" % (labelA, labelB, p_to_ast(pw)))
        nanmask = (np.isnan(A)==0) & (np.isnan(B)==0)
        print("Number of data %s: %u, %s: %u\n" % (labelA, np.sum(nanmask), labelB, np.sum(nanmask)))
    else:
        p1 = annotate_ttest_p(A, B, 0, 1, ax, force=force)
        print("Un-paired t-test (%s)-(%s):\t%s" % (labelA, labelB, p_to_ast(p1)))
        print("Number of data %s: %u, %s: %u\n" % (labelA, np.sum(np.isnan(A)==0), labelB, np.sum(np.isnan(B)==0)))
    sns.despine()

    return f, ax


def plot_corr_timedist(FF, NN, FN, maxdist, region, savepath):
    f, axs = plt.subplots(1,3,figsize=(11,3), sharey=True)
    x = list(range(1, min(maxdist, len(FF)-2)))
    means_FF = [bn.nanmean(FF[i]) for i in x]
    stds = [np.nanstd(FF[i])/np.sqrt(len(FF[i])) for i in x]
    try:
        r, pcorr = pearsonr(x, means_FF)
    except:
        r, pcorr = np.nan, np.nan
    axs[0].errorbar(x, means_FF, stds, marker='o')
    axs[0].set_xlabel('time distance between laps')
    axs[0].set_ylabel('$\\vec{r}$ correlation')
    axs[0].set_title('|||-|||')
    ys = axs[0].get_ylim()
    axs[0].text(np.mean(x), ys[0]+(ys[1]-ys[0])*0.9, 'R = %.2f, p = %.3f' % (r, pcorr), ha='center')

    x = list(range(1, min(maxdist, len(NN)-2)))
    means_NN = [bn.nanmean(NN[i]) for i in x]
    stds = [np.nanstd(NN[i])/np.sqrt(len(NN[i])) for i in x]
    try:
        r, pcorr = pearsonr(x, means_NN)
    except:
        r, pcorr = np.nan, np.nan
    axs[1].errorbar(x, means_NN, stds, marker='o')
    axs[1].set_xlabel('time distance between laps')
    # axs[1].set_ylabel('$\\vec{r}$ correlation')
    axs[1].set_title('%s-%s' % (labels_novel[region], labels_novel[region]))
    ys = axs[1].get_ylim()
    axs[1].text(np.mean(x), ys[0]+(ys[1]-ys[0])*0.9, 'R = %.2f, p = %.3f' % (r, pcorr), ha='center')

    x = list(range(1, min(maxdist, len(FN)-2)))
    means_FN = [bn.nanmean(FN[i]) for i in x]
    stds = [np.nanstd(FN[i])/np.sqrt(len(FN[i])) for i in x]
    try:
        r, pcorr = pearsonr(x, means_FN)
    except:
        r, pcorr = np.nan, np.nan
    axs[2].errorbar(x, means_FN, stds, marker='o')
    axs[2].set_xlabel('time distance between laps')
    # axs[2].set_ylabel('$\\vec{r}$ correlation')
    axs[2].set_title('|||-%s' % labels_novel[region])
    ys = axs[2].get_ylim()
    axs[2].text(np.mean(x), ys[0]+(ys[1]-ys[0])*0.9, 'R = %.2f, p = %.3f' % (r, pcorr), ha='center')

    sns.despine(fig=f)
    plt.savefig(savepath)
    plt.close()

    return means_FF, means_NN, means_FN


def mkdir(path):
    if os.path.exists(path)==0:
        os.mkdir(path)

def test_data(dataA, dataB, parametric=True):
    data1 = np.asarray(dataA)
    data2 = np.asarray(dataB)
    data1 = data1[np.isnan(data1)==0]
    data2 = data2[np.isnan(data2)==0]
    if len(data1) and len(data2):
        if parametric:
            return ttest(data1, data2)[1]
        else:
            return mannwhitneyu(data1, data2, alternative='two-sided')[1] # Added 2022-11-15, CSH
    else:
        return np.nan

def annotate_ttest_p(dataA, dataB, x1, x2, ax, pairplot=False, force=False, size=7, parametric=True, pfixed=None):
    if pfixed is None:
        p = test_data(dataA, dataB, parametric=parametric)
    else:
        p = pfixed
    if not np.isnan(p):
        if (p<0.05) or (force):
            ys = ax.get_ylim() # [np.min([np.min(data1), np.min(data2)]), np.max([np.max(data1), np.max(data2)])]
            y = ys[1] + (ys[1]-ys[0])*0.1
            dy = (ys[1]-ys[0])*0.02
            dx = np.abs(x1-x2)*0.02
            ax.plot([x1+dx, x1+dx, x2-dx, x2-dx], [y, y+dy, y+dy, y], 'k')
            ax.text((x1+x2)/2., y+2*dy, p_to_text(p), ha='center', va='bottom', color='k', fontsize=size)
            
        if pairplot:
            draw_pair_plot(data1, data2, x1, x2, ax)
        return p
    else:
        return np.nan

def annotate_wilcoxon_p(dataA, dataB, x1, x2, ax, pairplot=False, force=False):
    data1 = np.asarray(dataA)
    data2 = np.asarray(dataB)
    mask = (np.isnan(data1)==0) & (np.isnan(data2)==0)
    data1 = data1[mask]
    data2 = data2[mask]
    if np.sum(mask):
        p = wilcoxon(data1, data2)[1]
        if (p<0.05) or force:
            ys = [np.min([np.min(data1), np.min(data2)]), np.max([np.max(data1), np.max(data2)])]
            y = ys[1] + (ys[1]-ys[0])*0.1
            dy = (ys[1]-ys[0])*0.02
            dx = np.abs(x1-x2)*0.02
            ax.plot([x1+dx, x1+dx, x2-dx, x2-dx], [y, y+dy, y+dy, y], 'k')
            ax.text((x1+x2)/2., y+2*dy, p_to_ast(p), ha='center', va='bottom', color='r', fontsize=7)
        if pairplot:
            draw_pair_plot(data1, data2, x1, x2, ax)
        return p
    else:
        return np.nan

def annotate_ttestpaired_p(dataA, dataB, x1, x2, ax, pairplot=True, force=False):
    data1 = np.asarray(dataA)
    data2 = np.asarray(dataB)
    mask = (np.isnan(data1)==0) & (np.isnan(data2)==0)
    data1 = data1[mask]
    data2 = data2[mask]
    if np.sum(mask):
        p = ttest_p(data1, data2)[1]
        if (p<0.05) or force:
            ys = [np.min([np.min(data1), np.min(data2)]), np.max([np.max(data1), np.max(data2)])]
            y = ys[1] + (ys[1]-ys[0])*0.1
            dy = (ys[1]-ys[0])*0.02
            dx = np.abs(x1-x2)*0.02
            ax.plot([x1+dx, x1+dx, x2-dx, x2-dx], [y, y+dy, y+dy, y], 'k')
            ax.text((x1+x2)/2., y+2*dy, p_to_ast(p), ha='center', va='bottom', color='r', fontsize=7)
        if pairplot:
            draw_pair_plot(data1, data2, x1, x2, ax)
        return p
    else:
        return np.nan

def annotate_wilcoxon_p_single(dataA, x1, x2, ax, force=False):
    data1 = np.asarray(dataA)
    mask = (np.isnan(data1)==0)
    data1 = data1[mask]
    if np.sum(mask):
        p = wilcoxon(data1)[1]
        if (p<0.05) or force:
            ys = [np.min(data1), np.max(data1)]
            y = ys[1] + (ys[1]-ys[0])*0.1
            dy = (ys[1]-ys[0])*0.02
            dx = np.abs(x1-x2)*0.02
            ax.plot([x1+dx, x1+dx, x2-dx, x2-dx], [y, y+dy, y+dy, y], 'k')
            ax.text((x1+x2)/2., y+2*dy, p_to_ast(p), ha='center', va='bottom', color='r', fontsize=7)
        return p
    else:
        return np.nan


def draw_pair_plot(data1, data2, x1, x2, ax, showmeans=False):
    mask = (np.isnan(data1)==0) & (np.isnan(data2)==0)
    for i in range(len(data1)):
        ax.plot([x1, x2], [data1[i], data2[i]], color='k', alpha=0.5)
    if showmeans:
        plt.errorbar([x1, x2], [bn.nanmean(data1[mask]), bn.nanmean(data2[mask])], [np.nanstd(data1[mask])/np.sqrt(len(data1[mask])), np.nanstd(data2[mask])/np.sqrt(len(data2[mask]))], linestyle='', marker='o', color=pltcolors[0], capsize=5)

def p_to_ast_only(p):
    if 0.01<p<0.05:
        return '*' % p
    if 0.001 < p < 0.01:
        return '**' % p
    if 0.0001 < p < 0.001:
        return '***' % p
    if p < 0.0001:
        return '***+' % p
    else:
        return 'NS' % p

def p_to_ast(p):
    if 0.01<p<0.05:
        return '* (p = %.2f)' % p
    if 0.001 < p < 0.01:
        return '** (p = %.3f)' % p
    if 0.0001 < p < 0.001:
        return '*** (p = %.4f)' % p
    if p < 0.0001:
        return '***+ (p = %.1e)' % p
    else:
        return 'NS (p = %.2f)' % p

def p_to_text(p):
    if 0.01<p<0.05:
        return 'p = %.2f' % p
    if 0.001 < p < 0.01:
        return 'p = %.3f' % p
    if 0.0001 < p < 0.001:
        return 'p = %.4f' % p
    if 0.00001 < p < 0.0001:
        return 'p = %.5f' % p
    if p < 0.00001:
        return 'p = %.1e' % p
    else:
        return 'p = %.2f' % p

def pov(ratemaps1, ratemaps2, percell=False, VR=True, nmin=2, flip=False):
    if flip:
        maps2 = ratemaps2.copy()[:, ::-1]
    else:
        maps2 = ratemaps2.copy()
    maps1 = np.transpose(ratemaps1)
    maps2 = np.transpose(maps2)
    if percell: # so elegant that I'm throwing up
        maps1 = np.transpose(maps1)
        maps2 = np.transpose(maps2)
    nbins = maps1.shape[0]
    ps = np.zeros(nbins)

    for i in range(nbins):
        if percell:
            ps[i] = ratemap_correlation(maps1[i], maps2[i], 0.05, periodic=VR, nmin=nmin)
        else:
            mask = (np.isnan(maps1[i]) == 0) & (np.isnan(maps2[i]) == 0)
            if len(maps1[i][mask]) < nmin and len(maps2[i][mask]) < nmin:
                ps[i] = np.nan
            else:
                ps[i] = pearsonr(maps1[i][mask], maps2[i][mask])[0]
    return ps

def manufy(ax, ymax=1.0, yrand=0, data = [], ylim=[]):
    if len(data):
        mins = [np.min(d[np.isnan(d)==0]) for d in data]
        maxs = [np.max(d[np.isnan(d)==0]) for d in data]
        mindata = np.min(mins)
        maxdata = np.max(maxs)
    else:
        ys = ax.get_ylim()
        mindata = ys[0]
        maxdata = ys[1]

    dy = (ymax - yrand) * 0.2
    yhigh = dy * ceil(maxdata / dy) + dy #TODO: understand why does not go to 120 in spatial laps
    ylow = dy * floor(mindata / dy)
    ny = (yhigh - ylow)/dy + 1
    yticks = np.linspace(ylow, yhigh, ny)
    ax.set_yticks(yticks)
    # ylabels = []
    # for y in yticks:
    # 	if y < yrand or y > ymax:
    # 		ylabels.append('')
    # 	elif ymax < 10:
    # 		ylabels.append('%.2f' % y)
    # 	else:
    # 		ylabels.append('%u' % y)
    ax.set_ylim([ylow, yhigh])
    if len(ylim):
        ax.set_ylim(ylim)
    # ax.set_yticklabels(ylabels)
    sns.despine()
    ax.axhline([yrand], linestyle='--', color='k')
    ax.spines['left'].set_position(('outward', 5))

def pointplotmio(ddf, xs, ylabel, ax, legend):
    ys = np.zeros(len(xs))
    ys_err = np.zeros(len(xs))
    for n in range(len(xs)):
        mask = (np.isnan(ddf[ylabel].values) == 0) & (np.isnan(ddf['session_number'].values) == 0) & (ddf['session_number'].values == xs[n])
        ys[n] = np.mean(ddf[ylabel][mask])
        ys_err[n] = np.std(ddf[ylabel][mask])/np.sqrt(len(ddf[ylabel][mask]))
    ax.errorbar(xs, ys, ys_err, marker='o', linestyle='-', label=legend)
    ax.set_xlabel('session number')
    return ys

def ratemap_correlation(rate1, rate2, peak_fraction, periodic=True, nmin=2):
    ibins_peaksearch = int(len(rate1)*peak_fraction)
    mask = (np.isnan(rate1) == 0) & (np.isnan(rate2) == 0)
    map1 = rate1[mask]
    map2 = rate2[mask]
    if len(map1) < nmin and len(map2) < nmin:
        return np.nan

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
            peak_cc_at_center = pearsonr(map1, map2)[0]
            return peak_cc_at_center
    #axs[i].set_ylabel(keys[i])

def paired_topo_day(db, quantity, region='all', norm=False):
    if region != 'all':
        db_region = db[(db['region']==region)]
        db_region_CL = db[(db['region']==region+'_CL')]
    else:
        db_region = db[(db['region']=='DG') | (db['region']=='CA1') | (db['region']=='CA3')]
        db_region_CL = db[(db['region']=='DG_CL') | (db['region']=='CA1_CL') | (db['region']=='CA3_CL')]

    topo_day_keys_grat = np.asarray([row['topo_name']+row['session_name'][8:16] for index, row in db_region.iterrows()])
    topo_day_keys_CL = np.asarray([row['topo_name']+row['session_name'][8:16] for index, row in db_region_CL.iterrows()])
    topo_day_keys = []
    topo_day_keys_unique = np.unique(topo_day_keys_grat)
    values_grat = []
    values_CL = []

    for tdk in topo_day_keys_unique:
        grat = db_region[(topo_day_keys_grat == tdk)][quantity].values
        CL = db_region_CL[(topo_day_keys_CL == tdk)][quantity].values

        if len(grat)>0 and len(CL)>0:
            # print tdk, grat, CL
            values_grat.append(np.mean(grat))
            values_CL.append(np.mean(CL))
            topo_day_keys.append(tdk)

    values_grat = np.asarray(values_grat)
    values_CL = np.asarray(values_CL)

    if norm:
        topokeys = np.unique(np.asarray([key[:6] for key in topo_day_keys]))
        for topokey in topokeys:
            topo_mask = np.asarray([topokey in key for key in topo_day_keys])
            values_topo_grat = values_grat[topo_mask]
            values_topo_CL = values_CL[topo_mask]
            # normalizing
            min_val = np.nanmin([np.nanmin(values_topo_grat), np.nanmin(values_topo_CL)])
            max_val = np.nanmax([np.nanmax(values_topo_grat), np.nanmax(values_topo_CL)])
            values_grat[topo_mask] = (values_topo_grat - min_val)/(max_val - min_val)
            values_CL[topo_mask] = (values_topo_CL - min_val)/(max_val - min_val)

    return values_grat, values_CL

def paired_analysis(db, neural_key, behavior_key, region):
    f, ax = plt.subplots(figsize=(4,3))
    hit_rate_N_grat, hit_rate_N_CL = paired_topo_day(db, behavior_key, region=region, norm=True)
    auc_grat, auc_CL = paired_topo_day(db, neural_key, region=region, norm=True)
    X = np.hstack([auc_grat, auc_CL])
    Y = np.hstack([hit_rate_N_grat, hit_rate_N_CL])
    nanmask = (np.isnan(X)==0) & (np.isnan(Y)==0)
    ress = linregress(X[nanmask], Y[nanmask])
    sns.regplot(X, Y, ax=ax)
    sns.despine()
    ax.set_title('R = %.2f, %s' % (ress.rvalue, p_to_text(ress.pvalue)))
    print(('\nLinear regression %s vs. %s - R = %.3f, p = %.1e \n' % (neural_key, behavior_key, ress.rvalue, ress.pvalue)))
    return f, ax, ress


def print_stats(data, name):
    d = np.asarray(data)
    m = bn.nanmean(d)
    std = np.nanstd(d)
    stder = sem(d[np.isnan(d)==0])
    print("> %s :\tmean %.3f +- %.3f SEM, std: %.3f" % (name, m, stder, std))

def running_mean_std(array, array_sort, window_width, window_step, min_n=2):
    index = np.argsort(array_sort)
    datay = array[index]
    datax = array_sort[index]
    x = []
    y = []
    yerr = []
    w = (np.max(datax) - np.min(datax)) * window_width
    dw = (np.max(datax) - np.min(datax)) * window_step
    nbins = int(1.0/window_step)

    for i in range(nbins):
        windex = (datax > i*dw) & (datax < i*dw+w)
        if np.sum(windex) > min_n:
            x.append(bn.nanmean(datax[windex]))
            y.append(bn.nanmean(datay[windex]))
            yerr.append(np.nanstd(datay[windex])/np.sqrt(len(datay[windex])-1))
        else:
            x.append(np.nan)
            y.append(np.nan)
            yerr.append(np.nan)

    return np.asarray(x), np.asarray(y), np.asarray(yerr)

def visualize_running_mean(X, Y, window_width, window_step, labelx, labely, ax='none', color_index=0, min_n=2):
    x, y, yerr = running_mean_std(Y, X, window_width, window_step, min_n=min_n)
    if ax=='none':
        f, ax = plt.subplots(figsize=(2.5,3.5))
    ax.plot(x, y, color=pltcolors[color_index], linewidth=2.0)
    ax.plot(x, y+yerr, color=pltcolors[color_index], linewidth=0.5)
    ax.plot(x, y-yerr, color=pltcolors[color_index], linewidth=0.5)
    ax.fill_between(x, y-yerr, y+yerr, facecolor=pltcolors[color_index], alpha=0.3)
    ax.set_xlabel(labelx)
    ax.set_ylabel(labely)
    sns.despine()
    return ax

def separate_mice(db, group1, group2, key, labelA, labelB, region):
    maskA = np.asarray([name in group1 for name in db['topo_name'].values], dtype=bool)
    maskB = np.asarray([name in group2 for name in db['topo_name'].values], dtype=bool)
    mask_region = db['region'] == region
    dbA = db[maskA & mask_region]
    dbB = db[maskB & mask_region]
    f, ax = box_comparison_two(dbA[key].values, dbB[key].values, labelA, labelB, key, force=True, bar=True)
    return f, ax


def visualize_spatial_maps_in_rois(session, savepath='none', VR = True, maptype='All', plotmean=True):

    mapkey1 = 'Smap_running'
    mapkey2 = 'ratemap_running'
    if maptype == 'All':
        mapkeys = [mapkey1, mapkey2]
    elif maptype == 'S':
        mapkeys = [mapkey1]
    elif maptype == 'F':
        mapkeys = [mapkey2]

    for mapkey in mapkeys:
        print(mapkey)
        N_roi = session.n_roi
        fig = plt.figure(figsize=(10,N_roi))
        fig.suptitle(session.session_name)
        print((session.session_name))

        map1_len = len(session.laps[0].s_maps[0])
        map2_len = len(session.laps[0].rate_maps[0])
        bin_len = len(session.bins)

        for nroi in range(N_roi):
            
            if nroi == 0:
                ax0obl = ax_obl = fig.add_subplot(N_roi,2,nroi*2+1)
                ax0vert = ax_vert = fig.add_subplot(N_roi,2,nroi*2+2, sharey=ax_obl)
            else:
                ax0obl = fig.add_subplot(N_roi,2,nroi*2+1, sharex=ax_obl)
                ax0vert = fig.add_subplot(N_roi,2,nroi*2+2, sharey=ax0obl, sharex=ax_vert)
            if nroi == 0:
                if VR:
                    if mapkey == 'Smap_running':
                        ax0obl.set_title("familiar (smap)")
                        ax0vert.set_title("novel (smap)")
                    else:
                        ax0obl.set_title("familiar (ratemap)")
                        ax0vert.set_title("novel (ratemap)")
                else:
                    if mapkey == 'Smap_running':
                        ax0obl.set_title("inbound (smap)")
                        ax0vert.set_title("outbound (smap)")
                    else:
                        ax0obl.set_title("inbound (ratemap)")
                        ax0vert.set_title("outbound (ratemap)")

            for lap_i, lap in enumerate(session.laps):

                if lap_i in session.incompletelaps:
                    ax0vert.plot([30],[0], alpha=0)
                    ax0obl.plot([30],[0], alpha=0)


                if lap in session.familiar_laps:
                    if VR:
                        print("familiar lap")
                    else:
                        print("inbound lap")
                    ax1_map = ax0vert
                    ax2_map = ax0obl

                elif lap in session.novel_laps:
                    if VR:
                        print("novel lap")
                    else:
                        print("outbound lap")
                    ax1_map = ax0obl
                    ax2_map = ax0vert

                else:
                    continue
                if mapkey == 'Smap_running':
                    #move.printA("roi["+str(nroi+1)+"]=>laps["+str(lap_i)+"].s_maps["+str(nroi)+"]",lap.s_maps[nroi])
                    ax2_map.plot(session.bins[:map1_len],lap.s_maps[nroi][:bin_len], alpha=0.5, label='lap %d' % lap_i)
                    ax1_map.plot(session.bins[:map1_len],lap.s_maps[nroi][:bin_len], alpha=0)
                else:
                    #move.printA("roi["+str(nroi+1)+"]=>laps["+str(lap_i)+"].rate_maps["+str(nroi)+"]",lap.rate_maps[nroi])
                    ax2_map.plot(session.bins[:map2_len],lap.rate_maps[nroi][:bin_len], alpha=0.5, label='lap %d' % lap_i)
                    ax1_map.plot(session.bins[:map2_len],lap.rate_maps[nroi][:bin_len], alpha=0)

            ax0obl.text(
                1.0, 0.5, "{0}".format(nroi),
                horizontalalignment='left',
                verticalalignment='center',
                transform=ax0obl.transAxes)
            if mapkey == 'Smap_running':
                if nroi in np.arange(len(session.spatial_cells_S))[session.spatial_cells_S]:
                    ax0obl.text(
                        1.1, 0.5, "PC",
                        horizontalalignment='right',
                        verticalalignment='center',
                        transform=ax0obl.transAxes,color='red',fontsize=15)
            else:
                if nroi in np.arange(len(session.spatial_cells_F))[session.spatial_cells_F]:
                    ax0obl.text(
                        1.1, 0.5, "PC",
                        horizontalalignment='right',
                        verticalalignment='center',
                        transform=ax0obl.transAxes,color='red',fontsize=15)
                    
            if nroi != N_roi-1:
                ax0vert.set_xticks([])
                ax0obl.set_xticks([])

            if plotmean:
                if mapkey == 'Smap_running':
                    meanvert = bn.nanmean([lap.s_maps[nroi]
                                    for lap in session.laps
                                       if lap in session.novel_laps], axis=0)
                    meanobl = bn.nanmean([lap.s_maps[nroi]
                                    for lap in session.laps
                                       if lap in session.familiar_laps], axis=0)
                    ax0vert.plot(session.bins[:map1_len], meanvert[:bin_len], '-r', lw=4)
                    ax0obl.plot(session.bins[:map1_len], meanobl[:bin_len], '-r', lw=4)
                else:
                    meanvert = bn.nanmean([lap.rate_maps[nroi]
                                    for lap in session.laps
                                       if lap in session.novel_laps], axis=0)
                    meanobl = bn.nanmean([lap.rate_maps[nroi]
                                    for lap in session.laps
                                       if lap in session.familiar_laps], axis=0)
                    ax0vert.plot(session.bins[:map2_len], meanvert[:bin_len], '-r', lw=4)
                    ax0obl.plot(session.bins[:map2_len], meanobl[:bin_len], '-r', lw=4)


        if savepath != 'none':
            print(savepath)
            if plotmean:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_w_mean.pdf')
                else:
                    plt.savefig(savepath+'_w_mean.pdf')
                    plt.close()
            else:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_wo_mean.pdf')
                else:
                    plt.savefig(savepath+'_wo_mean.pdf')
                    plt.close()


def visualize_spatial_maps_in_laps(session, savepath='none', VR = True, maptype='All', plotmean=True): 

    mapkey1 = 'Smap_running'
    mapkey2 = 'ratemap_running'
    if maptype == 'All':
        mapkeys = [mapkey1, mapkey2]
    elif maptype == 'S':
        mapkeys = [mapkey1]
    elif maptype == 'F':
        mapkeys = [mapkey2]

    for mapkey in mapkeys:
        print(mapkey)
        N_lap = session.nlaps
        fig = plt.figure(figsize=(10,N_lap))
        if mapkey == 'Smap_running':
            fig.suptitle(session.session_name+" (smap)")
        else:
            fig.suptitle(session.session_name+" (ratemap)")
        print((session.session_name))

        map1_len = len(session.laps[0].s_maps[0])
        map2_len = len(session.laps[0].rate_maps[0])
        bin_len = len(session.bins)

        N_roi = session.n_roi

        for nlap, lap in enumerate(session.laps):

            #ax0obl = fig.add_subplot(N_roi,2,nroi*2+1)
            #ax0vert = fig.add_subplot(N_roi,2,nroi*2+2, sharey=ax0obl)
            if nlap == 0:
                ax = ax0 = fig.add_subplot(N_lap, 1, nlap+1)
            else:
                ax = fig.add_subplot(N_lap, 1, nlap+1, sharex=ax0)
            if not VR:
                if session.laps[nlap].laptype == 'top':
                    ax.set_ylabel("inbound", color='green')
                else:
                    ax.set_ylabel("outbound", color='red')

            ax.text(
                1.0, 0.5, "lap{0}".format(nlap+1),
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)

            for nroi in range(N_roi):
                if mapkey == 'Smap_running':
                    ax.plot(session.bins[:map1_len],lap.s_maps[nroi][:bin_len], alpha=0.5, label='roi %d' % nroi)
                else:
                    ax.plot(session.bins[:map2_len],lap.rate_maps[nroi][:bin_len], alpha=0.5, label='roi %d' % nroi)

            if nlap!= N_lap-1:
                ax.set_xticks([])


            if plotmean:
                #print("bin_len",bin_len,"map1_len",map1_len,"map2_len",map2_len)
                if mapkey == 'Smap_running':
                    #move.printA("lap.s_maps",lap.s_maps)
                    meanlap = bn.nanmean(lap.s_maps, axis=0)
                    #move.printA(" meanlap", meanlap)
                    ax.plot(session.bins[:map1_len], meanlap[:bin_len], '-r', lw=4)
                else:
                    #move.printA("lap.rate_maps",lap.rate_maps)
                    meanlap = bn.nanmean(lap.rate_maps, axis=0)
                    #move.printA(" meanlap", meanlap)
                    ax.plot(session.bins[:map2_len], meanlap[:bin_len], '-r', lw=4)
                move.printA(" session.bins", session.bins)
        if savepath != 'none':
            print(savepath)
            if plotmean:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_w_mean.pdf')
                else:
                    plt.savefig(savepath+'_w_mean.pdf')
                    plt.close()
            else:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_wo_mean.pdf')
                else:
                    plt.savefig(savepath+'_wo_mean.pdf')
                    plt.close()

                    
def visualize_inference_in_laps(session, savepath='none', VR = True, maptype='All', plotmean=True): 

    mapkey1 = 'Smap_running'
    mapkey2 = 'ratemap_running'
    if maptype == 'All':
        mapkeys = [mapkey1, mapkey2]
    elif maptype == 'S':
        mapkeys = [mapkey1]
    elif maptype == 'F':
        mapkeys = [mapkey2]

    for mapkey in mapkeys:
        print(mapkey)
        N_lap = session.nlaps
        fig = plt.figure(figsize=(10,N_lap))
        if mapkey == 'Smap_running':
            fig.suptitle(session.session_name+" (S)")
        else:
            fig.suptitle(session.session_name+" (dF_F)")
        print((session.session_name))

        map1_len = len(session.laps[0].S[0])
        map2_len = len(session.laps[0].dF_F[0])
        bin_len = len(session.bins)


        N_roi = session.n_roi

        for nlap, lap in enumerate(session.laps):

            #ax0obl = fig.add_subplot(N_roi,2,nroi*2+1)
            #ax0vert = fig.add_subplot(N_roi,2,nroi*2+2, sharey=ax0obl)
            ax = fig.add_subplot(N_lap, 1, nlap+1)
            if VR:
                pass
            else:
                if mapkey == 'Smap_running':
                    if session.laps[nlap].laptype == 'top':
                        ax.set_ylabel("inbound")
                    else:
                        ax.set_ylabel("outbound")
                else:
                    if session.laps[nlap].laptype == 'top':
                        ax.set_ylabel("inbound")
                    else:
                        ax.set_ylabel("outbound")

            ax.text(
                1.0, 0.5, "lap{0}".format(nlap+1),
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes)

            for nroi in range(N_roi):
                if mapkey == 'Smap_running':
                    ax.plot(lap.S[nroi], alpha=0.5, label='roi %d' % nroi)
                else:
                    ax.plot(lap.dF_F[nroi], alpha=0.5, label='roi %d' % nroi)

            if nlap!= N_lap-1:
                ax.set_xticks([])


            if plotmean:
                print("bin_len",bin_len,"map1_len",map1_len,"map2_len",map2_len)
                if mapkey == 'Smap_running':
                    #move.printA("lap.S",lap.S)
                    meanlap = bn.nanmean(lap.S, axis=0)
                    #move.printA(" meanlap(S)", meanlap)
                    ax.plot(meanlap, '-r', lw=4)
                else:
                    #move.printA("lap.dF_F",lap.dF_F)
                    meanlap = bn.nanmean(lap.dF_F, axis=0)
                    #move.printA(" meanlap(dF_F)", meanlap)
                    ax.plot(meanlap, '-r', lw=4)
                move.printA(" session.bins", session.bins)
        if savepath != 'none':
            print(savepath)
            if plotmean:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_w_mean.pdf')
                else:
                    plt.savefig(savepath+'_w_mean.pdf')
                    plt.close()
            else:
                if maptype == 'All':
                    plt.savefig(savepath+'_'+mapkey+'_wo_mean.pdf')
                else:
                    plt.savefig(savepath+'_wo_mean.pdf')
                    plt.close()

def visualize_spatial_maps_neworder_three(
        maps1, maps2, maps3, maps4, maps5, maps6, min_cells=2, savepath='none', mode='or', HB=False, cmap='magma', plotfunc = lambda x: x, VR = True,
        mapname=["map1","map2","map3","map4","map5","map6"], nmin_corr=2, verbose=False):

    # plotting familiar vs. familiar
    if mode =='and':
        alive = (bn.nansum(maps1, 1) > 0) & (bn.nansum(maps2, 1) > 0)
    elif mode == 'or':
        alive = (bn.nansum(maps1, 1) > 0) | (bn.nansum(maps2, 1) > 0)
    elif mode == 'F':
        alive = (bn.nansum(maps1, 1) > 0)
    elif mode == 'all':
        alive = (bn.nansum(maps1, 1) != np.nan)

    if len(alive):
        maps1 = maps1[alive]
        maps2 = maps2[alive]

        masscenters = bn.nanargmax(maps1, 1)
        masscenters[bn.nansum(maps1, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps1, 1))
        maps1 = maps1[order]
        maps2 = maps2[order]

        if VR:
            for i in range(len(maps1)):
                if bn.nansum(maps1[i]):
                    maps1[i] /= bn.nanmax(maps1[i])
                if bn.nansum(maps2[i]):
                    maps2[i] /= bn.nanmax(maps2[i])

        if maps3.shape != maps4.shape:
            maps3 = []
            maps4 = []
        else:
            # plotting novel vs. novel
            if mode =='and':
                alive = (bn.nansum(maps3, 1) > 0) & (bn.nansum(maps4, 1) > 0)
            elif mode =='or':
                alive = (bn.nansum(maps3, 1) > 0) | (bn.nansum(maps4, 1) > 0)
            elif mode == 'F':
                alive = (bn.nansum(maps3, 1) > 0)
            elif mode == 'all':
                alive = (bn.nansum(maps3, 1) != np.nan)

            maps3 = maps3[alive]
            maps4 = maps4[alive]

            maps3_masked = np.ma.masked_array(maps3, np.isnan(maps3))
            masscenters = np.argmax(maps3_masked, 1)
            masscenters[bn.nansum(maps3, 1) == 0] = -1.
            order = np.argsort(masscenters)
            #order = np.argsort(np.argmax(maps3, 1))
            maps3 = maps3[order]
            maps4 = maps4[order]
            if VR:
                for i in range(len(maps3)):
                    if bn.nansum(maps3[i]):
                        maps3[i] /= bn.nanmax(maps3[i])
                    if bn.nansum(maps4[i]):
                        maps4[i] /= bn.nanmax(maps4[i])

        if maps5.shape != maps6.shape:
            n_place_cells_F = bn.nansum(bn.nansum(maps5, 1) > 0)
            n_place_cells_N = bn.nansum(bn.nansum(maps6, 1) > 0)
            maps5 = []
            maps6 = []
        else:
            # plotting familiar vs. novel
            if mode == 'and':
                alive = (bn.nansum(maps5, 1) > 0) & (bn.nansum(maps6, 1) > 0)
            elif mode == 'or':
                alive = (bn.nansum(maps5, 1) > 0) | (bn.nansum(maps6, 1) > 0)
            elif mode == 'F':
                alive = (bn.nansum(maps5, 1) > 0)
            elif mode == 'all':
                alive = (bn.nansum(maps5, 1) != np.nan)

            n_place_cells_F = bn.nansum(bn.nansum(maps5, 1) > 0)
            n_place_cells_N = bn.nansum(bn.nansum(maps6, 1) > 0)

            maps6 = maps6[alive]
            maps5 = maps5[alive]

            masscenters = bn.nanargmax(maps5, 1)
            masscenters[bn.nansum(maps5, 1) == 0] = -1.
            order = np.argsort(masscenters)
            #order = np.argsort(np.argmax(maps5, 1))
            maps5 = maps5[order]
            maps6 = maps6[order]

            if VR:
                for i in range(len(maps5)):
                    if bn.nansum(maps5[i]):
                        maps5[i] /= bn.nanmax(maps5[i])
                    if bn.nansum(maps6[i]):
                        maps6[i] /= bn.nanmax(maps6[i])
        if not VR and verbose:
            print("len(maps1) "+str(len(maps1)))
            print("len(maps2) "+str(len(maps2)))
            print("len(maps3) "+str(len(maps3)))
            print("len(maps4) "+str(len(maps4)))
            print("len(maps5) "+str(len(maps5)))
            print("len(maps6) "+str(len(maps6)))
            print("min_cells "+str(min_cells))
            move.printA("maps1 ",maps1)
            move.printA("maps2 ",maps2)
            move.printA("maps3 ",maps3)
            move.printA("maps4 ",maps4)
            move.printA("maps5 ",maps5)
            move.printA("maps6 ",maps6)

        if len(maps1) >= min_cells and len(maps3) >= min_cells and len(maps5) >=min_cells and savepath != 'none':
            print("start plot heatmap")
            xt = np.asarray([0, int(len(maps6[0])/2.), len(maps6[0])])
            xl = ['0', '60', '120']

            f, axs = plt.subplots(2, 6, figsize=(12, 4.5), gridspec_kw = {'height_ratios' : [5, 1]})
            axs[0, 0].pcolor(plotfunc(maps1), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 0].set_yticks([0, maps1.shape[0]])
            if VR:
                axs[0, 0].set_title('familiar even')
            else:
                axs[0, 0].set_title(mapname[0])
            axs[0, 0].set_ylabel('cell index', fontsize=12)
            axs[0, 0].set_xticks(xt)
            axs[0, 0].set_xticklabels(xl)
            axs[0 ,0].set_xlabel('distance (cm)')

            axs[0, 1].pcolor(plotfunc(maps2), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 1].set_yticks([0, maps1.shape[0]])
            if VR:
                axs[0, 1].set_title('familiar odd')
            else:
                axs[0, 1].set_title(mapname[1])
            axs[0, 1].set_xticks(xt)
            axs[0, 1].set_xticklabels(xl)
            axs[0 ,1].set_xlabel('distance (cm)')

            axs[0, 2].pcolor(plotfunc(maps3), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 2].set_yticks([0, maps3.shape[0]])
            if VR:
                axs[0, 2].set_title('novel even')
            else:
                axs[0, 2].set_title(mapname[2])
            axs[0, 2].set_xticks(xt)
            axs[0, 2].set_xticklabels(xl)
            axs[0 ,2].set_xlabel('distance (cm)')

            axs[0, 3].pcolor(plotfunc(maps4), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 3].set_yticks([0, maps3.shape[0]])
            if VR:
                axs[0, 3].set_title('novel odd')
            else:
                axs[0, 3].set_title(mapname[3])
            axs[0, 3].set_xticks(xt)
            axs[0, 3].set_xticklabels(xl)
            axs[0 ,3].set_xlabel('distance (cm)')

            axs[0, 4].pcolor(plotfunc(maps5), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 4].set_yticks([0, maps5.shape[0]])
            if VR:
                axs[0, 4].set_title('familiar (all)')
            else:
                axs[0, 4].set_title(mapname[4])
            axs[0, 4].set_xticks(xt)
            axs[0, 4].set_xticklabels(xl)
            axs[0 ,4].set_xlabel('distance (cm)')

            axs[0, 5].pcolor(plotfunc(maps6), edgecolor='face', cmap=cmap, rasterized=True)
            axs[0, 5].set_yticks([0, maps5.shape[0]])
            if VR:
                axs[0, 5].set_title('novel (all)')
            else:
                axs[0, 5].set_title(mapname[5])
            axs[0, 5].set_xticks(xt)
            axs[0, 5].set_xticklabels(xl)
            axs[0 ,5].set_xlabel('distance (cm)')

            pov_FF = pov(maps1, maps2, VR=VR, nmin=nmin_corr)
            axs[1, 1].plot(np.arange(len(maps5[0])), pov_FF, color=pltcolors[0])
            axs[1, 1].set_ylabel('PoV corr')
            axs[1, 1].set_ylim([-0.5, 1.1])
            axs[1, 1].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 1].axhline([0], color='k')
            mean_PoV_FF = bn.nanmean(pov_FF)
            axs[1, 1].set_title('<PoV corr> = %.2f' % mean_PoV_FF)

            pov_NN = pov(maps3, maps4, VR=VR, nmin=nmin_corr)
            axs[1, 3].plot(np.arange(len(maps5[0])), pov_NN, color=pltcolors[0])
            axs[1, 3].set_ylabel('PoV corr')
            axs[1, 3].set_ylim([-0.5, 1.1])
            axs[1, 3].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 3].axhline([0], color='k')
            mean_PoV_NN = bn.nanmean(pov_NN)
            axs[1, 3].set_title('<PoV corr> = %.2f' % mean_PoV_NN)

            pov_FN = pov(maps5, maps6, VR=VR, nmin=nmin_corr)
            axs[1, 5].plot(np.arange(len(maps5[0])), pov_FN, color=pltcolors[0])
            axs[1, 5].set_ylabel('PoV corr')
            axs[1, 5].set_ylim([-0.5, 1.1])
            axs[1, 5].set_yticks([-0.5, 0, 0.5, 1])
            axs[1, 5].axhline([0], color='k')
            mean_PoV_FN = bn.nanmean(pov_FN)
            axs[1, 5].set_title('<PoV corr> = %.2f' % mean_PoV_FN)

            if savepath != 'none':
                print(savepath)
                plt.savefig(savepath+'.pdf')
                plt.close()

            pov_FF_cells = pov(maps1, maps2, percell=HB, VR=VR, nmin=nmin_corr)
            pov_NN_cells = pov(maps3, maps4, percell=HB, VR=VR, nmin=nmin_corr)
            pov_FN_cells = pov(maps5, maps6, percell=HB, VR=VR, nmin=nmin_corr)

            if not VR:
                print("len(pov_FF_cells) "+str(len(pov_FF_cells)))
                print("len(pov_NN_cells) "+str(len(pov_NN_cells)))
                print("len(pov_FN_cells) "+str(len(pov_FN_cells)))

                move.printA("pov_FF_cells ",pov_FF_cells)
                move.printA("pov_NN_cells ",pov_NN_cells)
                move.printA("pov_FN_cells ",pov_FN_cells)

            zero_pov_FF_cells = move.change_nan_zero(pov_FF_cells)
            zero_pov_NN_cells = move.change_nan_zero(pov_NN_cells)
            zero_pov_FN_cells = move.change_nan_zero(pov_FN_cells)

            if len(mapname[0])+len(mapname[1]) > 6:
                label1 = mapname[0]+"\n-"+mapname[1]
            else:
                label1 = mapname[0]+"-"+mapname[1]
            if len(mapname[2])+len(mapname[3]) > 6:
                label2 = mapname[2]+"\n-"+mapname[3]
            else:
                label2 = mapname[2]+"-"+mapname[3]
            if len(mapname[4])+len(mapname[5]) > 6:
                label3 = mapname[4]+"\n-"+mapname[5]
            else:
                label3 = mapname[4]+"-"+mapname[5]

            f, ax = box_comparison_three(zero_pov_FF_cells, zero_pov_NN_cells, zero_pov_FN_cells, label1, label2, label3, 'Spatial correlation (PCs)', force=True, swarm=True, box=False, violin=False, bar=True)
            ax.set_ylim([-0.3, 1.5])
            ax.axhline([0], color='k', linestyle='--')
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title('dec: %.2f  #pc IN: %u OUT: %u' % (0.5*(np.mean(pov_FF_cells)+np.mean(pov_NN_cells))-np.mean(pov_FN_cells), len(maps1), len(maps3)))
            if savepath != 'none':
                plt.savefig(savepath+'_cell_corr.pdf')
            plt.close()

            f, ax = plt.subplots(figsize=(4,3))
            if VR:
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FF, color=pltcolors[4], label='|||-|||', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_NN, color=pltcolors[5], label='///-///', linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FN, color=pltcolors[6], label='|||-///', linewidth=2.0)
            else:
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FF, color=pltcolors[4], label=label1, linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_NN, color=pltcolors[5], label=label3, linewidth=2.0)
                ax.plot(np.linspace(0, 1.2, len(pov_FF)), pov_FN, color=pltcolors[6], label=label2, linewidth=2.0)
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('PoV Correlation')
            ax.set_xticks([0, 0.6, 1.2])
            ax.axhline([0], color='k', linestyle='--')
            ax.legend()
            ax.set_ylim([-0.1, 0.7])
            ax.set_yticks([-0.1, 0, 0.2, 0.4, 0.6])
            if savepath != 'none':
                plt.savefig(savepath+'_PoV.pdf')
            plt.close()

            return pov_FF_cells, pov_NN_cells, pov_FN_cells, bn.nanmean(pov_FF_cells), bn.nanmean(pov_NN_cells), bn.nanmean(pov_FN_cells), n_place_cells_F, n_place_cells_N
        else:
            return np.nan, np.nan, np.nan, n_place_cells_F, n_place_cells_N
    else:
        return np.nan, np.nan, np.nan, 0,0 #n_place_cells_F, n_place_cells_N

def visualize_spatial_maps_neworder_two(
        maps1, maps2, maps3, maps4, min_cells=2, savepath='none', mode='or', HB=False, cmap='magma',
        plotfunc = lambda x: x, VR = True, mapname=["map1","map2","map3","map4"], nmin_corr=2):

    # plotting familiar vs. familiar
    if mode =='and':
        alive = (np.sum(maps1, 1) > 0) & (np.sum(maps2, 1) > 0)
    elif mode == 'or':
        alive = (np.sum(maps1, 1) > 0) | (np.sum(maps2, 1) > 0)
    elif mode == 'F':
        alive = (np.sum(maps1, 1) > 0)
    elif mode == 'all':
        alive = (bn.nansum(maps1, 1) != np.nan)

    if len(alive):
        maps1 = maps1[alive]
        maps2 = maps2[alive]

        masscenters = np.argmax(maps1, 1)
        masscenters[np.sum(maps1, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps1, 1))
        maps1 = maps1[order]
        maps2 = maps2[order]

        for i in range(len(maps1)):
            if np.sum(maps1[i]):
                maps1[i] /= np.max(maps1[i])
            if np.sum(maps2[i]):
                maps2[i] /= np.max(maps2[i])

        # plotting novel vs. novel
        if mode =='and':
            alive = (np.sum(maps3, 1) > 0) & (np.sum(maps4, 1) > 0)
        elif mode =='or':
            alive = (np.sum(maps3, 1) > 0) | (np.sum(maps4, 1) > 0)
        elif mode == 'F':
            alive = (np.sum(maps3, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(maps3, 1) != np.nan)

        maps3 = maps3[alive]
        maps4 = maps4[alive]

        masscenters = np.argmax(maps3, 1)
        masscenters[np.sum(maps3, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps3, 1))
        maps3 = maps3[order]
        maps4 = maps4[order]

        for i in range(len(maps3)):
            if np.sum(maps3[i]):
                maps3[i] /= np.max(maps3[i])
            if np.sum(maps4[i]):
                maps4[i] /= np.max(maps4[i])

        if not VR:
            print("len(maps1) "+str(len(maps1)))
            print("len(maps3) "+str(len(maps3)))
            print("min_cells "+str(min_cells))


        if len(maps1) >= min_cells and len(maps3) >= min_cells and savepath != 'none':

            pov_FF_cells = pov(maps1, maps2, percell=HB, VR=VR, nmin=nmin_corr)
            pov_NN_cells = pov(maps3, maps4, percell=HB, VR=VR, nmin=nmin_corr)

            if len(mapname[0])+len(mapname[1]) > 6:
                label1 = mapname[0]+"\n-"+mapname[1]
            else:
                label1 = mapname[0]+"-"+mapname[1]
            if len(mapname[2])+len(mapname[3]) > 6:
                label2 = mapname[2]+"\n-"+mapname[3]
            else:
                label2 = mapname[2]+"-"+mapname[3]

            f, ax = box_comparison_two(pov_FF_cells, pov_NN_cells, label1, label2, 'Spatial correlation (PCs)', force=True, swarm=True, box=False, violin=False, bar=True)
            ax.set_ylim([-0.3, 1.5])
            ax.axhline([0], color='k', linestyle='--')
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title('#pc IN: %u OUT: %u' % (len(maps1), len(maps3)))
            if savepath != 'none':
                plt.savefig(savepath+'_cell_corr.pdf')
            plt.close()

            return bn.nanmean(pov_FF_cells), bn.nanmean(pov_NN_cells)
        else:
            return np.nan, np.nan
    else:
        return np.nan, np.nan

def visualize_spatial_maps_neworder_four(maps1, maps2, maps3, maps4, maps5, maps6, maps7, maps8, min_cells=2, savepath='none', mode='or', HB=False, cmap='magma', plotfunc = lambda x: x, VR = True, mapname=["map1","map2","map3","map4","map5","map6","map7","map8"], nmin_corr=2):

    # plotting familiar vs. familiar
    if mode =='and':
        alive = (bn.nansum(maps1, 1) > 0) & (bn.nansum(maps2, 1) > 0)
    elif mode == 'or':
        alive = (bn.nansum(maps1, 1) > 0) | (bn.nansum(maps2, 1) > 0)
    elif mode == 'F':
        alive = (bn.nansum(maps1, 1) > 0)
    elif mode == 'all':
        alive = (bn.nansum(maps1, 1) != np.nan)

    if len(alive):
        maps1 = maps1[alive]
        maps2 = maps2[alive]

        masscenters = bn.nanargmax(maps1, 1)
        masscenters[bn.nansum(maps1, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps1, 1))
        maps1 = maps1[order]
        maps2 = maps2[order]

        if VR:
            for i in range(len(maps1)):
                if bn.nansum(maps1[i]):
                    maps1[i] /= bn.nanmax(maps1[i])
                if bn.nansum(maps2[i]):
                    maps2[i] /= bn.nanmax(maps2[i])

        # plotting novel vs. novel
        if mode =='and':
            alive = (bn.nansum(maps3, 1) > 0) & (bn.nansum(maps4, 1) > 0)
        elif mode =='or':
            alive = (bn.nansum(maps3, 1) > 0) | (bn.nansum(maps4, 1) > 0)
        elif mode == 'F':
            alive = (bn.nansum(maps3, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(maps3, 1) != np.nan)

        maps3 = maps3[alive]
        maps4 = maps4[alive]

        if len(maps3[~np.isnan(maps3)]) > 0:
            maps3_masked = np.ma.masked_array(maps3, np.isnan(maps3))
            masscenters = np.argmax(maps3_masked, 1)
            masscenters[bn.nansum(maps3, 1) == 0] = -1.
            order = np.argsort(masscenters)
            #order = np.argsort(np.argmax(maps3, 1))
            maps3 = maps3[order]
            maps4 = maps4[order]

        if VR:
            for i in range(len(maps3)):
                if bn.nansum(maps3[i]):
                    maps3[i] /= bn.nanmax(maps3[i])
                if bn.nansum(maps4[i]):
                    maps4[i] /= bn.nanmax(maps4[i])

        # plotting familiar vs. novel
        if mode == 'and':
            alive = (bn.nansum(maps5, 1) > 0) & (bn.nansum(maps6, 1) > 0)
        elif mode == 'or':
            alive = (bn.nansum(maps5, 1) > 0) | (bn.nansum(maps6, 1) > 0)
        elif mode == 'F':
            alive = (bn.nansum(maps5, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(maps5, 1) != np.nan)

        maps6 = maps6[alive]
        maps5 = maps5[alive]

        masscenters = bn.nanargmax(maps5, 1)
        masscenters[bn.nansum(maps5, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps5, 1))
        maps5 = maps5[order]
        maps6 = maps6[order]

        if VR:
            for i in range(len(maps5)):
                if bn.nansum(maps5[i]):
                    maps5[i] /= bn.nanmax(maps5[i])
                if bn.nansum(maps6[i]):
                    maps6[i] /= bn.nanmax(maps6[i])

        # plotting Flip
        if mode == 'and':
            alive = (bn.nansum(maps7, 1) > 0) & (bn.nansum(maps8, 1) > 0)
        elif mode == 'or':
            alive = (bn.nansum(maps7, 1) > 0) | (bn.nansum(maps8, 1) > 0)
        elif mode == 'F':
            alive = (bn.nansum(maps7, 1) > 0)
        elif mode == 'all':
            alive = (bn.nansum(maps7, 1) != np.nan)

        maps8 = maps8[alive]
        maps7 = maps7[alive]

        masscenters = bn.nanargmax(maps7, 1)
        masscenters[bn.nansum(maps7, 1) == 0] = -1.
        order = np.argsort(masscenters)
        #order = np.argsort(np.argmax(maps5, 1))
        maps7 = maps7[order]
        maps8 = maps8[order]

        if VR:
            for i in range(len(maps7)):
                if bn.nansum(maps7[i]):
                    maps7[i] /= bn.nanmax(maps7[i])
                if bn.nansum(maps6[i]):
                    maps8[i] /= bn.nanmax(maps8[i])



        if not VR:
            print("len(maps1) "+str(len(maps1)))
            print("len(maps2) "+str(len(maps2)))
            print("len(maps3) "+str(len(maps3)))
            print("len(maps4) "+str(len(maps4)))
            print("len(maps5) "+str(len(maps5)))
            print("len(maps6) "+str(len(maps6)))
            print("len(maps7) "+str(len(maps7)))
            print("len(maps8) "+str(len(maps8)))
            print("min_cells "+str(min_cells))
            #move.printA("maps1 ",maps1)
            #move.printA("maps2 ",maps2)
            #move.printA("maps3 ",maps3)
            #move.printA("maps4 ",maps4)
            #move.printA("maps5 ",maps5)
            #move.printA("maps6 ",maps6)
            #move.printA("maps7 ",maps7)
            #move.printA("maps8 ",maps8)


        if len(maps1) >= min_cells and len(maps3) >= min_cells and len(maps5) >= min_cells and len(maps7) >= min_cells and savepath != 'none':

            pov_FF_cells = pov(maps1, maps2, percell=HB, VR=VR, nmin=nmin_corr)
            pov_NN_cells = pov(maps3, maps4, percell=HB, VR=VR, nmin=nmin_corr)
            pov_FN_cells = pov(maps5, maps6, percell=HB, VR=VR, nmin=nmin_corr)
            pov_Flip_cells = pov(maps7, maps8, percell=HB, VR=VR, nmin=nmin_corr)

            zero_pov_FF_cells = move.change_nan_zero(pov_FF_cells)
            zero_pov_NN_cells = move.change_nan_zero(pov_NN_cells)
            zero_pov_FN_cells = move.change_nan_zero(pov_FN_cells)
            zero_pov_Flip_cells = move.change_nan_zero(pov_Flip_cells)

            if len(mapname[0])+len(mapname[1]) > 6:
                label1 = mapname[0]+"\n-"+mapname[1]
            else:
                label1 = mapname[0]+"-"+mapname[1]
            if len(mapname[2])+len(mapname[3]) > 6:
                label2 = mapname[2]+"\n-"+mapname[3]
            else:
                label2 = mapname[2]+"-"+mapname[3]
            if len(mapname[4])+len(mapname[5]) > 6:
                label3 = mapname[4]+"\n-"+mapname[5]
            else:
                label3 = mapname[4]+"-"+mapname[5]
            if len(mapname[6])+len(mapname[7]) > 6:
                label4 = mapname[6]+"\n-"+mapname[7]
            else:
                label4 = mapname[6]+"-"+mapname[7]

            f, ax = box_comparison_four(zero_pov_FF_cells, zero_pov_NN_cells, zero_pov_FN_cells, zero_pov_Flip_cells, label1, label2, label3, label4, 'Spatial correlation (PCs)', force=True, swarm=True, box=False, violin=False, bar=True)
            ax.set_ylim([-0.3, 1.5])
            ax.axhline([0], color='k', linestyle='--')
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_title('#pc IN: %u OUT: %u' % (len(maps1), len(maps3)))
            if savepath != 'none':
                plt.savefig(savepath+'_cell_corr.pdf')
            plt.close()

            return pov_FF_cells, pov_NN_cells, pov_FN_cells, pov_Flip_cells, bn.nanmean(pov_FF_cells), bn.nanmean(pov_NN_cells), bn.nanmean(pov_FN_cells), bn.nanmean(pov_Flip_cells)
        else:
            return np.nan, np.nan, np.nan, np.nan
    else:
        return np.nan, np.nan, np.nan, np.nan

def change_nan_zero(spmaps):
    new_spmaps = np.zeros(len(spmaps))
    new_spmaps[np.isfinite(spmaps)] = spmaps[np.isfinite(spmaps)]
    return new_spmaps
