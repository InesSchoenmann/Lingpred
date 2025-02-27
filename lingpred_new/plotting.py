import sys
import os 
from pathlib import Path
import pickle 
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
import scipy 
# we do a cluster-based test to test if higher than zero 
from functools import partial
from mne.stats import permutation_cluster_1samp_test, ttest_1samp_no_p
from os import listdir
from os.path import isfile, join

sns.set_theme()
mpl.rcParams['pdf.fonttype'] = 42
sns.set_context("paper")

# time points in seconds:
times_100       = np.arange(-1.950, 1.951, .025)
pre_onset_times = times_100[:78]

# colour maps and colour dictionaries
cmap       = sns.cubehelix_palette(start=0.1, rot=-.75,  n_colors=11)
cmap_glove = sns.cubehelix_palette(start=0.5, rot=-.5,  n_colors=11)
colours    = {'Glove': cmap_glove[4],
              'GloVe': cmap_glove[4],
              'GPT': cmap[4],
              'arbitrary': cmap[8],
              'Predicting MEG Data (Acoustics)':cmap_glove[1],
              'Predicting Residual MEG Data (Acoustics)':cmap[7],
              'Top 1': 'teal',
              'Top 3': 'teal',
              'Top 5': 'teal',
              'Top 10': 'darkgoldenrod',
              'Not Predicted': 'crimson'}

ylim_dict = {1: (-0.01, 0.14),
            2: (-0.01, 0.1),
            3: ((-0.01, 0.18))}

acoustic_ylim_dict = {1: (-0.045, 0.14),
                        2: (-0.01, 0.1),
                        3: ((-0.01, 0.18))}


# -------------------------------------------------- #
#                          UTILS                     #
# -------------------------------------------------- #

def sem(x):
    '''
    computes standard error of the mean
    '''
    return np.std(x, axis=0)/np.sqrt(len(x))
    
def lowerCI(x):
    '''
    computes lower confidence interval based on alpha=5% and t(df>30)=1.96
    '''
    t = scipy.stats.t.ppf(q=0.972,df=x.shape[0]-1) # get critical t value for a 95% CI
    return np.mean(x, axis=0) - t * sem(x)

def upperCI(x):
    '''
    computes upper confidence interval based on alpha=5% and t(df>30)=1.96
    '''
    t = scipy.stats.t.ppf(q=0.972,df=x.shape[0]-1) # get critical t value for a 95% CI
    return np.mean(x, axis=0) + 1.96 * sem(x) 

def reshape(x):
    '''
    Reshapes array by collapsing one dimension
    x.shape = (30, 10, 157) --> (300, 157)
    '''
    return x.reshape(-1, x.shape[-1])

def get_signal_mask(subject=1, model='Glove', dataset='Armani'):
    '''
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    if dataset =='Armani':
        directory = '/project/3018059.03/Lingpred/data/Armani/sub-00{}/'.format(subject)
        filename  = directory + 'masks_sources_signal_Glove_GPT.pkl'
        mask      = pickle.load(open(filename, 'rb'))
        mask      = mask['signal_mask_{}'.format(model)]
    
    if dataset == 'Gwilliams':
        directory = '/project/3018059.03/Lingpred/data/Gwilliams/'
        filename  = directory + 'signal_masks_all_subjects_Glove_based.pkl'
        mask      = pickle.load(open(filename, 'rb'))
        
    return mask

def get_significant_nonzero_timepoints(corr_dict:dict, subject:int, percentile=95):
    '''
    Parameters
    ----------
    
    Returns
    -------
    
    '''
    significant_TPs_dict = {}
    
    for model, corr in corr_dict.items():
        
        T_obs, cl, cl_p_vals, H0 = clu = permutation_cluster_1samp_test(reshape(corr[get_signal_mask(subject)]), 
                                                                        adjacency=False, 
                                                                        n_jobs=1,
                                                                        stat_fun=partial(ttest_1samp_no_p),
                                                                        threshold=dict(start=0,step=0.2),
                                                                        verbose=True,
                                                                        n_permutations=10000,
                                                                        tail=1)

        significant_Tps = [nr for nr, i in zip(times_100, T_obs) if i > np.percentile(H0, percentile)]
        
        significant_TPs_dict[model] = significant_Tps
        
    return significant_TPs_dict


def get_significant_different_timepoints(corr_dict:dict, subject:int, percentile=95, return_all=False):
    '''
    Parameters
    ----------
    - return_all: bool
        return p-values instead of significant time points
    Returns
    -------
    
    '''
    
    keys   = [k for k in corr_dict.keys()]
    corr_1 = corr_dict[keys[0]]
    corr_2 = corr_dict[keys[1]]
    
    # we are interested in the H1: "Is Top-1 larger than Not-Top-1", but only pre-onset:
    difference = reshape(corr_1[get_signal_mask(subject)]) - reshape(corr_2[get_signal_mask(subject)])
    difference = difference[:,:79]     
                                                                    
    T_obs, cl, cl_p_vals, H0 = clu = permutation_cluster_1samp_test(difference, 
                                                                    adjacency=False, 
                                                                    n_jobs=1,
                                                                    stat_fun=partial(ttest_1samp_no_p),
                                                                    threshold=dict(start=0,step=0.2),
                                                                    verbose=True,
                                                                    n_permutations=10000,
                                                                    tail=1)

    significant_Tps = [nr for nr, i in zip(times_100[:79], T_obs) if i > np.percentile(H0, percentile)]

    if return_all:
        return T_obs, cl, cl_p_vals, H0
    else:   
        return significant_Tps



def plot_base_effect(subject:int, models=['GPT', 'Glove', 'arbitrary'], dataset='Armani', legend=True, 
                     use_regressed_out=False, use_residualised_neural_data=False, plot_acoustic=False):
    '''
    Creates a plot for a single subject, showing all models in the list
    
    Parameters:
    -----------
    - subject: integer
        subject for whom to create the plot, for the Armeni dataset it's 1, 2, or 3
    - models: array-like
        list of models to plot. For the base effects this can be ['GPT', 'Glove', 'arbitrary']
    -legend: boolean
        whether to plot a legend for this plot
    -use_regressed_out: boolean
        whether to plot the effect computed from the regressed out vectors
    -use_residualised_neural_data: boolean
        whether to plot the effect computed from the regressed out vectors and neural data with acoustics removed
        
    Returns:
    --------
    shows plot
    '''
    
    # get the "brainscores" for all models in the list
    corr_dict = {}

    if plot_acoustic:
        directory = '/project/3018059.03/Lingpred/results/{}/after_regressing_out_acoustics/'.format(dataset)

        path = directory + 'corr_acoustic_8_mel_vectors_sub_{}.pkl'.format(subject)
        corr_dict['Predicting MEG Data (Acoustics)'] = pickle.load(open(path, 'rb'))
        
        path = directory + 'corr_acoustic_8_mel_vectors_residualised_neural_data_sub_{}.pkl'.format(subject)
        corr_dict['Predicting Residual MEG Data (Acoustics)'] = pickle.load(open(path, 'rb'))
    else:
        for model in models:
            directory = '/project/3018059.03/Lingpred/results/{}/grand_average/'.format(dataset)
            path      = directory + 'corr_{}_vectors_sub_{}.pkl'.format(model, subject)

            if use_regressed_out:
                path = directory + 'corr_regressed_out_one_{}_vectors_sub_{}.pkl'.format(model, subject)

            if use_residualised_neural_data:
                directory = '/project/3018059.03/Lingpred/results/{}/after_regressing_out_acoustics/'.format(dataset)
                path = directory + 'corr_regressed_out_one_{}_vectors_sub_{}.pkl'.format(model, subject)
            
            corr_dict[model] = pickle.load(open(path, 'rb'))
    
    # get their significant timepoints (non-zero): 
    significant_TPs_dict = get_significant_nonzero_timepoints(corr_dict, subject)
    
    if use_regressed_out:
        labels = dict(zip(models, ['Residualised '+m for m in models]))
    else: 
        labels = dict(zip(models, models))

    # plot:
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    offset  = -0.009
    
    for i, (model, corr) in enumerate(corr_dict.items()):
        ax.plot(times_100, reshape(corr[get_signal_mask(subject)]).mean(axis=0), c=colours[model], label=labels[model])
        ax.fill_between(times_100, lowerCI(reshape(corr[get_signal_mask(subject)])), 
                                   upperCI(reshape(corr[get_signal_mask(subject)])), color=colours[model], alpha=0.3)
        
        ax.scatter(significant_TPs_dict[model], 
                   np.repeat(offset, len(significant_TPs_dict[model])), 
                   marker=(5, 2), s=1, color=colours[model])
        offset += 0.003
        
    ax.set_xlabel('Time in Seconds')
    ax.set_ylabel('Crossvalidated Correlation')
    ax.set_ylim(ylim_dict[subject])
    if plot_acoustic:
        ax.set_ylim(acoustic_ylim_dict[subject])
    ax.axvspan(-.050, .050, color='gray',  alpha=0.3)
    ax.axhline(c='indianred',  alpha=0.3)
    if legend:
        ax.legend()

    if subject==1:
        fig_folder = 'figures/main/'
    else:
        fig_folder = 'figures/supplementary/'
    fig_path = fig_folder+'base_effect-subject_{}-residualised_{}.pdf'.format(subject, use_regressed_out)

    if not (use_residualised_neural_data or plot_acoustic): # those are not part of any figures, neither main nor supplementary
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        

def plot_prediction_split(subject:int, prediction='Top 3', not_prediction= 'Top 3', 
                          use_splitting_during_testing = False, 
                          use_regressed_out_Glove= False, 
                          dataset='Armani', legend=True):
    '''
    Creates a plot for a single subject, showing all models in the list
    
    Parameters:
    -----------
    - subject: integer
        subject for whom to create the plot, for the Armeni dataset it's 1, 2, or 3
    - prediction: string
        Which type of prediction to plot: Top 1, Top 3, Top 5 or Top 10 
    - not_prediction: string
        Which type of prediction complement to plot: Not (Top1, Top3, Top 5, Top 10) 
    - use_splitting_during_testing: boolean
        whether to plot the data that was split only during the testing split of the 10-fold CV
    - use_regressed_out_Glove: boolean
        whether to plot prediction split with the residualised GloVe vectors
    - dataset: string
        Which dataset to plot 'Armani' or 'Gwilliams'
    -legend: boolean
        whether to plot a legend for this plot
        
    Returns:
    --------
    shows plot
    '''
    
    # get the "brainscores" for all models in the list
    corr_dict = {}
    
    # models is a dictionary with the official name and the key for the corr_dict:
    prediction_type = {'Top 1': 'corr_top_1',
                       'Top 3': 'corr_top_3',
                       'Top 5': 'corr_top_5',
                       'Top 10': 'corr_top_10'}
    
    unprediction_type = {'Top 1': 'corr_not_top_1',
                         'Top 3': 'corr_not_top_3',
                         'Top 5': 'corr_not_top_5',
                        'Top 10': 'corr_not_top_10'}
        
    # load the dictionary with the correlations:
    directory      = '/project/3018059.03/Lingpred/results/{}/correct_incorrect_predictions/'.format(dataset)
    #path           = directory + 'prediction_split_sub_{}_Glove.pkl'.format(subject)
    #corr_dict_1_10 = pickle.load(open(path, 'rb'))
    #path           = directory + 'prediction_split_top_3_and_5_sub_{}_Glove.pkl'.format(subject)
    #corr_dict_3_5  = pickle.load(open(path, 'rb'))
    
    if prediction=='Top 3':
        path = directory + 'prediction_split_top_3_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 5':
        path = directory + 'prediction_split_top_5_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 1':
        path = directory + 'prediction_split_top_1_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 1':
        corr_dict_1_10 = pickle.load(open(path, 'rb'))
    else:
        corr_dict_3_5 = pickle.load(open(path, 'rb'))  

    if use_splitting_during_testing:
        # load the testing split
        path = directory + 'prediction_split_split_in_testing_set_top_3_sub_{}_Glove.pkl'.format(subject)
        corr_dict_3_testing = pickle.load(open(path, 'rb'))  
        path = directory + 'prediction_split_split_in_testing_set_top_1_sub_{}_Glove.pkl'.format(subject)
        corr_dict_1_testing = pickle.load(open(path, 'rb'))  
        path = directory + 'prediction_split_split_in_testing_set_top_10_sub_{}_Glove.pkl'.format(subject)
        corr_dict_10_testing = pickle.load(open(path, 'rb'))  

    if use_regressed_out_Glove:
        if prediction=='Top 3':
            path = directory + 'prediction_split_top_3_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 5':
            path = directory + 'prediction_split_top_5_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 1':
            path = directory + 'prediction_split_top_1_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 1':
            corr_dict_1_10 = pickle.load(open(path, 'rb'))
        else:
            corr_dict_3_5 = pickle.load(open(path, 'rb'))  


    # plot:
    # -----
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    offset  = -0.005
    
    if prediction in ['Top 5', 'Top 3']:
        corr_dict[prediction] = corr_dict_3_5[prediction_type[prediction]]
        if use_splitting_during_testing:
            corr_dict[prediction] = corr_dict_3_testing[prediction_type[prediction]]
    else:
        corr_dict[prediction] = corr_dict_1_10[prediction_type[prediction]]
        if use_splitting_during_testing and prediction == 'Top 1':
            corr_dict[prediction] = corr_dict_1_testing[prediction_type[prediction]]
        if use_splitting_during_testing and prediction == 'Top 10':
            corr_dict[prediction] = corr_dict_10_testing[prediction_type[prediction]]
    
    if not_prediction in ['Top 5', 'Top 3']:
        corr_dict['Not Predicted'] = corr_dict_3_5[unprediction_type[not_prediction]]
        if use_splitting_during_testing and not_prediction=='Top 3':
            corr_dict['Not Predicted'] = corr_dict_3_testing[unprediction_type[not_prediction]]
    else:
        corr_dict['Not Predicted'] = corr_dict_1_10[unprediction_type[not_prediction]]
        if use_splitting_during_testing and not_prediction=='Top 1':
            corr_dict['Not Predicted'] = corr_dict_1_testing[unprediction_type[not_prediction]]
        if use_splitting_during_testing and not_prediction=='Top 10':
            corr_dict['Not Predicted'] = corr_dict_10_testing[unprediction_type[not_prediction]]
    
    # get the significant pre-onset timepoints, were top-1 is larger than not-top-1
    significant_TPs = get_significant_different_timepoints(corr_dict, subject, )
       
    
    if use_regressed_out_Glove:
        labels = dict(zip(corr_dict.keys(), [m+' (Residualised)' for m in corr_dict.keys()]))
    else: 
        labels = dict(zip(corr_dict.keys(), [m for m in corr_dict.keys()]))

    for model, corr in corr_dict.items():
        
        ax.plot(times_100, reshape(corr[get_signal_mask(subject)]).mean(axis=0), c=colours[model], label=labels[model])
        ax.fill_between(times_100, lowerCI(reshape(corr[get_signal_mask(subject)])), 
                                   upperCI(reshape(corr[get_signal_mask(subject)])), color=colours[model], alpha=0.3)
    ax.scatter(significant_TPs, np.repeat(offset, len(significant_TPs)), marker=(5, 2), s=1, color='darkslategrey')
        
        
    ax.set_xlabel('Time in Seconds')
    ax.set_ylabel('Crossvalidated Correlation')
    
    ax.set_ylim(ylim_dict[subject])
    ax.axvspan(-.050, .050, color='gray',  alpha=0.3)
    ax.axhline(c='indianred',  alpha=0.3)
    if legend:
        ax.legend()

    if subject==1:
        fig_folder = 'figures/main/'
    else:
        fig_folder = 'figures/supplementary/'

    fig_path = fig_folder+'prediction_split-subject_{}-residualised_Glove_{}-{}.pdf'.format(subject, 
                                                                                            use_regressed_out_Glove, 
                                                                                            prediction)
    if not use_splitting_during_testing or use_uneven_trials:
        plt.savefig(fig_path, format='pdf', bbox_inches='tight')


def plot_predicting_acoustics(dataset='Armani', 
                              plot_split=False, 
                              top=1, 
                              vectors='Glove', 
                              countinuous_spectrogram = False,
                              use_regressed_out=True):

    if countinuous_spectrogram:
        type_of_acoustic_y_matrix = 'neural_data'
        vectors = '_'+vectors
    else:
        type_of_acoustic_y_matrix = 'selfpred'
        if vectors == 'Glove':
            vectors =''
        else:
            vectors = '_'+vectors

    # load results:
    results_dir      = '/project/3018059.03/Lingpred/results/{}/Predicting_acoustics/'.format(dataset)
    results_file_name= 'regressed_out_vectors_GPT_Glove_arbitrary_y_matrix_like_{}'.format(type_of_acoustic_y_matrix)
    if not use_regressed_out:
        results_file_name= 'original_vectors_GPT_Glove_arbitrary_y_matrix_like_{}'.format(type_of_acoustic_y_matrix)
    if plot_split:
        results_file_name= 'top_{}_split_regressed_out_vectors_y_matrix_like_{}{}'.format(top, 
                                                                                          type_of_acoustic_y_matrix, 
                                                                                          vectors)
        if not use_regressed_out:
            results_file_name= 'top_{}_split_original_vectors_y_matrix_like_{}{}'.format(top, 
                                                                                         type_of_acoustic_y_matrix, 
                                                                                         vectors)
    results_file_path= results_dir + results_file_name
    results = pickle.load(open(results_file_path, 'rb'))

    # get model names
    models = [k.split(sep='_')[1] for k in results.keys()]
    
    if plot_split and top==5:
        print(results.keys())
        models = ['Top 5', 'Not Predicted']

    if plot_split and top==1:
        print(results.keys())
        models = ['Top 1', 'Not Predicted']

    if plot_split:
        if not use_regressed_out:
            labels = models
        if use_regressed_out:
            labels = [m +' (Residualised)' for m in models]
    else:
        if  use_regressed_out:
            labels = ['Residualised '+m for m in models]
        if not use_regressed_out:
            labels = models
    
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))

    for i, key in enumerate(results.keys()):
        ax1.plot(times_100[:78], results[key].mean(axis=0).mean(axis=0)[:78], c=colours[models[i]], label=labels[i])   
        ax1.fill_between(times_100[:78], lowerCI(reshape(results[key][:, :, :78])), 
                                upperCI(reshape(results[key][:, :, :78])), color=colours[models[i]], alpha=0.3)

    ax1.legend()
    #ax1.set_ylim([-0.005, 0.05])
    ax1.set_xlabel('Time in Seconds', fontsize=12)
    ax1.set_ylabel('Cross-validated Correlation', fontsize=12)
    ax1.axhline(c='indianred',  alpha=0.3)
    #ax1.set_title(dataset+':' + vectors)

    if top==1 and dataset=='Armani':
        fig_folder = 'figures/main/'
        if not use_regressed_out:
            fig_folder = 'figures/supplementary/'       
    else:
        fig_folder = 'figures/supplementary/'

    fig_path = fig_folder+'Acoustics_{}_with_residualised_vectors_{}-prediction_split_{}-top_{}.pdf'.format(dataset,
                                                                                                  use_regressed_out,
                                                                                                  plot_split, 
                                                                                                  top)
    plt.savefig(fig_path, format='pdf',bbox_inches='tight')


def get_and_threshold_Gwilliams_base_results(use_regressed_out=False):

    # set the type of results you want to plot:
    if use_regressed_out:
        res_type = 'Grand-Avg/regressed_out/'
    else:
        res_type = 'Grand-Avg/'

    # get file names:
    res_dir    = '/project/3018059.03/Lingpred/results/Gwilliams/'
    path       = res_dir + res_type
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    # initialise new correlation arrays for all participants
    n_subjects = len(file_names)

    if res_type == 'Grand-Avg/regressed_out/':
        corr_GPT       = np.empty(shape=(n_subjects, 208, 10, 157))
        corr_Glove     = np.empty(shape=(n_subjects, 208, 10, 157))
        corr_arbitrary = np.empty(shape=(n_subjects, 208, 10, 157))
    elif res_type == 'Grand-Avg/':
        corr_GPT       = np.empty(shape=(n_subjects, 208, 157))
        corr_Glove     = np.empty(shape=(n_subjects, 208, 157))
        corr_arbitrary = np.empty(shape=(n_subjects, 208, 157))

    # fill the arrays with the results for each subject
    for i, file in enumerate(file_names):
        results           = pickle.load(open(path+file, 'rb'))
        corr_GPT[i]       = results['corr_GPT']
        corr_Glove[i]     = results['corr_Glove']
        corr_arbitrary[i] = results['corr_arbitrary']
    
    # when folds have been saved, average over folds first: 
    if len(corr_GPT.shape) == 4:
        corr_GPT       = corr_GPT.mean(axis=2)
        corr_Glove     = corr_Glove.mean(axis=2)
        corr_arbitrary = corr_arbitrary.mean(axis=2)
        print(corr_Glove.shape)
        
    corr_dict = dict(zip(['corr_GPT', 'corr_Glove', 'corr_arbitrary'], [corr_GPT, corr_Glove, corr_arbitrary]))
    
    # get signal masks for each subject:
    masks = get_signal_mask(dataset='Gwilliams')

    
    for key, corr in corr_dict.items():
        #initialise empty array
        corr_thresholded = np.empty(shape=(corr.shape[0], 157))

        # threshold per participant based on Glove
        subjects_without_encoding = []
        for i in range(corr.shape[0]):
            temp = corr[i][masks[i]]
            # if there is no encoding save i and continue
            if len(temp)==0 : 
                print('participant {} has no channels > the threshold'.format(i))
                subjects_without_encoding.append(i)
                continue
            corr_thresholded[i] = temp.mean(axis=0)
        # drop participants without encoding:
        if len(subjects_without_encoding)>0: 
            corr_thresholded = np.delete(corr_thresholded, subjects_without_encoding, axis=0)
        corr_dict[key] = corr_thresholded

    return corr_dict


def get_and_threshold_Gwilliams_prediction_split(use_regressed_out=False):

    # set the type of results you want to plot:
    if use_regressed_out:
        res_type = 'correct_incorrect_predictions/regressed_out/'
    else:
        res_type = 'correct_incorrect_predictions/'

    # get file names:
    res_dir    = '/project/3018059.03/Lingpred/results/Gwilliams/'
    path       = res_dir + res_type
    file_names = [f for f in listdir(path) if isfile(join(path, f))]

    # initialise new correlation arrays for all participants
    n_subjects = len(file_names)

    corr_top_1     = np.empty(shape=(n_subjects, 208, 157))
    corr_not_top_1 = np.empty(shape=(n_subjects, 208, 157))
    corr_top_5     = np.empty(shape=(n_subjects, 208, 157))
    corr_not_top_5 = np.empty(shape=(n_subjects, 208, 157))

    # fill the arrays with the results for each subject
    for i, file in enumerate(file_names):
        results           = pickle.load(open(path+file, 'rb'))
        corr_top_1[i]     = results['corr_top_1'].mean(axis=1)
        corr_not_top_1[i] = results['corr_not_top_1'].mean(axis=1)
        corr_top_5[i]     = results['corr_top_5'].mean(axis=1)
        corr_not_top_5[i] = results['corr_not_top_5'].mean(axis=1)
        
    corr_dict = dict(zip(['corr_top_1', 'corr_not_top_1', 'corr_top_5', 'corr_not_top_5'], 
                         [corr_top_1, corr_not_top_1, corr_top_5, corr_not_top_5]))
    
    # get signal masks for each subject:
    masks = get_signal_mask(dataset='Gwilliams')
    
    for key, corr in corr_dict.items():
        #initialise empty array
        corr_thresholded = np.empty(shape=(corr.shape[0], 157))

        # threshold per participant based on Glove
        subjects_without_encoding = []
        for i in range(corr.shape[0]):
            temp = corr[i][masks[i]]
            # if there is no encoding save i and continue
            if len(temp)==0 : 
                print('participant {} has no channels > the threshold'.format(i))
                subjects_without_encoding.append(i)
                continue
            corr_thresholded[i] = temp.mean(axis=0)
        # drop participants without encoding:
        if len(subjects_without_encoding)>0: 
            corr_thresholded = np.delete(corr_thresholded, subjects_without_encoding, axis=0)
        corr_dict[key] = corr_thresholded

    return corr_dict


def plot_base_effect_Gwilliams(use_regressed_out=False):
        
    corr_dict = get_and_threshold_Gwilliams_base_results(use_regressed_out=use_regressed_out)
    
    models = [k.split(sep='_')[1] for k in corr_dict.keys()]
    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))

    if use_regressed_out:
        labels = dict(zip(models, ['Residualised '+m for m in models]))
    else: 
        labels = dict(zip(models, [m for m in models]))
    
    for model, key in zip(models, corr_dict.keys()):
        ax1.plot(times_100, corr_dict[key].mean(axis=0), c=colours[model], label=labels[model])   
        ax1.fill_between(times_100, lowerCI(corr_dict[key]), 
                                upperCI(corr_dict[key]), color=colours[model], alpha=0.3)

    ax1.legend()
    ax1.set_ylim([-0.01, 0.06])
    ax1.set_xlabel('Time in Seconds', fontsize=12)
    ax1.set_ylabel('Cross-validated Correlation', fontsize=12)
    ax1.axhline(c='indianred',  alpha=0.3)

    fig_folder = 'figures/supplementary/'
    fig_path   = fig_folder+'base_effect-Gwilliams_data-residualised_{}.pdf'.format(use_regressed_out)

    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.show()
        
        
def plot_prediction_split_Gwilliams(use_regressed_out=False, top=5):
        
    corr_dict = get_and_threshold_Gwilliams_prediction_split(use_regressed_out=use_regressed_out)
    
    if top==5:
        models = ['Top 5', 'Not Predicted']
        corrs = dict(zip(models, [corr_dict['corr_top_5'], corr_dict['corr_not_top_5']]))
    
    if top==1:
        models = ['Top 1', 'Not Predicted']
        corrs = dict(zip(models, [corr_dict['corr_top_1'], corr_dict['corr_not_top_1']]))
    

    if use_regressed_out:
        labels = dict(zip(models, [m+' (Residualised)' for m in models]))
    else: 
        labels = dict(zip(models, [m for m in models]))

    fig, (ax1) = plt.subplots(1, 1, figsize=(4, 4))

    for model, corr in corrs.items():
        ax1.plot(times_100, corr.mean(axis=0), c=colours[model], label=labels[model])   
        ax1.fill_between(times_100, lowerCI(corr), 
                                upperCI(corr), color=colours[model], alpha=0.3)

    ax1.legend()
    ax1.set_ylim([-0.01, 0.06])
    ax1.set_xlabel('Time in Seconds', fontsize=12)
    ax1.set_ylabel('Cross-validated Correlation', fontsize=12)
    ax1.axhline(c='indianred',  alpha=0.3)

    fig_folder = 'figures/supplementary/'
    fig_path = fig_folder+'prediction_split-Gwilliams_data-residualised_{}-Top_{}.pdf'.format(use_regressed_out, 
                                                                                              top)

    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.show()
    

def get_quantifications_prediction_split(subject:int, prediction='Top 1', 
                                        use_regressed_out_Glove= False, 
                                        dataset='Armani'):
    '''
    Prints quanitfications of the difference between predicted and unpredicted words' encodings
    
    Parameters:
    -----------
    - subject: integer
        subject for whom to create the plot, for the Armeni dataset it's 1, 2, or 3
    - prediction: string
        Which type of prediction to plot: Top 1, Top 3, Top 5 or Top 10 
    - use_regressed_out_Glove: boolean
        whether to plot prediction split with the residualised GloVe vectors
    - dataset: string
        Which dataset to plot 'Armani' or 'Gwilliams'
        
    Returns:
    --------
    prints stats on the largest cluster before onset of predicted > not_predicted
    and the largest significant p value of those clusters
    '''

    # we always want the same not-predictionas prediction type
    not_prediction= prediction

    # get the "brainscores" for all models in the list
    corr_dict = {}
    
    # models is a dictionary with the official name and the key for the corr_dict:
    prediction_type = {'Top 1': 'corr_top_1',
                       'Top 3': 'corr_top_3',
                       'Top 5': 'corr_top_5',
                       'Top 10': 'corr_top_10'}
    
    unprediction_type = {'Top 1': 'corr_not_top_1',
                         'Top 3': 'corr_not_top_3',
                         'Top 5': 'corr_not_top_5',
                        'Top 10': 'corr_not_top_10'}
        
    # load the dictionary with the correlations:
    directory      = '/project/3018059.03/Lingpred/results/{}/correct_incorrect_predictions/'.format(dataset)
    
    if prediction=='Top 3':
        path = directory + 'prediction_split_top_3_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 5':
        path = directory + 'prediction_split_top_5_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 1':
        path = directory + 'prediction_split_top_1_sub_{}_Glove_without_same_nr_of_trials.pkl'.format(subject)
    if prediction=='Top 1':
        corr_dict_1_10 = pickle.load(open(path, 'rb'))
    else:
        corr_dict_3_5 = pickle.load(open(path, 'rb'))  

    if use_regressed_out_Glove:
        if prediction=='Top 3':
            path = directory + 'prediction_split_top_3_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 5':
            path = directory + 'prediction_split_top_5_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 1':
            path = directory + 'prediction_split_top_1_sub_{}_Glove_regressed_out.pkl'.format(subject)
        if prediction=='Top 1':
            corr_dict_1_10 = pickle.load(open(path, 'rb'))
        else:
            corr_dict_3_5 = pickle.load(open(path, 'rb'))  

    
    if prediction in ['Top 5', 'Top 3']:
        corr_dict[prediction] = corr_dict_3_5[prediction_type[prediction]]
    else:
        corr_dict[prediction] = corr_dict_1_10[prediction_type[prediction]]
    
    if not_prediction in ['Top 5', 'Top 3']:
        corr_dict['Not Predicted'] = corr_dict_3_5[unprediction_type[not_prediction]]
    else:
        corr_dict['Not Predicted'] = corr_dict_1_10[unprediction_type[not_prediction]]
    
    # get the significant pre-onset timepoints, were top-1 is larger than not-top-1
    T_obs, cl, cl_p_vals, H0 = get_significant_different_timepoints(corr_dict, subject, return_all=True)
       
    
    if use_regressed_out_Glove:
        labels = dict(zip(corr_dict.keys(), [m+' (Residualised)' for m in corr_dict.keys()]))
    else: 
        labels = dict(zip(corr_dict.keys(), [m for m in corr_dict.keys()]))

    # compute difference between predictable and unpredictable
    diff = corr_dict[prediction].mean(axis=1)[get_signal_mask(subject)].mean(axis=0) - corr_dict['Not Predicted'].mean(axis=1)[get_signal_mask(subject)].mean(axis=0) 

    # initialise list to contain cluster closest before TP=0
    cluster = []

    # loop through difference-array starting at TP=0 and going backwards 
    # for subject 3 we choose a later  index, since we want to jump the closest cluster which only has 5 TPs
    start_index = len(pre_onset_times)+1
    if subject == 3:
        start_index = 69

    for i in np.flip(range(start_index)):

        # make sure this is indeed TP=0:
        if i == list(range(start_index))[-1]:
            latest_index = i
            print('We start at index: ', i)
            print('This corresponds to time point: {:.3f}s'.format(times_100[i]))

        if diff[i]<0 and diff[i+1]<0: # ignore time points that are negative
            continue

        if diff[i]<0  and diff[i+1]>0:
            print(i+1," was the earliest time point of the cluster")
            print('This corresponds to time point: {:.3f}s'.format(times_100[i+1]))
            earliest_index = i+1
            break
        else:  # now our current time point value is positive
            if diff[i+1]<0:  # however, if the one before was negative this is the closest to zero:
                print(i," is the lastest index")
                print('This corresponds to time point: {:.3f}s'.format(times_100[i]))
                latest_index = i
            cluster.append(diff[i])

    # print the info:
    print('-'*40)
    print('There are ', len(cluster), ' time points in the cluster')
    print('On average, being predictable improved encoding performance by {:.3f}'.format(np.mean(cluster)))
    print('range: [{:.3f} - {:.3f}]'.format(np.min(cluster), np.max(cluster)))
    print ('With a SD of {:.3f}'.format(np.std(cluster)))
    print('-'*40)

    # now let's compute that in terms of percentage improvement from baseline (=unpredictable)
    encoding_unpred    = corr_dict['Not Predicted'].mean(axis=1)[get_signal_mask(subject)].mean(axis=0)
    percentage_diff    = (diff / encoding_unpred) * 100
    percentage_cluster = percentage_diff[earliest_index:latest_index+1]

    # print the info:
    print('There are ', len(percentage_cluster), ' time points in the cluster')
    print('On average, being predictable improved encoding performance by {:.3f} percent wrt unpredictable encoding'.format(np.mean(percentage_cluster)))
    print('range: [{:.3f} - {:.3f}]'.format(np.min(percentage_cluster), np.max(percentage_cluster)))
    print ('With a SD of {:.3f} percent'.format(np.std(percentage_cluster)))

    if subject ==3: 
        print(np.mean(encoding_unpred[22:67]))
        print(np.mean(cluster) / np.mean(encoding_unpred[22:67]))

    # compute the p-values of significant time points
    significant_pvals = [pval for pval, tval in zip(cl_p_vals, T_obs) if tval > np.percentile(H0, 95)]

    print('-'*40)
    print('The maximum p value for all significant timepoint is: ', np.max(significant_pvals))

def plot_selfpredictability(dataset = 'Armani', plot_split = False, top=1):
    
    dir_path = '/project/3018059.03/results/{}/self_predictability/'.format(dataset)
    if plot_split: 
        if dataset == 'Armani':
            file = 'new_in_correct_selfpredictability_Glove_first_session.pkl'
            if top==1:
                file = 'new_in_correct_selfpredictability_Glove_top_1_first_session_1.pkl'
        if dataset == 'Gwilliams':
            file = 'new_in_correct_selfpredictability_Glove.pkl'
            if top==1:
                file = 'new_in_correct_selfpredictability_Glove_top_1.pkl'
    else:
        if dataset == 'Armani':
            file = 'Glove_GPT_Arbitrary_session_1.pkl'
        if dataset == 'Gwilliams':
            file = 'GPT_Glove_arbitrary.pkl'
    path     = dir_path + file   
    corr_dict= pickle.load(open(path, 'rb'))
    
    # plot:
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    
    models = [k.split(sep='_')[1] for k in corr_dict.keys()]
    
    if plot_split:
        models = ['Top 5', 'Not Predicted']
        if top ==1:
            models = ['Top 1', 'Not Predicted']
    
    for model, key in zip(models, corr_dict.keys()):
        ax.plot(pre_onset_times, reshape(corr_dict[key]).mean(axis=0)[0:78], c=colours[model], label=model)
        ax.fill_between(pre_onset_times, lowerCI(reshape(corr_dict[key])[:, 0:78]), 
                                   upperCI(reshape(corr_dict[key])[:, 0:78]), color=colours[model], alpha=0.3)
        
        
    ax.set_xlabel('Time in Seconds')
    ax.set_ylabel('Crossvalidated Correlation')
    ax.axhline(c='indianred',  alpha=0.3)
    ax.legend()

    fig_folder = 'figures/supplementary/'

    if dataset == 'Armani':
        if plot_split ==False:
            fig_folder = 'figures/main/'
        if plot_split == True and top==1:
            fig_folder = 'figures/main/'

    fig_path = fig_folder+'Selfpredictability-{}-prediction_split_{}-top_{}.pdf'.format(dataset, plot_split, top)

    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
        
def plot_regressed_out_selfpredictability():
    
    # get the results:
    dir_path       = '/project/3018059.03/Lingpred/results/Armani/X-Y-Flip/'
    corr_Glove     = corr_dict= pickle.load(open(dir_path + 'regressed_out_one_glove.pkl', 'rb'))
    corr_GPT       = corr_dict= pickle.load(open(dir_path + 'regressed_out_one_gpt.pkl', 'rb'))
    corr_arbitrary = corr_dict= pickle.load(open(dir_path + 'regressed_out_arbitrary.pkl', 'rb'))
    
    # make dictionary:
    models    = ['Glove', 'GPT', 'arbitrary']
    corr_dict = dict(zip(models, [corr_Glove, corr_GPT, corr_arbitrary]))
    labels    = dict(zip(models, ['Residualised '+m for m in models]))

    # plot:
    fig, ax = plt.subplots(1,1, figsize=(4, 4))
    
    for model, corr in corr_dict.items():
        ax.plot(times_100, reshape(corr).mean(axis=0), c=colours[model], label=labels[model])
        ax.fill_between(times_100, lowerCI(reshape(corr)), 
                                   upperCI(reshape(corr)), color=colours[model], alpha=0.3)
        
    ax.set_xlabel('Time in Seconds')
    ax.set_ylabel('Crossvalidated Correlation')
    ax.axvspan(-.050, .050, color='gray',  alpha=0.3)
    ax.axhline(c='indianred',  alpha=0.3)
    ax.legend()


    fig_folder = 'figures/main/'
    fig_path = fig_folder+'Selfpredictability_regressed_out_True.pdf'

    plt.savefig(fig_path, format='pdf', bbox_inches='tight')
    plt.show()       