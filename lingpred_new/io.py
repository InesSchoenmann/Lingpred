import glob 
import shutil
import os 
import sys
from pathlib import Path
import numpy as np
import pickle
from typing import Sequence,Union,Optional
import imp
import pandas as pd
import tgt
import mne 
import mne_bids 
import h5py
from itertools import compress

from scipy.io import loadmat


from mh.utils import loadmat_struct2dict 
from mh.mne.utils import _neighbours_from_coords

PROJ_ROOT = '/project/3018059.03'

# -----------------------------------------------------------#
#                 Routine to load text                       #
# -----------------------------------------------------------#

def get_text_per_session(dataset: str, session: int, subject: int):
    
    """
    Loads text for a given dataset, session, and subject.

    Parameters
    ----------
    dataset : str
        Name of the dataset. Options: "Gwilliams" or "Armani"
    session : int
        Session for which the text is supposed to be loaded.
        - `range(11)` for Armani
        - `range(4)` for Gwilliams
    subject : 
        Subject for whom the text is supposed to be loaded.

    Returns
    -------
    str
        The text for all runs as a single string.
    """
    
    if dataset =='Armani':
        runs = _runs_in_session(sess_i=session,sub_i=subject)

        text_all_runs = ''

        for run in runs:
            this_run_text = _load_full_text(sess_i = session, run_i = run ,without_breaks=True)
            text_all_runs = text_all_runs + ' ' + this_run_text
        return text_all_runs   
    
    if dataset == 'Gwilliams':
            
        if str(session) =='0':
            textname = 'lw1.txt'
        if str(session) == '1':
            textname = 'cable_spool_fort.txt'
        if str(session) =='2':
            textname = 'easy_money.txt'
        if str(session) == '3':
            textname = 'the_black_willow.txt'

        fname = '/project/3018059.03/Lingpred/data/Gwilliams/stimuli/text/' + textname

        with open(fname, 'r') as file:
            text = file.read()
        return text 

# -----------------------------------------------------------#
#       NEW ROUTINES TO LOAD GWILLIAMS OR ARMANI MEG DATA    #
# -----------------------------------------------------------#

def get_neural_data(dataset: str, sessions: list, subject: int, task = 'compr', datatype='source', channels=None, window_size=100, band=(0.1, 40), baseline=None):
    
    """
    Computes the average activation at each channel and lag relative to word onset.


    Parameters
    ----------
    dataset : str
        Either "Gwilliams", "Sherlock", or "Armani".
    subject : str or int
        Subject identifier. For Armani dataset: e.g., "001". For Gwilliams dataset: e.g., "01".
    task : str or int
        Task identifier. For Armani dataset: "compr". For Gwilliams dataset: 0, 1, 2, or 3, corresponding to:
        - 0 = lw1
        - 1 = cable spool fort
        - 2 = easy money
        - 3 = black willow
    channels : str, optional
        Can be 'None', 'language', 'signal', 'pre-onset', or 'post-onset'.  
        - `None`: Returns all channels.  
        - `'language'`: Returns only language-related channels.  
        - `'signal'` (subject-specific): Returns only channels with relevant encoding (max value > 25% of max overall value).  
        - `'pre-onset'` (subject-specific): Returns only channels with predominantly pre-word-onset encoding.  
        - `'post-onset'` (subject-specific): Returns only channels with predominantly post-word-onset encoding.  
    window_size : int, optional
        Window size for averaging the neural data with lag = 25 ms. If `None`, no averaging is performed.
    band : tuple of float
        Bandpass filter range to be used. Can be (0.5, 8) or (1, 40).
    baseline : bool or None
        Whether to perform baseline correction. If `None`, no correction is performed.

    Returns
    -------
    numpy.ndarray
        An array containing the average activation at each channel and lag relative to word onset.
        The returned array has shape (nr_channels, nr_epochs, nr_lags).
    pandas.DataFrame
        A dataframe containing the word of each epoch, the onset time, and additional information.
    list
        A list of dropped epochs.
    """

 
    if dataset == "Gwilliams":
        task = task
                
    sessions = sessions
    #get_sessions(dataset, subject)      # These methods are still missing
    
    if channels == 'language':
        sources = get_language_channels(dataset)
        
    if channels == 'signal':
        sources = get_signal_channels(dataset, subject)
    
    if channels == 'pre-onset':
        sources = get_signal_channels(dataset, subject, pre=True)
        
    if channels == 'post-onset':
        sources = get_signal_channels(dataset, subject, post=True)
        
    else: sources = None # this includes all channels

    for session in sessions:
        
        bad_epochs_all_runs = []      # list with indices of dropped epochs
        length_all_epochs   = 0       # constant to be added to the indices for each run
        
        runs = get_runs(dataset, subject, session) # This now only returns [1] for Gwilliams & Armani

        for run in runs:

            # load raw data and get a dataframe for annotations incl. word ID and onset/offset
            raw_data       = load_raw_data(dataset, subject, session, run, task, datatype=datatype, band=band)
            annotations_df = get_words_onsets_offsets(raw_data, dataset, subject, session, run)
            onset_words    = annotations_df.onset


            # create events
            events         = np.c_[onset_words * raw_data.info['sfreq'], 
                                   np.zeros((len(onset_words), 1)), 
                                   np.ones((len(onset_words), 1))].astype(int)
            
            # now down-sample the data to 200Hz for computational reasons: 
            # by passing the events array we make sure our timing is not jittered:
            raw_data, events = raw_data.resample(sfreq=200, events=events)
            

            # create epochs
            epochs = mne.Epochs(raw_data, 
                                events, 
                                tmin=-2, 
                                tmax=2, 
                                baseline=baseline,            # (-2, 0) would be the default baseline
                                picks=sources,
                                metadata = annotations_df,
                                event_repeated='drop',
                                preload=True)
            
            
            # get indices of bad epochs:
            bad_epochs_this_run = [index + length_all_epochs for index, dl in enumerate(epochs.drop_log) if len(dl)]
            print(bad_epochs_this_run)
            length_all_epochs = length_all_epochs + len(epochs) + len(bad_epochs_this_run)
            bad_epochs_all_runs = bad_epochs_all_runs + bad_epochs_this_run


            # get average per lag
            if window_size:
                data_per_lag = get_mean_data_lag(epochs, window_size)    # array of shape (nr_epochs, nr_channels, nr_lags)
            else:
                data_per_lag = epochs.get_data()


            # stack on top for all runs in one session & append annotations dataframe
            if run == runs[0] : # first run in session 
                data_per_lag_all_runs = data_per_lag
                comb_annotation_df    = epochs.metadata
            else:
                data_per_lag_all_runs = np.vstack((data_per_lag_all_runs, # array of shape:
                                                   data_per_lag))         # (nr_all_epochs, nr_channels, nr_lags)
                comb_annotation_df   = comb_annotation_df.append(epochs.metadata)

        print(bad_epochs_all_runs)
        
        # stack on top for all sessions & append annotations dataframe
        if session == 1 or len(sessions)==1: 
            data_per_lag_all_sess    = data_per_lag_all_runs
            annotation_df_all_sess   = comb_annotation_df
        else:
            data_per_lag_all_sess    = np.vstack((data_per_lag_all_sess,  # array of shape:
                                                  data_per_lag_all_runs)) # (nr_all_epochs, nr_channels, nr_lags)
            annotation_df_all_sess   = annotation_df_all_sess.append(comb_annotation_df)
    
    # for subject 3, session 8 words 3841-4170 are scrambled: and need to be dropped from the df and the neural data:
    if session==8 and subject==3:
        annotation_df_all_sess.drop(index=annotation_df_all_sess[3841:4170].index, inplace=True)
        np.delete(data_per_lag_all_sess, np.arange(3841,4170))
        
    # change shape of the array such that one can easily loop over the channels: (nr_channels, nr_all_epochs, nr_lags)
    data_per_lag_all_sess = np.swapaxes(data_per_lag_all_sess,0,1)
    
    # return array with neural data and annotation data frame:
    return data_per_lag_all_sess, annotation_df_all_sess, bad_epochs_all_runs



# ------------------------------------------------------------#
#              Associated New Auxiliary Functions:            #
# ------------------------------------------------------------#

def drop_nans(y, X):

    '''
    Removes words containing NaN values from the neural data and GPT layer activations.

    Parameters
    ----------
    y : numpy.ndarray
        Neural data array of shape (channels, words, lags), containing the averaged MEG data.
    X : numpy.ndarray
        GPT layer activations, an array of shape (words, dimensions + 1).
    nan_ids : list of int
        List containing the indices of words that contain NaN values.

    Returns
    -------
    numpy.ndarray
        The `y` array with words containing NaNs removed.
    numpy.ndarray
        The `X` array with words containing NaNs removed.
    '''
    
    # get NaN values in neural data:
    nan_ids = get_nans(y)
    
    y = np.swapaxes(y,0,1) # swap axes such that rows == words
    
    # drop rows containing NaNs in y and X
    y = np.delete(y, nan_ids, axis=0)
    X = np.delete(X, nan_ids, axis=0)
    
    y = np.swapaxes(y,0,1) # swap axes such that rows == channels

    # print shapes
    print(X.shape, y.shape)
    
    return y, X


def get_nans(neural_data):
    """
    Identifies words containing NaN values in the neural data.

    Parameters
    ----------
    neural_data : numpy.ndarray
        Array of shape (channels, words, lags), containing the averaged MEG data.

    Returns
    -------
    list of int
        List containing the indices of words that contain NaN values (to be dropped).
    """

    # initialise counter and index list
    count    = 0
    nan_ids = []
    
    # loop over words (rows) to find rows containing NaNs
    for i, word in enumerate(neural_data[0]): # just do this for one channel
        if np.any(np.any(np.isnan(word))):
            count+=1
            nan_ids.append(i)
    print('There are {} words with NaNs which will be dropped.'.format(count))
    
    return nan_ids


def get_language_channels(dataset: str, get_areas=False):
    '''
    Retrieves the list of channel names related to the language system for a given dataset.

    Parameters
    ----------
    dataset : str
        The dataset name. Can be either 'sherlock', 'Armani', or 'Gwilliams'.

    Returns
    -------
    list of str
        A list of channel names related to the language system.
    '''

    if dataset=='sherlock' or dataset=='Armani':
        source_info = ASH_load_source_info()
        ch_names    = source_info['lbls_language']
        
        if get_areas:
            ch_names = list(compress(source_info['areas'], source_info['language_mask']))
        
    else: raise VallueError('get_language_channels is only implemented for the Armani dataset')
        
    return ch_names


def get_lags(window_size, lag=0.025, start=-2, end=2):
   
    '''
    Returns an list of shape (nr_lags, 3) each row containing (lag_index, start_window, end_window) in sec
    '''
   
    starts = np.round(np.arange(start, end-window_size+lag, lag), 3)
    ends   = np.round([s + window_size for s in starts], 3)

    arr = [[nr, start, end] for nr, (start, end) in enumerate(zip(starts, ends))]
       
    return arr


def get_mean_data_lag(epochs, window_size):
    '''
    Param: MNE epochs object
    Returns average data for each window: array of shape (nr_epochs, nr_channels, nr_lags)
    '''
    
    # transform window_size from ms to s (round to 2 decimals to ensure precision, i.e. 0.05, 0.10, 0.15, 0.2)
    window_size = round(window_size/1000, 2)
   
    lags       = get_lags(window_size=window_size)
   
    nr_lags     = len(lags)
   
    for i in range(nr_lags):
        nr, start, end = lags[i]
        data           = epochs.get_data(tmin=start, tmax=end) # ndarray of shape (nr_ep, nr_ch, nr_data)
        mean_data      = np.mean(data, axis=2)                 # ndarray of shape (nr_epochs, nr_channels)
        del data
        if i==0:
            com_data = mean_data
        else:
            com_data = np.dstack([com_data, mean_data])

    return com_data             # array of shape (nr_epochs, nr_channels, nr_lags)


def load_raw_data(dataset: str, subject: int, session: int, run: int, task: str, datatype='raw', band=(1, 40)):
    """
    Loads raw data object using the MNE.

    Parameters
    ----------
    dataset : str
        The dataset name. Can be "Gwilliams", "sherlock", or "Armani".
    subject : str or int
        The subject identifier for which the data should be loaded.
    session : str or int
        The session for which the data should be loaded.
    task : str
        The task for which the data should be loaded. '0' for Gwilliams and 'compr' for Armani.
    band : tuple of float
        A tuple specifying the frequency band of the filtered Sherlock data.
    datatype : str
        The type of data to load:
        - `'filtered'` if the dataset is Gwilliams (filtered sensor data).
        - `'source'` if the dataset is Armani (filtered source data).

    Returns
    -------
    mne.io.Raw
        An MNE raw object with the filtered data for Gwilliams and source localized data for Armani.
    """

    
    # set root to path:
    if dataset=="Gwilliams":
        root='/project/3018059.03/Lingpred/data/Gwilliams/derived/'
        
        sess = str(session)
        if subject < 10: 
            sub = '0' + str(subject)
        else:
            sub = str(subject)
        
    elif dataset=="Armani":
        root='/project/3018059.03/Lingpred/data/Armani/'
       
        sub = '00' + str(subject)
        
        if session < 10:
            sess = '00' + str(session)
        else:
             sess = '0' + str(session)
    
    
    # set path to raw data:
    if datatype == 'raw':
        
        datatype = 'meg'
        bids_path = mne_bids.BIDSPath(subject=sub, 
                                         session=sess, 
                                          task=task, 
                                          datatype=datatype, 
                                          root=root) 
        
    
    # load data:
    if dataset=="Armani":
        
        # if we want the raw source localised data we need to load it from the source foulder:
        if datatype=='source':
            
            # file name, e.g.: 1-1_lcmv-data_0.1-40raw.fif
            fname = str(subject) + '-' + str(session) + '_lcmv-data_' + str(band[0]) + '-' + str(band[1]) + 'raw.fif'
            
            # handle naming of session 10 
            if session < 10:
                session = '0'+ str(session)
            
            # full file path
            fif_path = root + 'sub-00' + str(subject) + '/ses-0' + str(session) + '/source/' + fname
            
            #read data
            raw = mne.io.read_raw_fif(fif_path) 
        
        
        # else if we want raw data:
        else:
            raw = mne.io.read_raw_ctf(bids_path) 
            
        
    if dataset=="Gwilliams":
        
        # if we want the filtered data:
        if datatype=='filtered':
            
            band=(0.1, 40)
            
            # file name, e.g.: 1-1_lcmv-data_0.1-40raw.fif
            fname = str(subject)+'-'+sess+'_filtered_data_'+str(band[0])+'-'+str(band[1])+'_task-'+task+'_'+'raw.fif'
            
            # full file path
            folder   = '/project/3018059.03/data/Gwilliams/'
            fif_path = folder + 'sub-' + sub + '/ses-' + sess + '/filtered/' + fname
            
            # return False if the file doesn't exist
            if not Path(fif_path).is_file():
                return False
            
            #read data
            raw = mne.io.read_raw_fif(fif_path) 
        
        # if we want the raw data:
        else: 
            raw = mne_bids.read_raw_bids(bids_path)
    
    if dataset not in ['Gwilliams', 'Armani']:
        raise ValueError('Dataset variable must be either "Gwilliams", or "Armani"')
    
    # load raw data
    raw.load_data()
    
    return raw


def get_words_onsets_offsets(raw_data, dataset:str, subject:int, session:int, run:int):
    """
    Loads word onset offsets for a given dataset, subject, session, and run.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw data object.
    dataset : str
        The dataset name. Can be "Gwilliams", "Armani", or "Sherlock".
    subject : str or int
        The subject identifier for which the offsets should be loaded.
    session : str or int
        The session for which the offsets should be loaded.
    run : str or int
        The run for which the offsets should be loaded.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing at least two columns: 'word' and 'onset'.
    """

    if dataset == 'Gwilliams':
        
        df     = raw_data.annotations.to_data_frame() 
        df     = pd.DataFrame(df.description.apply(eval).to_list())
        onsets = raw_data.annotations.onset
        
        # make a column for the onsets:
        df['onset'] = onsets

        # keep only word onset data
        df_words = df[df['kind']=='word']

        # keep only words which are part of the story (no pseudowords or wordlists)
        df_words = df_words[df_words['condition']=='sentence']
        
        # deal with shift:
        word_onsets = df.loc[df_words.index + 1].onset.values
        
        df_words['onset'] = word_onsets
        
        
    elif dataset == 'Armani':
        
        # handle naming of session 10: 
        if session < 10:
            sess = '00' + str(session)
        else: 
            sess = '0' + str(session)
        
        # get path to events file:
        dir_path = '/project/3018059.03/Lingpred/data/Armani/'
        filepath = 'sub-00' + str(subject) +'/' + 'ses-' + sess +'/'+ 'meg/'
        filename = 'sub-00' + str(subject) + '_ses-' + sess + '_task-compr_events.tsv'
        
        # read pandas DataFrame
        annotations = pd.read_csv(dir_path+filepath+filename, sep='\t')
        
        # type of event is separated by runs: there are at max 7 runs in each session:
        onset_names_list = ['word_onset_01', 'word_onset_02', 'word_onset_03', 'word_onset_04', 
                            'word_onset_05', 'word_onset_06', 'word_onset_07', 'word_onset_08']
        
        # get only words, rename columns (identically to Gwilliams) and clean data frame
        df_words = annotations[annotations.type.isin(onset_names_list)]
        df_words = df_words[df_words.value != 'sp']
        df_words.rename(columns={'value':'word'}, inplace=True)
        df_words = clean_events(df_words, subject, session, dataset)
        
    else: raise ValueError('Dataset variable must be either "Gwilliams", or "Armani"')
    
    return df_words



def get_phonemes_onsets_offsets(dataset:str, subject:int, session:int, run:int, 
                                only_word_inital_phonemes=True, correct_wav_onset=True):
    """
    Loads phoneme onset offsets for a given dataset, subject, session, and run.

    Parameters
    ----------
    raw_data : mne.io.Raw
        The raw data object.
    dataset : str
        The dataset name. Can be "Gwilliams", "Armani", or "Sherlock".
    subject : str or int
        The subject identifier for which the offsets should be loaded.
    session : str or int
        The session for which the offsets should be loaded.
    run : str or int
        The run for which the offsets should be loaded.
    only_word_initial_phonemes : bool, optional
        Whether to retrieve only word-initial phonemes. Defaults to `False`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing at least two columns: 'phoneme' and 'onset'.
    """

    if dataset == 'Armani':
        
        # handle naming of session 10: 
        if session < 10:
            sess = '00' + str(session)
        else: 
            sess = '0' + str(session)
        
        # get path to events file:
        dir_path = '/project/3018059.03/Lingpred/data/Armani/'
        filepath = 'sub-00' + str(subject) +'/' + 'ses-' + sess +'/'+ 'meg/'
        filename = 'sub-00' + str(subject) + '_ses-' + sess + '_task-compr_events.tsv'
        
        # read pandas DataFrame for the entire session:
        annotations = pd.read_csv(dir_path+filepath+filename, sep='\t')
        
        # get list with word_onsets for this run:
        word_onset_name = ['word_onset_0{}'.format(run)]
        df_words        = annotations[annotations.type.isin(word_onset_name)]
        df_words        = df_words[df_words.value != 'sp']
        word_onsets     = df_words.onset
        
        if correct_wav_onset:
            # now look for the timing of the audio onset for this run:
            index_first_word = df_words.index[0] # index for the first word in this run

            for i in np.arange(index_first_word, -1, -1):     # interate from there backwards
                if annotations.iloc[i].type == 'wav_onset':   # to the most recent wave onset
                    audio_onset = annotations.iloc[i].onset   # and get it's onset time
                    break                                     # break out of for loop
        
        # get list with phoneme_onsets for this run::
        onset_name = ['phoneme_onset_0{}'.format(run)]
        
        # get only phonemes and clean data frame
        df_phonemes = annotations[annotations.type.isin(onset_name)]
        
        if only_word_inital_phonemes:
            df_phonemes = df_phonemes[df_phonemes.onset.isin(word_onsets)]
        else: df_phonemes = df_phonemes[df_phonemes.value != 'sp']
        
        if correct_wav_onset:
            #convert times to audio onset times:
            df_phonemes.onset = df_phonemes.onset - audio_onset
        
        # add a column with the offsets:
        offsets               = df_phonemes.onset + df_phonemes.duration
        df_phonemes['offset'] = offsets
            
    else: raise ValueError('Dataset variable must be "Armani"')
    
    return df_phonemes

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DUMMY FUNCTIONS: TO BE IMPLEMENTED ONCE WE HAVE SOURCE LEVEL DATA FOR ALL DATASETS:
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# this function may be necessary to remove events which are not in the GPT-2 word embeddings
def clean_events(annotations_words, subject, session, dataset):
    
    return annotations_words


def get_sessions(dataset, subject):
    
    if dataset =='Armani':
        return np.arange(1,11) # Armani has 10 sessions: from 1-10
    else:
        raise ValueError("Only defined for the Armeni dataset at the moment.")
        
        

# get's runs when using Micha'methods, otherwise returns [1] as a list
def get_runs(dataset, subject, session):
    
    if dataset == "Armani":
        return _runs_in_session(session,sub_i=subject)
    else:
        return [1]

        
def _load_full_text(sess_i,run_i,without_breaks=True):
    ASH_sess2runs=lambda x : {1:7,2:7,4:8,6:7,8:7}.get(x,6)
    """Load the full text as a single string for each run."""
    sess_str='0'+str(sess_i) if sess_i<10 else str(sess_i)
    fname=PROJ_ROOT/'Lingpred'/'data' / 'Armani' /'stimuli'/ "{}_{}.txt".format(sess_str,run_i)
    with open(fname, 'r') as file:
        full_text = file.read().replace('\n', ' ') if without_breaks else file.read()
    full_text=full_text.replace('  ',' ')
    return(full_text)   
     

def _runs_in_session(sess_i:int,sub_i:Union[None,int])->list:
    """from session number (and, optionally, subject number), get list of runs"""
    def _sess2nruns(sess_i,sub_i=None):
        """
        (subject number has to be given to account for aborted runs in pilot sub, and sub-003).
        (in subj3-sess8, run3 (run 7 !?) is missing)
        """
        if sub_i is None: sub_i=1

        if sub_i<0:
            sess2runs=lambda x : {1:3,2:7,4:7,6:7,8:7}.get(x,6)
        elif sub_i==3:
            sess2runs=lambda x : {1:7,2:7,4:8,6:7}.get(x,6)
        else:
            sess2runs=lambda x : {1:7,2:7,4:8,6:7,8:7}.get(x,6)

        return(sess2runs(sess_i))

    runs=list(range(1, 1 + _sess2nruns(sess_i,sub_i=sub_i))) 
    return(runs)


##############################################################################################
""" !!! Legacy routines hinging upon there being a 'sherlock' folder with Micha's data !!!"""
##############################################################################################


def ASH_load_source_info():
    def _load_full_neighbour():
        neigh_full=PROJ_ROOT /'data'/'sherlock' / 'derived' / 'neighbours_all.pkl'
        with open(neigh_full,'rb') as pntr:
            nghbr=pickle.load(pntr)
        return(nghbr)

    """adv sher holm: load source info (chan names, language mask, etc)"""
    mask_info=loadmat_struct2dict(ASH_ATLAS['atlas_dir'] / 'mask.mat')
    lbls= [lbl[0] for lbl in mask_info['labels'] if (('???' not in lbl[0]) and ('MEDIAL' not in lbl[0]))]
    lbls_selected = [lbl[0] for lbl in mask_info['labels'][mask_info['selected'].astype(bool)]]
    language_mask =np.array([True if lbl in lbls_selected else False for lbl in lbls ])
    areas=[label2area(lbl) for lbl in lbls]

    # also include adjacency matrix to do cluster-based permutation testing 
    xyz=np.zeros((len(lbls),3))
    loc_dict=_ASH_get_loc_dict()
    for ch_i,ch_name in enumerate(lbls):
        xyz[ch_i,:]=np.mean(loc_dict[ch_name],axis=0)
    neighbours=_neighbours_from_coords(xyz)
    neighbours_all=_load_full_neighbour()
    return({'lbls':lbls,'areas':areas,'lbls_language':lbls_selected,'language_mask':language_mask,
           'neighbours':neighbours,'neighbours_all':neighbours_all})

def _ASH_load_meg_sess(sub_i,sess_i,dat_lvl='source',align=False,band=None):
    """load meg  (dict) + meg info (mne info) from disk.
    see: _ASH_get_run_raw for downstream handling 
    or : ASH_load_run_raw (convenience function, but slow if we do all runs anyway)
    """
    def correct_megdat(megdat_in,dat_lvl='sensor'):
        """insert a placeholder for run 3 of sub-003, session 8 (since data is missing)"""
        for k in ['runs','time']: # data runs: [1,2,[_missing_],4,5,6,7]
            if len(megdat_in[k])>6:raise ValueError('no run missing?!!!')
            if ('sensor' in dat_lvl):megdat_in[k]=np.insert(megdat_in[k],2,np.array(['missing']))
            elif ('source' in dat_lvl):megdat_in[k].insert(2,"missing")
        return(megdat_in)
    
    # parse inputs  
    # -------------
    
    # did we give a certain frequency band (e.g. 0.5-8 Hz) If none, default =1,40 (?)
    if band:band_str='{}-{}'.format(*(str(num_i).replace('.','') for num_i in band),)
    else: band_str='1-40'
    if sess_i > 1:
        suffix='_align' # and (align==True))  
    else: suffix=''
    
    # where are the data 
    megdir=PROJ_ROOT/'data'/'sherlock'/'derived'
    
    # path for this sub-session combination 
    sess_path=Path(f'sub-{str(sub_i).zfill(3)}')/f'ses-{str(sess_i).zfill(3)}'/'meg'
    
    # source or sensor level 
    if dat_lvl=='source':
        dat_file=megdir/sess_path/f'{str(sub_i)}-{str(sess_i)}_lcmv-data_{band_str}{suffix}.mat'
    elif dat_lvl=='sensor':
        dat_file=megdir/sess_path/f's{str(sub_i).zfill(2)}-{str(sess_i).zfill(2)}_preproc.mat'
    
    # load meg data (deal with annoying matlab files)
    megdat=_read_source_mat_v73(dat_file) if ('source' in dat_lvl) else _read_meg_mat(datfile)
    
    ##--------------------------##--------------------------##--------------------------
    # some ugly legacy stuff to deal with information about sensor-level data 
    # I put it here, but probably the definitve data is much better documented, so not recommended. 
    # Ines probably has no access to the OG /raw/ dir, so to be decided on what to do for sensor-level.

    if ('sensor' in dat_lvl):# define dir to load (wildcards because pilot naming highly inconsistent) 
        if (sub_i==1) and (sess_i==1):
            rawdir=glob.glob(str(_ASH_get_rawdir(sub_i,sess_i) / 'meg' /  
                            f'sub-{str(sub_i).zfill(3)}_ses-{str(sess_i).zfill(3)}_task-compr*.ds'))

        else:
            rawdir =glob.glob((_ASH_get_rawdir(sub_i,sess_i) / 'meg' / 
                   '{}0{}ses*{}*.ds'.format(sublet,abs(sub_i),sess_dir)).__str__())
            rawdir=glob.glob(str(_ASH_get_rawdir(sub_i,sess_i) / 'meg' /  
                            f'sub-{str(sub_i).zfill(3)}_ses-{str(sess_i).zfill(3)}_task-compr*.ds'))

        info = mne.io.read_raw_ctf(rawdir[0], preload=False).info
        info['bads']=natpred.config.ASH_BAD_CHAN_DICT.get(sub_i,[])
        
        # hilbert and power in get raw
    elif ('source' in dat_lvl):
        info=None # no info file 
    # -------------------------------------------------------------------------------------------------
    
    return(megdat,info)

from pdb import set_trace

def _read_source_mat_v73(fname):
    """load pre-computed source localised data using Kristijan/Jan-Matthijsses parcellation
    this wrapper reads the data from a v7.3 .mat file and returns fieldtrip-style dict 
    (i.e. dict with 'time', 'runs', 'label' and 'fsample' keys)
    """
    datt = h5py.File(fname,mode='r')   
    return(dict( # integer decoding v_73 style 
        runs =  [datt[ref[0]][()].T for ref in  datt[datt["data/trial"].ref][()]],
        time =  [datt[ref[0]][()] for ref in  datt[datt["data/time"].ref][()]],
        fsample= datt[datt["data/fsample"].ref][()][0][0],
        label=["".join([chr(int(i)) for i in ints]) for ints in 
               [datt[r][()] for r in datt[datt["data/label"].ref][()][0]] ]
    ))

def _read_meg_mat(file):
    """struct from v6 .mat with time, runs, label and fsample keys
    for generic version, see: mh.utils.loadmat_struct2dict
    """
    
    d = loadmat(file, struct_as_record=False)
    tmp = d["data"][0, 0]  # access the data

    return(dict(runs = tmp.trial.squeeze(),
             time = tmp.time.squeeze(),
             label = np.concatenate(tmp.label.tolist()).squeeze(),
             fsample = tmp.fsample[0]))

def _ASH_get_rawdir(sub_i,sess_i):
    """get posix path of raw data dir. negative subject number equals pilot
    -------------------------------------------------------------
    NB: points to the old sherlock data directory, ines progbably has not acces
    """
    sub_dir = 'sub-00{}'.format(sub_i) if sub_i>0 else 'pil-00{}'.format(sub_i*-1)
    sess_str='0'+str(sess_i) if sess_i<10 else str(sess_i)
    return(ASH_ROOT / 'data' / 'raw' / sub_dir / ('ses-0'+sess_str))

def _ASH_get_loc_dict():
    """adv sher holms: get dict with the 3d location of each vertex of each source parcel"""
    sub_parc= loadmat_struct2dict(ASH_ATLAS['atlas_parcels'])

    # create lookup table (so we don't have to search/loop through each time)
    parc_lbl2ix={lbl[0]:(lbl_i+1) for lbl_i,lbl in enumerate(sub_parc['parcellationlabel'])}

    # pos/loc dict
    return({this_lbl[0]: sub_parc['pos'][(sub_parc['parcellation']==parc_lbl2ix[this_lbl[0]]),:]
             for this_lbl in sub_parc['parcellationlabel']})

def ASH_load_onsets(sub,sess,run,correct_delay=True) -> pd.DataFrame:
    """load onsets as dataframe. 
    all exceptions and delays should be taken care of.
    remember, subject specific! 
    If correct_delay==True audiodelay is added (this can be switched off to prevent correcting twice)
    """
    #~~ the device making the dataframe
    def _create_DF(tier,delay_time,tier_name='word',correct_delay=True):
        delay_fact=1 if correct_delay else 0
        event_list=[];onsets=[];offsets=[]
        for annot in tier:
            if annot.text not in ['sp','silence']: # edit: 07-may-2020 -- delays
                onsets.append(round(annot.start_time + delay_fact*delay_time[0]/1000,6))
                offsets.append(round(annot.end_time  + delay_fact*delay_time[0]/1000,6))
                event_list.append(annot.text)
        return(pd.DataFrame({tier_name:event_list,'onset':onsets,'offset':offsets}))
    ##~~~

    words,phons=_load_full_text(sess,run)
    delay=_ASH_get_audiodelay(sub,sess,run) # easier to just say delay = 0?
    onsets={'words':_create_DF(words,delay,tier_name='word',correct_delay=correct_delay),
            'phons':_create_DF(phons,delay,tier_name='phon',correct_delay=correct_delay)}
    return(onsets)

def _ASH_get_audiodelay(sub_i,sess_i,run_i):
    datadir=_ASH_derdir(sub_i,sess_i)
    delayfile= 'audiodelay.mat'

    # identify run number (handle misalginment for sub003-sess008)
    runs_sub03sess08={1:1,2:2,4:3,5:4,6:4,7:6}
    if (sub_i==3) and (sess_i==8): # if sub003-sess008: account for missing run 3
        try: run_i={1:1,2:2,4:3,5:4,6:4,7:6}[run_i]
        except KeyError: raise ValueError("NO AUDIODELAY FOR THIS RUN (note run3missing)")
        
    return(loadmat((datadir /'audiodelay.mat').__str__())['delay'][run_i-1])

def _ASH_derdir(sub_i,sess_i):
    """For given subject and session, get derived dir (data/derived/sub-xxx/sess-xx, etc)"""
    return(PROJ_ROOT/'Lingpred'/'data'/'sherlock'/'derived'/f'sub-{str(sub_i).zfill(3)}' / 
           f'ses-{str(sess_i).zfill(3)}'/'meg')
