import sys
import os 
from pathlib import Path
import pandas as pd
import numpy as np
import pickle


# function returns the indices of the vectors to be used for the X matrix in the self-predictability per tasks
def get_indices_per_task(dataset = str, task='0', session=0, cutoff=None, acoustic_model=False):
    
    runs = get_runs(dataset, session, 1, task)
    
    if acoustic_model:
        
        # initialise empty DataFrame
        df_phonemes = pd.DataFrame()
        constant    = 0 
        for run in runs:
            
            # get phonemes:
            df_run = get_onsets_offsets_per_run(dataset=dataset, 
                                             task=task,
                                             subject=1, 
                                             session=session, 
                                             run=run, 
                                             phonemes = True,
                                             only_word_inital_phonemes=False)
            
            # onsets are counted per run --> add last onset time to make them unique
            df_run.onset  = df_run.onset + constant
            df_run.offset = df_run.offset + constant
            constant      = df_run.offset.to_list()[-1]
            
            # append run
            df_phonemes = df_phonemes.append(df_run)
            

        on_offsets_phonemes = np.vstack([df_phonemes.onset.to_numpy(), df_phonemes.offset.to_numpy()]).T
    # End if acoustic model ----------------------------------------- 
    
    # initialise empty DataFrame
    df_words = pd.DataFrame()
    constant = 0 
    
    if dataset == 'Goldstein': #Goldstein only has one 'run' so no need looping 
        df_words = get_words_onsets_offsets(dataset)

    else:
        for run in runs:  
            
            # get dataframe for this session:
            df_run = get_onsets_offsets_per_run(dataset=dataset, 
                                                task=task,
                                                subject=1, 
                                                session=0, 
                                                run=run, 
                                                phonemes = False,
                                                only_word_inital_phonemes=False)
            
            # onsets are counted per run --> add last onset time to make them unique
            df_run.onset  = df_run.onset + constant
            df_run.offset = df_run.offset + constant
            constant      = df_run.offset.to_list()[-1]
            
            # append run
            df_words = df_words.append(df_run)

    # make 2d onset-offset array
    on_offsets = np.vstack([df_words.onset.to_numpy(), df_words.offset.to_numpy()]).T
    
    if cutoff:
        on_offsets = on_offsets[0:cutoff]
        
    print('This is session {} and on_offsets has shape {}'.format(session, on_offsets.shape))
    
    # initialise array with indices for this session
    indices_one_session = np.ones(shape=(on_offsets.shape[0], 157))

    for index, word in enumerate(on_offsets):
    
        # get the times
        times = get_times(index, on_offsets)

        # get indices of words at timepoints:
        if acoustic_model: # for the acoustic models indices need to be taken from all phoneme on-/offsets
            indices_for_timepoints = get_indices_for_timepoints(times, on_offsets_phonemes)
        else:
            indices_for_timepoints = get_indices_for_timepoints(times, on_offsets)

        # replace -1's:
        indices_for_timepoints     = replace_minus_ones(indices_for_timepoints)

        indices_one_session[index] = indices_for_timepoints
        
    return indices_one_session.astype(int)


# returns next index and value in an array which is not -1
def find_next_real_nr(indices_for_timepoints):
    
    for nr, value in enumerate(indices_for_timepoints):
        if value != -1:
            return nr, value
        
def find_previous_real_nr(indices_for_timepoints):
    
    for nr, value in enumerate(reversed(indices_for_timepoints)):
        if value != -1:
            return len(indices_for_timepoints) - nr, value
        
        
def replace_with_prior_nr(indices_for_timepoints):
    
    for index, value in enumerate(indices_for_timepoints):
        if value == -1:
            prior_value = indices_for_timepoints[index-1]
            indices_for_timepoints[index] = prior_value
            
    return indices_for_timepoints
        
    
def replace_minus_ones(indices_for_timepoints):
    
    # replace initial -1s with first real index
    if indices_for_timepoints[0] == -1:
        
        index, nr = find_next_real_nr(indices_for_timepoints)
        indices_for_timepoints[0:index] = np.repeat(nr, index)
     
    # replace final -1s with last real index
    if indices_for_timepoints[-1] == -1:
        
        index, nr = find_previous_real_nr(indices_for_timepoints)
        indices_for_timepoints[index:] = np.repeat(nr, len(indices_for_timepoints) - index)
    
    indices_for_timepoints = replace_with_prior_nr(indices_for_timepoints)
    
    return indices_for_timepoints


def get_onsets_offsets_per_run(dataset:str, subject:int, session:int, run:int, task='0', phonemes=True,
                        only_word_inital_phonemes=True):
    '''
    Params:
    - raw_data object
    - dataset: dataset for which the offsets are supposed to be loaded: Gwilliams, Armani or sherlock
    - subject: subject for which the offsets are supposed to be loaded
    - session: session for which the offsets are supposed to be loaded
    - run: run for which the offsets are supposed to be loaded
    - only_word_inital_phonemes: whether or not to get only word-inital phonemes
    
    Returns:
    - pandas data frame with at least with 3 columns: phoneme, onset  
    
    '''
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
        
        #convert times to audio onset times:
        df_phonemes.onset = df_phonemes.onset - audio_onset
        
        # add a column with the offsets:
        offsets               = df_phonemes.onset + df_phonemes.duration
        df_phonemes['offset'] = offsets
        
    if dataset == 'Gwilliams':
        
        path_dir = '/project/3018059.03/Lingpred/data/Gwilliams/'
        file_name = 'annotation_task_'+ task + '.tsv'
        
        # read pandas DataFrame and keep only sentences (not word lists):
        annotations = pd.read_csv(path_dir+file_name, sep='\t')
        annotations = annotations[annotations['condition']=='sentence']
        
        # keep only this run
        story_id    = [float(run)]
        df_story    = annotations[annotations.sound_id.isin(story_id)]
        
        # get list with word_onsets for this run (i.e. story uid):
        df_words    = df_story[df_story['kind']=='word']
        word_onsets = df_words.start
        
        #re-name start column:
        df_words['onset'] = df_words.start
        
        # add a column with the offsets:
        offsets            = df_words.onset[1:].to_list()+[df_words.onset.to_list()[-1]+0.3]
        df_words['offset'] = offsets
        
        # get only phonemes 
        df_phonemes = df_story[df_story['kind']=='phoneme']
        
        #re-name start column:
        df_phonemes['onset'] = df_phonemes.start
        
        # add a column with the offsets:
        offsets               = df_phonemes.onset[1:].to_list()+[df_words.onset.to_list()[-1]+0.08]
        df_phonemes['offset'] = offsets
        
        # and now only keep the word_initial phonemes:
        if only_word_inital_phonemes:
            df_phonemes = df_phonemes[df_phonemes.start.isin(word_onsets)]
        
    else: raise ValueError('Dataset variable must be "Armani" or Gwilliams')
    
    if phonemes:
        return df_phonemes
    else:
        return df_words


def get_bigram_mask(df: pd.DataFrame, word_col: str = 'word') -> np.ndarray:
    """
    Returns indices of the first word in each unique bigram from a dataframe.
    
    Parameters:
    - df: pandas DataFrame containing a column of words
    - word_col: name of the column containing the words (default is 'word')
    
    Returns:
    - np.ndarray of indices corresponding to the first occurrence of each unique bigram
    """

    # create a column containing the bigram, e.g. Sherlock_Holmes
    df              = df.copy()
    df['next_word'] = df[word_col].shift(-1)
    df['bigram']    = df[word_col] + '_' + df['next_word']
    df_bigrams      = df[:-1]  # exclude last row with NaN bigram
    df_bigrams      = df[:-1].reset_index(drop=True) # reset index to make sure they are correct

    print(df.head())
    
    # get the indices used as mask:
    unique_bigram_indices = df_bigrams.drop_duplicates('bigram').index

    return unique_bigram_indices.to_numpy()


def get_phonemes_onsets_offsets(dataset:str, subject:int, session:int, run:int, task='0',
                                only_word_inital_phonemes=True):
    '''
    Params:
    - raw_data object
    - dataset: dataset for which the offsets are supposed to be loaded: Gwilliams, Armani or sherlock
    - subject: subject for which the offsets are supposed to be loaded
    - session: session for which the offsets are supposed to be loaded
    - run: run for which the offsets are supposed to be loaded
    - only_word_inital_phonemes: whether or not to get only word-inital phonemes
    
    Returns:
    - pandas data frame with at least with 3 columns: phoneme, onset  
    
    '''
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
        
        #convert times to audio onset times:
        df_phonemes.onset = df_phonemes.onset - audio_onset
        
        # add a column with the offsets:
        offsets               = df_phonemes.onset + df_phonemes.duration
        df_phonemes['offset'] = offsets
        
    if dataset == 'Gwilliams':
        
        path_dir = '/project/3018059.03/Lingpred/data/Gwilliams/'
        file_name = 'annotation_task_'+ task + '.tsv'
        
        # read pandas DataFrame and keep only sentences (not word lists):
        annotations = pd.read_csv(path_dir+file_name, sep='\t')
        annotations = annotations[annotations['condition']=='sentence']
        
        # keep only this run
        story_id    = [float(run)]
        df_story    = annotations[annotations.sound_id.isin(story_id)]
        
        # get list with word_onsets for this run (i.e. story uid):
        df_words    = df_story[df_story['kind']=='word']
        word_onsets = df_words.start
        
        # get only phonemes 
        df_phonemes = df_story[df_story['kind']=='phoneme']
        
        #re-name start column:
        df_phonemes['onset'] = df_phonemes.start
        
        # add a column with the offsets:
        offsets               = df_phonemes.onset[1:].to_list()+[0.08]
        df_phonemes['offset'] = offsets
        
        # and now only keep the word_initial phonemes:
        if only_word_inital_phonemes:
            df_phonemes = df_phonemes[df_phonemes.start.isin(word_onsets)]
    
    
    return df_phonemes

def get_runs(dataset, session, subject, task):
    
    if dataset == 'Armeni':
        if session in [1, 2, 6, 8]:
            runs = np.arange(1,8)
        if session in [3, 5, 7, 9, 10]:
            runs = np.arange(1,7)
        if session in [4]:
            runs = np.arange(1,9)
        
    if dataset == 'Gwilliams':
        if task == '0':
            runs = np.arange(0, 4)
        if task == '1':
            runs = np.arange(0, 6)
        if task == '2':
            runs = np.arange(0, 8)
        if task == '3':
            runs = np.arange(0, 12)
    
    if dataset=='Goldstein':
        runs = [0]
    return runs 


def get_words_onsets_offsets(dataset:str, subject:int, session:int, run:int, task='0', use_real_word_offsets=True):
    '''
    Params:
    - raw_data object
    - dataset: dataset for which the offsets are supposed to be loaded: Gwilliams, Armani or sherlock
    - subject: subject for which the offsets are supposed to be loaded
    - session: session for which the offsets are supposed to be loaded
    - run: run for which the offsets are supposed to be loaded
    
    Returns:
    - pandas data frame with at least with 3 columns: word, onset, offset 
    
    '''
    if dataset == 'Armeni':
        
        # handle naming of session 10: 
        if session < 10:
            sess = '00' + str(session)
        else: 
            sess = '0' + str(session)
        
        # get path to events file:
        dir_path = '../audio/Armeni/'
        #filepath = 'sub-00' + str(subject) +'/' + 'ses-' + sess +'/'+ 'meg/'
        filename = 'sub-00' + str(subject) + '_ses-' + sess + '_task-compr_events.tsv'
        
        # read pandas DataFrame for the entire session:
        annotations = pd.read_csv(dir_path+filename, sep='\t')
        
        # get list with word_onsets for this run:
        word_onset_name = ['word_onset_0{}'.format(run)]
        df_words        = annotations[annotations.type.isin(word_onset_name)]
        df_words        = df_words[df_words.value != 'sp']
        word_onsets     = df_words.onset
        
        # now look for the timing of the audio onset for this run:
        index_first_word = df_words.index[0] # index for the first word in this run
        
        for i in np.arange(index_first_word, -1, -1):     # interate from there backwards
            if annotations.iloc[i].type == 'wav_onset':   # to the most recent wave onset
                audio_onset = annotations.iloc[i].onset   # and get it's onset time
                break                                     # break out of for loop
        
        #convert times to audio onset times:
        df_words.onset = df_words.onset - audio_onset
        
        if not use_real_word_offsets:
            # add a column with the offsets:
            offsets            = [x - 0.005 for x in df_words.onset[1:].to_list()]+[df_words.onset.iloc[-1]+df_words.duration.iloc[-1]]
            df_words['offset'] = offsets
        else:
            # add a column with the offsets:
            offsets            = df_words.onset + df_words.duration
            df_words['offset'] = offsets

        df_words.rename(columns={'value': 'word'}, inplace=True)

        
    if dataset == 'Gwilliams':
        
        path_dir = '/project/3018059.03/data/Gwilliams/'
        file_name = 'annotation_task_'+ task + '.tsv'
        
        # read pandas DataFrame and keep only sentences (not word lists):
        annotations = pd.read_csv(path_dir+file_name, sep='\t')
        annotations = annotations[annotations['condition']=='sentence']
        
        # keep only this run
        story_id    = [float(run)]
        df_story    = annotations[annotations.sound_id.isin(story_id)]
        
        # get list with word_onsets for this run (i.e. story uid):
        df_words    = df_story[df_story['kind']=='word']
        
        #re-name start column:
        df_words.rename(columns={'start': 'onset'}, inplace=True)
        
        # add a column with the offsets:
        offsets            = df_words.onset[1:].to_list()+[df_words.onset.iloc[-1] + 0.08]
        df_words['offset'] = offsets
        
    if dataset == 'Goldstein':
        
        dir_path  = '../audio/Goldstein/'
        file_name =  'podcast_transcript.csv'
        filepath  = dir_path + file_name

        df_words = pd.read_csv(filepath, sep=',')
        df_words.rename(columns={'start': 'onset'}, inplace=True)

        if not use_real_word_offsets:
            # add a column with the offsets:
            offsets            = [x - 0.005 for x in df_words.onset[1:].to_list()]+[df_words.end.iloc[-1]]
            df_words['offset'] = offsets

            #sometimes the two people talk over one another, to avoid negative new_offset times I will replace them with the value in 'end'
            df_words['offset'] = np.where(df_words["offset"] < df_words["onset"], df_words["end"], df_words["offset"])

        else:
            df_words.rename(columns={'start': 'onset', 'end':'offset'}, inplace=True)

    return df_words




def get_times(ref_word_index, on_offsets):
    '''
    - ref_word_index: index of the reference word (onset = 0)
    - on_offsets: 2d array of dimensions (nr_words, 2) in which each row contains the onset & offset
    '''
    
    times_100 = np.arange(-1950, 1951, 25)/1000
    
    point_zero = on_offsets[ref_word_index][0]
    
    times = point_zero + times_100
    
    return times


# for each timepoint look at each row and decide whether that timepoint falls in the interval [onset, offset]
def get_indices_for_timepoints(times, on_offsets):
    '''
    Returns:
    --------
    array of length 157 with the indices of words heard for each timepoint. 
    If there was silence, the array contains the value -1
    
    Params:
    -------
    - times:      timepoints for the reference word onset +/- 2s
    - on_offsets: 2d array with on/offset of words in this session
    
    '''
    indices_for_timepoints = np.repeat(-1, 157)
    
    for nr, timepoint in enumerate(times):
        
        # search only close to the reference index (time reasons):
        for index, on_offset in enumerate(on_offsets):
                
            onset = on_offset[0]
            offset = on_offset[1]
            
            if timepoint >= onset:
                if timepoint <= offset:
                    indices_for_timepoints[nr] = (index)
                    continue
                
    return indices_for_timepoints


def make_y_matrix_per_run(X, indices, acoustic_model=False, use_random_vector_at_tp_zero=False):
    
    # initialise random generator:
    rng = np.random.default_rng(42)
    
    # initialise y 
    y = np.empty(shape=(indices.shape[0], indices.shape[1], X.shape[1], )) # n_words, n_timepoints, dim
    
    if acoustic_model: # ATTENTION: this here was only used in an earlier Phoneme model
        
        for word_index in range(indices.shape[0]):

            for nr, time_index in enumerate(indices[word_index]):

                onset_index = indices[word_index][78] # get the phoneme at timepoint 0 == [78]

                if onset_index == time_index:
                    random_vector = rng.normal(0, 1,size=(X.shape[1]))
                    y[word_index][nr] = random_vector
                else:
                    y[word_index][nr] = X[time_index]
        
        return y     
    
    for word_index in range(indices.shape[0]):

        for nr, time_index in enumerate(indices[word_index]):

            if use_random_vector_at_tp_zero:
                if word_index == time_index:
                    random_vector = rng.normal(0, 1, size=(X.shape[1]))
                    y[word_index][nr] = random_vector
                else:
                    y[word_index][nr] = X[time_index]
            else:
                y[word_index][nr] = X[time_index]
    return y


# function to make a pickle file name for saving based on the parameters:
def make_filename(dataset, subject, session, band, layer, Goldstein, datatype, channels, baseline, window_size):
    
    folder = '/project/3018059.03/Lingpred/results/' + dataset + '/'
    
    filename = 'subject-' + str(subject) + '_session-' + str(session) + '_band-' + str(band) +'_layer-' + str(layer) + '_goldstein-' + str(Goldstein) + '_datatype-' + datatype + '_channels-' + channels + 'baselinecorr-'+ str(baseline)+ '_window_size-'+ str(window_size) +'.pkl' 
    
    full_path = folder + filename
    
    return full_path
        
# loading a pickle file     
def load_data(dataset, subject, session, band, layer, Goldstein, datatype, channels, baseline, window_size):
    
    file_path = make_filename(dataset, subject, session, band, layer, Goldstein, datatype, channels, baseline, window_size)
    
    data = pickle.load(open(file_path, "rb" ))
    
    return data