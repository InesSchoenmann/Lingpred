import sys
import librosa
import numpy as np
import pandas as pd
import pickle

def word_avg_sg(subject:int, session:int, task= '0', dataset='Armani', n_mels=8, use_real_word_offsets=True):
    '''
    Params:
    -------
    - Dataset: 'Armeni' or Gwilliams or Goldstein
    - Subject
    - Session
    - n_mels: nr of bands for the mel spectogram
    
    Returns:
    --------
    Array of shape (nr_words, n_mels+1)
    containing the spectogram information averaged over time for each mel band 
    '''
    avg_sg_all_runs = np.empty(shape=(0, n_mels+1)) # +1 for the variance 


    if dataset == 'Armeni':
        audio_dir = '../audio/Armeni/'
        # sampling rate MEG data:
        sr_meg = 1200
        
    if dataset == 'Gwilliams':
        audio_dir = '../audio/Gwilliams/'
        # sampling rate MEG data:
        sr_meg = 1000

    if dataset=='Goldstein':
        audio_dir = '../audio/Goldstein/'
        sr_meg    = 512
        
    # get runs in this session
    runs = get_runs(dataset, session, subject, task)

    
    for run in runs:
        
        print(run)
        
        # get onset times of the initial phonemes adjusted to onset of the audiofile:
        df_words = get_words_onsets_offsets(dataset, subject, session, run, task, use_real_word_offsets=use_real_word_offsets)
        
        
        # audio file name 
        if session<10:
            audio_run = audio_dir + '0{}_{}.wav'.format(session, run)
        else:
            audio_run = audio_dir + '{}_{}.wav'.format(session, run)
            
        if dataset == 'Gwilliams':
            audio_run = audio_dir + df_words.sound.unique()[0]

        if dataset == 'Goldstein':
            audio_run = audio_dir + 'monkey_and_horse_corrected.wav'

        # waveform (scale) and sampling rate (sr)
        scale, sr = librosa.load(audio_run, sr=sr_meg*18)

        # make spectrogram 
        mel_sg   = librosa.feature.melspectrogram(y=scale, sr=sr, hop_length=int(sr/sr_meg), 
                                                  n_mels=n_mels)
        lm_sg    = librosa.power_to_db(mel_sg)
        lm_time  = np.arange(1,lm_sg.shape[1]+1)/sr_meg
        
        # average over the duration of the phoneme for each band:
        # resulting array is of shape (nr_phonemes, nr_bands)
        discrete_events = discretise_events(lm_sg, lm_time, onset_df=df_words)
        
        # stack for all runs:
        avg_sg_all_runs = np.vstack((avg_sg_all_runs, discrete_events))
        
    return avg_sg_all_runs

def make_acoustic_y_matrix(subject:int, session:int, task= '0', dataset='Armani', n_mels=8,
                        only_word_inital_phonemes=True):
    ''''
    Creates a Mel Spectogram with n_mels + the envelope and that resembles the shape of the neural data. 
    Hence, it will have the shape (n_mels+1, n_words, n_timepoints)
    '''
    if dataset == 'Armeni':
        audio_dir = '../audio/Armeni/stimuli/'
        # sampling rate MEG data:
        sr_meg = 1200
        
    if dataset == 'Gwilliams':
        audio_dir = '../audio/Gwilliams/'
        # sampling rate MEG data:
        sr_meg = 1000
    
    y_matrix_all_runs = np.empty(shape=(0, n_mels+1, sr_meg*4)) # +1 for the variance 
    
    # get runs in this session
    runs = get_runs(dataset, session, subject, task)

    for run in runs:
        
        print(run)
        # get onset times of the initial phonemes adjusted to onset of the audiofile:
        df_phonemes = get_phonemes_onsets_offsets(dataset, subject, session, run, task, only_word_inital_phonemes)
        
        # audio file name 
        if session<10:
            audio_run = audio_dir + '0{}_{}.wav'.format(session, run)
        else:
            audio_run = audio_dir + '{}_{}.wav'.format(session, run)
            
        if dataset == 'Gwilliams':
            audio_run = audio_dir + df_phonemes.sound.unique()[0]

        # waveform (scale) and sampling rate (sr)
        scale, sr = librosa.load(audio_run, sr=sr_meg*18)

        # make spectrogram 
        mel_sg   = librosa.feature.melspectrogram(y=scale, sr=sr, hop_length=int(sr/sr_meg), 
                                                  n_mels=n_mels)
        lm_sg    = librosa.power_to_db(mel_sg)
        lm_time  = np.arange(1,lm_sg.shape[1]+1)/sr_meg

        # average over the duration of the phoneme for each band:
        # resulting array is of shape (nr_epochs, nr_bands+1, nr_timepoints)
        epochs = epoch_events(lm_sg, lm_time, onset_df=df_phonemes, sr_meg=sr_meg)
        
        # stack for all runs:
        y_matrix_all_runs = np.vstack((y_matrix_all_runs, epochs))
        
    return y_matrix_all_runs

def init_phoneme_avg_sg(subject:int, session:int, task= '0', dataset='Armeni', n_mels=128, 
                        only_word_inital_phonemes=True):
    '''
    Params:
    -------
    - Dataset: 'Armeni' or Gwilliams
    - Subject
    - Session
    - n_mels: nr of bands for the mel spectogram
    - power
    - only_word_inital_phonemes: whether or not only to consider word-inital phonemes
    
    Returns:
    --------
    Array of shape (nr_initial_phonemes, n_mels)
    containing the spectogram information averaged over time for each mel band 
    '''
    avg_sg_all_runs = np.empty(shape=(0, n_mels+1)) # +1 for the variance 
    
    if dataset == 'Armani':
        audio_dir = '../audio/Armeni/stimuli/'
        # sampling rate MEG data:
        sr_meg = 1200
        
    if dataset == 'Gwilliams':
        audio_dir = '../audio/Gwilliams/'
        # sampling rate MEG data:
        sr_meg = 1000
        
    # get runs in this session
    runs = get_runs(dataset, session, subject, task)

    
    for run in runs:
        
        print(run)
        
        # get onset times of the initial phonemes adjusted to onset of the audiofile:
        df_phonemes = get_phonemes_onsets_offsets(dataset, subject, session, run, task, only_word_inital_phonemes)
        
        
        # audio file name 
        if session<10:
            audio_run = audio_dir + '0{}_{}.wav'.format(session, run)
        else:
            audio_run = audio_dir + '{}_{}.wav'.format(session, run)
            
        if dataset == 'Gwilliams':
            audio_run = audio_dir + df_phonemes.sound.unique()[0]

        # waveform (scale) and sampling rate (sr)
        scale, sr = librosa.load(audio_run, sr=sr_meg*18)

        # make spectrogram 
        mel_sg   = librosa.feature.melspectrogram(y=scale, sr=sr, hop_length=int(sr/sr_meg), 
                                                  n_mels=n_mels)
        lm_sg    = librosa.power_to_db(mel_sg)
        lm_time  = np.arange(1,lm_sg.shape[1]+1)/sr_meg
        
        # average over the duration of the phoneme for each band:
        # resulting array is of shape (nr_phonemes, nr_bands)
        discrete_events = discretise_events(lm_sg, lm_time, onset_df=df_phonemes)
        
        # stack for all runs:
        avg_sg_all_runs = np.vstack((avg_sg_all_runs, discrete_events))
        
    return avg_sg_all_runs


def epoch_events(spectogram, times, onset_df, sr_meg):
    '''
    Parameters
    ----------
    - spectogram: librosa mel spectrogram in db
        spectrogram of a given run, of shape (n_mels, sampling_rate*seconds) with sampling_rate = MEG_sr *18
    - times: numpy array
        containing the times in seconds corresponding to each time point in the spectogram
    - onset_df: pandas DataFrame
        containing a column 'onset' and 'offset' indicating the onset of each word-initial phoneme in seconds

    Returns
    -------
    - numpy ndarray of shape (n_epochs, n_mels+1, n_timepoints) with n_timepoints = 4s * MEG_sampling_rate
        containing the envelope and the mels for each epoch
    '''

    audio={'audio':spectogram,'times':times}
        
    # np array of shape (onsets, bands, time_points) 
    bands = np.zeros((len(onset_df),audio['audio'].shape[0], sr_meg*4))
    var   = np.zeros(shape=(len(onset_df), sr_meg*4))

    for ph_i,ph_row in enumerate(onset_df.iterrows()):
            
        # make a logical array for audio timepoints that are part of the epoch [onset-2s, onset+2s):
        samps_ix=np.logical_and(audio['times']>ph_row[1]['onset']-2,
                                audio['times']<ph_row[1]['onset']+2)
        
        # get samples for this epoch:
        temp = audio['audio'][:,samps_ix]

        # check if there are the right amount of samples for the epoch, i.e. 4s * MEG_sampling rate
        # and pad with zeros either to the left (beginning of audio run) or right (end of audio run)
        if temp.shape[1] < sr_meg*4:
            if ph_i < len(onset_df)/2: 
                temp = pad_left(temp, sr_meg*4)
            else:
                temp = pad_right(temp, sr_meg*4)
        
        bands[ph_i]= temp

    # compute variance over bands for each time point:
    var = np.var(bands, axis=1)

    # swap axis so mels are dimension 0  
    bands = np.swapaxes(bands, 0, 1)

    # make array to hold variance and bands, has shape (n_mels+1, n_epochs, n_timepoints)
    band_stats     = np.zeros(shape=(bands.shape[0]+1, bands.shape[1], bands.shape[2]))
    band_stats[0]  = var
    band_stats[1:] = bands
        
    # scrub nans 
    band_stats[np.isnan(band_stats)]=0

    # swap axes again to have epochs first ... easier for stacking:
    band_stats = np.swapaxes(band_stats, 0, 1)
    
    return(band_stats)


def discretise_events(spectogram, times, onset_df):

    # make dictionary of spectogram and meg sampled time points 
    audio={'audio':spectogram,'times':times}
    
    # np array of shape (onsets, bands) for the stats of each band 
    band_means = np.zeros((len(onset_df),audio['audio'].shape[0]))
    var        = np.zeros(shape=(len(onset_df)))

    for ph_i,ph_row in enumerate(onset_df.iterrows()):
        
        # make a logical array for audio timepoints that are part of the phoneme:
        samps_ix=np.logical_and(audio['times']>ph_row[1]['onset'],
                            audio['times']<ph_row[1]['offset'])
        
        # Average & variance over these timepoints for each band:
        band_means[ph_i,:]= np.mean(audio['audio'][:,samps_ix],axis=1)
        var[ph_i]         = np.var(audio['audio'][:,samps_ix])
        
        # Add as columns
        band_stats = np.c_[band_means, var]
        
    # scrub nans 
    band_stats[np.isnan(band_stats)]=0
    
    return(band_stats)


def get_phonemes_onsets_offsets(dataset:str, subject:int, session:int, run:int, task='0',
                                only_word_inital_phonemes=True):
    '''
    Params:
    - raw_data object
    - dataset: dataset for which the offsets are supposed to be loaded: Gwilliams, Armeni or sherlock
    - subject: subject for which the offsets are supposed to be loaded
    - session: session for which the offsets are supposed to be loaded
    - run: run for which the offsets are supposed to be loaded
    - only_word_inital_phonemes: whether or not to get only word-inital phonemes
    
    Returns:
    - pandas data frame with at least with 3 columns: phoneme, onset  
    
    '''
    if dataset == 'Armeni':
        
        # handle naming of session 10: 
        if session < 10:
            sess = '00' + str(session)
        else: 
            sess = '0' + str(session)
        
        # get path to events file:
        dir_path = '../data/Armeni/'
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
        
        path_dir = '../data/Gwilliams/'
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
    - dataset: dataset for which the offsets are supposed to be loaded: Gwilliams, Armeni or sherlock
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
        
        path_dir = '../data/Gwilliams/'
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


