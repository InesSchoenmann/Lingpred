import scipy
import sys
from pathlib import Path
from pdb import set_trace
import h5py
import mne_bids
import mne
import numpy as np



PROJ_ROOT = '/project/3018059.03/'

def create_sensor_data(subjects:list, 
                       sessions: list, 
                       destination_dir = '/project/3018059.03/Lingpred/data/Armani/',
                       source_dir = '/project/3018059.03/Lingpred/data/Armani/',
                       task = 'compr',
                       bandpass = [0.1, 40],
                      dataset = 'Armani'):
    '''
    Params: 
    - list of subjects 
    - list of sessions per subject
    - destination directory in which to save the filtered source data,
    - source directory from which to take the raw mne bids data
    - task: task for the mne bids path 
    
    Returns:
    - filtered MNE RAW object, saved at location destination_path + subject/session/meg/
    '''
    
    mne.set_log_level(verbose='error') # in order to suppress all those warnings about misc channels
    
    for sub in subjects:

        for sess in sessions:

            src_filt_info=load_spatial_filter(sub, sess) # load parcellation & label

            # Loading RAW sensor data using the MNE bids system:
            root = source_dir
            task = task
            datatype = 'meg'
            
            # set correct the session & subject names for the Armani and Gwilliams data:
            if task =='compr': # in the Armani dataset subjects are indexed 001, 002, 003
                subject = '00' + str(sub)
                if sess < 10:
                    session = '00' + str(sess)
                else:
                    session = '0' + str(sess)
                    
            else:             # in the Gwilliams dataset subjects are indexed 01, 02, ...
                if sub < 10: 
                    subject = '0' + str(sub)
                else: 
                    subject = str(sub)
                    
                session = str(sub)

            # make path to the raw files 
            bids_path = mne_bids.BIDSPath(subject=subject, 
                                          session=session, 
                                          task=task, 
                                          datatype=datatype, 
                                          root=root) 
            
            
            # channels to pick for the source reconstruction:
            if dataset=='Armani':
                raw = mne.io.read_raw_ctf(bids_path)  

                # get info about good and bad channels and adjust to naming in CTF:
                bad_and_good_ch = load_bads(sub, sess)

                picks = []

                for nr, channel in enumerate(raw.ch_names):
                    if '-' in channel:
                        name, loc = channel.split('-') 
                        if name in bad_and_good_ch['good']:
                            picks.append(channel)

                assert len(picks) == len(bad_and_good_ch['good'])
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # the picks is not yet decided/done for the Gwilliams data 
            else: 
                raw = mne_bids.read_raw_bids(bids_path)  
                picks = 'mag'

            
            # only keep good channels & load raw data into memory:
            raw.pick_channels(picks)
            raw.load_data()
            

            if not np.any(np.isnan(raw.get_data())): # if there are no NaNs in the data:

                # filter the data: bandpass from 0.1 to 40 Hz.
                raw_filtered = raw.filter(bandpass[0], bandpass[1], picks=raw.ch_names, phase='zero-double')
                del raw # delete raw to save memory

                # project onto sensor:
                sensor_data = src_filt_info['F_parc'] @ raw_filtered.get_data()
                print('Shape of sensor data for subject {}, session {} is : '.format(sub, sess), sensor_data.shape)
                print(raw_filtered.info)
                
                # make MNE RAW object:
                info = mne.create_info(ch_names = src_filt_info['label'], sfreq = raw_filtered.info['sfreq'])
                sensor_raw = mne.io.RawArray(sensor_data, info)
                del raw_filtered # delete raw_filtered to save memory
                
                # file name, e.g.: 1-1_lcmv_data_01-40_raw.fif:
                fname = str(sub) + '-' + str(sess) + '_lcmv-data_' + str(bandpass[0]) + '-' + str(bandpass[1]) + 'raw.fif'
                dir_path = destination_dir + 'sub-' + subject + '/ses-' + session + '/source/'
                path_fname =  dir_path + fname
                
                if not Path(dir_path).is_dir():
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                # save sensor data:
                sensor_raw.save(path_fname)
                
            else:
                raise ValueError('There are NaNs in the data for subject {}, session {}'.format(sub, sess))
            
            del sensor_raw # delete sensor_raw to save memory

            
            
def filter_data(subjects:list, 
                task: str,
                sessions = [0], 
                destination_dir = '/project/3018059.03/Lingpred/data/Gwilliams/',
                source_dir = '/project/3018059.03/Lingpred/data/Gwilliams/derived/',
                bandpass = [0.1, 40],
                dataset = 'Gwilliams'):
    '''
    Params: 
    - list of subjects 
    - list of sessions per subject, session 0 is enough for the Gwilliams data
    - destination directory in which to save the filtered source data,
    - source directory from which to take the raw mne bids data
    - task: task for the mne bids path: for Gwilliams this is: '0', '1', '2' or '3' 
            0 = lw1, 1 = cable spool fort, 2 = easy money, 3 = black willow
    
    Returns:
    - filtered MNE RAW object, saved at location destination_path + subject/session/meg/
    '''
    
    mne.set_log_level(verbose='error') # in order to suppress all those warnings about misc channels
    
    for sub in subjects:

        for sess in sessions:

            # Loading RAW sensor data using the MNE bids system:
            root = source_dir
            task = task
            datatype = 'meg'
            # in the Gwilliams dataset subjects are indexed 01, 02, ...
            if sub < 10: 
                subject = '0' + str(sub)
            else: 
                subject = str(sub)
                    
            session = str(sess)

            # make path to the raw files 
            bids_path = mne_bids.BIDSPath(subject=subject, 
                                          session=session, 
                                          task=task, 
                                          datatype=datatype, 
                                          root=root) 
            
            
            # the pick only Magnetometers (exclude reference and misc channels)
            raw = mne_bids.read_raw_bids(bids_path)  
            picks = []

            for nr, channel in enumerate(raw.ch_names):
                if raw.get_channel_types()[nr] == 'mag':
                    picks.append(channel)

            
            # only keep good channels & load raw data into memory:
            raw.pick_channels(picks)
            raw.load_data()
            

            if not np.any(np.isnan(raw.get_data())): # if there are no NaNs in the data:

                # filter the data: bandpass from 0.1 to 40 Hz.
                raw_filtered = raw.filter(bandpass[0], bandpass[1], picks=raw.ch_names, phase='zero-double')
                print(raw_filtered.info)
                del raw
                
                # file name, e.g.: 1-0_filtered_data_01-40_task-0_raw.fif:
                fname = str(sub) + '-' + str(sess) + '_filtered_data_' + str(bandpass[0]) + '-' + str(bandpass[1]) + '_task-' + task + '_' 'raw.fif'
                dir_path = destination_dir + 'sub-' + subject + '/ses-' + session + '/filtered/'
                path_fname =  dir_path + fname
                
                if not Path(dir_path).is_dir():
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                # save sensor data:
                raw_filtered.save(path_fname)
                
            else:
                raise ValueError('There are NaNs in the data for subject {}, session {}'.format(sub, sess))
            
            del raw_filtered # delete sensor_raw to save memory
            
            

def load_spatial_filter(sub_i:int,sess_i:int):
    """from given subject and session number, load the parcellated spatial filter.
    
    in:
    - sub,sess
    out:
    source_info:dict
        dictionary with keys:
        -label:  list of strings, with labels of each parcel.
        -F_parc: parcellated spatial filter, (N_sources X N_sensors)
        
    ------
    NB: In subject-3, session[3,4,8], N_sensors in the spatial filter is less than 269!
    This is because there are bad channels omitted. It is important to check that this is also the case in the
    sensor level data, otherwise remove them manually (see how to load "good" vs "bad" sensors, below)
    """
    this_derdir=(PROJ_ROOT / 'data/sherlock/derived/' f'sub-00{sub_i}'
                 / ('ses-0' + f'{sess_i}'.zfill(2) )/'meg')
    this_spatfilt_file=this_derdir/ f'{sub_i}-{sess_i}_lcmv-filt-parc_1-40_readable.mat'
    source_filt_info=loadmat_struct2dict(this_spatfilt_file)
    source_filt_info['label']=[ar[0] for ar in source_filt_info['label']]
    return(source_filt_info)



def load_bads(sub_i:int,sess_i:int):
    """load the bad channel info"""
    this_derdir=(PROJ_ROOT / 'data/sherlock/derived/' f'sub-00{sub_i}'
                 / ('ses-0' + f'{sess_i}'.zfill(2) )/'meg')
    chansel_info=loadmat_struct2dict(this_derdir/'chansel.mat')
    for k in chansel_info: # reformat awkward arrays as lists of strings
        chansel_info[k]=[ar[0] for ar in chansel_info[k]]
    return(chansel_info)



####################################################################################
############################ dict TOOLS ##########################################
####################################################################################
def loadmat_struct2dict(fname,structnames=None):
    """load structs from matlab v6 .mat files into dicts
    in:
    -fname (str or posixpath)
    -structnames(None or list of strings) 
        if you want a specific struct instead of all vars
    """
    def _getdct(fl_open,strctnm):
        "dict maker"
        tmp=fl_open[strctnm][0,0]
        dct_out={nm:getattr(tmp,nm) for nm in tmp._fieldnames} # get rid of redundant dims
        return({k:v.squeeze()  if isinstance(v,np.ndarray) else v for k,v in dct_out.items()})
    
    # open file & get name(s) of struct(s)
    fl=scipy.io.loadmat(fname,struct_as_record=False)
    if not structnames: structnames=[k for k in list(fl.keys()) if '__' not in k]

    all_dcts={struct_name:_getdct(fl,struct_name) for struct_name in structnames}
    
    # return nested if more than one struct found; else only a dict
    if len(all_dcts)==1:return(all_dcts[structnames[0]])
    else: return(all_dcts)
