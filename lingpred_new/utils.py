import sys
import os 
from pathlib import Path

import pickle



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