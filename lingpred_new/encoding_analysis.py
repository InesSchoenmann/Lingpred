import sys
import os 
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy import stats
import spacy
from sklearn.preprocessing import StandardScaler

# make sure you add the path of the Lingpred folder to be able to import the gpt2 module:
#sys.path.append(os.path.abspath(".."))
#from gpt2 import model  # or use `import gpt2` if you import the whole package




# make arbitrary static vectors: values in vectors come from a normal distribution with mean=0.1, and SD=1.1
# these values were chosen as vectors are standard scaled in each fold of the ridge regression 
def make_arbitrary_static_vectors(annotation_df, dim):
    
    # get word in neural data:
    all_words            = [i.lower() for i in annotation_df.word.to_numpy()]
    unique_words         = np.unique(all_words)
    
    # make word specific vectors: 
    rng = np.random.default_rng(42)
    arbitrary_embeddings = rng.normal(0.1, 1.1, size=(unique_words.shape[0], dim))
    dictionary           = dict(zip(unique_words, arbitrary_embeddings))
    
    #initialise empty array:
    arbitrary_vectors = np.empty(shape=(len(all_words), dim))
    
    # look-up in dictionary and fill in 
    for i in range(len(all_words)):
        
        word      = all_words[i]
        embedding = dictionary['{}'.format(word)]
        
        arbitrary_vectors[i] = embedding
        
    return arbitrary_vectors   

def regress_out_X(X, y, alpha=5000):
    '''
    Returns an y with X regressed out

    '''
    print('y has shape: ', y.shape)
    print('First dimension should be channels!')

    assert y.shape[0] in [100, 208]

    residuals = np.empty_like(y)

    for nr, data in enumerate(y):
        # fit Ridge:
        regression = Ridge(alpha=alpha, fit_intercept=True)
        regression.fit(X, y[nr])

        y_hat         = regression.predict(X)
        residuals[nr] = y[nr] - y_hat 
    
    return residuals

def regress_out_one(X, alpha=5000):
    '''
    Returns an X matrix with the previous vectors regressed out
    The returned matrix is one word smaller (as there is no previous vector for v_0)
    '''
    
    X_matrix = X[:-1]  # drop last
    y        = X[1:]   # drop first

    # fit Ridge:
    regression = Ridge(alpha=alpha, fit_intercept=True)
    regression.fit(X_matrix, y)

    y_hat     = regression.predict(X_matrix)
    residuals = y - y_hat 
    
    return residuals



# computes the brainscore without saving coefficients per channel:
def brainscore_no_coef_per_channel(X, y, alpha, print_ind, test_mask = None):
    '''
    Params: 
    - X: X matrix for ridge regression of shape (nr of words, GPT-dim + 1)
    - y: neural data from one channel with x nr of lags per epoch. Shape: (nr of words, nr of lags)
    
    Returns:
    - list: correlations between GPT embeddings and neural data per lag
    - array of shape (GPT-dim+1 * 10, nr_lags) containing the coefficients for all 10 folds
    '''
    kf = KFold(n_splits=10)
    
    # initialise arry to hold the correlations for each fold
    corr_all_folds = np.empty(shape=(10, y.shape[1])) # y.shape[1] = nr. of lags

    for nr, (train_index, test_index) in enumerate(kf.split(X)):
        
        if type(test_mask) == list:
            
            # set all but the test indices for this fold to False :
            test_index_mask             = np.repeat(False, len(test_mask)) # initialise new list
            test_index_mask[test_index] = np.array(test_mask)[test_index]  # populate with booleans from mask at given indices
            
            # split in training and testing data using the mask only for testing
            X_train = X[train_index]
            y_train = y[train_index]
            X_test = X[test_index_mask]
            y_test = y[test_index_mask]
            
            if print_ind == 0:
                print('We are in channel 0 and in fold', nr)
                print('You split only testing data according to the mask provided')
                print('Hence X_train and Y_train have dimensions:', X_train.shape, y_train.shape)
                print('While X_test and Y_test have dimensions:', X_test.shape, y_test.shape)
                print('')
        
        
        else:
            
            # split in training and testing data
            X_train = X[train_index]
            y_train = y[train_index]
            X_test  = X[test_index]
            y_test  = y[test_index]

        
        # initialise Standard Scaler:
        scaler = StandardScaler()
        
        # scale for each fold, such that for X, each dim has mean of 0 and SD of 1:
        fit     = scaler.fit(X_train)
        X_train = fit.transform(X_train)
        fit     = scaler.fit(X_test)
        X_test  = fit.transform(X_test)
              

        # fit Ridge:
        regression = Ridge(alpha=alpha, fit_intercept=True)
        regression.fit(X_train, y_train)
        
        y_hat     = regression.predict(X_test)
        
        # transpose: as correlation is computed for each lag
        y_hat  = y_hat.T
        y_test = y_test.T
        
        # compute correlation for each lag:
        corr_list = []
        for lag in range(y_test.shape[0]):
            r = stats.pearsonr(y_hat[lag], y_test[lag]).statistic
            corr_list.append(r)

        # compute correlation for each lag:
        corr_all_folds[nr] = corr_list
            
    return corr_all_folds


# computes the brainscore without saving coefficients (due to memory concerns) for all channels:
def brainscore_no_coef(X, y, test_mask = None, alpha=5000):
    '''
    Params: 
    - X: X matrix for ridge regression of shape (nr of words, GPT-dim + 1)
    - y: neural data from all channel with 153 lags per epoch. Shape: (nr of channels, nr of words, 153)
    - test_mask: a list to be used as mask when splitting during the testing fold, i.e. is_top_1_prediction
    
    Returns:
    - 2D array: correlations between GPT embeddings and neural data per lag for each channel. Shape: (nr channels, 153)
    - 3D array: coefficents for all channels and lags. Shape: (nr_channels, (GPT-dim+1)*10, nr_lags)
    '''
    
    if test_mask is not None and not isinstance(test_mask, list):
        raise ValueError("test_mask must be of type list.")
    
    print('We are now in the brainscore_no_coef method, and y has shape:')
    print(y.shape)
    
    corr_all_channels   = np.empty(shape=(y.shape[0], 10, y.shape[2]))  # array of shape (100, 10, nr_lags)

    channels = range(y.shape[0])  # y.shape[0] = nr. of channels

    for i, channel in enumerate(channels):
        y_ch                       = y[channel] # get one channel
        corr_all_channels[channel] = brainscore_no_coef_per_channel(X, y_ch, alpha=alpha, print_ind = i, test_mask=test_mask) # compute brainscore per channel
        
    return corr_all_channels


