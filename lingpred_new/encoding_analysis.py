import sys
import os 
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy import stats
import spacy
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
import pandas as pd

# make sure you add the path of the Lingpred folder to be able to import the gpt2 module:
sys.path.append(os.path.abspath("/project/3018059.03/Lingpred"))
sys.path.append(os.path.abspath("/Users/ines/research/Lingpred"))
import gpt2
import lingpred_new

# wrapper functions for the Armeni dataset for the different models:
def compute_corr(subjects:list, sessions:list, model:str, regress_out=True, bigrams_removed=False):

    warnings.simplefilter(action='ignore')

    model = model.lower() # make sure the spelling is all lower case
    
    if model=='glove':
        compute_corr_glove(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)
    if model=='arbitrary':
        compute_corr_arbitrary(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)
    if model=='gpt':
        compute_corr_gpt(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)


def compute_corr_glove(subjects, sessions, regress_out=True, bigrams_removed=False):

    project_dir = '/project/3018059.03/Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armani/sub-00{}/'.format(subject)
        file_path   = subject_dir + 'y_all_language.pkl'
        y_all       = pickle.load(open(file_path, 'rb'))

        if subject == 3: sessions = [1, 2, 3, 4, 5, 6, 7, 9, 10]

        # get dataframe to make Glove embeddings:
        df_words_all = pd.DataFrame()

        for session in sessions:
            df_words = lingpred_new.io.get_words_onsets_offsets(None, 
                                                        dataset='Armani', 
                                                        subject=subject, 
                                                        session=session, 
                                                        run=1)
            df_words_all = df_words_all.append(df_words)

        print(len(df_words_all))

        # get GloVe vectors from spacy:
        # spacy.prefer_gpu()
        nlp           = spacy.load('en_core_web_lg')
        Glove_vectors = np.vstack([nlp(word).vector for word in df_words_all.word]) # np array of shape (nr_words, 300)
        Glove_vectors = np.array(Glove_vectors) # transform from cupy to numpy array for stacking later on 

        if subject == 3: 
            Glove_vectors = Glove_vectors[9:]
            y_all = np.swapaxes(y_all, 0, 1)

        print(y_all.shape, Glove_vectors.shape)

        if regress_out and bigrams_removed:
            bigram_mask   = lingpred_new.utils.get_bigram_mask(df_words_all)
            y_all         = y_all[:,bigram_mask,:] # drop bigrams
            y_all         = y_all[:,1:,:]          # drop first
            Glove_vectors = regress_out_one(Glove_vectors)
            # now I need to adjust the bigram mask so that all integers are one smaller and negative ints are dropped:
            bigram_mask   = [x - 1 for x in bigram_mask if x - 1 >= 0]
            Glove_vectors = Glove_vectors[bigram_mask]
            print(y_all.shape, Glove_vectors.shape)

        elif regress_out:
            Glove_vectors = regress_out_one(Glove_vectors)
            # swapaxes such that words is dimension 0, drop the first row and swap back
            y_all = np.swapaxes(np.swapaxes(y_all, 0, 1)[1:], 0, 1)
            print(y_all.shape, Glove_vectors.shape)
        
        elif bigrams_removed:
            # get bigram mask and drop all bigrams that repeat:
            bigram_mask   = lingpred_new.utils.get_bigram_mask(df_words_all)
            Glove_vectors = Glove_vectors[bigram_mask]
            y_all         = y_all[:,bigram_mask,:]
            print(y_all.shape, Glove_vectors.shape)

        #saving results:
        filename = 'Glove_vectors_sub_{}.pkl'.format(subject)
        if regress_out and bigrams_removed:
            filename = 'regressed_out_one_bigrams_removed_Glove_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'regressed_out_one_Glove_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'bigrams_removed_Glove_vectors_sub_{}.pkl'.format(subject)

        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(Glove_vectors,f)
        f.close()


        # compute brainscore:
        corr = brainscore_no_coef(y= y_all, X = Glove_vectors)

        #saving results:
        results_dir = project_dir + '/results/Armani/grand_average/'
        filename    = 'corr_Glove_vectors_sub_{}.pkl'.format(subject)
        if regress_out and bigrams_removed:
            filename = 'corr_regressed_out_one_bigrams_removed_Glove_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'corr_regressed_out_one_Glove_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'corr_bigrams_removed_Glove_vectors_sub_{}.pkl'.format(subject)

        file_path = results_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(corr,f)
        f.close()


def compute_corr_arbitrary(subjects, sessions, regress_out=True, bigrams_removed=False):

    project_dir = '/project/3018059.03/Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armani/sub-00{}/'.format(subject)
        file_path   = subject_dir + 'y_all_language.pkl'
        y_all       = pickle.load(open(file_path, 'rb'))

        if subject == 3: sessions = [1, 2, 3, 4, 5, 6, 7, 9, 10]

        # get dataframe to make arbitrary embeddings:
        df_words_all = pd.DataFrame()

        for session in sessions:
            df_words = lingpred_new.io.get_words_onsets_offsets(None, 
                                                        dataset='Armani', 
                                                        subject=subject, 
                                                        session=session, 
                                                        run=1)
            df_words_all = df_words_all.append(df_words)

        print(len(df_words_all))

        arbitrary_vectors = make_arbitrary_static_vectors(df_words_all, dim = 300)

        if subject == 3: 
            arbitrary_vectors = arbitrary_vectors[9:]
            y_all = np.swapaxes(y_all, 0, 1)

        print(y_all.shape, arbitrary_vectors.shape)

        if regress_out and bigrams_removed:
            bigram_mask       = lingpred_new.utils.get_bigram_mask(df_words_all)
            y_all             = y_all[:,bigram_mask,:] # drop bigrams
            y_all             = y_all[:,1:,:]          # drop first
            arbitrary_vectors = regress_out_one(arbitrary_vectors)
            # now I need to adjust the bigram mask so that all integers are one smaller and negative ints are dropped:
            bigram_mask       = [x - 1 for x in bigram_mask if x - 1 >= 0]
            arbitrary_vectors = arbitrary_vectors[bigram_mask]
            print(y_all.shape, arbitrary_vectors.shape)

        elif regress_out:
            arbitrary_vectors = regress_out_one(arbitrary_vectors)
            # swapaxes such that words is dimension 0, drop the first row and swap back
            y_all = np.swapaxes(np.swapaxes(y_all, 0, 1)[1:], 0, 1)
            print(y_all.shape, arbitrary_vectors.shape)
        
        elif bigrams_removed:
            # get bigram mask and drop all bigrams that repeat:
            bigram_mask       = lingpred_new.utils.get_bigram_mask(df_words_all)
            arbitrary_vectors = arbitrary_vectors[bigram_mask]
            y_all             = y_all[:,bigram_mask,:]
            print(y_all.shape, arbitrary_vectors.shape)

        #saving vectors:
        filename = 'arbitrary_vectors_sub_{}.pkl'.format(subject)
        if regress_out and bigrams_removed:
            filename = 'regressed_out_one_bigrams_removed_arbitrary_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'regressed_out_one_arbitrary_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'bigrams_removed_arbitrary_vectors_sub_{}.pkl'.format(subject)

        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(arbitrary_vectors,f)
        f.close()
        
        corr = brainscore_no_coef(y= y_all, X = arbitrary_vectors)

        #saving results:
        results_dir = project_dir + '/results/Armani/grand_average/'
        filename    = 'corr_arbitrary_vectors_sub_{}.pkl'.format(subject)

        if regress_out and bigrams_removed:
            filename = 'corr_regressed_out_one_bigrams_removed_arbitrary_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'corr_regressed_out_one_arbitrary_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'corr_bigrams_removed_arbitrary_vectors_sub_{}.pkl'.format(subject)

        file_path = results_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(corr,f)
        f.close()


def compute_corr_gpt(subjects, sessions, regress_out=True, bigrams_removed=False):

    project_dir = '/project/3018059.03/Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armani/sub-00{}/'.format(subject)
        file_path   = subject_dir + 'y_all_language.pkl'
        y_all       = pickle.load(open(file_path, 'rb'))

        if subject == 3: sessions = [1, 2, 3, 4, 5, 6, 7, 9, 10]

        # get dataframe to make GPT embeddings:
        df_words_all = pd.DataFrame()

        for session in sessions:
            df_words = lingpred_new.io.get_words_onsets_offsets(None, 
                                                        dataset='Armani', 
                                                        subject=subject, 
                                                        session=session, 
                                                        run=1)
            df_words_all = df_words_all.append(df_words)

        print(len(df_words_all))
        
        # load GPT vectors
        file_name = 'data_all_sess_signal_sources_01-40Hz.pkl'
        if subject == 3: 
            file_name = 'data_all_sess_language_sources_01-40Hz.pkl'
        data        = pickle.load(open(subject_dir + file_name, 'rb'))
        GPT_vectors = data['X_GPT']

        if subject == 3: 
            y_all = np.swapaxes(y_all, 0, 1)

        print(y_all.shape, GPT_vectors.shape)


        if regress_out and bigrams_removed:
            bigram_mask = lingpred_new.utils.get_bigram_mask(df_words_all)
            y_all       = y_all[:,bigram_mask,:] # drop bigrams
            y_all       = y_all[:,1:,:]          # drop first
            GPT_vectors = regress_out_one(GPT_vectors)
            # now I need to adjust the bigram mask so that all integers are one smaller and negative ints are dropped:
            bigram_mask   = [x - 1 for x in bigram_mask if x - 1 >= 0]
            GPT_vectors   = GPT_vectors[bigram_mask]
            print(y_all.shape, GPT_vectors.shape)

        elif regress_out:
            GPT_vectors = regress_out_one(GPT_vectors)
            # swapaxes such that words is dimension 0, drop the first row and swap back
            y_all = np.swapaxes(np.swapaxes(y_all, 0, 1)[1:], 0, 1)
            print(y_all.shape, GPT_vectors.shape)

        elif bigrams_removed:            
            # get bigram mask and drop all bigrams that repeat:
            bigram_mask = lingpred_new.utils.get_bigram_mask(df_words_all)
            GPT_vectors = GPT_vectors[bigram_mask]
            y_all       = y_all[:,bigram_mask,:]
            print(y_all.shape, GPT_vectors.shape)

        #saving vectors:
        filename = 'GPT_vectors_sub_{}.pkl'.format(subject)
        if regress_out and bigrams_removed:
            filename = 'regressed_out_one_bigrams_removed_GPT_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'regressed_out_one_GPT_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'bigrams_removed_GPT_vectors_sub_{}.pkl'.format(subject)

        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(GPT_vectors,f)
        f.close()
        
        corr = brainscore_no_coef(y= y_all, X = GPT_vectors)

        #saving results:
        results_dir = project_dir + '/results/Armani/grand_average/'
        filename    = 'corr_GPT_vectors_sub_{}.pkl'.format(subject)
        if regress_out and bigrams_removed:
            filename = 'corr_regressed_out_one_bigrams_removed_GPT_vectors_sub_{}.pkl'.format(subject)
        elif regress_out:
            filename = 'corr_regressed_out_one_GPT_vectors_sub_{}.pkl'.format(subject)
        elif bigrams_removed:
            filename = 'corr_bigrams_removed_GPT_vectors_sub_{}.pkl'.format(subject)

        file_path = results_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(corr,f)
        f.close()

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


# making the y matrix based on indices and the corresponding X matrix
def make_y_matrix_for_self_pred(X, indices, acoustic_model=False):
    
    # initialise random generator:
    rng = np.random.default_rng(42)
    
    # initialise y 
    y = np.empty(shape=(indices.shape[0], indices.shape[1], X.shape[1], )) # n_words, n_timepoints, dim
    
    if acoustic_model:
        
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

            if word_index == time_index:
                random_vector = rng.normal(0, 1, size=(X.shape[1]))
                y[word_index][nr] = random_vector
            else:
                y[word_index][nr] = X[time_index]
    return y


# The difference to the brainscore method is that for the self-predictability X and y are standard scaled
def self_pred_per_dim(X, y, alpha):
    '''
    Params: 
    - X: X matrix for ridge regression of shape (nr of words, dimensionality)
    - y: vectors for each lags per epoch. Shape: (nr of words, dimensionality)
    
    Returns:
    - list: crossvalidated correlations between y_hat based on X and y
    '''
    kf = KFold(n_splits=10)

    y_hat_all_folds  = np.empty(shape=(0, y.shape[1])) # y.shape[1] = nr. of lags
    y_test_all_folds = np.empty(shape=(0, y.shape[1]))

    for nr, (train_index, test_index) in enumerate(kf.split(X)):

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
        
        # Scale y:
        fit     = scaler.fit(y_train)
        y_train = fit.transform(y_train)
        fit     = scaler.fit(y_test)
        y_test  = fit.transform(y_test)
              

        # fit Ridge:
        regression = Ridge(alpha=alpha, fit_intercept=True)
        regression.fit(X_train, y_train)
        
        y_hat     = regression.predict(X_test)

        # concatenate prediction, test sets, intercepts, and coefficients
        y_hat_all_folds  = np.vstack((y_hat_all_folds, y_hat))
        y_test_all_folds = np.vstack((y_test_all_folds, y_test))

    # transpose: as correlation is computed for each lag
    y_hat_all_folds  = y_hat_all_folds.T
    y_test_all_folds = y_test_all_folds.T

    # compute correlation for each lag:
    corr_list = []
    for lag in range(y_test_all_folds.shape[0]):
        r = stats.pearsonr(y_hat_all_folds[lag], y_test_all_folds[lag]).statistic
        corr_list.append(r)
            
    return corr_list


def self_predictability(X, y, alpha=5000):
    '''
    Params: 
    - X: X matrix for ridge regression of shape (nr of words, dimensionality)
    - y: vectors for each lags per epoch. Shape: (nr of words, dimensionality)
    
    Returns:
    - list: crossvalidated correlations between y_hat based on X and y
    '''
    
    print('We are now in the regression, and y has shape:')
    print(y.shape)
    
    corr_all_dimensions   = np.empty(shape=(0, y.shape[2]))               # array of shape (0, nr_lags)

    dimensions = range(y.shape[0])  # y.shape[0] = nr. of dimensions

    for dimension in dimensions:
        y_ch               = y[dimension] # get one dimension
        corr_ch            = self_pred_per_dim(X, y_ch, alpha=alpha) # compute self-pred per dimension
        corr_all_dimensions  = np.vstack((corr_all_dimensions, corr_ch))      # stack for all dimensions
        
    return corr_all_dimensions


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
    - y: neural data from all channel with 157 lags per epoch. Shape: (nr of channels, nr of words, 153)
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

    
# adds a column of ones for the intercept to GPT's layer activation 
def make_X(layer_act):
    intercept = np.ones(shape=(layer_act.shape[0]))
    X         = np.column_stack((intercept, layer_act))
    return X
