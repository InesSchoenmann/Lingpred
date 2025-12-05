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
def compute_corr(subjects:list, sessions:list, model:str, regress_out=True, bigrams_removed=False, residualise=False):

    warnings.simplefilter(action='ignore')

    model = model.lower() # make sure the spelling is all lower case
    
    if model=='glove':
        compute_corr_glove(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)
    if model=='arbitrary':
        compute_corr_arbitrary(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)
    if model=='gpt':
        compute_corr_gpt(subjects, sessions, regress_out=regress_out, bigrams_removed=bigrams_removed)

def compute_corr_glove(subjects, sessions, regress_out=False, bigrams_removed=False):

    project_dir = '../Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armeni/sub-00{}/'.format(subject)
        file_path   = subject_dir + 'y_all_language.pkl'
        y_all       = pickle.load(open(file_path, 'rb'))

        if subject == 3: sessions = [1, 2, 3, 4, 5, 6, 7, 9, 10]

        # get dataframe to make Glove embeddings:
        df_words_all = pd.DataFrame()

        for session in sessions:
            df_words = lingpred_new.io.get_words_onsets_offsets(None, 
                                                        dataset='Armeni', 
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

        '''
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
            print(y_all.shape, Glove_vectors.shape)'''

        # saving vectors:
        filename  = 'Glove_vectors_sub_{}.pkl'.format(subject)
        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(Glove_vectors,f)
        f.close()

        if bigrams_removed:
            bigrams_mask = lingpred_new.utils.get_bigram_mask(df_words_all)
        else:
            bigrams_mask = None

        # compute brainscore:
        corr = brainscore(y= y_all, X = Glove_vectors, residualise=regress_out, indx_top_1=bigrams_mask)

        #saving results:
        results_dir = project_dir + '/results/Armeni/grand_average/'
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

    project_dir = '../Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armeni/sub-00{}/'.format(subject)
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

        '''if regress_out and bigrams_removed:
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
            print(y_all.shape, arbitrary_vectors.shape)'''

        #saving vectors:
        filename  = 'arbitrary_vectors_sub_{}.pkl'.format(subject)
        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(arbitrary_vectors,f)
        f.close()
        

        if bigrams_removed:
            bigrams_mask = lingpred_new.utils.get_bigram_mask(df_words_all)
        else:
            bigrams_mask = None
            
        corr = brainscore(y= y_all, X = arbitrary_vectors, residualise=regress_out, indx_top_1=bigrams_mask)

        #saving results:
        results_dir = project_dir + '/results/Armeni/grand_average/'
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

    project_dir = '../Lingpred'

    for subject in subjects:

        # load neural data:
        subject_dir = project_dir + '/data/Armeni/sub-00{}/'.format(subject)
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

        '''
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
            print(y_all.shape, GPT_vectors.shape)'''

        #saving vectors:
        filename  = 'GPT_vectors_sub_{}.pkl'.format(subject)
        file_path = subject_dir + filename
        f         = open(file_path,"wb")
        pickle.dump(GPT_vectors,f)
        f.close()

        # get the bigrams mask if we only want to keep the first occurrance of each bigram:
        if bigrams_removed:
            bigrams_mask = lingpred_new.utils.get_bigram_mask(df_words_all)
        else:
            bigrams_mask = None
        
        # compute the correlation
        corr = brainscore(y= y_all, X = GPT_vectors, residualise=regress_out, indx_top_1=bigrams_mask)

        #saving results:
        results_dir = project_dir + '/results/Armeni/grand_average/'
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

# computes the brainscore without saving coefficients per channel:
def brainscore_old_per_channel(X, y, alpha, print_ind, residualise=False, test_mask = None):
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

            # ---------------------------
            # 1. Optional residualisation
            # ---------------------------
            if residualise:
                X_train, X_test = residualise_within_fold(X_train, X_test, alpha=alpha)

                # Align y if needed — if you drop the first row of X, 
                # you must drop the first y too:
                if len(y_train) > 1:
                    y_train = y_train[1:]
                if len(y_test) > 1:
                    y_test = y_test[1:]


        
        # initialise Standard Scaler:
        scaler = StandardScaler()
        
        # scale for each fold, such that for X, each dim has mean of 0 and SD of 1:
        # fit on the training data only and apply to both train and test
        fit     = scaler.fit(X_train)
        X_train = fit.transform(X_train)
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
def brainscore_old(X, y, residualise=False, test_mask = None, alpha=5000):
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
    
    print('We are now in the brainscore method, and y has shape:')
    print(y.shape)
    
    corr_all_channels   = np.empty(shape=(y.shape[0], 10, y.shape[2]))  # array of shape (100, 10, nr_lags)

    channels = range(y.shape[0])  # y.shape[0] = nr. of channels

    for i, channel in enumerate(channels):
        y_ch                       = y[channel] # get one channel
        corr_all_channels[channel] = brainscore_no_coef_per_channel(X, y_ch, residualise=residualise, alpha=alpha, print_ind = i, test_mask=test_mask) # compute brainscore per channel
        
    return corr_all_channels


def brainscore_per_channel(X, y, alpha, print_ind,
                            residualise=False, indx_top_1=None):

    kf = KFold(n_splits=10)
    corr_all_folds = np.empty(shape=(10, y.shape[1]))  # y.shape[1] = nr lags

    for nr, (train_index, test_index) in enumerate(kf.split(X)):

        # -----------------------------------------------------
        # 1. Extract full fold data per split
        # -----------------------------------------------------
        X_train_full = X[train_index]
        y_train_full = y[train_index]

        X_test_full  = X[test_index]
        y_test_full  = y[test_index]

        # -----------------------------------------------------
        # 2. Residualise FIRST (preserves adjacency)
        # -----------------------------------------------------
        if residualise:
            X_train_res, X_test_res = residualise_within_fold(
                X_train_full, X_test_full, alpha=alpha
            )

            # y must be aligned with X (drop first row)
            if len(y_train_full) > 1:
                y_train_res = y_train_full[1:]
            else:
                y_train_res = y_train_full

            if len(y_test_full) > 1:
                y_test_res = y_test_full[1:]
            else:
                y_test_res = y_test_full

        else:
            X_train_res = X_train_full
            y_train_res = y_train_full

            X_test_res = X_test_full
            y_test_res = y_test_full

        # -----------------------------------------------------
        # 3. Apply predictability mask AFTER residualisation
        # -----------------------------------------------------
        if indx_top_1 is not None:
            mask_train = np.isin(train_index, indx_top_1)
            mask_test  = np.isin(test_index, indx_top_1)
        else:
            mask_train = slice(None)
            mask_test  = slice(None)

        X_train = X_train_res[mask_train]
        y_train = y_train_res[mask_train]

        X_test  = X_test_res[mask_test]
        y_test  = y_test_res[mask_test]

        # If a fold has no predictable words → skip (set nan)
        if len(X_test) == 0 or len(X_train) == 0:
            corr_all_folds[nr, :] = np.nan
            continue

        # -----------------------------------------------------
        # 4. Scale X (train scaler only)
        # -----------------------------------------------------
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        # -----------------------------------------------------
        # 5. Ridge + correlation per lag
        # -----------------------------------------------------
        regression = Ridge(alpha=alpha, fit_intercept=True)
        regression.fit(X_train, y_train)

        y_hat = regression.predict(X_test)

        # transpose for lag-wise correlation:
        y_hat = y_hat.T
        y_test = y_test.T

        corr_all_folds[nr] = [
            stats.pearsonr(y_hat[lag], y_test[lag]).statistic
            for lag in range(y_test.shape[0])
        ]

    return corr_all_folds

def brainscore(X, y, residualise=False, indx_top_1=None, alpha=5000):
    '''
    Params: 
    - X: X matrix for ridge regression of shape (nr of words, GPT-dim + 1)
    - y: neural data from all channel with 157 lags per epoch. 
         Shape: (nr of channels, nr of words, 157)
    - indx_top_1: numpy array of indices of "predictable" words
    - residualise: whether to remove preceding-word information within each fold

    Returns:
    - 3D array: correlations for all channels and lags.
      Shape: (nr_channels, 10, nr_lags)
    '''

    print('We are now in the brainscore method, and y has shape:')
    print(y.shape)
    
    corr_all_channels = np.empty(shape=(y.shape[0], 10, y.shape[2]))

    channels = range(y.shape[0])  # y.shape[0] = nr. of channels

    for i, channel in enumerate(channels):
        y_ch = y[channel]  # (nr_words, nr_lags)

        corr_all_channels[channel] = brainscore_per_channel(
            X,
            y_ch,
            residualise=residualise,
            alpha=alpha,
            print_ind=i,
            indx_top_1=indx_top_1
        )

    return corr_all_channels


# adds a column of ones for the intercept to GPT's layer activation 
def make_X(layer_act):
    intercept = np.ones(shape=(layer_act.shape[0]))
    X         = np.column_stack((intercept, layer_act))
    return X

def residualise_within_fold(X_train, X_test, alpha=5000):
    """
    Residualise X by regressing X[t] on X[t-1] *within the fold*.

    X_train, X_test: arrays of shape (N, D)

    Returns:
    - X_train_res, X_test_res: residualised versions of X
    """

    # Build (previous, current) pairs for training
    X_prev_train = X_train[:-1]
    X_curr_train = X_train[1:]

    # Fit regression only on training data
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(X_prev_train, X_curr_train)

    # Residualise training set
    X_curr_hat_train = reg.predict(X_prev_train)
    X_train_res = X_curr_train - X_curr_hat_train

    # For the test set, we must construct (prev, curr) pairs too
    if len(X_test) > 1:
        X_prev_test = X_test[:-1]
        X_curr_test = X_test[1:]

        X_curr_hat_test = reg.predict(X_prev_test)
        X_test_res = X_curr_test - X_curr_hat_test
    else:
        # Degenerate case: test set of size 1
        X_test_res = X_test.copy()

    return X_train_res, X_test_res

def self_pred_per_dim_CVsafe(X, y, alpha=5000, residualise=False):
    """
    X: (N, D)  -- N = nr. of words, D = embedding dimensionality
    y: (N, L)  -- L = nr. lags
    """

    kf = KFold(n_splits=10)

    y_hat_all = []
    y_test_all = []

    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------------------------
        # 1. Optional residualisation
        # ---------------------------
        if residualise:
            X_train, X_test = residualise_within_fold(X_train, X_test, alpha=alpha)

            # Align y if needed — if you drop the first row of X, 
            # you must drop the first y too:
            if len(y_train) > 1:
                y_train = y_train[1:]
            if len(y_test) > 1:
                y_test = y_test[1:]

        # ---------------------------
        # 2. Scale X using train stats
        # ---------------------------
        scalerX = StandardScaler()
        scalerX.fit(X_train)

        X_train = scalerX.transform(X_train)
        X_test  = scalerX.transform(X_test)

        # ---------------------------
        # 3. Scale y using train stats
        # ---------------------------
        scalerY = StandardScaler()
        scalerY.fit(y_train)

        y_train = scalerY.transform(y_train)
        y_test = scalerY.transform(y_test)

        # ---------------------------
        # 4. Fit ridge
        # ---------------------------
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(X_train, y_train)

        # Predict
        y_hat = reg.predict(X_test)

        y_hat_all.append(y_hat)
        y_test_all.append(y_test)

    # Concatenate across folds
    y_hat_all = np.vstack(y_hat_all)
    y_test_all = np.vstack(y_test_all)

    # Compute per-lag correlation
    corrs = []
    for lag in range(y.shape[1]):
        r = stats.pearsonr(y_hat_all[:, lag], y_test_all[:, lag]).statistic
        corrs.append(r)

    return corrs

def self_predictability_CVsafe(X, Y, alpha=5000, residualise=False):
    """
    X: (N, D)
    Y: (D, N, L)  -- like in your code: dimension × time × lag

    Returns: array (D, L)
    """
    print('Y should have shape (Dim, Nr_Words, Lags), and has shape', Y.shape)

    nr_dims = Y.shape[0]
    nr_lags = Y.shape[2]

    corr_all = np.empty((nr_dims, nr_lags))

    for d in range(nr_dims):
        y_d         = Y[d]  # shape (N, L)
        corr_d      = self_pred_per_dim_CVsafe(X, y_d, alpha=alpha, residualise=residualise)
        corr_all[d] = corr_d

    return corr_all
