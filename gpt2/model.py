from pdb import set_trace
from pathlib import Path
import numpy as np
import torch 
import string
from sklearn.preprocessing import StandardScaler

from transformers import GPT2Tokenizer, GPT2Model, GPT2Config, AutoTokenizer, GPT2LMHeadModel

# define locations for cached models:
small_model_folder = Path(__file__).parent/'models'/'small_GPT2/' 
medium_model_folder = Path(__file__).parent/'models'/'medium_GPT2/' 
large_model_folder = Path(__file__).parent/'models'/'large_GPT2/' 
xl_model_folder = Path(__file__).parent/'models'/'xl_GPT2/' 


class MyGPT2():
    
    def __init__(self, model_size='S', pretrained=True):
        
        # set context window size:
        self.context_len = 1024
               
        # set tokeniser
        self.tokeniser = AutoTokenizer.from_pretrained('gpt2')
              
        # set device: GPU if available else CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # set standard scaler to scale embedding activation:
        self.scaler = StandardScaler() 
        
        
        # get model depending on model size
        if pretrained == True:
        
            if model_size == 'S': 
                self.model = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir = small_model_folder)

            elif model_size == 'M': 
                self.model = GPT2LMHeadModel.from_pretrained('gpt2-medium', cache_dir = medium_model_folder)

            elif model_size == 'L': 
                self.model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir = large_model_folder)

            elif model_size == 'XL': 
                self.model = GPT2LMHeadModel.from_pretrained('gpt2-xl', cache_dir = xl_model_folder)

            else: 
                raise ValueError('The parameter model_size must be S, M, L or XL')
                
        else: 
            # untrained model: random weights 
            config         = GPT2Config() # config as in the paper
            self.model     = GPT2Model(config)
            self.tokeniser = GPT2Tokenizer.from_pretrained('gpt2')
            
    
        # initialise model:
        self.model.to(self.device)
        self.model.eval()
        
        
    def get_aligned_layer_act(self, text, annotation_df, dropped_epochs, dataset='Armani', subject=1, session=1, 
                          min_context=512, layer=8, scaled=True):
        """
        Extracts contextualized GPT embeddings for a given MEG dataset, subject, and session.

        Parameters
        ----------
        text : str
            The text for the given session.
        annotation_df : pandas.DataFrame
            A DataFrame containing the words for the epoch.
        dataset : str
            The MEG dataset. Can be 'sherlock', 'Gwilliams', or 'Armani'.
        subject : str or int
            The MEG subject identifier.
        session : str or int
            The MEG session identifier.
        min_context : int
            The minimum context size for GPT embeddings. Each embedding has at least a context of 512.
        layer : int
            The GPT layer from which to extract the embeddings.
        scaled : bool, optional
            Whether to apply standard scaling to the embeddings over all dimensions. Defaults to `False`.

        Returns
        -------
        numpy.ndarray
            An array of shape (words, dimensions) containing the contextualized embeddings of the specified GPT layer.
        """


        # encode text for this session
        encoded_text = self.tokeniser.encode(text, return_tensors="pt")

        # preprocess the BPEs (merging tokens like men + 's to men's) corresponding to the Sherlock data
        # and get a list with merged words, stripped of whitespaces
        merged_words, initial_final_BPEs = self.get_words_from_BPEs(encoded_text)

        # check alignment of words for GPT and MEG annotations data frame
        merged_words, initial_final_BPEs  = check_alignment(merged_words, 
                                                            initial_final_BPEs, 
                                                            annotation_df, 
                                                            dropped_epochs,
                                                            dataset=dataset, 
                                                            subject=subject, 
                                                            session=session)

        # model forward pass for the entire text with a minimum context size of 512 --> get embeddings for layer 8 using
        # function: get_windowed_layer_act(self, encoded_text, min_context=512, layer=8, scaled=True)
        layer_act_all_words   = self.get_windowed_layer_act(encoded_text, min_context=min_context, 
                                                             layer=layer, scaled=scaled)
        layer_act_word_finals = layer_act_all_words[initial_final_BPEs.T[1]]

        return layer_act_word_finals
    
    def get_windowed_layer_act(self, encoded_text, min_context=512, layer=8, scaled=True):
        """
        Extracts contextualized GPT-2 embeddings from encoded text.

        Parameters
        ----------
        encoded_text : torch.Tensor
            A tensor containing the entire encoded text.
        min_context : int
            The minimum amount of context each word's embedding relies on. Defaults to 512.
        layer : int
            The GPT-2 layer from which the activation is extracted.
        scaled : bool, optional
            Whether to standardize the activation over all dimensions of GPT-2. Defaults to `False`.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (nr_words, GPT2-dim) containing the contextualized embeddings for the encoded text 
            at the specified layer.
        """

        
        windows = get_windows(min_context=min_context, 
                              window_size=self.context_len, 
                              start=0, 
                              end=encoded_text.size()[1])
        
        with torch.no_grad():   # we do not need the gradient
            
            # do a forward pass for each window and get layer activation:
            for i in range(len(windows)):
                _, start, end = windows[i]

                # get layer (default = 8) embedding for this window of 1024 from start:end
                model_output = self.model.forward(encoded_text[0, start:end].to(self.device), output_hidden_states=True)

                layer_act = model_output.hidden_states[layer]          # get layer activation
                layer_act = layer_act[0].cpu().numpy()                 # reduce to 2D, send back to CPU & convert to np array
                                                                       # a numpy array is needed for scaling --> to CPU necessary

                # stack activations, such that each word in the final array has at least a context window of 512 words
                if i == 0:
                    layer_act_all_words = layer_act
                else:
                    layer_act_all_words = np.vstack((layer_act_all_words, layer_act[min_context:self.context_len]))
            
            
        # scale if scaled is set to True: I am now doing scaling in the brainscore_per_channel method
        #if scaled:
            #fit                 = self.scaler.fit(layer_act_all_words)
            #layer_act_all_words = fit.transform(layer_act_all_words)
                
        return layer_act_all_words
     
    # function that returns logits for the entire text 
    def get_windowed_logits(self, encoded_text, min_context=512):

        logits_all_words = np.empty((0, 50257))

        # get windows 
        windows = get_windows(min_context=min_context, 
                              window_size=self.context_len, 
                              start=0, 
                              end=encoded_text.size()[1])

        with torch.no_grad():   # we do not need the gradient

            # do a forward pass for each window and get layer activation:
            for i in range(len(windows)):
                _, start, end = windows[i]

                # get layer (default = 8) embedding for this window of 1024 from start:end
                model_output = self.model.forward(encoded_text[0, start:end].to(self.device))

                logits = model_output.logits  # get logits
                logits = logits.cpu().numpy() # cast to numpy for stacking

                # stack logits, such that each word in the final array has at least a context window of 512 words
                if i == 0:
                    logits_all_words = logits
                else:
                    logits_all_words = np.vstack((logits_all_words, logits[min_context:self.context_len]))

        return logits_all_words
    
    def get_prediction(self, encoded_text, min_context=512):
        """
        Computes the top-1 word prediction probabilities and indices from encoded text.

        Parameters
        ----------
        encoded_text : torch.Tensor
            A tensor containing the entire encoded text.
        min_context : int
            The minimum amount of context each word's embedding relies on. Defaults to 512.

        Returns
        -------
        numpy.ndarray
            An array of shape (nr_BPEs,) containing the probabilities of the top-1 word prediction (excluding punctuation).
        numpy.ndarray
            An array of shape (nr_BPEs,) containing the BPE indices of the top-1 word prediction (excluding punctuation).
        """

        windows = get_windows(min_context=min_context, 
                              window_size=self.context_len, 
                              start=0, 
                              end=encoded_text.size()[1])
        
        # make a mask to filter out BPEs which are only punctuation
        all_BPEs         = [self.tokeniser.decode(i) for i in range(50257)]
        punctuation_mask = [not contains_only_whitespace_punctuation(BPE) for BPE in all_BPEs]
        
        with torch.no_grad():   # we do not need the gradient
            
            # do a forward pass for each window and get layer activation:
            for i in range(len(windows)):
                _, start, end = windows[i]

                # get model output for this window of 1024 from start:end
                model_output = self.model.forward(encoded_text[0, start:end].to(self.device))

                 # filter out BPEs which are only punctuation
                logits_no_punct = model_output.logits.T[punctuation_mask].T
                no_punct_BPEs   = np.arange(model_output.logits.shape[1])[punctuation_mask]

                # get softmax of logits
                softmax_logits = torch.softmax(logits_no_punct,dim=-1)  
                softmax_logits = softmax_logits.cpu().numpy()           # send back to CPU & convert to np array
                
                # get the predicted BPE (i.e. index of the highest probability) and probability
                predicted_BPE = []
                probabilities = []

                # need to loop through all rows in order to get the correct BPE which isn't punctuation
                for word in softmax_logits: 

                    max_prob  = np.max(word)
                    ind_max   = np.argmax(word)
                    pred_BPE  = no_punct_BPEs[ind_max]
                    predicted_BPE.append(pred_BPE)
                    probabilities.append(max_prob)

                # stack activations, such that each word in the final array has at least a context window of 512 words
                if i == 0:
                    predicted_BPE_all = predicted_BPE
                    probabilities_all = probabilities
                else:
                    predicted_BPE_all = np.concatenate((predicted_BPE_all, predicted_BPE[min_context:self.context_len]))
                    probabilities_all = np.concatenate((probabilities_all, probabilities[min_context:self.context_len]))
                
        return predicted_BPE_all, probabilities_all

    def align_predictions(self, encoded_text, annotation_df, dropped_epochs:list, dataset:str, subject:int, session:int,
                      min_context=512):
        """
        Predicts words from encoded text and returns their probabilities.

        Parameters
        ----------
        encoded_text : torch.Tensor
            The encoded text.
        annotation_df : pandas.DataFrame
            A DataFrame containing the words.
        dropped_epochs : list
            A list of dropped epochs. Usually an empty list.
        dataset : str
            The dataset name. Can be 'Armani' or 'Gwilliams'.
        subject : int
            The subject identifier. For Armani, ranges from 1 to 3.
        session : int
            The session identifier. For Armani, ranges from 1 to 10.
        min_context : int
            The minimum contextual window for GPT.

        Returns
        -------
        list of str
            A list containing the predicted words in uppercase.
        list of float
            A list containing the probabilities of the predicted words.
        """

        # get predicted BPEs and their probabilities (softmax of logits):
        predicted_BPEs, probabilities = self.get_prediction(encoded_text)

        # get the word-final BPEs and check they are properly aligned with the annotations DataFrame:
        merged_words, initial_final_BPEs = self.get_words_from_BPEs(encoded_text)

        merged_words, initial_final_BPEs = check_alignment(merged_words, initial_final_BPEs, annotation_df, 
                                                            dropped_epochs, dataset='Armani', subject=subject, 
                                                            session=session)

        # keep only predictions after word-final BPEs:
        predicted_BPEs = predicted_BPEs[initial_final_BPEs.T[1]]
        probabilities  = probabilities[initial_final_BPEs.T[1]]

        # Decode the BPE and get rid off the whitespace denoting that it's a word-initial BPE 
        predicted_words = [self.tokeniser.decode(BPE).upper() for BPE in predicted_BPEs]
        predicted_words = [word.strip(' ') for word in predicted_words]

        return predicted_words, probabilities
        
    def get_layer_act(self, index, encoded_text, layer=8, scaled=True):
        """
        Extracts the contextualized GPT-2 embedding for a specific word.

        Parameters
        ----------
        index : int
            The index of the last BPE token of the word of interest. For example, "Allan" is split into "All - an",
            so the index should refer to the last subword ("an").
        encoded_text : torch.Tensor
            A tensor containing the entire encoded text.
        layer : int
            The GPT-2 layer from which the activation is extracted.
        scaled : bool, optional
            Whether to standardize the activation over all dimensions of GPT-2. Defaults to `False`.

        Returns
        -------
        numpy.ndarray
            An array of shape (1, GPT2-dim) containing the contextualized embedding for the indexed word at the specified layer.
        """

        # set start and end of window
        if index > self.context_len:
            start_context = index - self.context_len
        else: 
            start_context = 0

        # get layer (default = 8) embedding for this window of 1024 from start_context:index
        model_output = self.model.forward(encoded_text[0, start_context:index].to(self.device), output_hidden_states=True)
 
        layer_act    = model_output.hidden_states[layer].detach()  # get layer activation
        #print(layer_act.shape)
        layer_act      = layer_act[0].cpu().numpy()   # reduce to 2D, send back to CPU & convert to np array
                                                      # a numpy array is needed for scaling --> to CPU necessary
            
        # get the last row, i.e. the context specific activation of our word    
        word_activation = layer_act[-1,:] 
    
        # reshape such that we have a 2d array with 1 row and 768 columns
        word_activation = word_activation.reshape(1, -1) 
    
        return word_activation   
       
    def token2word_indices(self, encoded_text):
        """
        Extracts word-initial and word-final BPE indices from encoded text.

        Parameters
        ----------
        encoded_text : torch.Tensor
            A tensor containing the entire encoded text.

        Returns
        -------
        numpy.ndarray
            A 2D array where each row contains the indices of the word-initial and word-final BPEs.

        Examples
        --------
        If the encoded text starts with the name "Allan", which is split into "All-an", then:

        >>> first_word_final_BPE = token2word_indices(encoded_text)[0]
        >>> first_word_final_BPE
        [0, 1]

        >>> tokenizer.decode(encoded_text[0][first_word_final_BPE[1]])
        'an'
        """


        index_word_end   = [] # list with the indices of word-final BPEs
        index_word_start = [] # list with the indices of word-initial BPEs

        length         = encoded_text.size()[1]                                             # nr. of BPEs
        bpe_text       = [self.tokeniser.decode(encoded_text[0][i]) for i in range(length)] # list of strings containing BPEs

        start_of_word = 0 
        
        # index_word_start.append(start_of_word)    # first BPE is always word-initial 

        for i in np.arange(0, length):

            if i <= length - 2:
            
                if contains_whitespace_or_puctuation(bpe_text[i]):  # if the current BPE contains whitespace or punctuation:
                    
                    if not contains_only_whitespace_punctuation(bpe_text[i-1]): # and the previous one is not only punctuation
                        end_of_word        = i-1                                # that means the previous BPE is word-final
                        index_word_end.append(end_of_word)
                        
                    if not contains_only_whitespace_punctuation(bpe_text[i]): # if current BPE is more than whitesp./punct.:
                        start_of_word = i                                     # that means the current BPE is word-initial
                        index_word_start.append(start_of_word)                
                        
                    if contains_only_whitespace_punctuation(bpe_text[i]):        # if the BPE contains only whitesp./punct.:
                        if not contains_whitespace_or_puctuation(bpe_text[i+1]): # and the one after that does not start with " "
                            for j in np.arange(i+1, length):                          # then the next BPE 
                                if not contains_whitespace_or_puctuation(bpe_text[j]):  # which contains only letters
                                    start_of_word = j                                   # is word-initial
                                    index_word_start.append(start_of_word)
                                    #print(bpe_text[j-1], bpe_text[j])
                                    break
            
            else: 
                index_word_end.append(i)                        # the last BPE will always be word-final
                if contains_whitespace_or_puctuation(bpe_text[i]): # if it starts with a whitespace or a punctuation:
                     index_word_start.append(i)                 # then its also the word-initial BPE
                elif contains_only_whitespace_punctuation(bpe_text[i-1]) and not contains_whitespace_or_puctuation(bpe_text[i]):
                    index_word_start.append(i) 


        return np.vstack((index_word_start, index_word_end)).T        
        
    def get_words_from_BPEs(self, encoded_text, preprocessed=True):     
        
        """
        Extracts words and their corresponding BPE indices from encoded text.

        Parameters
        ----------
        gpt2_instance : GPT2
            An instance of the GPT2 class.
        encoded_text : torch.Tensor
            A tensor containing the entire encoded text.

        Returns
        -------
        list of str
            A list of words extracted from the encoded text (used to check against the dataframe).
        numpy.ndarray
            A 2D array where each row contains the indices of the word-initial and word-final BPEs.
        """
    
        BPEs = [self.tokeniser.decode(encoded_text[0][i]) for i in range(encoded_text.size()[1])] # get BPEs
            
        initial_final_BPEs = self.token2word_indices(encoded_text)  # returns array of shape (nr_words, 2)
            
        merged_words = self.merge_words(initial_final_BPEs, BPEs)   # returns list with merged words, stripped of ' '

        if preprocessed:                                # merges words like men + 's --> men's (necessary for Sherlock)
            initial_final_BPEs = self.merge_BPEs(merged_words, initial_final_BPEs)
            merged_words       = self.merge_words(initial_final_BPEs, BPEs) 
            
        return merged_words, initial_final_BPEs
            
    def merge_words(self, initial_final_BPEs, BPEs):   
        """
        Merges BPE tokens into full words.

        Parameters
        ----------
        initial_final_BPEs : numpy.ndarray
            A 2D array of shape (nr_words, 2), where each row contains the word-initial and word-final BPE indices.
        BPEs : list of str
            A list of decoded BPE tokens from the encoded text.

        Returns
        -------
        list of str
            A list of merged words, stripped of whitespaces.
        """

        #initialise empty list
        merged_words = []

        for word_ind in initial_final_BPEs:
            start, end = word_ind
            if start == end:
                merged_words.append(BPEs[start])
            else:
                word = ''
                for i in np.arange(start, end+1):
                    word = word + BPEs[i]
                merged_words.append(word)
                    
        merged_words = [x.strip(' ') for x in merged_words]
            
        return merged_words
        
    def merge_BPEs(self, merged_words, initial_final_BPEs):
        """
        Identifies and merges word indices that need to be combined with their previous word, e.g., "men" + "'s" → "men's".

        Parameters
        ----------
        merged_words : list of str
            A list of words merged from the BPEs.
        initial_final_BPEs : numpy.ndarray
            A 2D array of shape (nr_words, 2), where each row contains the word-initial and word-final BPE indices.

        Returns
        -------
        numpy.ndarray
            A 2D array of shape (nr_words, 2), where necessary words with contractions have been merged.
        """

        index_list = []

        for nr, word in enumerate(merged_words):
            if word == "'t":
                index_list.append(nr)
            if word == "'s":
                index_list.append(nr)
            if word == "'ll":
                index_list.append(nr)
            if word == "'d":
                index_list.append(nr)
            if word == "'ve":
                index_list.append(nr)
            if word == "'re":
                index_list.append(nr)
            if word == "'m":
                index_list.append(nr)
            if word == "clock" and merged_words[nr-1] == "o":
                index_list.append(nr)
        
        # overwrite word-final index with new index:
        for index in index_list:
            initial_final_BPEs[index-1][1] = initial_final_BPEs[index][1] 
            
        # delete unnecessary rows, containing 's, 't, etc. from initial_final_BPEs
        initial_final_BPEs = np.delete(initial_final_BPEs,index_list, axis=0)
            
        return initial_final_BPEs
    
       
# --------------------------------------------------
#                HELPER FUNCTIONS
# --------------------------------------------------

def get_windows(min_context, window_size, start, end):
    
    """
    Generates a list of window indices with their start and end positions.

    Parameters
    ----------
    min_context : int
        The minimum context size for each window.
    window_size : int
        The size of each window.
    start : int
        The starting index for window segmentation.
    end : int
        The ending index for window segmentation.

    Returns
    -------
    list of tuple
        A list of shape (nr_windows, 3), where each row contains (window_index, start_window, end_window).
    """
    starts = np.arange(start, end-window_size+min_context, min_context)
    ends   = [s + window_size for s in starts]
    arr    = [[nr, start, end] for nr, (start, end) in enumerate(zip(starts, ends))]
       
    return arr


def contains_whitespace_or_puctuation(s):
    return True in [c in s for c in string.whitespace + string.punctuation + '–']

def contains_whitespace(s):
    return True in [c in s for c in string.whitespace]

def contains_only_whitespace_punctuation(s):
    return np.all([c in string.whitespace + string.punctuation + '–' for c in s])

# Checks and corrects the alignment between words corresponding to GPT's BPEs and the annotation dataframe for a given session
def check_alignment(merged_words, initial_final_BPEs, annotation_df, dropped_epochs, task='0',
                    dataset='Armani', subject=1, session=1):
    """
    Compares merged words from GPT with annotation data and identifies mismatches.

    Parameters
    ----------
    merged_words : list of str
        List of words obtained by merging all decoded BPEs in `initial_final_BPEs`.
    initial_final_BPEs : numpy.ndarray
        A 2D array of shape (words, 2), where each row contains the index of the word-initial and word-final BPE.
    annotation_df : pandas.DataFrame
        A DataFrame containing the words for each epoch.
    dataset : str
        The MEG dataset, either 'Gwilliams' or 'Armani'.
    dropped_epochs : list of int
        A list of MEG epoch indices that were dropped by MNE during epoching.
    task : str
        The task identifier in the Gwilliams dataset.
    subject : int or str
        The MEG subject identifier.
    session : int
        The MEG session identifier.

    Returns
    -------
    None
        Prints words that are not identical between the annotation DataFrame and the merged words from GPT.
    numpy.ndarray
        The updated `initial_final_BPEs`.
    list of str
        A list of merged words that needed to be dropped.
    """

    if dataset=='Armani': # these missing words have been checked for subject 1 & 2, other subjects might differ!!!
        if session==1: 
            missing_words = []
        if session==2:
            missing_words = [1561, 9209, 9211, 9215, 9216]
        if session==3:
            missing_words =  [105, 6979, 6982]
        if session==4:
            missing_words =  [3206]
        if session==5:
            missing_words =  []
        if session==6:
            missing_words =  [14, 1195, 7907]
        if session==7:
            missing_words =  [4932, 5085]
        if session==8:
            missing_words = [] # no missing words
            if subject ==3:
                missing_words = list(np.arange(3841, 4170)) # scrambled in the annotations, need to be dropped from there too! 
        if session==9:
            missing_words = [5]
        if session==10:
            missing_words = [5326]
        
    if dataset == 'Gwilliams':
        missing_words = get_indices_to_drop(merged_words, df_words, task)
        
    final_space     = [merged_words.index(merged_words[-1])]
    indices_to_drop = missing_words + final_space # combine lists 
        
    # drop indices from 2D array with BPEs and create new list of merged words
    merged_words_dropped = np.delete(merged_words, indices_to_drop)
    initial_final_BPEs   = np.delete(initial_final_BPEs, indices_to_drop, axis=0)
    
    # now drop the bad epochs from 2D array with BPEs and create new list of merged words
    merged_words_dropped = np.delete(merged_words_dropped, dropped_epochs)
    initial_final_BPEs   = np.delete(initial_final_BPEs, dropped_epochs, axis=0)
        
    if initial_final_BPEs.shape[0] == len(annotation_df.word):
        print('Both GPT and Neural data have {} word embeddings/epochs.'.format(initial_final_BPEs.shape[0]))
              
    # check that dropping worked correctly:
    print('The following words are not identical in annotations and GPT:')
    for nr, i in enumerate(range(len(annotation_df))):
        if annotation_df.word.to_numpy()[i].lower() != str(merged_words_dropped[i].lower()):
            print(nr, annotation_df.word.to_numpy()[i].lower(), merged_words_dropped[i].lower())
        
    return merged_words_dropped, initial_final_BPEs

def get_indices_to_drop(merged_words, df_words, task):
    """
    Identifies indices of words from GPT-2 that need to be dropped because they are not present in the annotation DataFrame.

    Parameters
    ----------
    merged_words : array-like
        The list or array containing merged words from GPT's initial-final BPEs.
    df_words : pandas.DataFrame
        A DataFrame containing a column 'word' with the words present in the neural data.

    Returns
    -------
    list of int
        A list of indices corresponding to words that need to be dropped from initial-final BPEs.
    """

    ind_list = []

    while get_index_to_drop(merged_words, df_words):
        ind = get_index_to_drop(merged_words, df_words)
        
        # For task 0, index 525 is it in the merged_words (526 is 's) and 'it's' in the df
        # which results in an endless loop --> just set merged_words[525] to it's and we're fine
        if task=='0':
            if ind == 525: 
                merged_words[ind] = df_words.word.to_numpy()[ind]
                continue
            
        ind_list.append(ind)
        merged_words = np.delete(merged_words, ind)

    indices_to_drop = [i+ ind for i, ind in enumerate(ind_list)]
    
    return indices_to_drop

# get next index to drop
def get_index_to_drop(merged_words, df_words):
    """
    Finds the next index of a word from GPT-2 that needs to be dropped because it is not present in the annotation DataFrame.

    Parameters
    ----------
    merged_words : array-like
        The list or array containing merged words from GPT's initial-final BPEs.
    df_words : pandas.DataFrame
        A DataFrame containing a column 'word' with the words present in the neural data.

    Returns
    -------
    int or bool
        The next index that needs to be dropped from initial-final BPEs, or `False` if no more indices need to be dropped.
    """

    for nr, i in enumerate(range(len(df_words))):
        if df_words.word.to_numpy()[i].lower() != str(merged_words[i].lower()): 
            #print(nr)
            return nr
            
    return False