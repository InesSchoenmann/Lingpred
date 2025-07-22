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
sys.path.append(os.path.abspath("/project/3018059.03/Lingpred"))
import gpt2

