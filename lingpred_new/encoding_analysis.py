import sys
import os 
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from scipy import stats
import spacy
from sklearn.preprocessing import StandardScaler