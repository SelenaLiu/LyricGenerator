from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model

import keras.backend as K
import numpy as np
import tensorflow as tf
import os, librosa

from utils import Utils

data_dir = "/mnt/d/Repositories/LyricGenerator/data"


#Define constants
use_gpu = (len(tf.config.experimental.list_physical_devices('GPU'))>0)?True:False
target_sr = 44100

audio_list = []
label_list = []
