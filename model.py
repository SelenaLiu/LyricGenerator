"""
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model

import keras.backend as K
"""
import numpy as np
import tensorflow as tf
import os, librosa

from utils import Utils
from preprocess import Preprocess

data_dir = "/mnt/d/Repositories/LyricGenerator/data"
audio_dir = os.path.join(data_dir,"Audio")

#Define constants
use_gpu = True if (len(tf.config.experimental.list_physical_devices('GPU'))>0) else False
target_sr = 44100

audio_list = []
label_list = []

p = Preprocess()
p.mp3_to_wav(audio_dir)
