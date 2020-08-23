"""
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model

import keras.backend as K
"""
import numpy as np
#import tensorflow as tf
import os, librosa

from utils import Utils
from preprocess import Preprocess

import torch
from torch.utils.data import TensorDataset, DataLoader

data_dir = "/mnt/d/Repositories/LyricGenerator/data"
audio_dir = os.path.join(data_dir,"Audio")

#Define constants
#use_gpu = True if (len(tf.config.experimental.list_physical_devices('GPU'))>0) else False
target_sr = 44100

audio_list = []
label_list = []

p = Preprocess()
p.mp3_to_wav(audio_dir)
train_audio = p.compile_audio(audio_dir)[0:29]
# p.dali_json_to_txt()
train_txt = p.dali_json_to_np()[0:29]

val_audio = p.compile_audio(audio_dir)[29:35]
# p.dali_json_to_txt()
val_txt = p.dali_json_to_np()[29:35]

num_epochs = 200
num_hidden = 50
num_layers = 1
batch_size = 128

train_seq_len = [(14553000, 128)]

train_audio_tensor = torch.from_numpy(train_audio)
train_txt_tensor = torch.from_numpy(train_txt)

train_dataset = TensorDataset(train_audio_tensor, train_txt_tensor)
train_dataloader = DataLoader(train_dataset)

val_audio_tensor = torch.from_numpy(val_audio)
val_txt_tensor = torch.from_numpy(val_txt)

val_dataset = TensorDataset(val_audio_tensor, val_txt_tensor)
val_dataloader = DataLoader(val_dataset)

model = Model(inputs, outputs)

