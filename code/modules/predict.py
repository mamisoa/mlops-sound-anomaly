# module for predicting if sound is anomalous
# parameters: wav sound file in 16kHz, 16 bits, mono
# return a tuple with 2 elements:
## 0 if normal, 1 if anomalous 
## probability

# check input format
# convert to correct wav format
# convert to MEL spectrogram in 224x224 RGB jpg format
# apply to model
# return prediction and probability

# common
import os, os.path, glob
import time

# pandas & numpy
import pandas as pd
import numpy as np

# matplotlib
import matplotlib.pyplot as plt
from matplotlib import image

# sound packages
import librosa
import librosa.display


# functions to generate from wav file
# 1 channel, precision 16bits, sampling 16kHz, 160000 frames, duration 10 sec (11 sec for toycar)
# to numpy array (224,224,3)
# from image file generated from MFCC or MEL spectrogram

def gen_mfcc_arr(file_in, render = False):
  """
  generate a numpy array of the wav file with shape (224,224,3) of spectrogram of mfccs and the 2 following derivatives
  Parameters
    ----------
    file_in: str
      input filename with path
    render: bool
      True: display image and shape
    Returns
    ----------
    numpy array of the image
  """
  signal, sr = librosa.load(file_in, sr=None)
  mfccs = librosa.feature.mfcc(y=signal, n_mfcc=13, sr=sr)
  delta_mfccs = librosa.feature.delta(mfccs)
  delta2_mfccs = librosa.feature.delta(mfccs, order=2)
  fig,axes = plt.subplots(3,1,figsize=(3.12,3.12))
  img0 = librosa.display.specshow(mfccs,ax=axes[0],  sr=sr)
  img1 = librosa.display.specshow(delta_mfccs,ax=axes[1],  sr=sr)
  img2 = librosa.display.specshow(delta2_mfccs,ax=axes[2], sr=sr)
  for axe in axes:
    axe.axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  fig.canvas.draw()
  img_arr = np.asarray(fig.canvas.renderer.buffer_rgba(),dtype='int16')[:,:,:3]
  if render == True:
    plt.show()
    print(f'Shape: {img_arr.shape}')
  plt.close(fig)
  return img_arr

def gen_mel_arr(file_in, render=False):
  """
  generate a numpy array of the wav file with shape (224,224,3) of MEL spectrogram
  Parameters
    ----------
    file_in: str
      input filename with path
    render: bool
      True: display image and shape
    Returns
    ----------
    numpy array of the image
  """
  signal, sr = librosa.load(file_in, sr=None)
  fig, ax = plt.subplots(figsize=(3.12,3.12))
  S = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128,fmax=8000)
  mel = librosa.display.specshow(librosa.power_to_db(S, ref=np.max),
                                 # x_axis='time',
                                 # y_axis='mel',
                                 sr=sr,
                                 fmax=8000,
                                 ax=ax)
  ax.axis('off')
  plt.subplots_adjust(wspace=0, hspace=0)
  fig.canvas.draw()
  img_arr = np.asarray(fig.canvas.renderer.buffer_rgba(),dtype='int16')[:,:,:3]
  if render == True:
    plt.show()
    print(f'Shape: {img_arr.shape}')
  plt.close(fig)
  return img_arr

# models
