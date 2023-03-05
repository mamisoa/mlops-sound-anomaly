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

# tensorflow
import tensorflow as tf

# similarity index
from skimage.metrics import structural_similarity as sk_ssim

# sound packages
import librosa
import librosa.display

# datasets

datasets=['toycar','toyconveyor','fan','pump','slider','valve']
machine_dict = {'fan':[0,2,4,6],'pump':[0,2,4,6],'slider':[0,2,4,6],'toycar':[1,2,3,4],'toyconveyor':[1,2,3],'valve':[0,2,4,6]}
threshold_dict = {'fan':[0.866819,0.840211,0.839655,0.573348],'pump':[0.822475,0.826182,0.811622,0.855862],'slider':[0.838541,0.836396,0.809553,0.835531],
    'toycar':[0.759323,0.895921,0.836600,0.834488],'toyconveyor':[0.900976,0.850254,0.837047],'valve':[0,2,4,6]}
contamination_dict = {'fan':0.7867,'pump':0.5327,'slider':0.6899,'toycar':0.4307,'toyconveyor': 0.3163,'valve':[0,2,4,6]}

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

def predict_id(machine, file_in):
    """
    predict the machine id
    Parameters
    ----------
    machine:
        string: indicates the machine
    Returns
    ----------
    tuple: (test_array,id, confidence)
        test_array: numpy array (224,224,3) of the MEL spectrogram
        id prediction: int
        confidence: float (percentage)
        (0,-1,0) if machine not known 
    """
    if machine not in datasets:
        return (0,-1,0)
    # load model from machine
    cnn = tf.keras.models.load_model(f'./models/id_predictions/{machine}/guess_id_{machine}')
    # generate MEL array
    test_array = gen_mel_arr(file_in, render=False)
    # predict machine id
    prediction = cnn.predict(test_array)
    prediction_id = np.argmax(prediction)
    confidence = prediction[prediction_id]
    return (test_array,prediction_id, confidence)

def ssim(input_img, output_img):
    '''
    similarity function
    '''
    return 1 - tf.reduce_mean(tf.image.ssim(input_img, tf.cast(output_img, tf.float32), max_val=1))

def predict_anomaly(test_array,machine,id):
    """
    predict if numpy array of MEL spectrogram is anomalous
    Parameters
    ----------
    test_array:
        numpy array: from MEL spectrogram
    machine:
        string: indicates the machine
    id:
        int: machine id
    Returns
    ----------
        prediction: boolean with 0 for normal 1 for anomaly
    """
    ae_conv = tf.keras.models.load_model(f'./models/anomaly_detection/{machine}/detect_id_{id}_{machine}', custom_objects={"ssim": ssim })
    test_autoencoded = ae_conv.predict(test_array)
    test_sim_index = sk_ssim(test_array,test_autoencoded,multichannel=True)
    if test_sim_index > threshold_dict[machine][machine_dict.index(id)]:
        prediction = 0 # normal
    else:
        prediction = 1 # anomaly
    return prediction

def predict(test_array,machine):
    """
    predict if numpy array of MEL spectrogram is anomalous
    Parameters
    ----------
    test_array:
        numpy array: from MEL spectrogram
    machine:
        string: indicates the machine
    Returns
    ----------
        prediction: boolean with 0 for normal 1 for anomaly
    """
    predicted_id, _ = predict_id(test_array,machine,id)
    return predict_anomaly(test_array,machine,predicted_id)
