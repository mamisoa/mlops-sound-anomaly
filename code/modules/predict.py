
# TODO: check input format parameters: wav sound file in 16kHz, 16 bits, mono


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

# path
modelDir = '../../models/'

# functions to generate from wav file
# 1 channel, precision 16bits, sampling 16kHz, 160000 frames, duration 10 sec (11 sec for toycar)
# to numpy array (224,224,3)
# from image file generated  MEL spectrogram

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
    # fig, ax = plt.subplots(figsize=(3.12,3.12))
    fig, ax = plt.subplots(figsize=(2.24,2.24))
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
    test_tmp = np.empty((0,224,224,3))
    test_id_isNormal_tmp = np.empty((0,))
    test_tmp = np.append(test_tmp,[img_arr/255], axis=0)
    # test_id_isNormal_tmp = np.append(test_id_isNormal_tmp,[0])
    print(f'Shape: {img_arr.shape}')
    #print(f'Array: {img_arr}')
    if render == True:
        plt.show()
        print(f'Shape: {img_arr.shape}')
    plt.close(fig)
    return test_tmp

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
    print(f'Loading model: {machine}')
    cnn = tf.keras.models.load_model(f'{modelDir}/id_predictions/{machine}/guess_id_{machine}')
    # generate MEL array
    test_array = gen_mel_arr(file_in, render=False)
    print(f'Test array shape: {test_array.shape}')
    # predict machine id
    prediction = cnn.predict(test_array)
    prediction_id = machine_dict[machine][np.argmax(prediction)]
    confidence = np.max(prediction)
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
    ae_conv = tf.keras.models.load_model(f'{modelDir}/anomaly_detection/{machine}/detect_id_{id}_{machine}', custom_objects={"ssim": ssim })
    test_autoencoded = ae_conv.predict(test_array)
    print(f'Test_autoencoded shape: {np.squeeze(test_autoencoded).shape}')
    print(f'Test_array shape: {np.squeeze(test_array).shape}')
    print(f'Test_autoencoded: {np.squeeze(test_autoencoded)}')
    test_sim_index = sk_ssim(np.squeeze(test_array),np.squeeze(test_autoencoded),multichannel=True)
    if test_sim_index > threshold_dict[machine][machine_dict.index(id)]:
        prediction = 0 # normal
    else:
        prediction = 1 # anomaly
    return prediction

def predict(machine, file_in):
    """
    predict if numpy array of MEL spectrogram is anomalous
    Parameters
    ----------
    file_in:
        string: path and file of the wav file
    machine:
        string: indicates the machine
    Returns
    ----------
        prediction: boolean with 0 for normal 1 for anomaly
    """
    test_array, predicted_id, _ = predict_id(machine,file_in)
    print(f'Predicted id: {predicted_id}')
    # return predict_anomaly(test_array,machine,predicted_id)
    return 0

# predict('fan','../../sounds/samples/normal/fan/normal_id_00_00000000.wav')
ds = 'fan'
file = f'../../sounds/samples/normal/{ds}/normal_id_06_00000013.wav'

# test_array, p, _ = predict_id(ds, file)
# id= machine_dict['fan'][p]
# print(f'Predicted id: {id}')
# isNormal = predict_anomaly(test_array,ds,id)
# result = 'Normal' if isNormal == 0 else 'Anomaly'
# print(f'Result: {result}')

isNormal = predict(ds,file)
result = 'Normal' if isNormal == 0 else 'Anomaly'
print(f'Result: {result}')