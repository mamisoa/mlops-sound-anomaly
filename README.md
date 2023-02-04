# Detection of anomalous sounds in industrial environnements
This repository contains a **Datascientest project in MLOPS** to deploy the best models to detect anomalous sounds in industrial environnements.

The original models are exposed in the following private repository:

[Datascientest DS project in Detection of sound anomalies in industrial parts](https://github.com/DataScientest-Studio/Py_ASD_ACM)

The original dataset is from Kaggle:

[Unsupervised Detection of Anomalous Sounds for Machine Condition Monitoring](https://dcase.community/challenge2020/task-unsupervised-detection-of-anomalous-sounds)

## Purpose
Demonstrate the ability to deploy a ML solution with:
* **testing**: if sound is anomalous
* **monitoring**: status, usage, results
* **training**: adding new data to train the model

## Architecture
Coding parts will use Python.
### Docker containers
* **nginx**: web UI
* **sqlite**: database
  * store path of spectrogram files
  * store image status: original or new
* **FastAPI**: REST API

## Models specifications
### Training data

Models are trained over batches of **normal sound wav files** from 6 differents machines, each machine having different serial ID.

Machines are: valve, pump, toyconveyor, toycar, slider, fan.

The wav file were converted in MEL spectrograms in jpg format 224 pixels x 224 pixels in RGB.

### Model pipeline

Models are **unsupervised**, using deep learning convolution neural networks that are trained from MEL-spectrograms created from the wav files.

#### Step 1: get the machine ID
A first model is applied to detect the machine ID in order to apply a model specific to ID in the second step of the pipeline
#### Step 2: detect anomalous sound
A second model specific to the machine ID is applied to detect if sound is anomalous.
#### Performance
Models offer very good precision (usually over 90%) and relative good recall (usually over 50%).

## Steps
Here are the steps to achieve this project
### REST API deployment
Code a REST API to:
* get a status
  * status
  * training data
  * new training data
* test a sound
  1. get format
  2. normalize format: convert to wav, crop to 10 sec, convert to spectrogram
  3. feed the pipeline
  4. get the result: normal/anormal, probability
* add new sounds
  * check format: wav, crop
  * if normal sounds: add to training data
  * if anomalous sounds: add to validation data
  * create specific database for external data
  * convert sound to spectrogram before adding to database
* list new sounds
  * list new sounds
  * counts
* delete new sounds
* launch new training
  * show performance
  * compare to original performance

### Web UI
* Code the web interface
* Enable live monitoring ?

## Dependancies
* Docker
* FastAPI
* Nginx
* Sqlite
* Tenserflow with Keras
