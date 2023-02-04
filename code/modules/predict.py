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