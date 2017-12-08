import argparse
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, LSTM, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import mne
import matplotlib.pyplot as plt
import math
import pickle
"""
Some constant hyperparameters of the model
"""

#length, in samples, of an observation
sampleLength = 256

#index of the chosen reference electrode
refChannel = 1

# Length of the starting period for which to use simple standardization instead of exponential moving average
init_block_size = 1000

# stabilizer for division-by-zero numerical problems
eps = 1e-6

#training batch size
train_batch_size = 32

convWindow = 50

plt.ion()

"""
Starting the GUI
"""
# gui = QtGui.QApplication([])

# win = pg.GraphicsLayoutWidget()
# win.show()
# win.setWindowTitle('EEG classifier trainer')
# view = win.addViewBox()


parser = argparse.ArgumentParser()
parser.add_argument("file", help="full path to the BDF file you wish to read and learn from")
args = parser.parse_args()


#TODO: might be too large? find way to load in steps or distributed

bdfData = mne.io.read_raw_edf(args.file, preload=True)

# re-sample if the input is too high
if bdfData.info['sfreq'] > 256:
    bdfData.resample(256.0)


#TODO: drop the stim channel for now, just learn to predict raw EEG
bdfData.pick_types(eeg=True,  misc=False, resp=False, exclude=['Status', "EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"])

bdfData.set_eeg_reference()                                 #applying eeg average referencing
bdfData.apply_proj()
channels = bdfData.info['nchan']        # Assume that one of the channels in the file is the Status Channel. we wont be learning from it since its scale means nothing
numSamples = len(bdfData) #assuming all channels have the same total number of samples for this recording




# bdfData.plot()
# plt.pause(100)

def ewm(dataArray):
    """
    Computes the exponential weighted running mean and updates the mne.Raw data in-place
    """

    # normalized = np.zeros(dataArray.shape)
    starting_means  = np.mean(dataArray[:init_block_size])
    starting_var    = np.var(dataArray[:init_block_size])
    averages    = np.copy(starting_means)
    variances   = np.copy(starting_var)

    for i in range(0, len(dataArray)):
        # for the first samples, there are not enough previous samples to warrant an exponential weighted averaging
        # simply substract the true average of the first samples
        if i < init_block_size:
            dataArray[i] = (dataArray[i] - starting_means) / np.maximum(eps, np.sqrt(starting_var))
        else:
            #update the rolling mean and variance
            averages = 0.999 * averages + 0.001 * dataArray[i]
            variances = 0.999 * variances + 0.001 * (np.square(dataArray[i] - averages))

            dataArray[i] = (dataArray[i] - averages) / np.maximum(eps, np.sqrt(variances))    

    return dataArray

bdfData.apply_function(ewm)


# bdfData.drop_channels(['Status'])

kernelSize = math.floor(bdfData.info['sfreq'] * (convWindow / 1000))

# EEG referencing probably not needed since we standardize every window by substrating the image mean and dividing by the standard deviation

# contructing our generator of EEG batches

def batch_generator(mne_Raw, window_len, batch_size, step=1):

    data_per_batch = window_len + ((batch_size -1) * step)
    remainder = len(mne_Raw) % data_per_batch
    counter = 0
                 #amount of contiguous EEG data seen within a batch of windows

    while True:
        for i in range(counter % remainder, len(mne_Raw) - data_per_batch-1,  data_per_batch):

            window = mne_Raw[:, i:i+data_per_batch][0].T
            strides = window.strides
            new_strides = (strides[0] * step, strides[0], window_len * strides[0])

            batch_X = np.lib.stride_tricks.as_strided(mne_Raw[:,i:i+data_per_batch][0], shape=(batch_size, window_len, mne_Raw.info['nchan']), strides=new_strides)
            batch_Y = mne_Raw[:, i+window_len: i+data_per_batch+1][0].T

            yield (batch_X, batch_Y)
            

        counter += 1




# X_training = np.zeros((numSamples - sampleLength, sampleLength, channels-1))
# Y_training = np.zeros((numSamples - sampleLength, channels-1))

# for i in range(0, numSamples - sampleLength, 1):

#     input_mat = bdfData[:, i:i+sampleLength][0]
#     output_vec = bdfData[:, i+sampleLength][0]
    
#     X_training[i] = input_mat.T
#     Y_training[i] = output_vec.squeeze()
#     pass

# Defined model architecture
model = Sequential()
model.add(Conv1D(32, kernelSize, activation='elu', input_shape=(256, bdfData.info['nchan'])))
model.add(MaxPool1D(3,1,))
model.add(Conv1D(50, 10, activation='elu'))
model.add(MaxPool1D(3,1,))
model.add(LSTM(64))
model.add(Dense(bdfData.info['nchan']))
model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])

# defining some keras.Callbacks to save weights as we train and stop when no more improvement
checkpoint = ModelCheckpoint('model.h5', 'loss', verbose=1)
early = EarlyStopping('loss', 0.001, verbose=1)

steps_per_epoch = (len(bdfData) // (256 + train_batch_size-1))

history = model.fit_generator(batch_generator(bdfData, 256, train_batch_size), steps_per_epoch=steps_per_epoch, epochs=100, callbacks=[checkpoint, early])

with open('trainHistoryDict.txt', 'wb+') as file_pi:
        pickle.dump(history.history, file_pi)


"""
Start the GUI event loop and display it
"""
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()