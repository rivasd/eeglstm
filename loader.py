import argparse
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, LSTM, Dense
# from keras.datasets import cifar10
import pyedflib
import numpy as np
import mne
import matplotlib.pyplot as plt
import math
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
eps = 1e-4

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

    normalized = np.zeros(dataArray.shape)
    starting_means  = np.mean(dataArray[:init_block_size])
    starting_var    = np.var(dataArray[:init_block_size])
    averages    = np.copy(starting_means)
    variances   = np.copy(starting_var)

    for i in range(0, len(dataArray)):
        # for the first samples, there are not enough previous samples to warrant an exponential weighted averaging
        # simply substract the true average of the first samples
        normalized[i] = (dataArray[i] - starting_means) / np.maximum(eps, np.sqrt(starting_var))
    else:
        #update the rolling mean and variance
        averages = 0.999 * averages + 0.001 * dataArray[i]
        variances = 0.999 * variances + 0.001 * (np.square(dataArray[i] - averages))

        normalized[i] = (dataArray[i] - averages) / np.maximum(eps, np.sqrt(variances))    

    return normalized

bdfData.apply_function(ewm)

#TODO: drop the stim channel for now, just learn to predict raw EEG
bdfData.pick_types(eeg=True,  exclude=['Status'])
# bdfData.drop_channels(['Status'])

kernelSize = math.floor(bdfData.info['sfreq'] * (convWindow / 1000))

# EEG referencing probably not needed since we standardize every window by substrating the image mean and dividing by the standard deviation

# contructing our training tensors

X_training = np.zeros((numSamples - sampleLength, sampleLength, channels-1))
Y_training = np.zeros((numSamples - sampleLength, channels-1))

for i in range(0, numSamples - sampleLength, 1):

    input_mat = bdfData[:, i:i+sampleLength][0]
    output_vec = bdfData[:, i+sampleLength][0]
    
    X_training[i] = input_mat.T
    Y_training[i] = output_vec.squeeze()
    pass

# Defined model architecture
model = Sequential()
model.add(Conv1D(32, kernelSize, activation='elu', input_shape=(256, X_training.shape[2])))
model.add(MaxPool1D(3,1,))
# model.add(Conv1D(50, 10, activation='elu'))
# model.add(MaxPool1D(3,1,))
model.add(LSTM(64))
model.add(Dense(bdfData.info['nchan']))
model.compile(optimizer='rmsprop', loss='cosine_proximity', metrics=['accuracy'])


hist = model.fit(X_training, Y_training, batch_size=32, epochs=100, verbose=1)
print(hist)


"""
Start the GUI event loop and display it
"""
# if __name__ == '__main__':
#     import sys
#     if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
#         QtGui.QApplication.instance().exec_()