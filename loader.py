import argparse
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from keras.models import Sequential
from keras.datasets import cifar10
import pyedflib
import numpy as np
"""
Some constant hyperparameters of the model
"""

#length, in samples, of an observation
sampleLength = 10

#index of the chosen reference electrode
refChannel = 1

"""
Starting the GUI
"""
gui = QtGui.QApplication([])

win = pg.GraphicsLayoutWidget()
win.show()
win.setWindowTitle('EEG classifier trainer')
view = win.addViewBox()


parser = argparse.ArgumentParser()
parser.add_argument("file", help="full path to the BDF file you wish to read and learn from")
args = parser.parse_args()

# load the specified file into memory
#TODO: might be too large? find way to load in steps or distributed

bdfData = pyedflib.EdfReader(args.file)
n = bdfData.signals_in_file
numSamples = bdfData.getNSamples()[0] #assuming all channels have the same total number of samples for this recording

#load all data in memory
#TODO: add hook to exclude some channels (like eye saccade ref)
fullData = np.zeros((n, numSamples))
for i in range(0, n):
    fullData[i] = bdfData.readSignal(i)



fullData = fullData.astype(np.float32, copy=False) #Probably no need for full double-precision ? verify

# EEG referencing probably not needed since we standardize every window by substrating the image mean and dividing by the standard deviation

# contructing our training tensors

X_training = np.zeros((numSamples - sampleLength, n, sampleLength, 1))
Y_training = np.zeros((numSamples - sampleLength, n, 1))

for i in range(0, numSamples - sampleLength, 1):

    # normalization
    input_mat = fullData[:, i:i+sampleLength, np.newaxis]
    input_mat = (input_mat - np.mean(input_mat)) / np.std(input_mat)  #substract mean and divide by standard deviation

    output_vec = fullData[:, i+sampleLength, np.newaxis]
    output_vec = (output_vec - np.mean(output_vec)) / np.std(output_vec)
    
    X_training[i] = input_mat  
    Y_training[i] = output_vec
    pass

#pg.image(X_training[0].squeeze().transpose()).setLevels(-4,5)
pg.image(fullData[:-1, :256])
model = Sequential()


"""
Start the GUI event loop and display it
"""
if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()