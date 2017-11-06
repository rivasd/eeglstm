import argparse
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
from keras.models import Sequential
# from keras.datasets import cifar10
import pyedflib
import numpy as np
"""
Some constant hyperparameters of the model
"""

#length, in samples, of an observation
sampleLength = 10

#index of the chosen reference electrode
refChannel = 1

# Length of the starting period for which to use simple standardization instead of exponential moving average
init_block_size = 1000

# stabilizer for division-by-zero numerical problems
eps = 1e-4



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


#TODO: might be too large? find way to load in steps or distributed

bdfData = pyedflib.EdfReader(args.file)
channels = bdfData.signals_in_file - 1            # Assume that one of the channels in the file is the Status Channel. we wont be learning from it since its scale means nothing
numSamples = bdfData.getNSamples()[0] #assuming all channels have the same total number of samples for this recording

#load all data in memory
#TODO: add hook to exclude some channels (like facial electrodes ref)
fullData = np.zeros((channels, numSamples))
for i in range(0, channels):
    if bdfData.getLabel(i) != "Status":
        fullData[i] = bdfData.readSignal(i)



# replace the data with its normalized version usign the pre processing of Schirrmeister et al. 2017. code freely copied from their github
other_axis = tuple(range(1, len(fullData.shape)))
starting_means  = np.mean(fullData[:, :init_block_size], axis=other_axis, keepdims=True).squeeze()
starting_var    = np.var(fullData[:, :init_block_size], axis=other_axis, keepdims=True).squeeze()

# we need to keep buffers for the running average and variance

averages    = np.copy(starting_means)
variances   = np.copy(starting_var)

for i in range(0, numSamples):

    if i<init_block_size:
        # for the first samples, there are not enough previous samples to warrant an exponential weighted averaging
        # simply substract the true average of the first samples
        fullData[:, i] = (fullData[:, i] - starting_means) / np.maximum(eps, np.sqrt(starting_var))
    else:
        #update the rolling mean and variance
        averages = 0.999 * averages + 0.001 * fullData[:, i]
        variances = 0.999 * variances + 0.001 * (np.square(fullData[:, i] - averages))

        fullData[:, i] = (fullData[:, i] - averages) / np.maximum(eps, np.sqrt(variances))

        


# EEG referencing probably not needed since we standardize every window by substrating the image mean and dividing by the standard deviation

# contructing our training tensors

X_training = np.zeros((numSamples - sampleLength, channels, sampleLength, 1))
Y_training = np.zeros((numSamples - sampleLength, channels, 1))

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