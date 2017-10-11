import argparse
#from keras.models import Sequential
import pyedflib
import numpy as np

"""
Some constant hyperparameters of the model
"""

#length, in samples, of an observation
sampleLength = 10

#index of the chosen reference electrode
refChannel = 0

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

# normalization
fullData = (fullData - np.mean(fullData)) / np.std(fullData)    #substract mean and divide by standard deviation
#TODO: handle division by zero and subsequent NaN values

fullData = fullData.astype(np.float32, copy=False) #Probably no need for full double-precision ? verify
fullData = fullData - fullData[refChannel]          # as BDF stores raw voltage values, we need to reference the values. substract a channel (support for average referencing?)



X_training = np.zeros((numSamples - sampleLength, n, sampleLength, 1))
Y_training = np.zeros((numSamples - sampleLength, n, 1))

for i in range(0, numSamples - sampleLength, 1):
    pass