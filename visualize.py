import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import mne

init_block_size = 1000

eps= 1e-6

model = load_model('model-conv.h5')
bdfData = mne.io.read_raw_edf('testfiles/0403-CT.bdf')
numSamples = len(bdfData)

if numSamples > 25600:
    bdfData.crop(0.0, 10)

bdfData.pick_types(eeg=True,  misc=False, resp=False, exclude=['Status', "EXG1", "EXG2", "EXG3", "EXG4", "EXG5", "EXG6", "EXG7", "EXG8"])
bdfData.pick_channels(bdfData.info['ch_names'][:63])


bdfData.load_data()
if bdfData.info['sfreq'] > 256:
    bdfData.resample(256.0)



bdfData.set_eeg_reference()                                 #applying eeg average referencing
bdfData.apply_proj()

sampleLength = 256

channels = bdfData.info['nchan']

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


synthetic = np.zeros((2048, channels))
synthetic[0:256, :] = bdfData[:, 0:256][0].T

for i in range(255, 2047):

    batch = np.expand_dims(synthetic[i-255:i+1], axis=0)
    prediction = model.predict_on_batch(batch)
    synthetic[i+1] = prediction.squeeze()
    pass


# plotting

ax = plt.axes()
ax.set_xlim

t = bdfData[:, 256:2048][1]
channel_to_plot = 1
plt.plot(t, synthetic[256:2048, channel_to_plot], 'r-', t, bdfData[channel_to_plot, 256:2047][0], 'b-')
plt.pause(10000)