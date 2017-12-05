import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import mne

eps= 1e-6

model = load_model('model.h5')
bdfData = mne.io.read_raw_edf('testfiles/Newtest17-256.bdf', preload=True)
bdfData.pick_types(eeg=True,  exclude=['Status'])
bdfData.set_eeg_reference()                                 #applying eeg average referencing
bdfData.apply_proj()

sampleLength = 256
numSamples = len(bdfData)
channels = bdfData.info['nchan']


def ewm(dataArray):
    """
    Computes the exponential weighted running mean and updates the mne.Raw data in-place
    """

    normalized = np.zeros(dataArray.shape)
    starting_means  = np.mean(dataArray[:256])
    starting_var    = np.var(dataArray[:256])
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

synthetic = np.zeros((1024, 16))
synthetic[0:256, :] = bdfData[:, 0:256][0].T

for i in range(255, 1023):

    batch = np.expand_dims(synthetic[i-255:i+1], axis=0)
    prediction = model.predict_on_batch(batch)
    synthetic[i+1] = prediction.squeeze()
    pass


# plotting

ax = plt.axes()
ax.set_xlim


plt.plot(bdfData[:, :1024][1], synthetic)
plt.pause(10000)