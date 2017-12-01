import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np
import mne



model = load_model('model.h5')
bdfData = mne.io.read_raw_edf('testfiles/Newtest17-256.bdf', preload=True)
bdfData.pick_types(eeg=True,  exclude=['Status'])

sampleLength = 256
numSamples = len(bdfData)
channels = bdfData.info['nchan']

X_training = np.zeros((numSamples - sampleLength, sampleLength, channels))
Y_training = np.zeros((numSamples - sampleLength, channels))

for i in range(0, len(bdfData) - 256, 1):

    input_mat = bdfData[:, i:i+sampleLength][0]
    output_vec = bdfData[:, i+sampleLength][0]
    
    X_training[i] = input_mat.T
    Y_training[i] = output_vec.squeeze()
    pass

synthetic = np.zeros((1000, 256, 16))
synthetic[0] = bdfData[:, 0].T

for i in range(0, 1001):

    batch = synthetic[i].
    synthetic[i+1] = model.predict_on_batch(batch)

    pass

plt.plot(bdfData[:, :1000][1], synthetic)
plt.pause(10000)