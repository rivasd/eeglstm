#%% [markdown]
#
# We will load EEG data from the lab and attemp to build a classifier that distinguishes between learners and non-learners

#%%
import mne
import numpy as np
import os.path
import glob
import re
import pandas as pd

# try to enable cuda support to speed up filtering, make sure the MNE_USE_CUDA environment variable is set to true
mne.cuda.init_cuda()


DATA_DIR = "../../EEGdata/Fish_5Block"

event_dict = {
    "cat":{
        "1": 20,
        "2": 21
    }
}

data_path = os.path.join(DATA_DIR, "Tail/Learner/126670_EXP_FISH.bdf")

test_data = mne.io.read_raw_edf(data_path, preload=True)
# find the related behavioral data
participant_number = re.search(r"^(\d+)_EXP_FISH", os.path.basename(data_path))[1]
behav_path = [filename for filename in glob.glob(os.path.join(DATA_DIR, "EXP_fish2_Tomy/Cat_data/*.csv")) if participant_number in filename][0]
behav_df = pd.read_csv(behav_path)
learning_curve  = behav_df["Resultat"].rolling(20).mean()   # our in house definition of current learning performance
learning_time = (learning_curve >= 0.8).idxmax()            # using a 80% correct categorization threshold 

#%% [markdown]
# We now need to find the event times and give the same code to all stimulus presentation events since we don't want to differentiate among category 1 or 2

#%%
events = mne.find_events(test_data)
events = np.array(events)
events[events[:,2]==event_dict["cat"]["2"],2] = 20
events = events.tolist()

#%% [markdown]
# visualize data

#%%
#test_data.plot()

#%%
test_data.set_eeg_reference("average", projection=False)
test_data.filter(0.1, 50.0, n_jobs="cuda")
stim_epochs = mne.Epochs(test_data, events=events, event_id={"stimulus presentation":20}, tmin=-0.2, tmax=0.8, reject={"eeg":200-6})
# do basic cleaning by bandpass filtering, we will need to load the data
stim_epochs.load_data()
stim_epochs.resample(256)


#%% building the pytorch model

pass
