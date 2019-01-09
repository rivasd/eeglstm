from mne.io import read_raw_edf
from mne import find_events
import os.path


def load_mne(path):
    raw = read_raw_edf(path)
    events = find_events(raw)


def patch_events(evts):
    """change the value of some triggers depending on later trigger values, to propagate backwards in time if the subject got the answer right or not
    
    Arguments:
        evts {[type]} -- [description]
    """
    pass