
import numpy as np




class PreProcessor:
    """
    A simple class to encapsulate a particular EEG pre-processing pipeline, each instance providing a way to reliably do an undo the preprocessing
    """

    def __init__(self, init_block_size=1000, alpha=0.999, eps=1e-10):
        self.init_block_size    = init_block_size
        self.alpha              = alpha
        self.eps                = eps

