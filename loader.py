import argparse
import keras
import pyedflib

parser = argparse.ArgumentParser()
parser.add_argument("file", help="full path to the BDF file you wish to read and learn from")
args = parser.parse_args()

# load the specified file into memory
#TODO: might be too large? find way to load in steps or distributed

bdfData = pyedflib.EdfReader(args.file)
n = bdfData.signals_in_file


