import keras
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("file", help="full path to the BDF file you wish to read and learn from")
args = parser.parse_args()

