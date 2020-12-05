import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import numpy as np
import pickle
import sys
import os

# Set device
def get_default_device():
    if torch.cuda.is_available():
        print("Got CUDA! But still using cpu")
        return torch.device('cpu')
    else:
        print("No CUDA found")
        return torch.device('cpu')

def LtoM(L, feature_dim, max_frame_len):
    # Make mask from lengths matrix
    M = []
    for spk in lengths:
        print(spk)
        m = [[[1]*feature_dim]*l + [[0]*feature_dim]]*(max_frame_len - l) for l in spk]
        M.append(m)
    return M

# Get command line arguments
commandLineParser = argparse.ArgumentParser()
commandLineParser.add_argument('PAIRS', default=1.0, type=float, help='Specify pkl file with siamese pairs data')

args = commandLineParser.parse_args()
pkl_file = args.PAIRS

# Save the command run
if not os.path.isdir('CMDs'):
    os.mkdir('CMDs')
with open('CMDs/train_siamese.cmd', 'a') as f:
    f.write(' '.join(sys.argv)+'\n')

# set the device
device = get_default_device()

# Define a max length of number of frames in a phone instance
max_frame_len = 100
mfcc_dim = 13

# Load the data
pkl = pickle.load(open(pkl_file, "rb"))
ind = [k for k in pkl if pkl[k][0].shape[0] < max_len and pkl[k][1].shape[0] < max_frame_len]
X1_raw = [pkl[k][0] for k in inds]
X2_raw = [pkl[k][1] for k in inds]
y = [pkl[k][2] for k in inds]
lengths1 = [pkl[k][0].shape[0] for k in inds]
lengths2 = [pkl[k][1].shape[0] for k in inds]

# Make masks
M1 = LtoM(lengths1, mfcc_dim, max_frame_len)
